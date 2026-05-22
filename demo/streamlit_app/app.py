"""
Streamlit live-only demo for the IU X-ray V6.1 project.

This application loads the trained V6.1 chest X-ray neural network,
reconstructs Platt calibration from the calibration split, and runs live
inference on an uploaded frontal chest X-ray.

Important: this is an academic demonstration, not a medical device.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import streamlit as st

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchxrayvision as xrv
except Exception as exc:  # pragma: no cover - user-facing runtime guard
    xrv = None
    XRV_IMPORT_ERROR = exc
else:
    XRV_IMPORT_ERROR = None

from sklearn.linear_model import LogisticRegression


APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR / "model"

MODEL_PATH = MODEL_DIR / "main_best_image_model_v6_1.pt"
CONFIG_PATH = MODEL_DIR / "config_v6_1.json"
MANIFEST_PATH = MODEL_DIR / "manifest_v6_1.json"
THRESHOLDS_PATH = MODEL_DIR / "thresholds_main_v6_1.csv"
CALIBRATION_PATH = MODEL_DIR / "predictions_calibration_main_v6_1.csv"
FUSION_WEIGHT_PATH = MODEL_DIR / "fusion_weight_selected_main_v6_1.json"

IMAGE_SIZE = 320
FEATURE_DIM = 256
DROPOUT = 0.25
EPS = 1e-6


st.set_page_config(
    page_title="IU X-ray V6.1 Live Demo",
    page_icon="🩻",
    layout="centered",
)


class ImageEncoder(nn.Module):
    """Image encoder matching the V6.1 training notebook architecture."""

    def __init__(self, out_dim: int = FEATURE_DIM):
        super().__init__()
        if xrv is None:
            raise RuntimeError(
                "torchxrayvision is required for this model because V6.1 used "
                "the xrv_densenet121 backbone. Install it with: pip install torchxrayvision"
            ) from XRV_IMPORT_ERROR

        cnn = xrv.models.DenseNet(weights=None)
        self.features = cnn.features
        in_dim = cnn.classifier.in_features
        self.target_layer = getattr(self.features, "denseblock4", self.features[-1])

        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        feats = F.relu(feats, inplace=True)
        pooled = F.adaptive_avg_pool2d(feats, (1, 1)).flatten(1)
        return self.proj(pooled)


class ImageOnlyClassifier(nn.Module):
    """Binary classifier used in V6.1."""

    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder(out_dim=FEATURE_DIM)
        self.head = nn.Sequential(
            nn.LayerNorm(FEATURE_DIM),
            nn.Dropout(DROPOUT),
            nn.Linear(FEATURE_DIM, 1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        emb = self.image_encoder(image)
        return self.head(emb).squeeze(1)


def logit_transform(p: np.ndarray | float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1.0 - EPS)
    return np.log(p / (1.0 - p))


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess a user-uploaded image as in the xrv_densenet121 pipeline."""
    if xrv is None:
        raise RuntimeError("torchxrayvision is not installed.")

    img = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(img).astype(np.float32)
    arr = xrv.datasets.normalize(arr, 255)
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    return tensor


@st.cache_data(show_spinner=False)
def load_tables() -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict]:
    thresholds = pd.read_csv(THRESHOLDS_PATH)
    cal_df = pd.read_csv(CALIBRATION_PATH)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    fusion_weight = json.loads(FUSION_WEIGHT_PATH.read_text(encoding="utf-8"))
    return thresholds, cal_df, config, manifest, fusion_weight


@st.cache_resource(show_spinner=True)
def load_model_and_calibrator():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if xrv is None:
        raise RuntimeError(
            "torchxrayvision is required but could not be imported. "
            f"Original error: {type(XRV_IMPORT_ERROR).__name__}: {XRV_IMPORT_ERROR}"
        )

    _, cal_df, *_ = load_tables()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageOnlyClassifier().to(device)
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Reconstruct Platt calibration from the stored calibration split.
    y_cal = cal_df["label"].astype(int).values
    p_raw = cal_df["prob_image_raw"].astype(float).values
    calibrator = LogisticRegression(max_iter=1000, solver="lbfgs")
    calibrator.fit(logit_transform(p_raw).reshape(-1, 1), y_cal)

    return model, calibrator, device


def calibrate_probability(calibrator: LogisticRegression, p_raw: float) -> float:
    return float(calibrator.predict_proba(logit_transform(p_raw).reshape(1, -1))[0, 1])


def get_threshold(model_name: str, target_recall: float) -> float:
    thresholds, *_ = load_tables()
    row = thresholds[
        (thresholds["model"] == model_name)
        & (thresholds["threshold_type"] == f"min_fpr_recall_{target_recall:.2f}")
    ]
    if row.empty:
        row = thresholds[
            (thresholds["model"] == model_name)
            & (thresholds["target_recall"].round(2) == round(target_recall, 2))
        ]
    if row.empty:
        raise ValueError(f"No threshold found for {model_name}, recall={target_recall}")
    return float(row.iloc[0]["threshold"])


def predict(image: Image.Image, target_recall: float) -> Dict[str, float | str | bool]:
    model, calibrator, device = load_model_and_calibrator()
    x = preprocess_image(image).to(device)
    with torch.no_grad():
        raw_logit = float(model(x).detach().cpu().item())
        prob_raw = float(torch.sigmoid(torch.tensor(raw_logit)).item())
    prob_cal = calibrate_probability(calibrator, prob_raw)
    threshold = get_threshold("image_calibrated", target_recall)
    is_positive = prob_cal >= threshold
    return {
        "raw_logit": raw_logit,
        "prob_raw": prob_raw,
        "prob_calibrated": prob_cal,
        "threshold": threshold,
        "target_recall": target_recall,
        "is_positive": is_positive,
        "decision": "Abnormal / non-incidental finding suspected" if is_positive else "No non-incidental abnormality flagged",
    }


def render_result(prob: float, threshold: float, decision: str):
    st.markdown("### Prediction result")
    col1, col2 = st.columns(2)
    col1.metric("Calibrated probability", f"{prob:.3f}")
    col2.metric("Operational threshold", f"{threshold:.3f}")
    st.progress(min(max(prob, 0.0), 1.0))
    if prob >= threshold:
        st.error(f"Decision: {decision}")
    else:
        st.success(f"Decision: {decision}")


# ----------------------------- UI -----------------------------

st.title("🩻 Live inference with an uploaded chest X-ray")
st.caption("IU X-ray V6.1 academic neural network demo. Not for clinical use.")

thresholds, cal_df, config, manifest, fusion_weight = load_tables()

with st.sidebar:
    st.header("Inference settings")
    target_recall = st.selectbox(
        "Operational threshold",
        options=[0.80, 0.85, 0.90],
        index=2,
        format_func=lambda x: f"Minimum-FPR threshold for recall ≥ {x:.2f}",
    )
    st.markdown("---")
    st.write("**V6.1 final model**")
    st.write(f"Backbone: `{config.get('backbone', 'xrv_densenet121')}`")
    st.write(f"Image size: `{config.get('image_size', IMAGE_SIZE)} × {config.get('image_size', IMAGE_SIZE)}`")
    st.write(f"Positive prevalence: `{manifest.get('uid_prevalence', 0):.3f}` by UID")
    st.write(
        f"Fusion selected: image `{fusion_weight.get('image_weight', 1.0):.2f}` + "
        f"text `{fusion_weight.get('text_weight', 0.0):.2f}`"
    )
    st.info("In V6.1, the selected text weight was 0. The live prediction uses the calibrated visual model.")

if xrv is None:
    st.error(
        "torchxrayvision is not installed. Install the requirements first: "
        "`pip install -r requirements_streamlit.txt`."
    )
    st.stop()

uploaded = st.file_uploader("Upload a frontal chest X-ray image", type=["png", "jpg", "jpeg"])
indication = st.text_area(
    "Clinical indication/context",
    placeholder="Example: Dyspnea, chest pain, preoperative evaluation...",
)
st.caption(
    "The clinical indication is displayed as contextual information. In V6.1, "
    "the final selected model uses image weight = 1.0, so the text does not change the decision."
)

if uploaded is None:
    st.info("Upload a PNG/JPG frontal chest X-ray to run the neural network.")
else:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", use_container_width=True)
    with st.spinner("Running neural network inference..."):
        try:
            result = predict(image, target_recall)
        except Exception as exc:
            st.exception(exc)
            st.stop()

    render_result(result["prob_calibrated"], result["threshold"], result["decision"])

    with st.expander("Technical details"):
        st.json(result)
        st.write("Clinical indication/context:", indication or "[empty]")

st.markdown("---")
st.caption(
    "This interface is intended for academic demonstration only. It is not validated "
    "for diagnosis or clinical decision-making."
)
