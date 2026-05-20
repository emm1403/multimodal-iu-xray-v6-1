# Multimodal IU X-Ray Abnormality Detection — V6.1

This repository contains a reproducible academic project for binary classification of frontal chest X-rays from the Indiana University Chest X-ray Collection. The final V6.1 system evaluates whether a study should be flagged as a possible non-incidental abnormality.

The project started as a multimodal pipeline using chest X-ray images plus clinical indication text. After validation, the final V6.1 weighted-fusion selector assigned `image_weight = 1.0` and `text_weight = 0.0`; therefore, the recommended final model is the calibrated visual model. The clinical indication is still displayed in the demo as contextual information.

> This project is for research and academic demonstration only. It is not intended for clinical diagnosis or medical decision-making.

---

## Repository contents

```text
.
├── README.md
├── requirements.txt
├── data/
│   └── README_dataset.md
├── notebooks/
│   ├── diagnostico_multimodal_iu_xray_v6_1_colab.ipynb
│   ├── diagnostico_multimodal_iu_xray_v6_1_reporting_github_demo_colab.ipynb
│   └── extract_test_demo_set_v6_1_colab.ipynb
├── src/
│   ├── metrics.py
│   └── make_report.py
├── demo/
│   ├── README.md
│   └── streamlit_app/
│       ├── app.py
│       ├── requirements_streamlit.txt
│       ├── model/
│       ├── assets/
│       └── demo_data/
├── docs/
│   ├── pipeline.md
│   ├── presentation_summary_v6_1.md
│   ├── demo_script.md
│   ├── STREAMLIT_DEMO_COLAB.md
│   ├── GITHUB_UPLOAD_STEPS.md
│   └── REPRODUCIBILITY_CHECKLIST.md
├── results/
└── assets/
```

---

## Project pipeline

| Stage | Description |
|---|---|
| Data input | Frontal chest X-ray image and clinical indication/context. |
| Preprocessing | UID-based split, image resizing/normalization, label audit, exclusion of ambiguous and benign nodular cases. |
| Model | DenseNet-based image model, Platt calibration, text/fusion evaluated as exploratory branches. |
| Training | Train/validation/calibration/threshold/test partitions, plus UID-based cross-validation. |
| Evaluation | ROC-AUC, PR-AUC, Brier score, sensitivity, specificity, FPR, confusion matrix, bootstrap CI, McNemar test. |
| Final output | Calibrated abnormality probability and threshold-based research flag. |

---

## Main V6.1 result

Recommended model: `image_calibrated`  
Operational threshold: `min_fpr_recall_0.90`  
Evaluation level: held-out test set aggregated by UID.

| Metric | Value |
|---|---:|
| ROC-AUC | 0.857 |
| PR-AUC | 0.744 |
| Brier score | 0.132 |
| Sensitivity | 92.5% |
| Specificity | 51.2% |
| FPR | 48.8% |

The stricter V6.1 labeling reduced benign/calcified nodular false positives in the target definition. The text-only branch was weak, and the final weighted fusion selected the image model only.

---

## Dataset

The raw images are not included in this repository due to size and licensing considerations. Download instructions are provided in:

```text
data/README_dataset.md
```

Use the Indiana University Chest X-ray Collection via Kaggle/KaggleHub and keep the expected structure used by the training notebook.

---

## Reproducing the full experiment

Recommended environment: Google Colab with GPU.

1. Download or mount the IU X-Ray dataset according to `data/README_dataset.md`.
2. Open and run:

```text
notebooks/diagnostico_multimodal_iu_xray_v6_1_colab.ipynb
```

3. Save the generated experiment archive:

```text
outputs_multimodal_iuxray_v6_1.zip
```

4. Generate final figures, tables, and repository assets with:

```text
notebooks/diagnostico_multimodal_iu_xray_v6_1_reporting_github_demo_colab.ipynb
```

5. Optional: extract test-set examples for a live presentation with:

```text
notebooks/extract_test_demo_set_v6_1_colab.ipynb
```

---

## Running the Streamlit live demo locally

```bash
cd demo/streamlit_app
pip install -r requirements_streamlit.txt
streamlit run app.py
```

The app loads the V6.1 checkpoint and calibration files included in `demo/streamlit_app/model/`.

---

## Running the Streamlit demo in Colab

See:

```text
docs/STREAMLIT_DEMO_COLAB.md
```

Short version:

```python
%cd /content/multimodal-iu-xray-v6-1/demo/streamlit_app
!pip install -q streamlit torchxrayvision scikit-learn pillow pandas numpy matplotlib
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &>/content/logs.txt &
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8501
```

Open the generated `trycloudflare.com` URL.

---

## Suggested live demonstration

Use held-out test-set examples extracted with `extract_test_demo_set_v6_1_colab.ipynb`.

Recommended order:

1. True Negative: normal case correctly classified.
2. True Positive: abnormal case correctly flagged.
3. False Positive or False Negative: model limitation.

During the demo, report:

- uploaded X-ray,
- clinical indication/context,
- calibrated probability,
- operational threshold,
- final prediction.

---

## Limitations

- This is a research prototype, not a clinical device.
- The final V6.1 decision relies on the calibrated visual model because indication text did not improve performance robustly.
- The model was trained and evaluated on IU X-Ray data; external validation is required before any real-world use.
- False positives and false negatives remain clinically important limitations.
