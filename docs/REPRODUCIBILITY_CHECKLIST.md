# Reproducibility checklist

The repository includes the elements requested for a reproducible academic project.

## Required elements

| Requirement | Location |
|---|---|
| Source code | `src/`, `demo/streamlit_app/app.py` |
| README file | `README.md` |
| Dataset link or instructions | `data/README_dataset.md` |
| Execution steps | `README.md`, `docs/STREAMLIT_DEMO_COLAB.md`, notebooks |
| Training notebook | `notebooks/diagnostico_multimodal_iu_xray_v6_1_colab.ipynb` |
| Reporting notebook | `notebooks/diagnostico_multimodal_iu_xray_v6_1_reporting_github_demo_colab.ipynb` |
| Demo set extraction notebook | `notebooks/extract_test_demo_set_v6_1_colab.ipynb` |
| Demo app | `demo/streamlit_app/` |
| Results/tables/figures | `results/`, `assets/`, `docs/` |

## Suggested workflow for reproduction

1. Download the IU X-Ray dataset following `data/README_dataset.md`.
2. Run the V6.1 training notebook in Colab with GPU.
3. Save `outputs_multimodal_iuxray_v6_1.zip`.
4. Run the reporting notebook to regenerate tables and figures.
5. Run the Streamlit app for live inference.

## Demo-only workflow

If the goal is only to test the deployed demo, the user can run:

```bash
cd demo/streamlit_app
pip install -r requirements_streamlit.txt
streamlit run app.py
```

The included checkpoint and calibration files are enough to run the interface without retraining.
