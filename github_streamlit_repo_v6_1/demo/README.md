# Demo

This folder contains the live inference demo for the final V6.1 model.

## Streamlit app

```bash
cd streamlit_app
pip install -r requirements_streamlit.txt
streamlit run app.py
```

The app allows uploading a frontal chest X-ray and entering clinical indication/context. It outputs the calibrated visual probability, selected threshold, and final binary research flag.

## Recommended live presentation

Use examples extracted from the held-out test set with:

```text
notebooks/extract_test_demo_set_v6_1_colab.ipynb
```

Show one true negative, one true positive, and one limitation case such as false positive or false negative.

This system is for academic demonstration only and is not a clinical device.
