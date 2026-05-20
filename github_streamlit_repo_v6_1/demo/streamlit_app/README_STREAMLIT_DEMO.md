# IU X-ray V6.1 Streamlit Demo — Live Only

This version contains only the live inference page. The retrospective validation and presentation tabs were removed for a cleaner project demonstration.

## Run locally

```bash
pip install -r requirements_streamlit.txt
streamlit run app.py
```

## Run in Colab

```python
%cd /content/streamlit_demo_v6_1_live_only
!pip install -q streamlit torchxrayvision scikit-learn pillow pandas numpy matplotlib
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &>/content/logs.txt &
```

Then expose it using localtunnel or Cloudflare Tunnel.

## Notes

- This is an academic demonstration, not a medical device.
- V6.1 selected image weight = 1.0 and text weight = 0.0, so the clinical indication is shown only as context.
