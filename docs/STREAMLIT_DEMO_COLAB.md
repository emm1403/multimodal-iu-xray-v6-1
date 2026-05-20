# Running the Streamlit demo in Colab

The Streamlit demo is located in:

```text
demo/streamlit_app/
```

It performs live inference with an uploaded frontal chest X-ray. The interface intentionally shows only the live inference page.

## 1. Enter the app folder

```python
%cd /content/drive/MyDrive/Demo/multimodal-iu-xray-v6-1/demo/streamlit_app
```

If running from a cloned repository in Colab:

```python
%cd /content/multimodal-iu-xray-v6-1/demo/streamlit_app
```

## 2. Install dependencies

```python
!pip install -q streamlit torchxrayvision scikit-learn pillow pandas numpy matplotlib
```

## 3. Launch Streamlit

```python
!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &>/content/logs.txt &
```

## 4. Open with Cloudflare Tunnel

```python
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!./cloudflared tunnel --url http://localhost:8501
```

Open the generated `trycloudflare.com` link.

## What to show during the demo

1. Upload a frontal chest X-ray from the held-out test demo set.
2. Paste the corresponding clinical indication from `test_demo_cases_short_v6_1.csv`.
3. Run inference.
4. Report the calibrated probability, threshold, and final prediction.

The text indication is shown as context. In V6.1, the selected weighted fusion assigned `image_weight = 1.0` and `text_weight = 0.0`, so the final decision is based on the calibrated visual model.

This app is for academic demonstration only and is not intended for clinical diagnosis.
