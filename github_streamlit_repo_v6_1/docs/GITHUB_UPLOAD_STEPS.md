# GitHub upload steps

This repository is prepared to be uploaded to GitHub and includes source code, notebooks, results, documentation, dataset instructions, and a Streamlit live demo.

## Option A: Upload using the GitHub web interface

1. Create a new repository in GitHub.
2. Name suggestion: `multimodal-iu-xray-v6-1`.
3. Keep it public or private according to your course requirements.
4. Do not initialize with another README if you are uploading this package, because a README is already included.
5. Unzip `multimodal_iuxray_v6_1_github_live_only_ready.zip` on your computer.
6. Open the extracted folder and upload all files/folders to the GitHub repository.
7. Wait until the upload finishes.
8. Copy the repository URL and use it as your GitHub Link.

## Option B: Upload using Git commands

```bash
git init
git add .
git commit -m "Initial reproducible V6.1 IU X-Ray project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/multimodal-iu-xray-v6-1.git
git push -u origin main
```

## Important notes

- Raw IU X-Ray images are not included in this repository.
- Dataset download instructions are provided in `data/README_dataset.md`.
- The trained model checkpoint for the demo is included under `demo/streamlit_app/model/`.
- The checkpoint file is below GitHub's normal file size limit, so Git LFS is not required unless your repository grows later.
