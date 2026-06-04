# STF Image Compression Demo

Local reproduction demo for **The Devil Is in the Details: Window-Based Attention for Image Compression**. The project uses official pretrained STF/CNN+WAM checkpoints, Kodak images, CompressAI baselines, and a React dashboard to compare local-detail reconstruction quality.

## What It Shows

- Original image, paper method reconstruction, and baseline reconstruction.
- Local crop zoom for text, edge, and texture details.
- bpp, PSNR, MS-SSIM, runtime, and cached result status.
- RD points for paper methods, CompressAI learned baselines, and JPEG.
- Likelihood-based bit allocation heatmap for learned models.

## Project Layout

```text
backend/       FastAPI inference and metric service
frontend/      React + Vite dashboard
scripts/       Dataset and checkpoint download helpers
data/kodak/    Kodak images after download
checkpoints/   Official checkpoints after download
outputs/       Cached reconstructions and metrics
```

## Setup

Install backend requirements:

```powershell
python -m pip install -r backend/requirements.txt
python -m pip install --target external/compressai_zoo_pkg --no-deps compressai==1.2.8
```

Download Kodak:

```powershell
python scripts/download_kodak.py
```

Upload or copy the demo checkpoints to:

```text
checkpoints/stf/cnn_0018.pth.tar
checkpoints/stf/cnn_0035.pth.tar
checkpoints/stf/stf_0035.pth.tar
```

Optional: download other official checkpoints as needed:

```powershell
python scripts/download_stf_checkpoints.py --model stf --lambdas 0.0035
python scripts/download_stf_checkpoints.py --model cnn --lambdas 0.0018 0.0035
```

Install frontend dependencies:

```powershell
cd frontend
npm.cmd install
```

## Run

Single-port production-style preview:

```powershell
cd frontend
npm.cmd install
npm.cmd run build
cd ..
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Development mode with Vite is still available:

```powershell
cd frontend
npm.cmd run dev
```

Then open:

```text
http://127.0.0.1:5173
```

## Cloud Studio Preview

The repository includes `.vscode/preview.yml` for cloud preview. It uses a single-port setup:

- Web demo and API: port `8000`

Cloud Studio builds `frontend/dist` first, then FastAPI serves both the React page and API from port `8000`. This avoids iframe/proxy issues that can happen with Vite dev-server previews.

For Cloud Studio imports, upload the three checkpoint files above into `checkpoints/stf/` before running paper-model inference. The preview will still open without them, but STF/CNN+WAM quality entries stay disabled until the files are present.

## Notes

This is a pretrained inference reproduction, not full OpenImages retraining. Learned-model bpp is estimated from likelihoods:

```text
bpp = (sum -log2 p(y_hat) + sum -log2 p(z_hat)) / (H * W)
```

Demo checkpoints are excluded from Git because they exceed ordinary GitHub file limits.
