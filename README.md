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

Download official checkpoints as needed:

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

Backend:

```powershell
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Frontend:

```powershell
cd frontend
npm.cmd run dev
```

Open:

```text
http://127.0.0.1:5173
```

## Cloud Studio Preview

The repository includes `.vscode/preview.yml` for cloud preview:

- Backend app: port `8000`
- Frontend app: port `5173`

The frontend uses Vite proxy rules for `/api`, `/data`, and `/outputs`, so the browser can access the backend through the frontend preview origin. If your Cloud Studio workspace keeps this repository inside a parent folder named `the-devil-is-in-the-details`, put the same preview config at the workspace root and set app roots to `./the-devil-is-in-the-details` and `./the-devil-is-in-the-details/frontend`.

## Notes

This is a pretrained inference reproduction, not full OpenImages retraining. Learned-model bpp is estimated from likelihoods:

```text
bpp = (sum -log2 p(y_hat) + sum -log2 p(z_hat)) / (H * W)
```

The model checkpoints are intentionally excluded from git because they are large and exceed ordinary GitHub file limits.
