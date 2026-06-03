from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KODAK_DIR = ROOT / "data" / "kodak"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "cache"
STF_CHECKPOINT_DIR = ROOT / "checkpoints" / "stf"
ZOO_PACKAGE_DIR = ROOT / "external" / "compressai_zoo_pkg"
TORCH_HUB_CHECKPOINTS = Path.home() / ".cache" / "torch" / "hub" / "checkpoints"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STF_GOOGLE_DRIVE = {
    "cnn-wam": {
        "0.0018": "1RPdtyxTtfosuDe1-xtl5JzvnCU2vYnHD",
        "0.0035": "1L7xvei3Wj4BeSQ3lDBL-pyjEy13RKsjn",
        "0.0067": "1DDCFFWBUa5cYOgJ9D9HPcwoOigzoJK31",
        "0.025": "1LrAWPlBE6WJUfjiDPGFO8ANSaP5BFEQI",
    },
    "stf": {
        "0.0018": "15ujpSjif628iwVEay3mAWN-Vyqls3r23",
        "0.0035": "1OFzZoEaofNgsimBuOPHtgOJiGsR_RS-M",
        "0.0067": "1SjhqcKyP3SqVm4yhJQslJ6HgY1E8FcBL",
        "0.013": "1mupv4vcs8wpNdXCPclXghliikJyYjgj-",
        "0.025": "1rsYgEYuqSYBIA4rfvAjXtVSrjXOzkJlB",
        "0.0483": "1cH5cR-0VdsQqCchyN3DO62Sx0WGjv1h8",
    },
}

MODEL_NAMES = {
    "stf": "STF (Ours)",
    "cnn-wam": "CNN+WAM (Ours)",
    "mbt2018-mean": "Minnen2018 mean-scale",
    "cheng2020-attn": "Cheng2020 attention",
    "jpeg": "JPEG",
}

MODEL_FAMILIES = {
    "stf": "paper",
    "cnn-wam": "paper",
    "mbt2018-mean": "learned_sota",
    "cheng2020-attn": "learned_sota",
    "jpeg": "traditional",
}

CHENG2020_MSE_WEIGHTS = {
    1: "cheng2020_attn-mse-1-465f2b64.pth.tar",
    2: "cheng2020_attn-mse-2-e0805385.pth.tar",
    3: "cheng2020_attn-mse-3-2d07bbdf.pth.tar",
    4: "cheng2020_attn-mse-4-f7b0ccf2.pth.tar",
    5: "cheng2020_attn-mse-5-26c8920e.pth.tar",
    6: "cheng2020_attn-mse-6-730501f2.pth.tar",
}


def _lambda_tag(quality: str) -> str:
    quality = str(quality)
    if quality.startswith("0."):
        return quality.split(".", 1)[1]
    return quality.replace(".", "")


def _checkpoint_for(model_id: str, quality: str) -> Path:
    if model_id == "cnn-wam":
        return STF_CHECKPOINT_DIR / f"cnn_{_lambda_tag(quality)}.pth.tar"
    if model_id == "stf":
        return STF_CHECKPOINT_DIR / f"stf_{_lambda_tag(quality)}.pth.tar"
    raise ValueError(f"Model {model_id!r} does not use STF checkpoints.")


def _quality_entry(value: str, label: str, available: bool, note: str = "") -> dict[str, Any]:
    return {"value": value, "label": label, "available": available, "note": note}


def available_models() -> list[dict[str, Any]]:
    zoo_available = ZOO_PACKAGE_DIR.exists()
    models: list[dict[str, Any]] = []

    for model_id in ("stf", "cnn-wam"):
        qualities = []
        for value in STF_GOOGLE_DRIVE[model_id]:
            checkpoint = _checkpoint_for(model_id, value)
            if checkpoint.exists():
                note = f"已检测到 {checkpoint.name}"
                available = True
            else:
                file_id = STF_GOOGLE_DRIVE[model_id][value]
                note = (
                    f"未下载权重。可用 gdown 下载 Google Drive 文件 {file_id} "
                    f"到 checkpoints/stf/{checkpoint.name}"
                )
                available = False
            qualities.append(_quality_entry(value, f"lambda={value}", available, note))
        models.append(
            {
                "id": model_id,
                "name": MODEL_NAMES[model_id],
                "family": MODEL_FAMILIES[model_id],
                "description": "论文官方预训练权重，前向推理计算重建图、bpp、PSNR 和 MS-SSIM。",
                "qualities": qualities,
            }
        )

    zoo_note = (
        "使用隔离安装的 CompressAI model zoo；首次运行会自动下载官方预训练权重。"
        if zoo_available
        else "未检测到 external/compressai_zoo_pkg；运行 pip install --target external/compressai_zoo_pkg --no-deps compressai==1.2.8。"
    )
    models.append(
        {
            "id": "mbt2018-mean",
            "name": MODEL_NAMES["mbt2018-mean"],
            "family": MODEL_FAMILIES["mbt2018-mean"],
            "description": "旧学习式 SOTA baseline：Joint Autoregressive and Hierarchical Priors，mean-scale hyperprior 版本。",
            "qualities": [
                _quality_entry(str(q), f"quality={q}", zoo_available, zoo_note)
                for q in range(1, 9)
            ],
        }
    )
    cheng_note = (
        "可选 baseline。当前只在检测到 Torch cache 中已有 Cheng2020 权重时启用；"
        "本机自动下载该权重较慢，主 demo 已使用可跑通的 Minnen2018 baseline。"
    )
    models.append(
        {
            "id": "cheng2020-attn",
            "name": MODEL_NAMES["cheng2020-attn"],
            "family": MODEL_FAMILIES["cheng2020-attn"],
            "description": "旧学习式 SOTA baseline：Cheng2020 attention 模型，来自 CompressAI zoo。",
            "qualities": [
                _quality_entry(
                    str(q),
                    f"quality={q}",
                    zoo_available and (TORCH_HUB_CHECKPOINTS / CHENG2020_MSE_WEIGHTS[q]).exists(),
                    cheng_note,
                )
                for q in range(1, 7)
            ],
        }
    )
    models.append(
        {
            "id": "jpeg",
            "name": MODEL_NAMES["jpeg"],
            "family": MODEL_FAMILIES["jpeg"],
            "description": "传统有损压缩基线，用 PIL/libjpeg 编码后再解码展示。",
            "qualities": [
                _quality_entry(str(q), f"Q={q}", True, "JPEG 质量因子，数值越大码率越高。")
                for q in (20, 35, 50, 70, 90)
            ],
        }
    )
    return models


def list_kodak_images() -> list[dict[str, Any]]:
    items = []
    for path in sorted(KODAK_DIR.glob("kodim*.png")):
        with Image.open(path) as image:
            width, height = image.size
        items.append(
            {
                "id": path.stem,
                "name": path.stem,
                "url": f"/data/{path.name}",
                "width": width,
                "height": height,
            }
        )
    return items


def _find_model(model_id: str) -> dict[str, Any]:
    for model in available_models():
        if model["id"] == model_id:
            return model
    raise ValueError(f"Unknown model: {model_id}")


def _assert_quality_available(model_id: str, quality: str) -> None:
    model = _find_model(model_id)
    for entry in model["qualities"]:
        if entry["value"] == str(quality):
            if entry["available"]:
                return
            raise RuntimeError(entry["note"])
    raise ValueError(f"Quality {quality!r} is not defined for {model['name']}.")


def _image_path(image_id: str) -> Path:
    safe_id = Path(image_id).stem
    path = KODAK_DIR / f"{safe_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Kodak image not found: {image_id}")
    return path


def _source_signature(model_id: str, quality: str) -> str:
    if model_id in {"stf", "cnn-wam"}:
        checkpoint = _checkpoint_for(model_id, quality)
        if checkpoint.exists():
            stat = checkpoint.stat()
            return f"{checkpoint.name}:{stat.st_mtime_ns}:{stat.st_size}"
    if model_id in {"mbt2018-mean", "cheng2020-attn"}:
        return "compressai-zoo-1.2.8"
    return "pillow-jpeg"


def _cache_key(image_id: str, model_id: str, quality: str) -> str:
    payload = f"{image_id}|{model_id}|{quality}|{_source_signature(model_id, quality)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]


def _relative_url(path: Path) -> str:
    return "/" + path.relative_to(ROOT).as_posix()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_codec(image_id: str, model_id: str, quality: str, force: bool = False) -> dict[str, Any]:
    quality = str(quality)
    image_path = _image_path(image_id)
    _assert_quality_available(model_id, quality)

    key = _cache_key(image_id, model_id, quality)
    result_dir = CACHE_DIR / key
    result_path = result_dir / "result.json"
    if result_path.exists() and not force:
        result = _read_json(result_path)
        result["cached"] = True
        return result

    result_dir.mkdir(parents=True, exist_ok=True)
    if model_id == "jpeg":
        result = _run_jpeg(image_path, model_id, quality, result_dir, key)
    else:
        result = _run_worker(image_path, model_id, quality, result_dir, key)

    _write_json(result_path, result)
    return result


def _run_jpeg(image_path: Path, model_id: str, quality: str, result_dir: Path, key: str) -> dict[str, Any]:
    import numpy as np
    import torch
    from pytorch_msssim import ms_ssim
    from torchvision.transforms.functional import to_tensor

    started = time.perf_counter()
    quality_int = int(quality)
    recon_path = result_dir / "reconstruction.png"
    encoded_path = result_dir / "encoded.jpg"

    original = Image.open(image_path).convert("RGB")
    width, height = original.size
    original.save(encoded_path, "JPEG", quality=quality_int, optimize=True)
    reconstructed = Image.open(encoded_path).convert("RGB")
    reconstructed.save(recon_path)

    x = to_tensor(original).unsqueeze(0)
    x_hat = to_tensor(reconstructed).unsqueeze(0)
    mse = torch.mean((x - x_hat) ** 2).item()
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    msssim = ms_ssim(x_hat, x, data_range=1.0).item()
    bpp = encoded_path.stat().st_size * 8.0 / (width * height)

    diff = torch.mean(torch.abs(x - x_hat), dim=1, keepdim=True)
    arr = diff.squeeze().mul(255).clamp(0, 255).byte().numpy()
    diff_image = Image.fromarray(arr, mode="L")
    heatmap = Image.new("RGB", diff_image.size)
    heatmap.paste(Image.fromarray(np.asarray(diff_image), mode="L"))
    heatmap_path = result_dir / "bit_heatmap.png"
    diff_image.save(heatmap_path)

    return {
        "id": key,
        "image_id": image_path.stem,
        "model_id": model_id,
        "model_name": MODEL_NAMES[model_id],
        "quality": quality,
        "quality_label": f"Q={quality}",
        "reconstruction_url": _relative_url(recon_path),
        "heatmap_url": _relative_url(heatmap_path),
        "bpp": bpp,
        "psnr": psnr,
        "ms_ssim": msssim,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "cached": False,
        "notes": "JPEG baseline; heatmap is absolute reconstruction error rather than likelihood bit allocation.",
    }


def _run_worker(image_path: Path, model_id: str, quality: str, result_dir: Path, key: str) -> dict[str, Any]:
    started = time.perf_counter()
    cmd = [
        sys.executable,
        "-m",
        "backend.inference_worker",
        "--image",
        str(image_path),
        "--model",
        model_id,
        "--quality",
        quality,
        "--output-dir",
        str(result_dir),
        "--result-id",
        key,
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        cmd,
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=900,
    )
    worker_json = result_dir / "worker_result.json"
    if completed.returncode != 0 or not worker_json.exists():
        detail = (completed.stderr or completed.stdout or "Unknown worker failure").strip()
        raise RuntimeError(f"Inference worker failed for {model_id} {quality}: {detail[-3000:]}")

    result = _read_json(worker_json)
    result["elapsed_ms"] = (time.perf_counter() - started) * 1000
    result["cached"] = False
    return result


def get_cached_result(result_id: str) -> dict[str, Any]:
    result_path = CACHE_DIR / Path(result_id).name / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(result_id)
    return _read_json(result_path)


def batch_eval(
    selections: list[dict[str, str]],
    image_ids: list[str] | None = None,
    max_images: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    if image_ids is None:
        image_ids = [item["id"] for item in list_kodak_images()]
    if max_images is not None and max_images > 0:
        image_ids = image_ids[:max_images]

    curves = []
    errors = []
    for selection in selections:
        model_id = selection["model_id"]
        quality = str(selection["quality"])
        points = []
        for image_id in image_ids:
            try:
                points.append(run_codec(image_id, model_id, quality, force=force))
            except Exception as exc:  # Keep long batch runs useful even if one point fails.
                errors.append(
                    {
                        "image_id": image_id,
                        "model_id": model_id,
                        "quality": quality,
                        "error": str(exc),
                    }
                )
        if points:
            avg = {
                "model_id": model_id,
                "model_name": MODEL_NAMES.get(model_id, model_id),
                "quality": quality,
                "quality_label": points[0].get("quality_label", quality),
                "num_images": len(points),
                "bpp": sum(p["bpp"] for p in points) / len(points),
                "psnr": sum(p["psnr"] for p in points) / len(points),
                "ms_ssim": sum(p["ms_ssim"] for p in points) / len(points),
                "elapsed_ms": sum(p["elapsed_ms"] for p in points),
            }
            curves.append({"average": avg, "points": points})
    return {"image_ids": image_ids, "curves": curves, "errors": errors}
