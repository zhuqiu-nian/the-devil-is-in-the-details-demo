from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from pytorch_msssim import ms_ssim
from torchvision.transforms.functional import to_pil_image, to_tensor

ROOT = Path(__file__).resolve().parents[1]
STF_DIR = ROOT / "external" / "stf"
ZOO_DIR = ROOT / "external" / "compressai_zoo_pkg"
CHECKPOINT_DIR = ROOT / "checkpoints" / "stf"

MODEL_NAMES = {
    "stf": "STF (Ours)",
    "cnn-wam": "CNN+WAM (Ours)",
    "mbt2018-mean": "Minnen2018 mean-scale",
    "cheng2020-attn": "Cheng2020 attention",
}


def lambda_tag(quality: str) -> str:
    quality = str(quality)
    if quality.startswith("0."):
        return quality.split(".", 1)[1]
    return quality.replace(".", "")


def patch_compressai_extensions() -> None:
    """Bypass C++/rANS modules; this demo only needs differentiable forward passes."""
    cxx = types.ModuleType("compressai._CXX")
    cxx.pmf_to_quantized_cdf = lambda *args, **kwargs: []
    sys.modules["compressai._CXX"] = cxx

    ans = types.ModuleType("compressai.ans")

    class DummyAns:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def encode_with_indexes(self, *args: Any, **kwargs: Any) -> None:
            pass

        def flush(self) -> bytes:
            return b""

        def set_stream(self, *args: Any, **kwargs: Any) -> None:
            pass

        def decode_stream(self, *args: Any, **kwargs: Any) -> list[int]:
            return []

    ans.BufferedRansEncoder = DummyAns
    ans.RansDecoder = DummyAns
    ans.RansEncoder = DummyAns
    sys.modules["compressai.ans"] = ans


def prepare_compressai_import(package_dir: Path) -> None:
    for name in list(sys.modules):
        if name == "compressai" or name.startswith("compressai."):
            del sys.modules[name]
    for path in (str(STF_DIR), str(ZOO_DIR)):
        while path in sys.path:
            sys.path.remove(path)
    sys.path.insert(0, str(package_dir))
    patch_compressai_extensions()


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def load_model(model_id: str, quality: str, device: torch.device) -> torch.nn.Module:
    if model_id in {"stf", "cnn-wam"}:
        prepare_compressai_import(STF_DIR)
        from compressai.models import SymmetricalTransFormer, WACNN

        checkpoint_name = (
            f"stf_{lambda_tag(quality)}.pth.tar"
            if model_id == "stf"
            else f"cnn_{lambda_tag(quality)}.pth.tar"
        )
        checkpoint = CHECKPOINT_DIR / checkpoint_name
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
        loaded = torch.load(str(checkpoint), map_location="cpu")
        state = strip_module_prefix(loaded.get("state_dict", loaded))
        net = (
            SymmetricalTransFormer.from_state_dict(state)
            if model_id == "stf"
            else WACNN.from_state_dict(state)
        )
        return net.eval().to(device)

    if model_id in {"mbt2018-mean", "cheng2020-attn"}:
        prepare_compressai_import(ZOO_DIR)
        from compressai.zoo import cheng2020_attn, mbt2018_mean

        q = int(quality)
        net = mbt2018_mean(quality=q, pretrained=True) if model_id == "mbt2018-mean" else cheng2020_attn(quality=q, pretrained=True)
        return net.eval().to(device)

    raise ValueError(f"Unsupported neural model: {model_id}")


def pad_to_multiple(x: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, tuple[int, int]]:
    height, width = x.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (height, width)


def likelihood_bpp(likelihoods: dict[str, torch.Tensor], height: int, width: int) -> float:
    total = torch.zeros((), device=next(iter(likelihoods.values())).device)
    for likelihood in likelihoods.values():
        total = total + torch.log(likelihood.clamp_min(1e-9)).sum() / (-math.log(2))
    return (total / (height * width)).item()


def save_bit_heatmap(likelihoods: dict[str, torch.Tensor], height: int, width: int, output_path: Path) -> None:
    if not likelihoods:
        return
    source = likelihoods.get("y", next(iter(likelihoods.values())))
    bits = -torch.log2(source.clamp_min(1e-9)).mean(dim=1, keepdim=True)
    bits = F.interpolate(bits, size=(height, width), mode="bilinear", align_corners=False)
    values = bits[0, 0].detach().float().cpu()
    low = torch.quantile(values, 0.02).item()
    high = torch.quantile(values, 0.98).item()
    normalized = ((values - low) / max(high - low, 1e-6)).clamp(0, 1)
    gray = (normalized * 255).byte().numpy()
    image = Image.fromarray(gray, mode="L")
    colored = ImageOps.colorize(image, black="#132238", white="#ffbf47", mid="#2fb7a3")
    colored.save(output_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    image_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    x = to_tensor(image).unsqueeze(0).to(device)
    x_pad, (orig_h, orig_w) = pad_to_multiple(x)

    net = load_model(args.model, str(args.quality), device)
    with torch.no_grad():
        forward_started = time.perf_counter()
        output = net(x_pad)
        if device.type == "cuda":
            torch.cuda.synchronize()
        forward_ms = (time.perf_counter() - forward_started) * 1000

    x_hat = output["x_hat"][..., :orig_h, :orig_w].clamp(0, 1)
    target = x[..., :orig_h, :orig_w]
    mse = torch.mean((x_hat - target) ** 2).item()
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    msssim = ms_ssim(x_hat, target, data_range=1.0).item()
    bpp = likelihood_bpp(output.get("likelihoods", {}), height, width)

    recon_path = output_dir / "reconstruction.png"
    heatmap_path = output_dir / "bit_heatmap.png"
    to_pil_image(x_hat.squeeze(0).cpu()).save(recon_path)
    save_bit_heatmap(output.get("likelihoods", {}), height, width, heatmap_path)

    quality_label = f"lambda={args.quality}" if args.model in {"stf", "cnn-wam"} else f"quality={args.quality}"
    result = {
        "id": args.result_id,
        "image_id": image_path.stem,
        "model_id": args.model,
        "model_name": MODEL_NAMES[args.model],
        "quality": str(args.quality),
        "quality_label": quality_label,
        "reconstruction_url": "/" + recon_path.relative_to(ROOT).as_posix(),
        "heatmap_url": "/" + heatmap_path.relative_to(ROOT).as_posix() if heatmap_path.exists() else None,
        "bpp": bpp,
        "psnr": psnr,
        "ms_ssim": msssim,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "forward_ms": forward_ms,
        "device": str(device),
        "cached": False,
        "notes": "bpp is estimated from likelihoods, matching learned-compression evaluation practice.",
    }
    (output_dir / "worker_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--quality", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-id", required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
