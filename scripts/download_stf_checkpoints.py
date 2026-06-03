from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = ROOT / "checkpoints" / "stf"

FILES = {
    "cnn": {
        "0.0018": ("cnn_0018.pth.tar", "1RPdtyxTtfosuDe1-xtl5JzvnCU2vYnHD"),
        "0.0035": ("cnn_0035.pth.tar", "1L7xvei3Wj4BeSQ3lDBL-pyjEy13RKsjn"),
        "0.0067": ("cnn_0067.pth.tar", "1DDCFFWBUa5cYOgJ9D9HPcwoOigzoJK31"),
        "0.025": ("cnn_025.pth.tar", "1LrAWPlBE6WJUfjiDPGFO8ANSaP5BFEQI"),
    },
    "stf": {
        "0.0018": ("stf_0018.pth.tar", "15ujpSjif628iwVEay3mAWN-Vyqls3r23"),
        "0.0035": ("stf_0035.pth.tar", "1OFzZoEaofNgsimBuOPHtgOJiGsR_RS-M"),
        "0.0067": ("stf_0067.pth.tar", "1SjhqcKyP3SqVm4yhJQslJ6HgY1E8FcBL"),
        "0.013": ("stf_013.pth.tar", "1mupv4vcs8wpNdXCPclXghliikJyYjgj-"),
        "0.025": ("stf_025.pth.tar", "1rsYgEYuqSYBIA4rfvAjXtVSrjXOzkJlB"),
        "0.0483": ("stf_0483.pth.tar", "1cH5cR-0VdsQqCchyN3DO62Sx0WGjv1h8"),
    },
}


def download(model: str, lambdas: list[str]) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for value in lambdas:
        filename, file_id = FILES[model][value]
        target = CHECKPOINT_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {target.name} already exists")
            continue
        print(f"[download] {model} lambda={value} -> {target}")
        cmd = [
            sys.executable,
            "-m",
            "gdown",
            "--id",
            file_id,
            "-O",
            str(target),
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download official STF/CNN checkpoints.")
    parser.add_argument("--model", choices=["cnn", "stf"], required=True)
    parser.add_argument(
        "--lambdas",
        nargs="+",
        required=True,
        help="Lambda values, for example: 0.0018 0.0035 0.0067",
    )
    args = parser.parse_args()
    download(args.model, args.lambdas)


if __name__ == "__main__":
    main()

