from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "external"
STF_DIR = EXTERNAL / "stf"
ZOO_DIR = EXTERNAL / "compressai_zoo_pkg"


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("[run]", " ".join(command))
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_stf_repo() -> None:
    if STF_DIR.exists():
        print(f"[skip] {STF_DIR} already exists")
        return
    EXTERNAL.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--depth", "1", "https://github.com/Googolxx/STF", str(STF_DIR)])


def ensure_python_packages() -> None:
    run([sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"])
    ensure_stf_repo()
    run([sys.executable, "-m", "pip", "install", "-e", str(STF_DIR)])
    if (ZOO_DIR / "compressai").exists():
        print(f"[skip] {ZOO_DIR} already contains CompressAI zoo package")
        return
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--target",
            str(ZOO_DIR),
            "--no-deps",
            "compressai==1.2.8",
        ]
    )


def ensure_kodak() -> None:
    run([sys.executable, "scripts/download_kodak.py"])


def report_checkpoint_status() -> None:
    expected = [
        ROOT / "checkpoints" / "stf" / "cnn_0018.pth.tar",
        ROOT / "checkpoints" / "stf" / "cnn_0035.pth.tar",
        ROOT / "checkpoints" / "stf" / "stf_0035.pth.tar",
    ]
    missing = [path for path in expected if not path.exists() or path.stat().st_size < 1_000_000]
    if not missing:
        print("[ok] demo checkpoints are present")
        return
    print("[warn] demo checkpoints are missing; upload them manually in Cloud Studio:")
    for path in missing:
        print(f"       {path.relative_to(ROOT).as_posix()}")
    print("[warn] the web UI will still start, but paper models stay disabled until checkpoints are uploaded.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Cloud Studio runtime assets.")
    parser.parse_args()

    ensure_python_packages()
    ensure_kodak()
    report_checkpoint_status()


if __name__ == "__main__":
    main()
