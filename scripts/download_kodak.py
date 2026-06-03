from __future__ import annotations

import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KODAK_DIR = ROOT / "data" / "kodak"
BASE_URL = "https://r0k.us/graphics/kodak/kodak"


def main() -> None:
    KODAK_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(1, 25):
        filename = f"kodim{index:02d}.png"
        target = KODAK_DIR / filename
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {filename}")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"[download] {url}")
        urllib.request.urlretrieve(url, target)


if __name__ == "__main__":
    main()

