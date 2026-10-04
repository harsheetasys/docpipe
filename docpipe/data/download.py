"""Fetch the ICDAR-2019 SROIE scanned-receipt dataset (626 real receipts with text-line
boxes, transcriptions and key-field ground truth) from its GitHub mirror.

Usage: python -m docpipe.data.download
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from docpipe.config import RAW_DIR

MIRROR = "https://github.com/zzzDavid/ICDAR-2019-SROIE.git"


def download(dest: Path = RAW_DIR) -> Path:
    if (dest / "img").exists() and any((dest / "img").iterdir()):
        print(f"dataset already present at {dest}")
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        print(f"cloning {MIRROR} ...")
        subprocess.run(["git", "clone", "--depth", "1", MIRROR, tmp], check=True)
        for sub in ("img", "box", "key"):
            shutil.copytree(Path(tmp) / "data" / sub, dest / sub, dirs_exist_ok=True)
    n = len(list((dest / "img").glob("*.jpg")))
    print(f"done: {n} receipts in {dest}")
    return dest


if __name__ == "__main__":
    sys.exit(0 if download() else 1)
