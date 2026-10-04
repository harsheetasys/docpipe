"""Send traffic to the API to exercise monitoring.

  --mode baseline : clean test-split receipts (tag=baseline)  -> reference for prediction drift
  --mode drifted  : the same receipts degraded the way real field scans degrade
                    (phone-camera blur, under-exposure, skew, JPEG re-compression), tag=drifted

Usage: python -m docpipe.monitoring.simulate_traffic --mode drifted --n 60 --url http://localhost:8000
"""
from __future__ import annotations

import argparse
import io
import random

import duckdb
import httpx
from PIL import Image, ImageEnhance, ImageFilter

from docpipe.config import LAKE_DIR, RAW_DIR, SEED


def degrade(im: Image.Image, rng: random.Random) -> Image.Image:
    im = im.convert("RGB")
    im = im.filter(ImageFilter.GaussianBlur(radius=rng.uniform(1.2, 2.5)))        # out-of-focus phone shot
    im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.45, 0.7))               # under-exposed
    im = ImageEnhance.Contrast(im).enhance(rng.uniform(0.6, 0.85))                 # washed out
    im = im.rotate(rng.uniform(-4, 4), expand=True, fillcolor=(40, 40, 40))        # skew on dark desk
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=rng.randint(25, 45))                              # heavy re-compression
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def test_docs(n: int) -> list[str]:
    con = duckdb.connect()
    return [r[0] for r in con.execute(
        f"SELECT doc_id FROM read_parquet('{LAKE_DIR / 'docs.parquet'}/*.parquet') WHERE split='test' ORDER BY doc_id LIMIT ?", [n]
    ).fetchall()]


def run(mode: str, n: int, url: str, ocr: bool = True) -> list[dict]:
    rng = random.Random(SEED)
    out = []
    with httpx.Client(timeout=120) as c:
        for doc_id in test_docs(n):
            im = Image.open(RAW_DIR / "img" / f"{doc_id}.jpg")
            if mode == "drifted":
                im = degrade(im, rng)
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=90)
            r = c.post(f"{url}/predict", params={"tag": mode, "ocr": str(ocr).lower()}, files={"file": (f"{doc_id}.jpg", buf.getvalue(), "image/jpeg")})
            r.raise_for_status()
            j = r.json()
            out.append(j)
            s = j["summary"]
            print(f"{mode} {doc_id} det={s['n_detections']:3d} conf={s['det_conf_mean']:.2f} fields={s['fields_found']} total={j['fields'].get('total')} {s['latency_ms']['total']:.0f}ms")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "drifted"], default="baseline")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--no-ocr", action="store_true")
    a = ap.parse_args()
    run(a.mode, a.n, a.url, ocr=not a.no_ocr)
