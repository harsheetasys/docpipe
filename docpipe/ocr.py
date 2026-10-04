"""OCR engine wrapper + the preprocessing recipes we benchmark.

Tesseract is the engine (CPU-only, no weights to ship). The interesting ML-engineering
work is around it: which preprocessing gives measurable accuracy gains on *our* scans,
measured with character error rate against ground truth, tracked in MLflow.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageOps
from rapidfuzz.distance import Levenshtein

# Single text line, no page segmentation, English, whitelist nothing (receipts have symbols)
TESS_CONFIG_LINE = "--oem 1 --psm 7"


# --------------------------------------------------------------------------------------
# preprocessing recipes (name -> function)
# --------------------------------------------------------------------------------------
def pp_raw(im: Image.Image) -> Image.Image:
    return im.convert("L")


def pp_upscale2x(im: Image.Image) -> Image.Image:
    g = im.convert("L")
    return g.resize((g.width * 2, g.height * 2), Image.BICUBIC)


def _otsu_threshold(a: np.ndarray) -> int:
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    total = a.size
    sum_total = (np.arange(256) * hist).sum()
    best, thr, w_b, sum_b = 0.0, 127, 0.0, 0.0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        m_b, m_f = sum_b / w_b, (sum_total - sum_b) / w_f
        between = w_b * w_f * (m_b - m_f) ** 2
        if between > best:
            best, thr = between, t
    return thr


def pp_upscale_otsu(im: Image.Image) -> Image.Image:
    g = pp_upscale2x(im)
    a = np.asarray(g)
    return Image.fromarray(((a > _otsu_threshold(a)) * 255).astype(np.uint8))


def pp_upscale_denoise_otsu(im: Image.Image) -> Image.Image:
    g = pp_upscale2x(im).filter(ImageFilter.MedianFilter(3))
    g = ImageOps.autocontrast(g, cutoff=1)
    a = np.asarray(g)
    return Image.fromarray(((a > _otsu_threshold(a)) * 255).astype(np.uint8))


def pp_autocontrast(im: Image.Image) -> Image.Image:
    return ImageOps.autocontrast(im.convert("L"), cutoff=1)


def pp_otsu(im: Image.Image) -> Image.Image:
    a = np.asarray(im.convert("L"))
    return Image.fromarray(((a > _otsu_threshold(a)) * 255).astype(np.uint8))


RECIPES = {
    "raw": pp_raw,
    "autocontrast": pp_autocontrast,
    "otsu": pp_otsu,
    "upscale2x": pp_upscale2x,
    "upscale2x_otsu": pp_upscale_otsu,
    "upscale2x_denoise_otsu": pp_upscale_denoise_otsu,
}


# --------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------
@dataclass
class OcrResult:
    text: str
    confidence: float  # 0..1, mean word confidence from tesseract


TESS_CONFIG_AMOUNT = "--oem 1 --psm 7 -c tessedit_char_whitelist=0123456789.,RM$"


def ocr_line(crop: Image.Image, recipe: str = "upscale2x_otsu", pad: int = 4, numeric: bool = False) -> OcrResult:
    """OCR one text line. `numeric=True` restricts the engine to amount characters, which
    removes most 5<->8, 2<->9 confusions on money fields."""
    im = ImageOps.expand(RECIPES["autocontrast" if numeric else recipe](crop), border=pad, fill=255)
    data = pytesseract.image_to_data(im, config=TESS_CONFIG_AMOUNT if numeric else TESS_CONFIG_LINE,
                                     output_type=pytesseract.Output.DICT)
    words, confs = [], []
    for w, c in zip(data["text"], data["conf"], strict=False):
        if str(w).strip():
            words.append(str(w).strip())
            try:
                c = float(c)
                if c >= 0:
                    confs.append(c / 100.0)
            except ValueError:
                pass
    return OcrResult(" ".join(words), float(np.mean(confs)) if confs else 0.0)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------
def _norm_for_cer(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().upper())


def cer(pred: str, truth: str) -> float:
    t = _norm_for_cer(truth)
    if not t:
        return 0.0 if not _norm_for_cer(pred) else 1.0
    return Levenshtein.distance(_norm_for_cer(pred), t) / len(t)


def exact_match(pred: str, truth: str) -> bool:
    return _norm_for_cer(pred) == _norm_for_cer(truth)
