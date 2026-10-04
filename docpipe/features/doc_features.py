"""Document-level image features.

These are computed identically at training time (to build the reference distribution in
the feature store) and at serving time (logged per request). Drift detection compares the
two. Keeping ONE function for both paths is what prevents training/serving skew.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter, ImageOps

FEATURE_NAMES = [
    "width",
    "height",
    "aspect_ratio",
    "brightness_mean",
    "brightness_std",
    "contrast_rms",
    "blur_laplacian_var",
    "dark_pixel_frac",
    "edge_density",
]


def compute(image: Image.Image) -> dict[str, float]:
    g = ImageOps.exif_transpose(image).convert("L")
    w, h = g.size
    # downscale for speed; the statistics are scale-invariant enough for monitoring
    small = g.copy()
    small.thumbnail((512, 512))
    a = np.asarray(small, dtype=np.float32)

    lap = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    edges = (lap > 40).mean()
    # variance of Laplacian = standard blur score (low => blurry)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    pad = np.pad(a, 1, mode="edge")
    lap_var = float(
        sum(k[i, j] * pad[i : i + a.shape[0], j : j + a.shape[1]] for i in range(3) for j in range(3)).var()
    )
    return {
        "width": float(w),
        "height": float(h),
        "aspect_ratio": float(h / max(w, 1)),
        "brightness_mean": float(a.mean()),
        "brightness_std": float(a.std()),
        "contrast_rms": float(np.sqrt(((a - a.mean()) ** 2).mean())),
        "blur_laplacian_var": lap_var,
        "dark_pixel_frac": float((a < 64).mean()),
        "edge_density": float(edges),
    }
