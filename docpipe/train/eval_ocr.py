"""Benchmark OCR preprocessing recipes on ground-truth line crops, tracked in MLflow.

Each recipe is one MLflow run (same parent) so they can be compared side by side in
the UI. Metrics: character error rate (CER), exact-match rate, mean engine confidence,
latency per line. The winning recipe is written to data/reports/ocr_recipe.json and
picked up by the serving pipeline.

Usage: python -m docpipe.train.eval_ocr --lines 600
"""
from __future__ import annotations

import argparse
import json
import time

import duckdb
import mlflow
import numpy as np
from PIL import Image

from docpipe.config import LAKE_DIR, RAW_DIR, REPORT_DIR, SEED
from docpipe.ocr import RECIPES, cer, exact_match, ocr_line
from docpipe.tracking import setup_mlflow


def sample_lines(n: int, split: str = "test") -> list[tuple]:
    con = duckdb.connect()
    return con.execute(
        f"""
        SELECT doc_id, x1, y1, x2, y2, text, field_class
        FROM read_parquet('{LAKE_DIR / 'lines.parquet'}/*.parquet')
        WHERE split = ? AND length(trim(text)) >= 3
        ORDER BY hash(doc_id || x1 || y1 || '{SEED}') LIMIT ?
        """,
        [split, n],
    ).fetchall()


def evaluate(n_lines: int = 600, recipes: list[str] | None = None) -> dict:
    setup_mlflow()
    rows = sample_lines(n_lines)
    images: dict[str, Image.Image] = {}
    results = {}
    with mlflow.start_run(run_name="ocr-recipe-benchmark") as parent:
        mlflow.set_tags({"model": "ocr", "engine": "tesseract-5"})
        mlflow.log_params({"n_lines": len(rows), "split": "test", "engine": "tesseract", "psm": 7, "oem": 1})
        for recipe in recipes or list(RECIPES):
            with mlflow.start_run(run_name=f"ocr-{recipe}", nested=True):
                mlflow.set_tags({"model": "ocr", "recipe": recipe})
                cers, ems, confs, per_class = [], [], [], {}
                t0 = time.perf_counter()
                for doc_id, x1, y1, x2, y2, truth, cls in rows:
                    if doc_id not in images:
                        images[doc_id] = Image.open(RAW_DIR / "img" / f"{doc_id}.jpg").convert("RGB")
                    crop = images[doc_id].crop((x1, y1, x2, y2))
                    r = ocr_line(crop, recipe=recipe)
                    c = cer(r.text, truth)
                    cers.append(c)
                    ems.append(exact_match(r.text, truth))
                    confs.append(r.confidence)
                    per_class.setdefault(cls, []).append(c)
                latency = (time.perf_counter() - t0) / len(rows)
                m = {
                    "cer": float(np.mean(cers)),
                    "cer_median": float(np.median(cers)),
                    "exact_match": float(np.mean(ems)),
                    "mean_confidence": float(np.mean(confs)),
                    "latency_ms_per_line": latency * 1000,
                    **{f"cer_{k}": float(np.mean(v)) for k, v in per_class.items()},
                }
                mlflow.log_params({"recipe": recipe})
                mlflow.log_metrics(m)
                results[recipe] = m
                print(f"{recipe:28s} CER={m['cer']:.4f} EM={m['exact_match']:.3f} conf={m['mean_confidence']:.3f} {m['latency_ms_per_line']:.0f}ms")
        best = min(results, key=lambda k: results[k]["cer"])
        mlflow.log_metrics({"best_cer": results[best]["cer"], "baseline_cer": results["raw"]["cer"],
                            "cer_gain_vs_raw": results["raw"]["cer"] - results[best]["cer"]})
        mlflow.set_tag("best_recipe", best)
        out = {"best_recipe": best, "results": results, "n_lines": len(rows), "run_id": parent.info.run_id}
        (REPORT_DIR / "ocr_recipe.json").write_text(json.dumps(out, indent=2))
        print(json.dumps({"best_recipe": best, "cer": results[best]["cer"]}, indent=2))
        return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=600)
    a = ap.parse_args()
    evaluate(a.lines)
