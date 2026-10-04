"""End-to-end business metric: key-field extraction accuracy on the TEST split.

Runs the full champion pipeline (detect -> OCR -> post-process) on held-out receipts and
scores each field against SROIE ground truth:
    exact      normalised string equality
    fuzzy>=90  rapidfuzz ratio (tolerates one OCR slip in a long address)
Logged to MLflow so a promotion can also be judged on the metric the client pays for,
not only on mAP.

Usage: python -m docpipe.train.eval_end2end --n 60
"""
from __future__ import annotations

import argparse
import json

import duckdb
import mlflow
import numpy as np
from PIL import Image
from rapidfuzz import fuzz

from docpipe.config import LAKE_DIR, RAW_DIR, REPORT_DIR
from docpipe.ocr import _norm_for_cer
from docpipe.serving.pipeline import Pipeline
from docpipe.tracking import setup_mlflow

FIELDS = ("company", "date", "address", "total")


def evaluate(n: int = 60, alias: str = "champion") -> dict:
    setup_mlflow()
    pipe = Pipeline(alias=alias)
    con = duckdb.connect()
    docs = con.execute(
        f"SELECT doc_id, company, date, address, total FROM read_parquet('{LAKE_DIR / 'docs.parquet'}/*.parquet') "
        f"WHERE split='test' ORDER BY doc_id LIMIT ?", [n]
    ).fetchall()

    exact = {f: [] for f in FIELDS}
    fuzzy = {f: [] for f in FIELDS}
    found = {f: [] for f in FIELDS}
    lat = []
    for doc_id, *gt in docs:
        res = pipe.predict(Image.open(RAW_DIR / "img" / f"{doc_id}.jpg"))
        lat.append(res["summary"]["latency_ms"]["total"])
        for f, truth in zip(FIELDS, gt, strict=True):
            pred = res["fields"].get(f)
            found[f].append(pred is not None)
            p, t = _norm_for_cer(pred or ""), _norm_for_cer(truth or "")
            if f == "total":
                p, t = p.replace(",", ""), t.replace(",", "")
            exact[f].append(p == t)
            fuzzy[f].append(fuzz.ratio(p, t) >= 90)

    metrics = {}
    for f in FIELDS:
        metrics[f"e2e_exact_{f}"] = float(np.mean(exact[f]))
        metrics[f"e2e_fuzzy90_{f}"] = float(np.mean(fuzzy[f]))
        metrics[f"e2e_found_{f}"] = float(np.mean(found[f]))
    metrics["e2e_exact_macro"] = float(np.mean([metrics[f"e2e_exact_{f}"] for f in FIELDS]))
    metrics["e2e_fuzzy90_macro"] = float(np.mean([metrics[f"e2e_fuzzy90_{f}"] for f in FIELDS]))
    metrics["e2e_latency_ms_p50"] = float(np.percentile(lat, 50))
    metrics["e2e_latency_ms_p95"] = float(np.percentile(lat, 95))

    with mlflow.start_run(run_name=f"e2e-eval-{alias}"):
        mlflow.set_tags({"model": "pipeline", "alias": alias, **{f"{k}_version": v.version for k, v in pipe.info.items()}})
        mlflow.log_params({"n_docs": len(docs), "ocr_recipe": pipe.ocr_recipe, "split": "test"})
        mlflow.log_metrics(metrics)
    out = {"alias": alias, "models": {k: v.version for k, v in pipe.info.items()}, "n_docs": len(docs), **metrics}
    (REPORT_DIR / f"e2e_{alias}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--alias", default="champion")
    a = ap.parse_args()
    evaluate(a.n, a.alias)
