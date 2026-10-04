"""Drift detection + retraining trigger.

Three kinds of drift, each with its own signal:

  INPUT drift       image features in the inference log vs the training reference
                    distribution from the feature store (PSI + two-sample KS test).
  PREDICTION drift  detector / OCR confidence and fields_found vs the baseline window
                    (a model that is suddenly unsure is the earliest symptom of a new
                    receipt template; no labels needed).
  PERFORMANCE drift when labels arrive from the delivery pod (`labels.parquet`), exact
                    field accuracy on recent traffic vs the test-set baseline.

Severity -> action (configs/drift.yaml):
  ok        nothing
  warn      annotate; flag for review
  retrain   write data/reports/retrain_trigger.json (the k8s CronJob / CI picks it up and
            runs training -> promotion gate). Rollback is NOT automatic here: a drifted
            input distribution is a reason to retrain, not to revert.
  rollback  only for PREDICTION drift coinciding with a recent model change
            (regression introduced by the new champion) -> recommend rollback.

Usage: python -m docpipe.monitoring.drift [--window-hours 24] [--tag drifted]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from docpipe.config import REPORT_DIR, ROOT
from docpipe.features.doc_features import FEATURE_NAMES
from docpipe.features.store import DOC_IMAGE_VIEW, default_store
from docpipe.serving.inference_log import InferenceLog

DEFAULT_CFG = {
    "psi_warn": 0.1, "psi_retrain": 0.25,  # industry rule of thumb: <0.1 stable, 0.1-0.25 moderate, >0.25 major
    "ks_pvalue": 0.01,
    "min_requests": 30,
    "conf_drop_warn": 0.05, "conf_drop_retrain": 0.15,
    "fields_found_drop_retrain": 0.5,
    "features": ["brightness_mean", "contrast_rms", "blur_laplacian_var", "dark_pixel_frac", "edge_density", "aspect_ratio"],
}


def load_cfg(path: Path = ROOT / "configs" / "drift.yaml") -> dict:
    cfg = dict(DEFAULT_CFG)
    if path.exists():
        cfg.update(yaml.safe_load(path.read_text()) or {})
    return cfg


def psi(ref: np.ndarray, cur: np.ndarray, bins: int | None = None) -> float:
    """Population Stability Index with quantile bins from the reference.

    Bins adapt to the size of the *current* window: with 10 bins and 40 requests a single
    empty bin already contributes ~0.9 of PSI, which is pure sampling noise."""
    bins = bins or int(min(10, max(4, len(cur) // 20)))
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    r = np.histogram(ref, qs)[0] / len(ref)
    c = np.histogram(cur, qs)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def analyse(window_hours: float = 24, tag: str | None = None, cfg: dict | None = None, write: bool = True) -> dict:
    cfg = cfg or load_cfg()
    fs = default_store()
    ref = fs.get_historical_features(DOC_IMAGE_VIEW.name)
    ref = ref[ref["split"] == "train"] if "split" in ref else ref

    log = InferenceLog()
    cur = log.read(since_ts=time.time() - window_hours * 3600, tag=tag)
    report = {"ts": time.time(), "window_hours": window_hours, "tag": tag, "n_requests": int(len(cur)), "n_reference": int(len(ref))}

    if len(cur) < cfg["min_requests"]:
        report.update({"severity": "insufficient_data", "action": "none"})
        return _finish(report, write)

    # ---- input drift ------------------------------------------------------------------
    feats = {}
    worst_psi = 0.0
    for f in cfg["features"]:
        if f not in FEATURE_NAMES or f"feat_{f}" not in cur:
            continue
        r, c = ref[f].dropna().to_numpy(float), cur[f"feat_{f}"].dropna().to_numpy(float)
        p = psi(r, c)
        ks = stats.ks_2samp(r, c)
        feats[f] = {"psi": p, "ks_stat": float(ks.statistic), "ks_pvalue": float(ks.pvalue),
                    "ref_mean": float(r.mean()), "cur_mean": float(c.mean()),
                    "drifted": bool(p >= cfg["psi_warn"] and ks.pvalue < cfg["ks_pvalue"])}
        # PSI alone is noisy on small windows; a feature only counts when the KS test agrees
        if ks.pvalue < cfg["ks_pvalue"]:
            worst_psi = max(worst_psi, p)
    report["input_drift"] = feats
    report["worst_psi"] = worst_psi  # over KS-significant features only

    # ---- prediction drift -------------------------------------------------------------
    base = log.read(tag="baseline")
    pred = {}
    for col in ("pred_det_conf_mean", "pred_ocr_conf_mean", "fields_found", "pred_seg_confidence"):
        if col in cur and len(base) and col in base:
            b, c = base[col].dropna().astype(float), cur[col].dropna().astype(float)
            if len(b) and len(c):
                pred[col] = {"baseline_mean": float(b.mean()), "current_mean": float(c.mean()), "drop": float(b.mean() - c.mean())}
    report["prediction_drift"] = pred

    # ---- performance drift (only when labels exist) -----------------------------------
    labels_path = REPORT_DIR / "labels.parquet"
    if labels_path.exists() and "request_id" in cur:
        lab = pd.read_parquet(labels_path)
        merged = cur.merge(lab, on="request_id")
        if len(merged):
            report["performance"] = {"n_labelled": int(len(merged)),
                                     "field_accuracy": float(merged["correct"].mean()) if "correct" in merged else None}

    # ---- decision ---------------------------------------------------------------------
    severity, reasons = "ok", []
    if worst_psi >= cfg["psi_retrain"]:
        severity, reasons = "retrain", reasons + [f"input PSI {worst_psi:.3f} >= {cfg['psi_retrain']}"]
    elif worst_psi >= cfg["psi_warn"]:
        severity, reasons = "warn", reasons + [f"input PSI {worst_psi:.3f} >= {cfg['psi_warn']}"]
    conf_drop = pred.get("pred_det_conf_mean", {}).get("drop", 0.0)
    ff_drop = pred.get("fields_found", {}).get("drop", 0.0)
    if conf_drop >= cfg["conf_drop_retrain"] or ff_drop >= cfg["fields_found_drop_retrain"]:
        severity = "retrain"
        reasons.append(f"detector confidence dropped {conf_drop:.3f} / fields_found dropped {ff_drop:.2f}")
        # prediction drift WITHOUT input drift right after a model change => model regression => rollback
        if worst_psi < cfg["psi_warn"] and _model_changed_recently(cur):
            severity = "rollback"
            reasons.append("inputs stable but predictions degraded after model change -> suspect new champion")
    elif conf_drop >= cfg["conf_drop_warn"] and severity == "ok":
        severity, reasons = "warn", reasons + [f"detector confidence dropped {conf_drop:.3f}"]

    report["severity"] = severity
    report["reasons"] = reasons
    report["action"] = {"ok": "none", "warn": "flag_for_review", "retrain": "trigger_retraining", "rollback": "recommend_rollback",
                        }[severity]
    if severity == "retrain" and write:
        (REPORT_DIR / "retrain_trigger.json").write_text(json.dumps(
            {"requested_at": time.time(), "reason": reasons, "window_hours": window_hours, "tag": tag}, indent=2))
    return _finish(report, write)


def _model_changed_recently(cur: pd.DataFrame) -> bool:
    if "models" not in cur:
        return False
    versions = cur["models"].astype(str).nunique()
    return versions > 1


def _finish(report: dict, write: bool) -> dict:
    if write:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        (REPORT_DIR / "drift_latest.json").write_text(json.dumps(report, indent=2, default=float))
        with open(REPORT_DIR / "drift_history.jsonl", "a") as fh:
            fh.write(json.dumps({k: report[k] for k in ("ts", "tag", "n_requests", "severity", "action") if k in report}
                                | {"worst_psi": report.get("worst_psi")}) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("input_drift",)}, indent=2, default=float))
    if "input_drift" in report:
        print("per-feature:")
        for f, d in report["input_drift"].items():
            print(f"  {f:22s} psi={d['psi']:.3f} ks_p={d['ks_pvalue']:.2e} ref={d['ref_mean']:.2f} cur={d['cur_mean']:.2f} {'DRIFT' if d['drifted'] else ''}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-hours", type=float, default=24)
    ap.add_argument("--tag", default=None, help="only analyse requests with this tag (used by the demo)")
    a = ap.parse_args()
    analyse(a.window_hours, a.tag)
