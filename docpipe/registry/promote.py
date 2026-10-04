"""Promotion gate: challenger -> champion, with the previous champion kept for rollback.

A challenger is promoted only if ALL of these hold:
  1. its primary test metric beats the current champion by at least `min_gain`
     (or there is no champion yet),
  2. no per-class metric regressed by more than `max_class_regression`
     (a model that is +2% overall but -20% on `total` would silently break invoicing),
  3. it was trained on a dataset at least as large as the champion's (guards against an
     accidental "great" score from a tiny subset).

Decisions are written to data/reports/promotions.jsonl so there is an audit trail.

Usage:
  python -m docpipe.registry.promote --model docpipe-detector
  python -m docpipe.registry.promote --model docpipe-segmenter --metric test_iou
"""
from __future__ import annotations

import argparse
import json
import time

from mlflow import MlflowClient

from docpipe.config import (
    CHALLENGER_ALIAS,
    CHAMPION_ALIAS,
    DETECTOR_MODEL_NAME,
    FIELD_CLASSES,
    PREVIOUS_ALIAS,
    REPORT_DIR,
    SEGMENTER_MODEL_NAME,
)
from docpipe.tracking import setup_mlflow

PRIMARY_METRIC = {DETECTOR_MODEL_NAME: "test_mAP50", SEGMENTER_MODEL_NAME: "test_iou"}
CLASS_METRICS = {DETECTOR_MODEL_NAME: [f"test_mAP50_{c}" for c in FIELD_CLASSES], SEGMENTER_MODEL_NAME: []}


def _version_by_alias(client: MlflowClient, model: str, alias: str):
    try:
        return client.get_model_version_by_alias(model, alias)
    except Exception:
        return None


def _run_metrics(client: MlflowClient, run_id: str) -> dict:
    return client.get_run(run_id).data.metrics


def _run_params(client: MlflowClient, run_id: str) -> dict:
    return client.get_run(run_id).data.params


def decide(model: str, metric: str | None = None, min_gain: float = 0.0, max_class_regression: float = 0.05, dry_run: bool = False) -> dict:
    setup_mlflow()
    client = MlflowClient()
    metric = metric or PRIMARY_METRIC[model]
    chal = _version_by_alias(client, model, CHALLENGER_ALIAS)
    if chal is None:
        return {"model": model, "decision": "no_challenger"}
    champ = _version_by_alias(client, model, CHAMPION_ALIAS)
    cm = _run_metrics(client, chal.run_id)
    cp = _run_params(client, chal.run_id)
    decision = {"model": model, "metric": metric, "challenger_version": chal.version, "challenger_score": cm.get(metric),
                "champion_version": champ.version if champ else None, "ts": time.time()}

    reasons = []
    if champ is None:
        decision["champion_score"] = None
        reasons.append("no existing champion")
        promote = cm.get(metric) is not None
    else:
        if champ.version == chal.version:
            return {**decision, "decision": "already_champion"}
        km = _run_metrics(client, champ.run_id)
        kp = _run_params(client, champ.run_id)
        decision["champion_score"] = km.get(metric)
        gain = (cm.get(metric) or 0) - (km.get(metric) or 0)
        decision["gain"] = gain
        promote = gain >= min_gain
        reasons.append(f"gain {gain:+.4f} vs min_gain {min_gain}")
        for cmetric in CLASS_METRICS[model]:
            if cmetric in cm and cmetric in km and km[cmetric] - cm[cmetric] > max_class_regression:
                promote = False
                reasons.append(f"{cmetric} regressed {km[cmetric]:.3f} -> {cm[cmetric]:.3f}")
        if int(cp.get("n_train", 0)) < int(kp.get("n_train", 0)):
            promote = False
            reasons.append(f"trained on fewer samples ({cp.get('n_train')} < {kp.get('n_train')})")

    decision["reasons"] = reasons
    decision["decision"] = "promote" if promote else "reject"
    if promote and not dry_run:
        if champ is not None:
            client.set_registered_model_alias(model, PREVIOUS_ALIAS, champ.version)
        client.set_registered_model_alias(model, CHAMPION_ALIAS, chal.version)
        client.delete_registered_model_alias(model, CHALLENGER_ALIAS)
        client.set_model_version_tag(model, chal.version, "promoted_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(REPORT_DIR / "promotions.jsonl", "a") as fh:
        fh.write(json.dumps(decision) + "\n")
    print(json.dumps(decision, indent=2))
    return decision


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DETECTOR_MODEL_NAME)
    ap.add_argument("--metric", default=None)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--max-class-regression", type=float, default=0.05)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    decide(a.model, a.metric, a.min_gain, a.max_class_regression, a.dry_run)
