"""Rollback path.

Two layers, because a bad model and a bad container are different failures:

  MODEL rollback (this script): swap the `champion` alias back to `previous`. Serving
     pods reload the champion on their next refresh tick (or on restart), no image
     rebuild needed. Also usable to pin ANY version: --to-version 3.

  DEPLOYMENT rollback (k8s): `kubectl rollout undo deployment/docpipe-api` restores
     the previous image/config. See docs/RUNBOOK.md.

Usage:
  python -m docpipe.registry.rollback --model docpipe-detector
  python -m docpipe.registry.rollback --model docpipe-detector --to-version 2
"""
from __future__ import annotations

import argparse
import json
import time

from mlflow import MlflowClient

from docpipe.config import CHAMPION_ALIAS, DETECTOR_MODEL_NAME, PREVIOUS_ALIAS, REPORT_DIR
from docpipe.tracking import setup_mlflow


def rollback(model: str = DETECTOR_MODEL_NAME, to_version: str | None = None, reason: str = "manual") -> dict:
    setup_mlflow()
    client = MlflowClient()
    champ = client.get_model_version_by_alias(model, CHAMPION_ALIAS)
    if to_version is None:
        prev = client.get_model_version_by_alias(model, PREVIOUS_ALIAS)
        to_version = prev.version
    if str(to_version) == str(champ.version):
        out = {"model": model, "status": "noop", "champion_version": champ.version}
    else:
        client.set_registered_model_alias(model, PREVIOUS_ALIAS, champ.version)
        client.set_registered_model_alias(model, CHAMPION_ALIAS, to_version)
        client.set_model_version_tag(model, to_version, "rollback_reason", reason)
        out = {"model": model, "status": "rolled_back", "from_version": champ.version, "to_version": to_version,
               "reason": reason, "ts": time.time()}
    with open(REPORT_DIR / "promotions.jsonl", "a") as fh:
        fh.write(json.dumps({**out, "decision": "rollback"}) + "\n")
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DETECTOR_MODEL_NAME)
    ap.add_argument("--to-version", default=None)
    ap.add_argument("--reason", default="manual")
    a = ap.parse_args()
    rollback(a.model, a.to_version, a.reason)
