"""The retraining worker. Runs on a schedule (k8s CronJob / CI nightly):

  1. if data/reports/retrain_trigger.json exists -> rebuild dataset (new labels from the
     delivery pod land in data/raw), retrain the detector, run the promotion gate.
  2. otherwise exit 0 quietly.

Retraining and promotion are separate on purpose: a drift-triggered retrain that does
NOT beat the champion must never reach production.

Usage: python -m docpipe.monitoring.retrain_if_triggered [--force] [--epochs 12]
"""
from __future__ import annotations

import argparse
import json
import time

from docpipe.config import DETECTOR_MODEL_NAME, REPORT_DIR


def main(force: bool = False, epochs: int = 12) -> dict:
    trigger = REPORT_DIR / "retrain_trigger.json"
    if not trigger.exists() and not force:
        print("no retrain trigger present; nothing to do")
        return {"status": "idle"}
    reason = json.loads(trigger.read_text()) if trigger.exists() else {"reason": ["forced"]}
    print("retrain requested:", reason)

    from docpipe.data.spark_prep import build_lake
    from docpipe.data.yolo_export import export
    from docpipe.registry.promote import decide
    from docpipe.train.train_detector import train

    build_lake()
    data_yaml = export(1.0) / "dataset.yaml"
    result = train(data=data_yaml, epochs=epochs, run_name=f"detector-retrain-{int(time.time())}",
                   extra_tags={"trigger": "drift", "trigger_reason": ";".join(reason.get("reason", []))})
    decision = decide(DETECTOR_MODEL_NAME)
    trigger.unlink(missing_ok=True)
    (REPORT_DIR / "retrain_history.jsonl").open("a").write(json.dumps({"ts": time.time(), "reason": reason, "result": result, "decision": decision}) + "\n")
    return {"status": "done", "result": result, "decision": decision}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    a = ap.parse_args()
    main(a.force, a.epochs)
