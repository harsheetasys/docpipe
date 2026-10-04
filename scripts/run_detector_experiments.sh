#!/usr/bin/env bash
# Sequential detector runs: production candidate on 100% of labels, then the
# data-scaling experiment (25%, 50%) used in docs/EXPERIMENTS.md.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
EPOCHS="${EPOCHS:-12}"
python -m docpipe.train.train_detector --data data/yolo/dataset.yaml      --epochs "$EPOCHS" --run-name detector-f100 --tag fraction=1.0
python -m docpipe.train.train_detector --data data/yolo_f025/dataset.yaml --epochs "$EPOCHS" --run-name detector-f025 --tag fraction=0.25 --no-register
python -m docpipe.train.train_detector --data data/yolo_f050/dataset.yaml --epochs "$EPOCHS" --run-name detector-f050 --tag fraction=0.5  --no-register
