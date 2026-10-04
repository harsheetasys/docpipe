#!/usr/bin/env bash
# Data-scaling experiment only (25 %, 50 % of the labelled training receipts), same recipe
# as the production run. Val/test are identical across fractions. Not registered.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
EPOCHS="${EPOCHS:-12}"
python -m docpipe.train.train_detector --data data/yolo_f025/dataset.yaml --epochs "$EPOCHS" --run-name detector-f025 --tag fraction=0.25 --no-register
python -m docpipe.train.train_detector --data data/yolo_f050/dataset.yaml --epochs "$EPOCHS" --run-name detector-f050 --tag fraction=0.5  --no-register
