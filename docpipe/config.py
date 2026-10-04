"""Central configuration. Everything is overridable through environment variables so the
same code runs on a laptop, inside Docker Compose and inside Kubernetes without edits."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.getenv("DOCPIPE_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = Path(os.getenv("DOCPIPE_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"            # cloned SROIE repo (img/, box/, key/)
LAKE_DIR = DATA_DIR / "lake"          # parquet tables written by the Spark job
YOLO_DIR = DATA_DIR / "yolo"          # YOLO-format dataset exported from the lake
FEATURE_DIR = DATA_DIR / "features"   # offline feature store (parquet) + online store (sqlite)
LOG_DIR = DATA_DIR / "inference_log"  # prediction log written by the API, read by drift job
REPORT_DIR = DATA_DIR / "reports"     # drift reports, promotion decisions

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{(DATA_DIR / 'mlflow.db').as_posix()}")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "docpipe")

DETECTOR_MODEL_NAME = "docpipe-detector"
SEGMENTER_MODEL_NAME = "docpipe-segmenter"
CHAMPION_ALIAS = "champion"      # the model that serves traffic
CHALLENGER_ALIAS = "challenger"  # the latest trained candidate awaiting the promotion gate
PREVIOUS_ALIAS = "previous"      # last known-good champion, used for rollback

# Field classes for the detector. Every text line in a receipt is labelled with the key
# field it belongs to (or "other"). This turns plain text-line detection into a
# key-information-extraction problem, which is what a document-digitisation shop ships.
FIELD_CLASSES = ["company", "date", "address", "total", "other"]

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

for _d in (LAKE_DIR, YOLO_DIR, FEATURE_DIR, LOG_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
