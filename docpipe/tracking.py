"""MLflow helpers shared by every training / evaluation entrypoint."""
from __future__ import annotations

import os
import platform
import subprocess

import mlflow

from docpipe.config import DATA_DIR, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


def setup_mlflow(experiment: str = MLFLOW_EXPERIMENT) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=os.getenv("MLFLOW_ARTIFACT_ROOT", (DATA_DIR / "mlruns").as_uri()))
    mlflow.set_experiment(experiment)


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return os.getenv("GIT_SHA", "unknown")


def log_environment() -> None:
    """Everything needed to reproduce the run later, logged as tags."""
    import torch

    mlflow.set_tags(
        {
            "git_sha": git_sha(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "host": platform.node(),
        }
    )
