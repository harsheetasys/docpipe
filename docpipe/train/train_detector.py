"""Train the field detector (YOLOv8) and log everything to MLflow.

Every run:
  * logs hyper-parameters, dataset fingerprint, git SHA and environment tags
  * logs per-epoch losses and val mAP (via Ultralytics callbacks)
  * evaluates once on the held-out TEST split (never used for model selection)
  * logs weights + training curves as artifacts
  * registers the model in the MLflow Model Registry and tags it `challenger`
    (promotion to `champion` is a separate, gated step: docpipe.registry.promote)

Usage:
  python -m docpipe.train.train_detector --epochs 15 --imgsz 640
  python -m docpipe.train.train_detector --data data/yolo_f050/dataset.yaml --tag fraction=0.5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
import yaml

from docpipe.config import CHALLENGER_ALIAS, DETECTOR_MODEL_NAME, FIELD_CLASSES, SEED, YOLO_DIR
from docpipe.tracking import log_environment, setup_mlflow


def dataset_fingerprint(data_yaml: Path) -> dict:
    """Hash of the label files => the run is tied to an exact dataset version."""
    root = Path(yaml.safe_load(data_yaml.read_text())["path"])
    h = hashlib.sha256()
    counts = {}
    for split in ("train", "val", "test"):
        files = sorted((root / "labels" / split).glob("*.txt"))
        counts[split] = len(files)
        for f in files:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return {"dataset_sha256": h.hexdigest()[:16], **{f"n_{k}": v for k, v in counts.items()}}


class YoloPyfunc(mlflow.pyfunc.PythonModel):
    """Thin pyfunc wrapper so the registry holds a loadable model, not just a file."""

    def load_context(self, context):
        from ultralytics import YOLO

        self.model = YOLO(context.artifacts["weights"])

    def predict(self, context, model_input, params=None):
        # model_input: list of image paths or PIL images
        res = self.model.predict(model_input, verbose=False, conf=(params or {}).get("conf", 0.25))
        return [r.boxes.data.tolist() for r in res]


def train(
    data: Path = YOLO_DIR / "dataset.yaml",
    epochs: int = 15,
    imgsz: int = 640,
    batch: int = 8,
    base_weights: str = "yolov8n.pt",
    lr0: float = 0.01,
    run_name: str | None = None,
    extra_tags: dict | None = None,
    register: bool = True,
) -> dict:
    from ultralytics import YOLO, settings

    settings.update({"mlflow": False, "sync": False})  # we log explicitly; no telemetry
    setup_mlflow()

    data = Path(data)
    fp = dataset_fingerprint(data)
    work = Path(tempfile.mkdtemp(prefix="yolo_"))

    with mlflow.start_run(run_name=run_name or f"detector-{data.parent.name}") as run:
        log_environment()
        mlflow.set_tags({"model": "detector", "arch": base_weights, **(extra_tags or {})})
        params = {
            "epochs": epochs, "imgsz": imgsz, "batch": batch, "lr0": lr0, "seed": SEED,
            "base_weights": base_weights, "data": str(data), "classes": ",".join(FIELD_CLASSES),
            "mosaic": 0.0, "fliplr": 0.0, "optimizer": "auto", **fp,
        }
        mlflow.log_params(params)

        model = YOLO(base_weights)

        def on_epoch_end(trainer):
            m = {k.replace("(B)", "").replace("/", "_"): float(v) for k, v in trainer.metrics.items()}
            try:
                losses = trainer.label_loss_items(trainer.tloss, prefix="train")
                m.update({k.replace("/", "_"): float(v) for k, v in losses.items()})
            except Exception:
                pass
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, float)}, step=trainer.epoch)

        model.add_callback("on_fit_epoch_end", on_epoch_end)
        model.train(
            data=str(data), epochs=epochs, imgsz=imgsz, batch=batch, lr0=lr0, seed=SEED,
            device="cpu", workers=2, project=str(work), name="run", exist_ok=True,
            plots=True, verbose=False, amp=False,
            # receipts are never mirrored and mosaic destroys their reading order
            mosaic=0.0, fliplr=0.0, deterministic=True,
        )

        best = work / "run" / "weights" / "best.pt"
        # ---- held-out test evaluation (the number we make decisions on) ------------------
        test_metrics = YOLO(best).val(data=str(data), split="test", imgsz=imgsz, device="cpu", plots=False, verbose=False)
        final = {
            "test_mAP50": float(test_metrics.box.map50),
            "test_mAP50_95": float(test_metrics.box.map),
            "test_precision": float(test_metrics.box.mp),
            "test_recall": float(test_metrics.box.mr),
        }
        for i, c in enumerate(FIELD_CLASSES):
            try:
                final[f"test_mAP50_{c}"] = float(test_metrics.box.maps[i])
            except Exception:
                pass
        mlflow.log_metrics(final)

        # ---- artifacts -------------------------------------------------------------------
        for f in ("results.csv", "results.png", "confusion_matrix.png", "args.yaml"):
            p = work / "run" / f
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="training")
        mlflow.pyfunc.log_model(
            name="model",
            python_model=YoloPyfunc(),
            artifacts={"weights": str(best)},
            pip_requirements=["ultralytics", "torch", "pillow"],
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        version = None
        if register:
            mv = mlflow.register_model(model_uri, DETECTOR_MODEL_NAME)
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(DETECTOR_MODEL_NAME, CHALLENGER_ALIAS, mv.version)
            client.set_model_version_tag(DETECTOR_MODEL_NAME, mv.version, "test_mAP50", f"{final['test_mAP50']:.4f}")
            client.set_model_version_tag(DETECTOR_MODEL_NAME, mv.version, "dataset_sha256", fp["dataset_sha256"])
            version = mv.version
        out = {"run_id": run.info.run_id, "model_version": version, **final, **fp}
        print(json.dumps(out, indent=2))
    shutil.rmtree(work, ignore_errors=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(YOLO_DIR / "dataset.yaml"))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--tag", action="append", default=[], help="key=value, repeatable")
    ap.add_argument("--no-register", action="store_true")
    a = ap.parse_args()
    train(
        data=Path(a.data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, base_weights=a.weights,
        run_name=a.run_name, extra_tags=dict(t.split("=", 1) for t in a.tag), register=not a.no_register,
    )
