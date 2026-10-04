"""The inference pipeline: image -> field detections -> OCR -> structured fields.

Models are resolved from the MLflow Model Registry by alias (`champion`), so promotion
and rollback never require a redeploy. `refresh()` is called periodically by the API so a
running pod picks up a new champion within `DOCPIPE_MODEL_REFRESH_SECONDS`.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import numpy as np
import torch
from mlflow import MlflowClient
from PIL import Image, ImageOps

from docpipe.config import CHAMPION_ALIAS, DETECTOR_MODEL_NAME, FIELD_CLASSES, REPORT_DIR, SEGMENTER_MODEL_NAME
from docpipe.features.doc_features import compute as doc_features
from docpipe.ocr import ocr_line
from docpipe.postprocess import Detection, Extraction, extract, normalise_date, normalise_total
from docpipe.tracking import setup_mlflow

SEG_SIZE = (160, 320)


@dataclass
class ModelInfo:
    name: str
    version: str | None
    run_id: str | None
    loaded_at: float


class Pipeline:
    def __init__(self, alias: str = CHAMPION_ALIAS, conf: float | None = None, field_conf: float | None = None,
                 fallback: bool | None = None):
        setup_mlflow()
        self.alias = alias
        # `other` lines use the usual 0.25; the rare field classes keep lower-confidence boxes
        # (post-processing + regex validation filter the false positives).
        self.conf = conf if conf is not None else float(os.getenv("DOCPIPE_CONF", "0.25"))
        self.field_conf = field_conf if field_conf is not None else float(os.getenv("DOCPIPE_FIELD_CONF", "0.10"))
        # when the detector finds no `date`/`total`, OCR the remaining lines in the region
        # where those fields live and regex them (rules as a safety net under the model)
        self.fallback = fallback if fallback is not None else os.getenv("DOCPIPE_FALLBACK", "1") == "1"
        self.client = MlflowClient()
        self.lock = threading.Lock()
        self.detector = None
        self.segmenter = None
        self.info: dict[str, ModelInfo] = {}
        recipe_file = REPORT_DIR / "ocr_recipe.json"
        self.ocr_recipe = os.getenv("DOCPIPE_OCR_RECIPE") or (
            json.loads(recipe_file.read_text())["best_recipe"] if recipe_file.exists() else "raw"
        )
        self.refresh(force=True)

    # ---- model loading ----------------------------------------------------------------
    def _champion(self, name: str):
        try:
            return self.client.get_model_version_by_alias(name, self.alias)
        except Exception:
            return None

    def refresh(self, force: bool = False) -> dict:
        """Reload any model whose champion alias now points at a different version."""
        changed = {}
        for name in (DETECTOR_MODEL_NAME, SEGMENTER_MODEL_NAME):
            mv = self._champion(name)
            if mv is None:
                continue
            cur = self.info.get(name)
            if force or cur is None or cur.version != mv.version:
                with self.lock:
                    if name == DETECTOR_MODEL_NAME:
                        from ultralytics import YOLO, settings

                        settings.update({"sync": False})  # no telemetry from serving pods
                        local = mlflow.artifacts.download_artifacts(f"models:/{name}@{self.alias}")
                        weights = next(Path(local).rglob("best.pt"))
                        self.detector = YOLO(str(weights))
                    else:
                        seg = mlflow.pytorch.load_model(f"models:/{name}@{self.alias}", map_location="cpu")
                        try:
                            seg = seg.eval()
                        except NotImplementedError:
                            pass  # MLflow>=3 returns a torch.export graph: already frozen in eval mode
                        self.segmenter = seg
                    self.info[name] = ModelInfo(name, mv.version, mv.run_id, time.time())
                changed[name] = mv.version
        return changed

    @property
    def ready(self) -> bool:
        return self.detector is not None

    def _fallback_fields(self, image: Image.Image, dets: list[Detection], max_lines: int = 14, max_total_lines: int = 24) -> int:
        """Rules as a safety net under the model.

        If the detector produced no usable `date` / `total`, OCR up to `max_lines` of the
        `other` lines in the band where that field lives and regex them. Receipts print the
        TOTAL label and the amount as separate text segments, so for `total` we look for a
        line saying TOTAL and take the right-most amount on the same row. Returns the number
        of extra lines OCR'd (logged, so the latency cost stays visible)."""

        have = set()
        for d in dets:
            if d.cls == "date" and normalise_date(d.text):
                have.add("date")
            elif d.cls == "total" and normalise_total(d.text):
                have.add("total")
            elif d.cls in ("company", "address"):
                have.add(d.cls)
        h = image.height
        n = 0

        def _ocr(d: Detection) -> None:
            nonlocal n
            if not d.text and not d.ocr_conf:
                r = ocr_line(self._crop(image, d.box), recipe=self.ocr_recipe)
                d.text, d.ocr_conf = r.text, r.confidence
                n += 1

        if "date" not in have:
            cands = sorted((d for d in dets if d.cls == "other" and d.box[1] <= 0.45 * h), key=lambda d: d.box[1])
            for d in cands[:max_lines]:
                _ocr(d)
                if normalise_date(d.text):
                    d.cls, d.conf = "date", min(d.conf, 0.5)
                    break

        if "total" not in have:
            band = [d for d in dets if d.cls == "other" and d.box[1] >= 0.2 * h]
            band.sort(key=lambda d: -d.box[1])  # bottom-up
            for d in band[:max_lines]:
                _ocr(d)
            bad = re.compile(r"SUB|CHANGE|ROUND|CASH|TAX|GST|DISC|QTY|ITEM|SAVING|TENDER|BALANCE|PAID|EXCL|BEFORE")
            labels = [d for d in band[:max_lines]
                      if re.search(r"TOTAL|AMOUNT DUE|NET PAYABLE|GRAND", d.text.upper()) and not bad.search(d.text.upper())]
            labels.sort(key=lambda d: (not re.search(r"GRAND|NET|TOTAL AMOUNT|TOTAL\s*[:(]?\s*(RM|\$)?\s*$", d.text.upper()), d.box[1]))
            # every amount that sits on a TOTAL row is a candidate; the total is the largest of
            # them (GST-summary tables put the tax amount on the same row, to the right)
            cands: list[tuple[Detection, float]] = []
            for lab in labels:
                cy = (lab.box[1] + lab.box[3]) / 2
                row = [d for d in band if d is not lab and d.box[1] <= cy <= d.box[3] and d.box[0] >= lab.box[0]] + [lab]
                for d in row:
                    a = normalise_total(d.text)
                    if a:
                        cands.append((d, float(a)))
            picked = max(cands, key=lambda t: t[1])[0] if cands else None
            if picked is not None:
                r = ocr_line(self._crop(image, picked.box), numeric=True)   # second pass with a digit whitelist
                if normalise_total(r.text):
                    picked.text, picked.ocr_conf = r.text, r.confidence
                picked.cls, picked.conf = "total", min(picked.conf, 0.5)
        return n

    @staticmethod
    def _crop(image: Image.Image, box: tuple[int, int, int, int], pad_frac: float = 0.03, pad_px: int = 2) -> Image.Image:
        """Crop with a small margin: detector boxes are often 1-3 px tight and clipped
        digits are the main source of amount misreads."""
        x1, y1, x2, y2 = box
        px = int((x2 - x1) * pad_frac) + pad_px
        py = int((y2 - y1) * pad_frac) + pad_px
        return image.crop((max(0, x1 - px), max(0, y1 - py), min(image.width, x2 + px), min(image.height, y2 + py)))

    # ---- inference --------------------------------------------------------------------
    def predict(self, image: Image.Image, run_ocr: bool = True) -> dict:
        t0 = time.perf_counter()
        image = ImageOps.exif_transpose(image).convert("RGB")
        feats = doc_features(image)

        with self.lock:
            res = self.detector.predict(image, conf=min(self.conf, self.field_conf), verbose=False, imgsz=640)[0]
        dets: list[Detection] = []
        for x1, y1, x2, y2, conf, cls in res.boxes.data.tolist():
            name = FIELD_CLASSES[int(cls)]
            if name == "other" and conf < self.conf:
                continue
            dets.append(Detection(name, float(conf), (int(x1), int(y1), int(x2), int(y2))))
        t_det = time.perf_counter()

        n_fallback = 0
        if run_ocr:
            for d in dets:
                if d.cls == "other":
                    continue  # only OCR the lines we need; keeps p95 latency sane
                crop = self._crop(image, d.box)
                r = ocr_line(crop, recipe=self.ocr_recipe)
                d.text, d.ocr_conf = r.text, r.confidence
            for d in dets:
                if d.cls == "total":
                    r = ocr_line(self._crop(image, d.box), numeric=True)
                    if normalise_total(r.text):
                        d.text, d.ocr_conf = r.text, r.confidence
            if self.fallback:
                n_fallback = self._fallback_fields(image, dets)
        t_ocr = time.perf_counter()

        seg_cov, seg_conf = None, None
        if self.segmenter is not None:
            g = image.convert("L").resize(SEG_SIZE, Image.BILINEAR)
            x = torch.from_numpy(np.asarray(g, dtype=np.float32) / 255.0)[None, None]
            with torch.no_grad(), self.lock:
                p = torch.sigmoid(self.segmenter(x))
            seg_cov = float((p > 0.5).float().mean())
            seg_conf = float(p.sub(0.5).abs().mean() * 2)
        t_seg = time.perf_counter()

        fields: Extraction = extract(dets)
        field_dets = [d for d in dets if d.cls != "other"]
        summary = {
            "n_detections": len(dets),
            "n_field_detections": len(field_dets),
            "det_conf_mean": float(np.mean([d.conf for d in dets])) if dets else 0.0,
            "ocr_conf_mean": float(np.mean([d.ocr_conf for d in field_dets])) if (run_ocr and field_dets) else None,
            "seg_text_coverage": seg_cov,
            "seg_confidence": seg_conf,
            "fields_found": sum(v is not None for v in (fields.company, fields.date, fields.address, fields.total)),
            "n_fallback_ocr": n_fallback,
            "latency_ms": {"detect": (t_det - t0) * 1e3, "ocr": (t_ocr - t_det) * 1e3, "segment": (t_seg - t_ocr) * 1e3,
                           "total": (t_seg - t0) * 1e3},
        }
        return {
            "fields": {k: v for k, v in asdict(fields).items()},
            "detections": [asdict(d) for d in dets],
            "summary": summary,
            "features": feats,
            "models": {k: v.version for k, v in self.info.items()},
        }
