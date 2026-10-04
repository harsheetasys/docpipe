"""FastAPI serving layer.

  POST /predict            multipart image -> fields + detections
  GET  /health             liveness  (process is up)
  GET  /ready              readiness (champion models loaded)  -> k8s only routes traffic when 200
  GET  /models             which registry versions are live in this pod
  POST /admin/refresh      force a champion reload (called after promote/rollback)
  GET  /metrics            Prometheus exposition (latency histograms, confidences, counts)

Run: uvicorn docpipe.serving.app:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import io
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from docpipe import __version__
from docpipe.serving.inference_log import InferenceLog
from docpipe.serving.pipeline import Pipeline

REFRESH_SECONDS = int(os.getenv("DOCPIPE_MODEL_REFRESH_SECONDS", "60"))

REQS = Counter("docpipe_requests_total", "Prediction requests", ["status"])
LAT = Histogram("docpipe_latency_seconds", "End-to-end latency", ["stage"],
                buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 4, 8))
DET_CONF = Histogram("docpipe_detection_confidence", "Mean detector confidence per request", buckets=[i / 10 for i in range(11)])
OCR_CONF = Histogram("docpipe_ocr_confidence", "Mean OCR confidence per request", buckets=[i / 10 for i in range(11)])
FIELDS_FOUND = Histogram("docpipe_fields_found", "Key fields extracted per request (0-4)", buckets=(0, 1, 2, 3, 4))
MODEL_VERSION = Gauge("docpipe_model_version", "Registry version currently served", ["model"])

state: dict = {}


def _refresher(stop: threading.Event):
    while not stop.wait(REFRESH_SECONDS):
        try:
            changed = state["pipeline"].refresh()
            for name, v in changed.items():
                MODEL_VERSION.labels(model=name).set(float(v))
        except Exception as e:  # never let a registry hiccup kill serving
            state["last_refresh_error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["pipeline"] = Pipeline()
    state["log"] = InferenceLog()
    for name, info in state["pipeline"].info.items():
        MODEL_VERSION.labels(model=name).set(float(info.version))
    stop = threading.Event()
    threading.Thread(target=_refresher, args=(stop,), daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="DocPipe", version=__version__, lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/ready")
def ready():
    p = state.get("pipeline")
    if p is None or not p.ready:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ready", "models": {k: v.version for k, v in p.info.items()}}


@app.get("/models")
def models():
    p = state["pipeline"]
    return {k: {"version": v.version, "run_id": v.run_id, "loaded_at": v.loaded_at} for k, v in p.info.items()} | {
        "ocr_recipe": p.ocr_recipe, "last_refresh_error": state.get("last_refresh_error")}


@app.post("/admin/refresh")
def refresh():
    changed = state["pipeline"].refresh(force=True)
    return {"reloaded": changed or "unchanged"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), ocr: bool = Query(True), tag: str | None = Query(None)):
    p = state.get("pipeline")
    if p is None or not p.ready:
        raise HTTPException(503, "model not loaded")
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        REQS.labels(status="bad_input").inc()
        raise HTTPException(400, "not an image") from None
    t0 = time.perf_counter()
    try:
        result = p.predict(img, run_ocr=ocr)
    except Exception as e:
        REQS.labels(status="error").inc()
        raise HTTPException(500, f"inference failed: {e}") from e
    REQS.labels(status="ok").inc()
    s = result["summary"]
    for stage, ms in s["latency_ms"].items():
        LAT.labels(stage=stage).observe(ms / 1000)
    DET_CONF.observe(s["det_conf_mean"])
    if s["ocr_conf_mean"] is not None:
        OCR_CONF.observe(s["ocr_conf_mean"])
    FIELDS_FOUND.observe(s["fields_found"])
    rid = state["log"].write(result, tag=tag)
    return {"request_id": rid, **result, "wall_ms": (time.perf_counter() - t0) * 1e3}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
