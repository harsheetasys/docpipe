"""API contract tests with the pipeline stubbed out (no registry, no weights)."""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from docpipe.serving import app as app_module
from docpipe.serving.inference_log import InferenceLog


class FakePipeline:
    ready = True
    info = {}
    ocr_recipe = "raw"

    def predict(self, image, run_ocr=True):
        return {
            "fields": {"company": "ACME", "date": "01/01/2026", "address": None, "total": "9.00", "confidences": {}},
            "detections": [],
            "summary": {"n_detections": 3, "n_field_detections": 2, "det_conf_mean": 0.8, "ocr_conf_mean": 0.7,
                        "seg_text_coverage": 0.2, "seg_confidence": 0.9, "fields_found": 3,
                        "latency_ms": {"detect": 10.0, "ocr": 20.0, "segment": 5.0, "total": 35.0}},
            "features": {"width": 100.0, "height": 200.0, "aspect_ratio": 2.0, "brightness_mean": 200.0,
                         "brightness_std": 10.0, "contrast_rms": 10.0, "blur_laplacian_var": 500.0,
                         "dark_pixel_frac": 0.01, "edge_density": 0.1},
            "models": {"docpipe-detector": "3"},
        }

    def refresh(self, force=False):
        return {}


@pytest.fixture
def client(tmp_path, monkeypatch):
    app_module.state.clear()
    app_module.state["pipeline"] = FakePipeline()
    app_module.state["log"] = InferenceLog(tmp_path)
    # bypass lifespan (which would load real models)
    return TestClient(app_module.app, raise_server_exceptions=True)


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (100, 200), "white").save(buf, "PNG")
    return buf.getvalue()


def test_health_and_ready(client):
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/ready").status_code == 200


def test_predict_logs_and_exposes_metrics(client, tmp_path):
    r = client.post("/predict", params={"tag": "baseline"}, files={"file": ("a.png", _png(), "image/png")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"]["total"] == "9.00" and "request_id" in body
    df = InferenceLog(tmp_path).read()
    assert len(df) == 1 and df.iloc[0]["tag"] == "baseline" and df.iloc[0]["feat_width"] == 100.0
    m = client.get("/metrics").text
    assert 'docpipe_requests_total{status="ok"} 1.0' in m
    assert "docpipe_latency_seconds_bucket" in m


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_not_ready_returns_503(client):
    app_module.state["pipeline"].ready = False
    assert client.get("/ready").status_code == 503
    assert client.post("/predict", files={"file": ("a.png", _png(), "image/png")}).status_code == 503
