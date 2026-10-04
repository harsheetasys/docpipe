# Architecture

DocPipe takes a scanned receipt or invoice and returns four structured fields
(`company`, `date`, `address`, `total`) plus every text-line detection. Around that
inference path sits the full model lifecycle: data prep at scale, tracked training,
a gated registry, monitoring, automatic retraining, and two independent rollback paths.

```
                 ┌────────────────────────────── OFFLINE ──────────────────────────────┐
 raw scans +     │  Spark job          lake (parquet)       YOLO export    MLflow       │
 pod labels ───▶ │  spark_prep.py ───▶ lines / docs ──────▶ yolo_export ─▶ train_*.py  │
 (data/raw)      │        │                  │                              │  runs +   │
                 │        │                  ▼                              │  registry │
                 │        │        feature store (offline parquet ─▶ online sqlite)     │
                 │        │           doc_image_features = drift REFERENCE   │          │
                 │        ▼                                                  ▼          │
                 │   stats.json / SQL quality gates                promote.py (gate)    │
                 └─────────────────────────────────────────────── champion alias ───────┘
                                                                            │
                 ┌────────────────────────────── ONLINE ──────────────────────────────┐
                 │  FastAPI pod(s)  ── loads models:/<name>@champion, refreshes every 60s│
   image ──────▶ │  detector (YOLOv8n) ─▶ OCR (tesseract) ─▶ post-process ─▶ fields    │
                 │  segmenter (U-Net)  ─▶ text coverage / confidence (health signal)    │
                 │  /metrics (Prometheus)    inference_log/*.jsonl (features + preds)   │
                 └─────────────────────────────────────────────────────┬──────────────┘
                                                                        │ hourly CronJob
                 ┌────────────────────────────── MONITOR ──────────────▼──────────────┐
                 │  drift.py: PSI + KS on image features vs feature-store reference    │
                 │            confidence / fields_found vs baseline window             │
                 │  ok → warn → retrain_trigger.json → nightly retrain → promote gate  │
                 │  "predictions degraded, inputs stable, model just changed" → rollback│
                 └─────────────────────────────────────────────────────────────────────┘
```

## Components

| Layer | Module | Technology | Why |
|---|---|---|---|
| Data prep | `docpipe/data/spark_prep.py` | PySpark + Spark SQL | One code path from 626 receipts to 180M records; quality gates as SQL |
| Weak labelling | `docpipe/data/labels.py` | rapidfuzz | Pods deliver doc-level fields; the model needs line-level classes |
| Dataset versioning | `train_detector.dataset_fingerprint` | sha256 of label files | Every run is pinned to an exact dataset; the promotion gate checks `n_train` |
| Feature store | `docpipe/features/store.py` | DuckDB (offline) + SQLite (online) | Point-in-time joins, materialize, TTL. Feast-shaped API |
| Detector | `train_detector.py` | YOLOv8n (PyTorch) | 5-class text-line detection = key-information extraction |
| Segmenter | `models/unet.py` | PyTorch U-Net, 0.5M params | Text-region mask: catches skew, cheap health signal |
| OCR | `docpipe/ocr.py` | Tesseract 5 LSTM | No weights to ship; we benchmark preprocessing recipes instead |
| Tracking/registry | MLflow | sqlite locally, server in compose/k8s | Params, metrics per epoch, artifacts, aliases `champion/challenger/previous` |
| Serving | `serving/app.py` | FastAPI + Prometheus client | Liveness vs readiness split; hot model refresh |
| Monitoring | `monitoring/drift.py` | scipy (PSI, KS) | No labels needed for input/prediction drift |
| Orchestration | `k8s/` | Kustomize, HPA, CronJobs, PDB | Zero-downtime rollouts, scheduled drift + retrain |
| CI | `.github/workflows/ci.yml` | ruff, pytest, buildx, kubeconform | Manifests are validated like code |

## Key design decisions

**Models are resolved by alias, not by image tag.** The serving image never contains
weights. It asks the registry for `models:/docpipe-detector@champion` on start and every
60 s. Promotion and rollback are therefore registry operations that take effect in
≤ 60 s across every pod, with no redeploy. A bad *container* (dependency bug, config) is
rolled back separately with `kubectl rollout undo`.

**Promotion is a gate, not a default.** Training always registers a `challenger`.
`promote.py` compares it with the champion on the held-out test split, refuses per-class
regressions (> 5 pts on any field) and refuses models trained on fewer samples. Every
decision is appended to `reports/promotions.jsonl`.

**One feature function for training and serving.** `doc_features.compute()` builds the
reference distribution in the feature store and is called on every request. Drift is
the comparison of those two, so training/serving skew cannot hide inside the monitor.

**Drift ≠ rollback.** Input drift (new scanner, phone photos) means the world changed →
retrain. Prediction drift *without* input drift, right after a champion change, means we
broke it → rollback. The monitor distinguishes the two cases.

**Logs hold features, not images.** Receipts contain names and addresses. The inference
log stores image statistics and prediction summaries only.

## Request lifecycle

1. `POST /predict` receives the image; `doc_features.compute()` runs (≈5 ms).
2. YOLOv8n at 640 px returns boxes + class per text line.
3. Only non-`other` lines are cropped and OCR'd (`psm 7`, best recipe from the benchmark).
4. `postprocess.extract()` picks per-field winners and regex-normalises `total`/`date`.
5. U-Net produces text coverage and mean confidence.
6. Prometheus histograms are updated; one JSON line goes to the inference log.

## Repository layout

```
docpipe/            package (data, features, models, train, registry, serving, monitoring)
configs/            drift thresholds
docker/             Dockerfile.api, Dockerfile.train, docker-compose.yml, prometheus.yml
k8s/base            namespace, configmap, PVCs, mlflow, api (Deployment/Service/PDB), HPA, CronJobs
k8s/overlays/local  kind/minikube patch
sql/                DuckDB analytics over lake + inference log
scripts/            experiment runners
tests/              unit + API contract tests
docs/               this folder (+ adr/)
```
