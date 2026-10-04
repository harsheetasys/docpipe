# Interview guide: how to talk about DocPipe

Use this to prepare. Each section is a question you are likely to get, the 30-second
answer, and the follow-up depth if they dig. Numbers come from `EXPERIMENTS.md`.

## "Walk me through the project."

> DocPipe extracts company, date, address and total from scanned receipts. A YOLOv8n
> detector finds text lines and classifies them into field types, Tesseract reads only the
> lines we need, a small U-Net gives a text-coverage health signal, and rules normalise the
> output. Around that I built the lifecycle the job description describes: PySpark data
> prep with SQL quality gates, a feature store, MLflow tracking and a model registry with
> champion/challenger/previous aliases, a promotion gate, a FastAPI service with Prometheus
> metrics, a drift job (PSI + KS on image features, confidence drop on predictions) that
> writes a retraining trigger, a retrain CronJob that goes back through the gate, and two
> rollback paths: registry alias for a bad model, `kubectl rollout undo` for a bad
> container. Docker and Kustomize manifests for all of it, CI validates the manifests.

## "Why a detector with field classes instead of OCR-everything-then-NLP?"

End-to-end is simpler to monitor (one model, one mAP), cheaper at inference (we OCR ~4 of
~54 lines per receipt), and the labels were derivable for free from the key fields
(ADR 0001). The trade-off is label noise on `total`; post-processing absorbs it. If the
field set grew to 30 (line items, taxes), I would move to a layout-aware token classifier
(LayoutLM-style) and keep this detector as the region proposer.

## "How do you know more labelled data would help?" ← their core business

The data-scaling experiment: same val/test, nested 25 % / 50 % / 100 % train subsets,
identical recipe. mAP50 and per-field accuracy vs number of labelled receipts gives the
slope. If the curve is still steep at 100 %, the next rupee goes to annotation; if it is
flat, it goes to label *quality* or model capacity. That is the conversation I would have
with a delivery pod lead every batch.

## "Explain your drift detection."

Three signals. **Input**: per-feature PSI with quantile bins from the training reference
(from the feature store) plus a two-sample KS test; a feature is drifted only if PSI ≥ 0.1
*and* KS p < 0.01, retrain if PSI ≥ 0.25. **Prediction**: mean detector confidence and
fields-found vs a baseline window, no labels needed. **Performance**: when the pod sends
labels back, exact-field accuracy on recent traffic. Then the decision matrix: input
drift → retrain; prediction drift with stable inputs right after a model change → that's
us, roll back.

Follow-up — *why PSI and KS, not Evidently?* Evidently is a fine wrapper around the same
tests; writing 20 lines keeps the thresholds visible in a YAML file and the dependency
tree small for a CronJob image. I would adopt Evidently for the HTML reports if
stakeholders wanted them.

## "What stops a bad retrain from reaching production?"

The gate (`promote.py`): challenger must beat champion on held-out test mAP50, no field
may regress more than 5 points, and it must have trained on at least as many samples.
Rejections are logged with reasons. Retraining and promotion are deliberately separate
jobs.

## "Describe your rollback."

Model rollback: `champion` alias → `previous`; pods poll the registry every 60 s, no
restart, no image. Deployment rollback: `kubectl rollout undo`, zero downtime because
`maxUnavailable: 0` + readiness probe on `/ready` (model loaded) + PDB. Both are in the
runbook with exact commands; both are reversible; both leave an audit line.

## "Why Spark for 626 receipts?"

Because the job is written once. The same `spark_prep.py` runs on `local[*]` for a laptop
and `spark-submit` on a cluster for 180 M records; the only thing that changes is
`SPARK_MASTER`. The quality gates are Spark SQL, so they scale with the data. On a laptop
it is overkill and I say so in the docstring.

## "What's in your feature store and why not Feast?"

One feature view: image statistics per receipt (brightness, contrast, Laplacian blur,
dark-pixel fraction, edge density, aspect). Offline parquet + DuckDB with point-in-time
retrieval; online SQLite with TTL; `materialize()`. The API mirrors Feast so swapping is
mechanical; I did not pull in Feast's dependency tree for one view (ADR 0002). The real
value is that the *same function* computes features at training and serving time, which
is what makes drift comparisons honest.

## "Liveness vs readiness?"

Liveness = the process is alive (`/health`). Readiness = the champion model is loaded
(`/ready` returns 503 until then). A pod pulling a new model is alive but must not get
traffic. Startup probe gives the first model download time without the liveness probe
killing the pod.

## "What would you do with a GPU and a week?"

1. Fine-tune TrOCR or PaddleOCR-rec on the line crops — Tesseract CER is the ceiling on
   field accuracy now. 2. Move to YOLOv8s/m at 1024 px for the small-font lines.
3. Add a performance-drift feedback loop with the pod's corrections as labels.
4. Canary deployments (Argo Rollouts) with the end-to-end field accuracy as the analysis
   metric.

## "What went wrong / what did you learn?"

Be honest here; use the OCR benchmark. The "obvious" preprocessing (2× upscale + Otsu)
*hurt* CER on these scans because Tesseract's LSTM already handles grayscale well and
binarisation destroys thin strokes. Measuring before shipping a preprocessing step is the
lesson, and the benchmark is now part of the recipe so the next person doesn't re-learn it.

## Technology checklist vs the JD

| JD item | Where in the repo |
|---|---|
| PyTorch | U-Net (`models/unet.py`), YOLOv8 (PyTorch under the hood) |
| Computer vision: detection / segmentation / OCR | detector, segmenter, `ocr.py` |
| MLflow | tracking, registry, aliases, pyfunc + pytorch flavors |
| Docker | `docker/Dockerfile.api` (multi-stage, non-root), `Dockerfile.train`, compose |
| Kubernetes | Deployment (rolling, probes, PDB), HPA, CronJobs, Kustomize overlays |
| Feature store | `features/store.py` |
| Model monitoring | `/metrics`, `monitoring/drift.py` |
| Drift detection, retraining triggers, rollback paths | `drift.py`, `retrain_if_triggered.py`, `registry/rollback.py` + runbook |
| SQL | Spark SQL gates, DuckDB (`sql/analytics.sql`, feature store) |
| Spark | `data/spark_prep.py` |
| Big data | partitioned parquet lake, hash splits, same job laptop → cluster |
| Labelled data → measurable accuracy gains | data-scaling experiment |
| Document experiments, reproducible recipes | `EXPERIMENTS.md`, `TRAINING_RECIPE.md`, MLflow |
| Technical writing | this `docs/` folder, ADRs, runbook |
