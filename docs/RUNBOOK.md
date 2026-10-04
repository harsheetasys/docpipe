# Runbook: drift, retraining, rollback

Audience: whoever is on call for DocPipe. Everything here is a command you can paste.

## 0. Where things are

| What | Where |
|---|---|
| Live model versions | `GET /models` on any API pod; `mlflow models` UI → aliases |
| Latest drift report | `data/reports/drift_latest.json` (PVC `/data` in k8s) |
| Drift history | `data/reports/drift_history.jsonl` |
| Promotion / rollback audit trail | `data/reports/promotions.jsonl` |
| Pending retrain | `data/reports/retrain_trigger.json` exists |
| Metrics | `GET /metrics` → Prometheus → `docpipe_*` series |

## 1. Alerts you will see

| Alert (Prometheus rule) | Meaning | First action |
|---|---|---|
| `docpipe_requests_total{status="error"}` rate > 1 % | inference exceptions | `kubectl logs deploy/docpipe-api` ; check `/models` for `last_refresh_error` |
| `histogram_quantile(0.95, docpipe_latency_seconds{stage="total"})` > 4 s | slow | check HPA (`kubectl get hpa`), OCR stage latency, node CPU throttling |
| `avg(docpipe_detection_confidence)` drops ≥ 0.1 over 1 h | prediction drift | run §2 immediately instead of waiting for the hourly job |
| drift job severity `retrain` | input distribution moved | §3 |
| drift job severity `rollback` | regression after model change | §4 |

## 2. Run drift analysis by hand

```bash
# last 24h of traffic
python -m docpipe.monitoring.drift --window-hours 24
# only a tagged slice (e.g. one client)
python -m docpipe.monitoring.drift --tag client-acme
# in k8s
kubectl create job --from=cronjob/docpipe-drift drift-manual-$(date +%s) -n docpipe
```

Reading the report:

* `input_drift.<feature>.psi` – < 0.10 fine, 0.10–0.25 moderate, > 0.25 major. A feature
  counts as drifted only if the KS test also rejects (p < 0.01); this avoids crying wolf on
  tiny windows.
* `prediction_drift.pred_det_conf_mean.drop` – fall in mean detector confidence vs the
  `baseline` tagged window.
* `severity` / `action` – the decision. `reasons` tells you which rule fired.

## 3. Retraining path (severity = `retrain`)

The drift job has written `retrain_trigger.json`. The nightly `docpipe-retrain` CronJob
will pick it up. To run it now:

```bash
# local
python -m docpipe.monitoring.retrain_if_triggered          # honours the trigger file
python -m docpipe.monitoring.retrain_if_triggered --force  # retrain regardless
# k8s
kubectl create job --from=cronjob/docpipe-retrain retrain-manual-$(date +%s) -n docpipe
kubectl logs -f job/retrain-manual-... -n docpipe
```

What it does: rebuild lake (new labels in `data/raw` are included) → export YOLO →
train → register `challenger` → **promotion gate**. Watch for the decision line:

```
{"decision": "promote", "gain": 0.031, "reasons": ["gain +0.0310 vs min_gain 0.0"]}
{"decision": "reject",  "reasons": ["test_mAP50_total regressed 0.812 -> 0.701"]}
```

A rejected challenger stays in the registry with its metrics for inspection; nothing
changes in production. If promoted, API pods log `reloaded` within 60 s
(`GET /models` shows the new version). Force it: `POST /admin/refresh`.

### Getting more labelled data into the loop

New batches from the delivery pod go to `data/raw/{img,box,key}` with the same file
naming. `spark_prep` recomputes line labels and the hash-based split, so previously
held-out receipts stay held out. Use `sql/analytics.sql` §2 to send the pod a QA list of
receipts where no `total` line could be derived.

## 4. Rollback paths

There are two independent levers. Decide which failed.

### 4a. Bad **model** (quality regression, inputs stable) → registry rollback, ~60 s

```bash
python -m docpipe.registry.rollback --model docpipe-detector --reason "fields_found dropped after v5"
# or pin a specific version
python -m docpipe.registry.rollback --model docpipe-detector --to-version 3
curl -X POST http://docpipe-api/admin/refresh      # optional: don't wait for the tick
```

`champion` now points at the previous version, the previous champion becomes `previous`
(so a rollback is itself reversible), and the event is appended to `promotions.jsonl`.
No pods restart, no image changes.

### 4b. Bad **deployment** (crash loop, config, dependency) → Kubernetes rollback

```bash
kubectl rollout status  deployment/docpipe-api -n docpipe
kubectl rollout history deployment/docpipe-api -n docpipe
kubectl rollout undo    deployment/docpipe-api -n docpipe              # previous revision
kubectl rollout undo    deployment/docpipe-api -n docpipe --to-revision=4
```

The Deployment uses `maxUnavailable: 0`, a PodDisruptionBudget of `minAvailable: 1`
and a readiness probe on `/ready` (model loaded), so a bad image never receives traffic
and an undo is zero-downtime.

### 4c. Both

Do 4b first (restore a healthy process), then 4a.

## 5. Demo of the whole loop (10 minutes, laptop)

```bash
make serve &                                   # API on :8000, loads champion
make traffic-baseline N=40                     # clean receipts, tag=baseline
make traffic-drifted  N=40                     # blurred / dark / skewed / recompressed, tag=drifted
make drift TAG=drifted                         # -> severity retrain, retrain_trigger.json written
python -m docpipe.monitoring.retrain_if_triggered --epochs 3   # quick retrain + gate
make rollback                                  # demonstrate model rollback
curl -s localhost:8000/models | jq
```

Expected: `drifted` traffic shows PSI > 0.25 on `blur_laplacian_var` and
`brightness_mean`, detector confidence and `fields_found` fall, severity = `retrain`.
See docs/EXPERIMENTS.md for the numbers from the reference run.

## 6. Safety rails

* The retrain worker never promotes on its own merit; the gate does.
* `concurrencyPolicy: Forbid` on both CronJobs: one drift job, one retrain at a time.
* Logs contain no images, only features and prediction summaries.
* Every alias change is appended to `promotions.jsonl` with a timestamp and reason.
