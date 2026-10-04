.PHONY: data prep features train train-seg ocr-eval promote rollback serve drift traffic-baseline traffic-drifted test lint docker k8s-local mlflow-ui

PY ?= python
export MLFLOW_DISABLE_AGENT_HINT=1

data:            ## download SROIE receipts
	$(PY) -m docpipe.data.download
prep:            ## Spark: raw -> parquet lake + YOLO exports
	$(PY) -m docpipe.data.spark_prep && $(PY) -m docpipe.data.yolo_export
features:        ## populate + materialize the feature store
	$(PY) -m docpipe.features.build
train:           ## detector (production candidate, registered as challenger)
	$(PY) -m docpipe.train.train_detector --epochs $${EPOCHS:-12}
train-seg:       ## U-Net text segmenter
	$(PY) -m docpipe.train.train_segmenter --epochs $${EPOCHS:-8}
scaling:         ## data-scaling experiment (25/50/100 % of labels)
	scripts/run_detector_experiments.sh
ocr-eval:        ## benchmark OCR preprocessing recipes
	$(PY) -m docpipe.train.eval_ocr --lines $${LINES:-600}
promote:         ## promotion gate challenger -> champion
	$(PY) -m docpipe.registry.promote --model docpipe-detector && $(PY) -m docpipe.registry.promote --model docpipe-segmenter
rollback:        ## champion -> previous
	$(PY) -m docpipe.registry.rollback --model docpipe-detector
serve:           ## run the API locally
	uvicorn docpipe.serving.app:app --host 0.0.0.0 --port 8000
traffic-baseline:
	$(PY) -m docpipe.monitoring.simulate_traffic --mode baseline --n $${N:-40}
traffic-drifted:
	$(PY) -m docpipe.monitoring.simulate_traffic --mode drifted --n $${N:-40}
drift:           ## run drift analysis on the last 24h
	$(PY) -m docpipe.monitoring.drift --window-hours 24 $(if $(TAG),--tag $(TAG),)
mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///data/mlflow.db --port 5000
test:
	pytest
lint:
	ruff check docpipe tests
docker:
	docker build -f docker/Dockerfile.api -t docpipe-api:local . && docker build -f docker/Dockerfile.train -t docpipe-train:local .
k8s-local:       ## deploy to kind/minikube
	kubectl apply -k k8s/overlays/local
help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'
