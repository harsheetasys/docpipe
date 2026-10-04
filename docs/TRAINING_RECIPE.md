# Reproducible training recipe

Goal of this document: a second engineer, on a fresh machine, reproduces the registered
models and the numbers in `EXPERIMENTS.md` to within run-to-run noise, without asking
anyone anything. Everything is pinned: data version (hash), split (hash), seed, hyper-
parameters, and environment (image).

## 0. Environment

```bash
# Python 3.10–3.13, Tesseract 5, Java 17+ (Spark)
sudo apt-get install -y tesseract-ocr default-jre-headless           # Debian/Ubuntu
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt
# or: docker build -f docker/Dockerfile.train -t docpipe-train:local .
```

The reference run used: Python 3.13, torch 2.14 (CPU), ultralytics 8.4, MLflow 3.16,
PySpark 4.2, Tesseract 5.3.4, 2 vCPU / 7 GB RAM. Nothing requires a GPU; a GPU only
changes wall-clock.

## 1. Data

**Source.** ICDAR-2019 SROIE task 1/2/3: 626 scanned receipts, one CSV of text-line
quadrilaterals + transcriptions per receipt, one JSON of key fields (`company`, `date`,
`address`, `total`). Public mirror: `github.com/zzzDavid/ICDAR-2019-SROIE`.

```bash
make data            # python -m docpipe.data.download  -> data/raw/{img,box,key}
```

**Lake.** The Spark job (`make prep`) does, in order:

1. parses boxes (`min/max` of the quadrilateral → axis-aligned box),
2. assigns each line a field class with `labels.assign_field()` (fuzzy match against the
   document-level key fields; rules documented in the docstring; thresholds `85`),
3. splits by `xxhash64(doc_id || seed) mod 1000` → 70 / 15 / 15 train/val/test.
   The split is a pure function of `doc_id`, so adding receipts never moves an existing
   one between splits,
4. drops invalid boxes via a SQL gate and writes `data/lake/{lines,docs}.parquet` +
   `stats.json`.

Reference `stats.json`:

```json
{"n_docs": 626, "n_lines": 33626, "dropped_invalid_boxes": 0,
 "split_counts": {"train": 438, "val": 91, "test": 97},
 "class_counts": {"other": 28848, "address": 1694, "total": 1578, "date": 762, "company": 744},
 "docs_missing_total_line": 2}
```

If your counts differ, stop: the dataset or the labelling rules changed.

**YOLO export.** `python -m docpipe.data.yolo_export [--fraction f]` writes symlinked
images and normalised `cls cx cy w h` labels to `data/yolo` (or `data/yolo_f050`, …).
Subsets are nested (25 % ⊂ 50 % ⊂ 100 %) and val/test are identical across fractions,
which is what makes the data-scaling curve comparable. `dataset_sha256` of the 100 %
export in the reference run: `d4a58864752e58b7` (25 %: `0e12762db269718a`).

**Feature store.** `make features` computes `doc_image_features` for every receipt and
materializes the online store. The train-split rows are the drift reference.

## 2. Detector

```bash
python -m docpipe.train.train_detector --epochs 12 --imgsz 640 --batch 8
```

| Setting | Value | Why |
|---|---|---|
| Architecture | YOLOv8n, COCO-pretrained (`yolov8n.pt`) | 3 M params, CPU-trainable, fine-tunes fast on dense small boxes |
| Classes | company, date, address, total, other | detection + key-information extraction in one model |
| Image size | 640 (long side) | receipts are ~460×1000; 640 keeps ~8 px x-height legible |
| Batch | 8 | CPU memory |
| Epochs | 12 (reference); 30 for the nightly retrain | mAP50 still rising at 12 on CPU budget; see curve |
| Optimizer / LR | Ultralytics `auto` (SGD, lr0 0.01, warm-up 3 epochs) | default, not tuned |
| Augmentation | `mosaic=0`, `fliplr=0`; default HSV/scale/translate kept | receipts are never mirrored; mosaic destroys reading order |
| Seed | 42, `deterministic=True` | |
| Model selection | best `val` fitness epoch (Ultralytics default) | |
| Reported metric | `test_mAP50`, per-class `test_mAP50_<field>` | test split never touched during training |

Each run logs to MLflow: params above + `dataset_sha256`, `n_train/val/test`, git SHA,
torch version; per-epoch `metrics_mAP50`, `val_box_loss`, …; artifacts
`results.csv`, `results.png`, `confusion_matrix.png`; a pyfunc model wrapping `best.pt`;
and registers version N with alias `challenger`.

## 3. Segmenter

```bash
python -m docpipe.train.train_segmenter --epochs 6
```

U-Net (base 16 channels, 3 down/up levels, ~0.5 M params), grayscale input resized to
160×320, target = union of line boxes rasterised to a mask. Loss = BCE + Dice.
AdamW lr 2e-3, OneCycle, batch 8, seed 42. Best epoch by val IoU; reported `test_iou`.

## 4. OCR recipe benchmark

```bash
python -m docpipe.train.eval_ocr --lines 600
```

600 ground-truth line crops sampled deterministically from the test split
(`hash(doc_id||x1||y1||seed)`), length ≥ 3 chars. Each recipe in `ocr.RECIPES` is one
nested MLflow run. Metric: CER (Levenshtein / len(truth), upper-cased, whitespace
collapsed). The lowest-CER recipe is written to `data/reports/ocr_recipe.json` and used
by the serving pipeline (override with `DOCPIPE_OCR_RECIPE`).

## 5. Promotion

```bash
python -m docpipe.registry.promote --model docpipe-detector
python -m docpipe.registry.promote --model docpipe-segmenter
```

Rules: see `docpipe/registry/promote.py` docstring (gain ≥ `min_gain`, no field
regresses > 5 pts, `n_train` not smaller than the champion's). First promotion has no
champion to compare with and goes through.

## 6. End-to-end evaluation

```bash
python -m docpipe.train.eval_end2end --n 97      # whole test split
```

Runs the champion pipeline on test receipts and scores the four fields against SROIE
ground truth (exact and fuzzy ≥ 90). This is the number the client cares about; mAP is
only a proxy.

## 7. Full reproduction in one go

```bash
make data prep features
scripts/run_detector_experiments.sh      # 100 %, 25 %, 50 %  (EPOCHS=12)
make train-seg ocr-eval promote
python -m docpipe.train.eval_end2end --n 97
make mlflow-ui                           # compare runs at http://localhost:5000
```

Wall-clock on 2 vCPU: ~55 min for the 100 % detector, ~15 + 30 min for the 25/50 %
runs, ~10 min segmenter, ~10 min OCR benchmark.

## 8. What is *not* deterministic

* Tesseract's LSTM is deterministic for a given binary; a different Tesseract build will
  move CER by a few tenths of a point.
* Ultralytics on CPU with `deterministic=True` is bit-reproducible on the same machine;
  across CPU microarchitectures expect ±0.01 mAP50.
* Wall-clock numbers depend on the box; nothing else does.
