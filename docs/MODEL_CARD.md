# Model card: DocPipe field detector (`docpipe-detector`) and text segmenter (`docpipe-segmenter`)

## Intended use

Extract `company`, `date`, `address`, `total` from scanned retail receipts and simple
invoices, as a first-pass automation in a human-in-the-loop digitisation workflow.
Outputs carry per-field confidences so low-confidence fields can be routed to a reviewer.
Not intended for: handwritten documents, non-Latin scripts, legal or financial decisions
without review.

## Training data

ICDAR-2019 SROIE: 626 scanned receipts from Malaysian retailers, English, thermal and
laser prints, 2016–2019. Split 438 / 91 / 97 (train / val / test) by hash of `doc_id`.
Line-level field labels are derived from document-level key fields (see ADR 0001), so
they carry noise, most visibly on `total`.

## Models

| | Detector | Segmenter |
|---|---|---|
| Architecture | YOLOv8n, 3.0 M params | U-Net, base 16, ~0.5 M params |
| Input | RGB, 640 px long side | grayscale 160×320 |
| Output | boxes + 5 classes | text/background mask |
| Training | see `TRAINING_RECIPE.md` | see `TRAINING_RECIPE.md` |
| Registry | `docpipe-detector` | `docpipe-segmenter` |

## Evaluation (held-out test split, 97 receipts)

See `EXPERIMENTS.md` for the full tables and the data-scaling curve. Headline numbers of
the current champion are reproduced there, including per-field mAP50, end-to-end field
accuracy (exact and fuzzy), and OCR CER.

## Known limitations and failure modes

* **OCR digit confusion on thermal prints** (7↔1, 5↔8, 3↔8) is the largest source of
  wrong `total` values; the detector finds the right line more often than Tesseract reads
  it correctly. A fine-tuned recognizer is the planned fix.
* **`date` and `total` are rare classes** (2 % and 5 % of lines) and the detector is
  CPU-budget trained; the serving pipeline uses a lower confidence threshold for field
  classes plus a rule-based fallback (TOTAL-row matching, regex validation) to recover
  misses. The fallback's cost is logged (`n_fallback_ocr`).
* **Receipts outside the SROIE distribution** (phone photos, dark backgrounds, strong
  skew) degrade confidence sharply — this is exactly what the drift monitor is tuned to
  catch (see RUNBOOK §5 for the measured effect).
* **Address** is multi-line; exact match is strict and fuzzy ≥ 90 is the fairer metric.

## Ethical considerations

Receipts contain personal data (names, addresses, sometimes card fragments). The
inference log deliberately stores only image statistics and prediction summaries; images
are never persisted by the service. The models do not infer anything about people.

## Maintenance

Owner: ML platform. Retraining: nightly CronJob when the drift job raises a trigger, or
on new labelled batches. Promotion requires passing the gate in `registry/promote.py`.
Every alias change is recorded in `data/reports/promotions.jsonl`.
