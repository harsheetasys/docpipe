# ADR 0001: Derive line-level field labels from document-level key fields

**Status:** accepted

## Context
SROIE (and most pod-delivered digitisation batches) gives us, per document, (a) every text
line with its box and transcription and (b) the extracted key fields. It does not say which
line produced which field. Training a detector that outputs field classes needs line-level
labels.

## Options
1. Pay for a second annotation pass that tags lines with fields.
2. Train a plain text-line detector and classify fields downstream with heuristics on OCR text.
3. Derive line labels automatically by fuzzy-matching line transcriptions against the key
   fields, accept some label noise, and let the detector learn field classes directly.

## Decision
Option 3. It costs nothing, produces 4,778 field-labelled lines out of 33,626, and keeps the
detector end-to-end (one model, one set of boxes, no second classifier to monitor). The
matching rules live in one pure function (`labels.assign_field`) with unit tests, so the
delivery pod can review and tune them.

## Consequences
* Label noise: `total` is over-assigned (~2.5 lines/doc; sub-totals that share the amount).
  Post-processing prefers lines that literally say TOTAL. A second annotation pass would
  tighten this; the data-scaling curve tells us whether more labels or cleaner labels is the
  better next spend.
* 2 receipts have no derivable `total` line; `sql/analytics.sql` §2 produces the QA list.
