-- DuckDB queries over the lake, the feature store and the inference log.
-- Run:  duckdb < sql/analytics.sql        (or paste into `python -c "import duckdb; ..."`)
-- Paths are relative to the repo root.

-- 1. Label distribution per split (did the hash split keep class balance?)
SELECT split, field_class, COUNT(*) AS n,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY split), 2) AS pct
FROM read_parquet('data/lake/lines.parquet/*.parquet')
GROUP BY 1, 2 ORDER BY 1, 3 DESC;

-- 2. Receipts where weak labelling found no `total` line (annotation QA queue for the pod)
SELECT d.doc_id, d.total AS gt_total, d.n_lines
FROM read_parquet('data/lake/docs.parquet/*.parquet') d
WHERE NOT EXISTS (
    SELECT 1 FROM read_parquet('data/lake/lines.parquet/*.parquet') l
    WHERE l.doc_id = d.doc_id AND l.field_class = 'total');

-- 3. Image-quality reference distribution (what the drift job compares against)
SELECT split,
       ROUND(AVG(brightness_mean), 1)        AS brightness,
       ROUND(AVG(contrast_rms), 1)           AS contrast,
       ROUND(MEDIAN(blur_laplacian_var), 0)  AS blur_median,
       ROUND(QUANTILE_CONT(blur_laplacian_var, 0.05), 0) AS blur_p05,
       COUNT(*) AS n
FROM read_parquet('data/features/offline/doc_image_features.parquet')
GROUP BY 1;

-- 4. Inference log: per-hour traffic, latency p95, confidence, fields found
SELECT date_trunc('hour', to_timestamp(ts))              AS hour,
       tag,
       COUNT(*)                                          AS requests,
       ROUND(QUANTILE_CONT(latency_ms, 0.95), 0)         AS p95_ms,
       ROUND(AVG(pred_det_conf_mean), 3)                 AS det_conf,
       ROUND(AVG(pred_ocr_conf_mean), 3)                 AS ocr_conf,
       ROUND(AVG(fields_found), 2)                       AS fields_found
FROM read_json_auto('data/inference_log/predictions-*.jsonl', union_by_name=true)
GROUP BY 1, 2 ORDER BY 1, 2;

-- 5. Baseline vs drifted traffic, side by side (the demo in docs/RUNBOOK.md)
SELECT tag,
       ROUND(AVG(feat_brightness_mean), 1)   AS brightness,
       ROUND(AVG(feat_blur_laplacian_var), 0) AS blur,
       ROUND(AVG(pred_det_conf_mean), 3)     AS det_conf,
       ROUND(AVG(pred_n_detections), 1)      AS detections,
       ROUND(AVG(fields_found), 2)           AS fields_found
FROM read_json_auto('data/inference_log/predictions-*.jsonl', union_by_name=true)
GROUP BY 1;

-- 6. Which model version served each request (needed to attribute a regression)
SELECT json_extract_string(models, '$."docpipe-detector"') AS detector_version,
       COUNT(*) AS requests, ROUND(AVG(pred_det_conf_mean), 3) AS det_conf
FROM read_json_auto('data/inference_log/predictions-*.jsonl', union_by_name=true)
GROUP BY 1;
