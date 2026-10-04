"""Large-scale data preparation with PySpark.

Reads the raw annotation files (one CSV of boxes + one JSON of key fields per receipt),
builds two governed parquet tables in the "lake", and computes dataset statistics with
Spark SQL:

    lake/lines.parquet  -> one row per text line  (doc_id, box, text, field_class, split)
    lake/docs.parquet   -> one row per receipt    (doc_id, width, height, n_lines, split, ...)

The same job runs unchanged on a laptop (local[*]) or on a cluster (spark-submit) because
nothing here touches the driver's filesystem except reading the raw files and the final
write. On 626 receipts this is overkill; on 180M records it is the only thing that works,
and keeping one code path for both is the point.

Usage: python -m docpipe.data.spark_prep
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from docpipe.config import LAKE_DIR, RAW_DIR, SEED, TEST_FRACTION, VAL_FRACTION
from docpipe.data.labels import assign_field

LINE_SCHEMA = T.StructType(
    [
        T.StructField("doc_id", T.StringType()),
        T.StructField("x1", T.IntegerType()),
        T.StructField("y1", T.IntegerType()),
        T.StructField("x2", T.IntegerType()),
        T.StructField("y2", T.IntegerType()),
        T.StructField("text", T.StringType()),
    ]
)


def spark_session(app: str = "docpipe-prep") -> SparkSession:
    return (
        SparkSession.builder.appName(app)
        .master(os.getenv("SPARK_MASTER", "local[*]"))
        .config("spark.sql.shuffle.partitions", os.getenv("SPARK_SHUFFLE_PARTITIONS", "8"))
        .config("spark.driver.memory", os.getenv("SPARK_DRIVER_MEMORY", "2g"))
        .config("spark.ui.enabled", "false")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )


def _parse_box_file(path: str) -> list[tuple]:
    """Each row: x1,y1,x2,y1,x2,y2,x1,y2,text  (text may itself contain commas)."""
    doc_id = Path(path).stem
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.rstrip("\n")
            if not raw:
                continue
            parts = raw.split(",", 8)
            if len(parts) < 9:
                continue
            xs = [int(parts[i]) for i in (0, 2, 4, 6)]
            ys = [int(parts[i]) for i in (1, 3, 5, 7)]
            rows.append((doc_id, min(xs), min(ys), max(xs), max(ys), parts[8]))
    return rows


def build_lake(raw_dir: Path = RAW_DIR, lake_dir: Path = LAKE_DIR) -> dict:
    spark = spark_session()
    sc = spark.sparkContext

    box_files = sorted(str(p) for p in (raw_dir / "box").glob("*.csv"))
    key_files = sorted(str(p) for p in (raw_dir / "key").glob("*.json"))

    # ---- lines: parallelise file parsing across executors -------------------------------
    lines_rdd = sc.parallelize(box_files, numSlices=16).flatMap(_parse_box_file)
    lines = spark.createDataFrame(lines_rdd, LINE_SCHEMA)

    # ---- key fields (document-level weak labels) ----------------------------------------
    def _read_key(path: str):
        d = json.load(open(path, encoding="utf-8"))
        return (Path(path).stem, d.get("company", ""), d.get("date", ""), d.get("address", ""), d.get("total", ""))

    keys = spark.createDataFrame(
        sc.parallelize(key_files, numSlices=16).map(_read_key),
        "doc_id string, company string, date string, address string, total string",
    )

    # ---- image dimensions (needed to normalise boxes for YOLO) --------------------------
    def _img_dims(doc_id: str):
        from PIL import Image

        with Image.open(raw_dir / "img" / f"{doc_id}.jpg") as im:
            return (doc_id, im.width, im.height)

    dims = spark.createDataFrame(
        sc.parallelize([Path(p).stem for p in box_files], numSlices=16).map(_img_dims),
        "doc_id string, width int, height int",
    )

    # ---- line-level field labels via UDF ------------------------------------------------
    @F.udf(T.StringType())
    def field_udf(text, company, date, address, total):
        return assign_field(text or "", {"company": company, "date": date, "address": address, "total": total})

    lines = (
        lines.join(keys, "doc_id", "left")
        .withColumn("field_class", field_udf("text", "company", "date", "address", "total"))
        .drop("company", "date", "address", "total")
    )

    # ---- deterministic document-level split (hash of doc_id, so it never changes) -------
    docs = (
        lines.groupBy("doc_id")
        .agg(
            F.count("*").alias("n_lines"),
            F.sum(F.when(F.col("field_class") != "other", 1).otherwise(0)).alias("n_field_lines"),
            F.avg(F.col("y2") - F.col("y1")).alias("mean_line_height"),
            F.avg(F.length("text")).alias("mean_chars_per_line"),
        )
        .join(dims, "doc_id")
        .join(keys, "doc_id")
        .withColumn("bucket", F.pmod(F.xxhash64(F.concat(F.col("doc_id"), F.lit(str(SEED)))), F.lit(1000)))
        .withColumn(
            "split",
            F.when(F.col("bucket") < TEST_FRACTION * 1000, "test")
            .when(F.col("bucket") < (TEST_FRACTION + VAL_FRACTION) * 1000, "val")
            .otherwise("train"),
        )
        .drop("bucket")
    )
    lines = lines.join(docs.select("doc_id", "split", "width", "height"), "doc_id")

    # ---- quality gates expressed in SQL -------------------------------------------------
    lines.createOrReplaceTempView("lines")
    docs.createOrReplaceTempView("docs")
    bad_boxes = spark.sql(
        "SELECT COUNT(*) AS n FROM lines WHERE x2<=x1 OR y2<=y1 OR x1<0 OR y1<0 OR x2>width OR y2>height"
    ).first()["n"]
    lines = lines.filter("x2>x1 AND y2>y1 AND x1>=0 AND y1>=0 AND x2<=width AND y2<=height")

    stats = {
        "n_docs": docs.count(),
        "n_lines": lines.count(),
        "dropped_invalid_boxes": int(bad_boxes),
        "split_counts": {r["split"]: r["n"] for r in spark.sql("SELECT split, COUNT(*) n FROM docs GROUP BY split").collect()},
        "class_counts": {
            r["field_class"]: r["n"]
            for r in spark.sql("SELECT field_class, COUNT(*) n FROM lines GROUP BY field_class ORDER BY n DESC").collect()
        },
        "docs_missing_total_line": spark.sql(
            "SELECT COUNT(*) n FROM docs d WHERE NOT EXISTS (SELECT 1 FROM lines l WHERE l.doc_id=d.doc_id AND l.field_class='total')"
        ).first()["n"],
    }

    lake_dir.mkdir(parents=True, exist_ok=True)
    lines.coalesce(1).write.mode("overwrite").parquet(str(lake_dir / "lines.parquet"))
    docs.coalesce(1).write.mode("overwrite").parquet(str(lake_dir / "docs.parquet"))
    with open(lake_dir / "stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    spark.stop()
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    build_lake()
    sys.exit(0)
