"""Export the lake (parquet) into the YOLO directory layout the detector trainer expects.

    data/yolo/images/{train,val,test}/<doc_id>.jpg   (symlinks, no copies)
    data/yolo/labels/{train,val,test}/<doc_id>.txt   (class cx cy w h, normalised)
    data/yolo/dataset.yaml

`fraction` lets the data-scaling experiment train on a deterministic subset of the
training split while keeping val/test fixed, so accuracy numbers are comparable.

Usage: python -m docpipe.data.yolo_export [--fraction 0.5]
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import duckdb
import yaml

from docpipe.config import FIELD_CLASSES, LAKE_DIR, RAW_DIR, SEED, YOLO_DIR


def export(fraction: float = 1.0, out_dir: Path | None = None, lake_dir: Path = LAKE_DIR) -> Path:
    out_dir = out_dir or (YOLO_DIR if fraction >= 1.0 else YOLO_DIR.parent / f"yolo_f{int(fraction * 100):03d}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW lines AS SELECT * FROM read_parquet('{lake_dir / 'lines.parquet'}/*.parquet')")
    con.execute(f"CREATE VIEW docs AS SELECT * FROM read_parquet('{lake_dir / 'docs.parquet'}/*.parquet')")

    # Deterministic subset of train: order by a hash so 25% ⊂ 50% ⊂ 100%.
    docs = con.execute(
        f"""
        WITH ranked AS (
            SELECT doc_id, split, width, height,
                   ROW_NUMBER() OVER (PARTITION BY split ORDER BY hash(doc_id || '{SEED}')) AS rn,
                   COUNT(*)    OVER (PARTITION BY split) AS n
            FROM docs)
        SELECT doc_id, split, width, height FROM ranked
        WHERE split <> 'train' OR rn <= CEIL(n * {fraction})
        """
    ).fetchall()

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    cls_idx = {c: i for i, c in enumerate(FIELD_CLASSES)}
    n_written = {"train": 0, "val": 0, "test": 0}
    for doc_id, split, w, h in docs:
        src = RAW_DIR / "img" / f"{doc_id}.jpg"
        dst = out_dir / "images" / split / f"{doc_id}.jpg"
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy(src, dst)
        rows = con.execute(
            "SELECT field_class, x1, y1, x2, y2 FROM lines WHERE doc_id = ?", [doc_id]
        ).fetchall()
        with open(out_dir / "labels" / split / f"{doc_id}.txt", "w") as fh:
            for cls, x1, y1, x2, y2 in rows:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                fh.write(f"{cls_idx[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        n_written[split] += 1

    meta = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: c for i, c in enumerate(FIELD_CLASSES)},
    }
    with open(out_dir / "dataset.yaml", "w") as fh:
        yaml.safe_dump(meta, fh)
    print(f"exported {n_written} (train fraction={fraction}) -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, default=1.0)
    export(ap.parse_args().fraction)
