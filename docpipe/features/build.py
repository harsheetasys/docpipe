"""Populate the feature store from the lake + raw images, then materialize to online.

Usage: python -m docpipe.features.build
"""
from __future__ import annotations

import duckdb
import pandas as pd
from PIL import Image

from docpipe.config import LAKE_DIR, RAW_DIR
from docpipe.features.doc_features import compute
from docpipe.features.store import DOC_IMAGE_VIEW, DOC_LAYOUT_VIEW, default_store


def build() -> dict:
    fs = default_store()
    con = duckdb.connect()
    docs = con.execute(
        f"SELECT doc_id, split, n_lines, n_field_lines, mean_line_height, mean_chars_per_line "
        f"FROM read_parquet('{LAKE_DIR / 'docs.parquet'}/*.parquet')"
    ).df()

    rows = []
    for doc_id in docs.doc_id:
        with Image.open(RAW_DIR / "img" / f"{doc_id}.jpg") as im:
            f = compute(im)
        f["doc_id"] = doc_id
        rows.append(f)
    img_df = pd.DataFrame(rows).merge(docs[["doc_id", "split"]], on="doc_id")
    img_df["event_ts"] = pd.Timestamp.now(tz='UTC')

    n_img = fs.write_offline(DOC_IMAGE_VIEW.name, img_df)
    n_lay = fs.write_offline(DOC_LAYOUT_VIEW.name, docs.assign(event_ts=pd.Timestamp.now(tz='UTC')))
    n_on = fs.materialize(DOC_IMAGE_VIEW.name)
    summary = {"offline_image_rows": n_img, "offline_layout_rows": n_lay, "online_rows": n_on}
    print(summary)
    return summary


if __name__ == "__main__":
    build()
