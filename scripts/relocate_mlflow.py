"""Point the bundled MLflow store at its new location after moving/unzipping the repo.

MLflow stores *absolute* artifact URIs in its backend. When `data/mlflow.db` + `data/mlruns`
travel with the repo (so you can serve the trained champion without retraining), every
`file:///old/path/...` must be rewritten to the current path. This script does that in
place and is safe to run repeatedly.

Usage: python scripts/relocate_mlflow.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "mlflow.db"
NEW_ROOT = (ROOT / "data" / "mlruns").resolve().as_uri()  # e.g. file:///C:/Users/me/docpipe/data/mlruns

COLUMNS = [
    ("experiments", "artifact_location"),
    ("runs", "artifact_uri"),
    ("model_versions", "source"),
    ("model_versions", "storage_location"),
    ("logged_models", "artifact_location"),
]


def main() -> int:
    if not DB.exists():
        print(f"no MLflow db at {DB}; nothing to relocate")
        return 0
    con = sqlite3.connect(DB)
    # find the old root from the experiment row(s)
    olds = {r[0].split("/mlruns")[0] + "/mlruns" for r in con.execute("SELECT artifact_location FROM experiments") if r[0] and "/mlruns" in r[0]}
    if not olds:
        print("no file-based artifact locations found; nothing to do")
        return 0
    changed = 0
    for old in olds:
        if old == NEW_ROOT:
            continue
        for table, col in COLUMNS:
            try:
                cur = con.execute(f"UPDATE {table} SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?", (old, NEW_ROOT, old + "%"))
                changed += cur.rowcount
            except sqlite3.OperationalError:
                pass  # table absent in this MLflow version
        print(f"{old}  ->  {NEW_ROOT}")
    con.commit()
    con.close()
    print(f"rewrote {changed} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
