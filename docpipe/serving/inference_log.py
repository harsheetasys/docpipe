"""Append-only prediction log (one JSON line per request, rotated daily).

The drift job reads this; it holds image features + prediction summaries, never the
image itself (scans may contain PII). In production this is the same schema shipped to
a warehouse table via the log collector.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path

import duckdb
import pandas as pd

from docpipe.config import LOG_DIR


class InferenceLog:
    def __init__(self, root: Path = LOG_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _file(self) -> Path:
        return self.root / f"predictions-{time.strftime('%Y%m%d', time.gmtime())}.jsonl"

    def write(self, result: dict, request_id: str | None = None, tag: str | None = None) -> str:
        rid = request_id or uuid.uuid4().hex
        row = {
            "request_id": rid,
            "ts": time.time(),
            "tag": tag,
            "models": result.get("models", {}),
            **{f"feat_{k}": v for k, v in result["features"].items()},
            **{f"pred_{k}": v for k, v in result["summary"].items() if k != "latency_ms"},
            "latency_ms": result["summary"]["latency_ms"]["total"],
            "fields_found": result["summary"]["fields_found"],
        }
        with self._lock, open(self._file(), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return rid

    def read(self, since_ts: float | None = None, tag: str | None = None) -> pd.DataFrame:
        files = sorted(self.root.glob("predictions-*.jsonl"))
        if not files:
            return pd.DataFrame()
        con = duckdb.connect()
        where = []
        if since_ts:
            where.append(f"ts >= {since_ts}")
        if tag:
            where.append(f"tag = '{tag}'")
        q = f"SELECT * FROM read_json_auto({[str(f) for f in files]}, union_by_name=true)"
        if where:
            q += " WHERE " + " AND ".join(where)
        return con.execute(q).df()
