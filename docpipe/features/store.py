"""A small but real feature store with the two halves every production store has:

  * OFFLINE store  -> parquet files queried with DuckDB SQL. Used for training sets,
                      reference distributions and point-in-time correct joins.
  * ONLINE store   -> SQLite key/value table keyed by entity id. Used at serving time
                      for low-latency lookups (e.g. per-vendor priors) and by the drift
                      job to read the latest feature snapshot.

The interface (`FeatureView`, `materialize`, `get_online_features`,
`get_historical_features`) mirrors Feast on purpose, so swapping in Feast/Tecton is a
config change, not a rewrite. See docs/adr/0002-feature-store.md for why we didn't
pull Feast into a 2-model project.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import pandas as pd

from docpipe.config import FEATURE_DIR


@dataclass(frozen=True)
class FeatureView:
    name: str
    entity: str                       # column that identifies the entity (doc_id)
    features: list[str]
    ttl_seconds: int = 30 * 24 * 3600
    description: str = ""
    tags: dict = field(default_factory=dict)


class FeatureStore:
    def __init__(self, root: Path = FEATURE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.online_db = self.root / "online.sqlite"
        self._views: dict[str, FeatureView] = {}

    # ---- registry ---------------------------------------------------------------------
    def register(self, view: FeatureView) -> None:
        self._views[view.name] = view
        reg = self.root / "registry.json"
        data = json.loads(reg.read_text()) if reg.exists() else {}
        data[view.name] = {
            "entity": view.entity,
            "features": view.features,
            "ttl_seconds": view.ttl_seconds,
            "description": view.description,
            "tags": view.tags,
        }
        reg.write_text(json.dumps(data, indent=2))

    def view(self, name: str) -> FeatureView:
        if name not in self._views:
            reg = json.loads((self.root / "registry.json").read_text())[name]
            self._views[name] = FeatureView(name=name, **reg)
        return self._views[name]

    # ---- offline ----------------------------------------------------------------------
    def _offline_path(self, view: str) -> Path:
        return self.root / "offline" / f"{view}.parquet"

    def write_offline(self, view: str, df: pd.DataFrame, event_time_col: str = "event_ts") -> int:
        """Append rows to the offline store. Each row carries an event timestamp so that
        historical retrieval can be point-in-time correct."""
        v = self.view(view)
        missing = {v.entity, *v.features} - set(df.columns)
        if missing:
            raise ValueError(f"missing columns for view {view}: {missing}")
        df = df.copy()
        if event_time_col not in df:
            df[event_time_col] = pd.Timestamp.now(tz='UTC')
        df["created_ts"] = pd.Timestamp.now(tz='UTC')
        path = self._offline_path(view)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df.to_parquet(path, index=False)
        return len(df)

    def get_historical_features(self, view: str, entity_df: pd.DataFrame | None = None, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        """Latest feature value per entity as of `as_of` (point-in-time join), optionally
        restricted to the entities in `entity_df`."""
        v = self.view(view)
        path = self._offline_path(view)
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        con.execute(f"CREATE VIEW f AS SELECT * FROM read_parquet('{path}')")
        where = f"WHERE event_ts <= TIMESTAMP '{as_of}'" if as_of is not None else ""
        q = f"""
            SELECT * EXCLUDE (rn) FROM (
              SELECT *, ROW_NUMBER() OVER (PARTITION BY {v.entity} ORDER BY event_ts DESC) rn
              FROM f {where}) WHERE rn = 1
        """
        out = con.execute(q).df()
        if entity_df is not None:
            out = entity_df[[v.entity]].merge(out, on=v.entity, how="left")
        return out

    def sql(self, query: str) -> pd.DataFrame:
        """Ad-hoc SQL over every offline view (each view is exposed as a table)."""
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        for p in (self.root / "offline").glob("*.parquet"):
            con.execute(f"CREATE VIEW {p.stem} AS SELECT * FROM read_parquet('{p}')")
        return con.execute(query).df()

    # ---- online -----------------------------------------------------------------------
    def materialize(self, view: str, as_of: pd.Timestamp | None = None) -> int:
        """Copy the latest offline values into the online store (what Feast calls
        `feast materialize`)."""
        v = self.view(view)
        latest = self.get_historical_features(view, as_of=as_of)
        con = sqlite3.connect(self.online_db)
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {view} (entity_id TEXT PRIMARY KEY, features TEXT, event_ts REAL, written_ts REAL)"
        )
        rows = [
            (str(r[v.entity]), json.dumps({f: (None if pd.isna(r[f]) else float(r[f])) for f in v.features}),
             pd.Timestamp(r["event_ts"]).timestamp(), time.time())
            for _, r in latest.iterrows()
        ]
        con.executemany(f"INSERT OR REPLACE INTO {view} VALUES (?,?,?,?)", rows)
        con.commit()
        con.close()
        return len(rows)

    def get_online_features(self, view: str, entity_ids: list[str]) -> dict[str, dict | None]:
        v = self.view(view)
        con = sqlite3.connect(self.online_db)
        try:
            qs = ",".join("?" * len(entity_ids))
            cur = con.execute(f"SELECT entity_id, features, event_ts FROM {view} WHERE entity_id IN ({qs})", entity_ids)
            now = time.time()
            found = {eid: (json.loads(f) if now - ts <= v.ttl_seconds else None) for eid, f, ts in cur.fetchall()}
        except sqlite3.OperationalError:
            found = {}
        finally:
            con.close()
        return {e: found.get(e) for e in entity_ids}


# ----------------------------------------------------------------------------------------
# Feature views used by DocPipe
# ----------------------------------------------------------------------------------------
from docpipe.features.doc_features import FEATURE_NAMES  # noqa: E402

DOC_IMAGE_VIEW = FeatureView(
    name="doc_image_features",
    entity="doc_id",
    features=FEATURE_NAMES,
    description="Image quality statistics per receipt; reference distribution for drift detection",
    tags={"owner": "ml-platform", "pii": "false"},
)

DOC_LAYOUT_VIEW = FeatureView(
    name="doc_layout_features",
    entity="doc_id",
    features=["n_lines", "n_field_lines", "mean_line_height", "mean_chars_per_line"],
    description="Layout statistics derived from annotations (training-time only)",
    tags={"owner": "ml-platform", "pii": "false"},
)


def default_store() -> FeatureStore:
    fs = FeatureStore()
    fs.register(DOC_IMAGE_VIEW)
    fs.register(DOC_LAYOUT_VIEW)
    return fs
