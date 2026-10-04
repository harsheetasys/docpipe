# ADR 0002: A minimal in-repo feature store instead of Feast

**Status:** accepted

## Context
The job of a feature store here is narrow: hold the reference distribution of image
features computed by *the same function* used at serving time, support point-in-time
retrieval for reproducible training sets, and expose an online lookup for per-entity
priors. The platform has two models and one feature view.

## Decision
Implement `FeatureStore` (~150 lines): offline = parquet queried with DuckDB SQL (latest
value per entity as of a timestamp), online = SQLite keyed by entity with TTL, a registry
JSON, and `materialize()`. The API deliberately mirrors Feast (`FeatureView`,
`get_historical_features`, `get_online_features`, `materialize`).

## Why not Feast now
Feast brings a registry service, provider abstraction and ~40 transitive dependencies for
one feature view. The cost is paid in image size, cold-start and CI time, with no benefit
until there are several teams sharing features. Because the interface is Feast-shaped, the
swap is mechanical when that day comes.

## Consequences
* No cross-team discovery UI. Acceptable at this scale.
* Point-in-time correctness is tested (`tests/test_units.py::test_feature_store_roundtrip`).
