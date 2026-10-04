# ADR 0003: Resolve models by registry alias; promote and roll back by moving aliases

**Status:** accepted

## Context
Two ways to get a model into a serving pod: bake weights into the image (immutable,
rollback = image rollback) or fetch from a registry at runtime (mutable pointer).

## Decision
Fetch by alias. Pods load `models:/<name>@champion` on start and poll every 60 s.
Three aliases: `challenger` (latest trained), `champion` (serving), `previous`
(last good). Promotion = move `champion` after the gate passes; rollback = move
`champion` back to `previous`. Deployment rollback (`kubectl rollout undo`) remains for
container-level failures.

## Why
* A model rollback should not need a build, a registry push and a rollout. 60 seconds
  and one command, auditable in `promotions.jsonl`.
* Model and code have different failure modes and different owners; separate levers.
* The registry already stores lineage (run, dataset hash, metrics) next to the alias.

## Consequences
* Serving depends on the registry at start-up (readiness probe covers this; the image
  does not become ready until the model is loaded).
* Two pods can briefly serve different versions during a refresh window. The inference log
  records the version per request so a regression can be attributed.
