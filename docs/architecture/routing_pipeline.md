# Routing pipeline & background systems

Companion to `docs/glossary.md` (vocabulary) and `exploration_quota.md`
(the coverage floor). This is the canonical map of *one router, layered* — there
is a single routing decision wrapped in pre-filter gates and pluggable
strategies, not multiple competing routers.

## One request, end to end

### Entry
1. HTTP handler — `POST /codex`, `/v1/responses`, or `/v1/chat/completions`
   (`app.py`) → `_dispatch_route`.
2. `_dispatch_route` — resolves `session_id`, `api_key`/`user_id` →
   `_dispatch_internal`.

### Routing decision (`_dispatch_internal`)
3. Read intent — requested `(model, effort)` from the body.
4. `parse_selector` — a `callosum:` id → `SelectorDecision` (strategy or pin);
   non-`callosum:` → legacy pass-through; malformed → 400.
5. Snapshot live cells — `live_cells_fn()`.
6. **Gate stack** narrows the candidate pool *before* the router:
   operator/selector mode (backend-kind filter) → canary redirect → concrete
   pin → denylist. Empty pool → 503 (specific reason).
7. **`router.route(body, cells_now)`** — the one decision (`routing/router.py`):
   `extract_features` (+ embedding) → `CapabilityFilter` (empty → 400) →
   `QualityPredictor.predict` (uniform=0.5 / knn=learned) →
   `CostWeightedSelector.select` → `RoutingDecision`.
8. **Exploration quota** (`exploration_quota.md`) — deficit-fill: on a
   text-eligible, non-hard turn, steer to an under-floor cell.
9. Rewrite body (`model`, effort) → per-cell transforms
   (`_TRANSFORM_REGISTRY.apply_request`) → stamp provenance (predictor id,
   predictions, embedding, candidate cells capped at `MAX_CELL_ATTEMPTS=3`).

### Backend selection + execution
10. `_active_pool` (+ selector source / `forced_backend_id` narrowing) →
    session stickiness (`session_registry.get`).
11. Peer-quality injection (stream + capture-enabled only) — `<<qop>>`
    instruction + provenance tags, or a recorded `skip_reason`.
12. `_dispatch_{stream,nonstream}_with_cell_retry` — walk up to 3 candidate
    cells, reroute on 5xx → inner `_dispatch_{stream,nonstream}`:
    pick backend (honoring blocking-meters/cooldown/quota) → call upstream →
    stream: tee SSE, `capture.apply` strips markers and forwards cleaned text;
    nonstream: buffer + apply response transforms → on success
    `_remember_binding` (session→backend).

### Record + output
13. `usage_log.record` (request row) → peer-quality opinions/metrics →
    finalize cost estimate → emit routing event → output (already streamed, or
    buffered+cleaned).

## Background systems (no request involved)

**Startup one-shot (`lifespan`):** cold-boot catalog refresh; catalog
boot-resync task; kNN predictor reload (knn only); startup smoke test;
capability harness + light probe one-shot; reset-aware quota/cooldown pass.

**Periodic loops:** `smoke_tester` (liveness/model probing), `cooldown_prober`
(release backends when quota resets), `codex_catalog_reconciler` (regenerate
the Codex `/model` picker catalog), `probe_scheduler` sweep (tool-call shape
verification), `periodic_harness` (capability findings + adapter hints).

**Passive surfaces:** routing-events SSE (one event per recorded request),
label UI, admin routes.

**Offline jobs (operator/cron):** `peer_quality_shadow_report`,
`apply_peer_quality_labels`, retention pruning.
