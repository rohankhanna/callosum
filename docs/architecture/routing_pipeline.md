# Routing pipeline & background systems

Companion to `docs/glossary.md` (vocabulary) and `min_coverage_quota.md`
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
5. Snapshot live cells — `live_cells_fn(include_hidden=False)` for automatic
   and strategy routing; concrete selector pins call it with
   `include_hidden=True` so hidden-but-supported upstream lanes are reachable
   only when explicitly named.
6. **Gate stack** narrows the candidate pool *before* the router:
   operator/selector mode (backend-kind filter) → canary redirect → concrete
   pin → denylist. Hidden cells never enter free routing; an explicit pin may
   target them and then either dispatch normally or return a specific 503 if no
   backend serves the requested model/effort. Empty pool → 503 (specific
   reason). The backend-kind filter's remote set is two distinct kinds:
   `codex_auth_vault` (Codex Plus/Pro) and `ollama_cloud` (cloud models via the
   local ollama daemon under `ollama signin`; `BackendKind="ollama_cloud"`,
   env-gated via `CALLOSUM_OLLAMA_CLOUD_ENABLED`, `CLOUD_PRIORITY_OFFSET=1_000`).
   Every `=="litellm_gateway"` (local-affirmative) site excludes `ollama_cloud`
   by omission, so cloud cells land in remote lanes and stay out of
   local-only/probing paths without a per-site predicate.
7. **`router.route(body, cells_now)`** — the one decision (`routing/router.py`):
   `extract_features` → `CapabilityFilter` (empty → 400) →
   `QualityPredictor.predict` (`cell_majority_prior` = learned per-cell
   majority-baseline prior; `uniform` 0.5 is the cold-start fallback) →
   `CostWeightedSelector.select` → `RoutingDecision`. Cold-start (no
   differentiated prediction) explores randomly among capability-filtered,
   window-fitting, **feasibility-eligible** cells (predicted p95 latency under
   the stall-guard first-byte budget; cold cells with no measured fit are
   graced — see `feasibility.py` / ). Local cells also
   carry capability-matrix metadata from `local-llm capabilities --json`
   (`quantization`, host-fit status, and measured/estimated tokens/s when the
   hub has it). The local backend converts throughput into an explicit GPU
   opportunity-cost signal (`seconds/token`) so local cells remain nearly-free
   relative to remote quota cost while slower local models are still costlier
   inside the local fleet. The cost selector uses that local cost only after
   quality and ETA, then falls back to raw throughput. Inside the admitted
   local fleet, the final deterministic tie-break is exact-fit size: when
   quality and performance are otherwise tied, Callosum spends the smaller
   admitted local model before a larger one. The forward time estimator also has
   a local-only parametric
   performance hook (`routing/local_performance.py`) for TTFT, decode-rate,
   pool-edge, overhang, and thread-growth estimates. When the local model registry
   exposes enough structured evidence, Callosum seeds that hook from the hub's
   capability matrix and host profile; when it does not, the existing
   request-log time estimator remains the fallback. Separately, the curated
   local fleet can be projected through `local_model_catalog.py`, which applies
   Callosum-owned admission rules over hub discovery: runnable-on-host,
   responses-capable, training precision (no post-hoc down/up casts; a native
   released quant the model was post-trained at — e.g. model-a0d2 mxfp4 — counts
   as training precision), full-context memory-fit on this host (via the
   model-fit probe; see `model_fit_probe.md`), and an optional throughput
   floor. The local backend now exposes that curation as a stable consumer
   surface: admitted model ids plus per-model rejection reasons, so hierarchy
   work can consume the fleet view directly instead of re-encoding catalog
   policy.
8. **Minimum-coverage quota** (`min_coverage_quota.md`) — deficit-fill: on a
   text-eligible, non-hard turn, steer to an under-floor cell. Candidates are
   **feasibility-filtered** first (don't force onto a cell predicted to time
   out), and a cell in **post-timeout cooldown** (recent `min_coverage_quota`
   failure, no completed sample since) is skipped so the quota does not
   re-target it — the doom-loop fix ().
9. Rewrite body (`model`, effort) → per-cell transforms
   (`_TRANSFORM_REGISTRY.apply_request`) → stamp provenance (predictor id,
   predictions, candidate cells capped at `MAX_CELL_ATTEMPTS=3`).

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
boot-resync task; data-backed predictor reload (`cell_majority_prior`
from the request log; stays cold-start `uniform` on reload failure);
startup smoke test;
capability harness + light probe one-shot; reset-aware quota/cooldown pass.

**Periodic loops:** `smoke_tester` (liveness/model probing), `cooldown_prober`
(release backends when quota resets), `codex_catalog_reconciler` (regenerate
the Codex `/model` picker catalog), `probe_scheduler` sweep (tool-call shape
verification), `periodic_harness` (capability findings + adapter hints),
`model_probe_spawner` (idle-gated; launches the local-model full-context fit
probe as a short-lived subprocess when the operator is idle and memory
headroom is sufficient — see `model_fit_probe.md`).

**Passive surfaces:** routing-events SSE (one event per recorded request),
label UI, admin routes.

**Offline jobs (operator/cron):** `peer_quality_shadow_report`,
`apply_peer_quality_labels`, `model_probe` (per-model full-context fit probe;
`scripts/probe_model_fit.py`), retention pruning.
