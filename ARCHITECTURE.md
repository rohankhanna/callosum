# Architecture

This document is the version-controlled architecture source for
callosum. It describes the runtime topology, the major modules, the
contracts between them, and where to find the source for each. It is
the artifact a future operator or contributor should be able to read
without tribal knowledge to understand what callosum does and why it
is shaped the way it is.

For the product's purpose and intended outcome, see `README.md`.

## Runtime topology

```
   any OpenAI-compatible client (Codex CLI, IDE plugins, curl)
                          │
                          ▼  HTTP on 127.0.0.1
                   ┌──────────────┐
                   │   callosum   │  FastAPI app, uvicorn-served,
                   │    proxy     │  systemd --user managed
                   └──────┬───────┘
                          │
   ┌──────────────────┬───┴───────────────┬──────────────────┐
   ▼                  ▼                   ▼                  ▼
 codex_auth_vault  LocalModelRegistryBackend  LiteLLMGatewayBackend  OllamaCloudBackend
  (remote: Codex)   (local, PREFERRED:   (local, FALLBACK:    (remote, ENV-GATED:
   Plus/Pro vaults   per-model endpoints  LiteLLM gateway —    cloud models served
   rotated by health, via local-llm CLI    registered only     by the LOCAL ollama
   quota, cooldown)  discovery)            when LocalModelRegistry    daemon under
                                         Backend is NOT       `ollama signin`; callosum
                                         available)           holds NO credential)
   │                  │                   │                    │
   ▼                  ▼                   ▼                    ▼
 chatgpt.com      ollama / vllm / etc  LiteLLM proxy       local ollama daemon
 /backend-api     per-model endpoints  /v1/chat/completions /v1/chat/completions
 /codex/responses                                            (daemon injects cloud auth)
```

The two local-cell backends (`LocalModelRegistryBackend`,
`LiteLLMGatewayBackend`) are MUTUALLY EXCLUSIVE at registration
time — they cannot both run in the same process. The preferred
backend (`LocalModelRegistryBackend`) is registered whenever the
`local-llm` CLI is on PATH; the gateway backend is registered as a
fallback only when the CLI isn't installed. See
`src/callosum/__main__.py` for the gating and
`docs/decisions/...-keep-litellmgatewaybackend-as-fallback...`
for the historical rationale.

`OllamaCloudBackend` is a fourth, INDEPENDENT backend — not
mutually exclusive with the local pair. It is registered only when
`CALLOSUM_OLLAMA_CLOUD_ENABLED=1` (default off). Callosum holds NO
credential: the local ollama daemon (`localhost:11434`) authenticates
to the cloud under `ollama signin` and callosum dispatches to the
daemon's OpenAI-compatible `/v1/chat/completions` with no
`Authorization` header. Catalog is `/api/tags` filtered to
`:cloud`-suffixed models (so cloud and local models partition
cleanly); capabilities come from `/api/show`. It sits in the remote
band (`CLOUD_PRIORITY_OFFSET=1_000`, after Codex but before the local
`10_000` floor) and is classified `BackendKind="ollama_cloud"`, which
every `=="litellm_gateway"` (local-affirmative) site excludes by
omission — cloud cells land in remote lanes and stay out of
local-only/probing paths.

callosum runs loopback-only on a single operator's machine. It is
not multi-tenant. Each request enters the proxy on
`http://127.0.0.1:<port>`, gets routed to a cell `(model,
reasoning_effort)`, dispatched to the cell's backend, and the
response is streamed back to the client.

## Major modules

All paths are under `src/callosum/`.

- `app.py` — FastAPI app factory + request dispatch. Wires backends,
  router, operator state, usage log, admin surface, and the auth
  middleware. The longest file in the repo; structured by
  request-lifecycle phase rather than by feature.
- `routing/` — the per-request routing pipeline. `features.py`
  extracts prompt-shape facts from the request body. `capability.py`
  filters cells against modality and tool-support requirements.
  `predictor/` predicts quality per cell (`cell_majority_prior` — a
  per-cell majority-baseline prior that deliberately ignores prompt
  embeddings; `cell_mean_prior` — a shadow-only per-cell mean prior that
  preserves outcome magnitude, not the default; the embedding/KNN
  predictor was evaluated and removed. `uniform` returns 0.5 and is the
  cold-start fallback). `selector/`
  picks one cell from the predicted-and-scored candidate set
  (cost-weighted today). `router.py` orchestrates the pipeline.
  `cost_model.py` derives each remote model's compatibility `cost_rank`
  from MEASURED weekly-quota burn, using serialized integer-meter tick
  windows built from `weekly_used_percent` movement in the request log
  instead of a flat constant. `/v1/usage` and the forward routing cost
  path build the same class of quantized windows independently for both
  Codex quota ledgers (`five_hourly_used_percent` and
  `weekly_used_percent`) and report per-cell five-hourly rates, weekly
  rates, and empirical five-hourly/weekly ratios. Backend routing is
  composite quota-aware: it scores estimated burn against the remaining
  headroom of each independent ledger and ranks by the worst pressure,
  so either five-hourly or weekly exhaustion can block a route. Catalog `priority` is
  only the cold-start prior, and an operator override map wins
  outright (`CostRankProvider`, overlaid onto backend capabilities in
  `app.py`). `routing/quota.py` + `routing/coverage.py` implement the
  per-cell **minimum-coverage quota** on *organic* traffic: when a
  compatible `(model, reasoning_effort)` cell is below its even-split
  floor of `min_coverage_budget_pct` over the rolling window, the
  router steers one turn to the least-sampled eligible cell so coverage
  accumulates evenly across the grid (the engine that feeds the quality
  predictor and the measured cost model); the contextual-bandit plan was
  superseded — selection is cost-weighted, not a bandit. Otherwise,
  organic routing keeps the cost/quality-optimal pick. Gated
   `usage_estimate.py` +
  `cost_estimator.py` are the FORWARD usage estimators (the twin of the
  backward-looking `cost_model.py`): `usage_estimate.py` holds the
  estimator-agnostic contract (`OutputTokenForecast`,
  `OutputTokenForecaster`, `EstimateInput`, range-first `Estimate`,
  `UsageEstimator` protocol). The forecaster predicts a request's
  output-token distribution ONCE (cold-start global output:input ratio
  yielding to per-cell measured ratios) so the cost AND time estimators
  consume the same forecast and cannot diverge. `cost_estimator.py`
  implements the cost half: `usage% ≈ rate × (input + output)` per cell,
  reading the same meter-tick windows for five-hourly and weekly meters,
  local cells taking an explicit `0` branch. The estimate is a range
  (`p50`→`p95`) consumed by composite backend routing and post-hoc —
  `finalize()` (hooked at the `_log_attempt`
  logging path) records the realized quota delta, marking a 0 integer-%
  delta `verifiable=False` so it feeds aggregate calibration only, never
  a per-request point fit; `aggregate_cost_accuracy()` is the
  many-rows predicted-vs-actual check. `time_estimator.py` implements
  the time half from the SAME forecast and substrate, fitting
  `t ≈ a·input + b·output + c` (v1 blends `a == b` into one slope per
  total token + intercept, against `latency_ms`; the TTFB/decode split
  is deferred). Unlike cost, local cells are NOT zeroed — they are often
  the slow path (5–50× remote in the live log) — and latency is always
  observable, so every estimate/`finalize()` is `verifiable=True` (no
  integer-resolution case). Its range carries BOTH output-length
  uncertainty and per-cell residual dispersion (load/concurrency), and
  the point uses the median residual (latency is right-skewed). It
  publishes on `app.state.time_estimator` and finalizes at the same
  `_log_attempt` path; `aggregate_time_accuracy()` is its many-rows
  predicted-vs-actual check. The static stall-guard timeouts
  (`CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S` / `..._IDLE_TIMEOUT_S`) remain
  separate GUARD thresholds that could later become data-driven per-cell
  from this estimator. `probe.py` provides a tool-call
  verification probe used by `callosum probe-tools` (or the legacy
  `callosum-ctl probe-tools` alias).
- `backends/` — one module per backend kind. `codex_auth_vault.py`
  rotates Codex Plus/Pro `auth.json` vaults by health and quota.
  `local_direct.py` (`LocalModelRegistryBackend`) is the PREFERRED local
  backend: discovers local models via the `local-llm` CLI and
  dispatches to per-model endpoints (responses-proxy lanes, ollama
  direct, vllm direct). `litellm_gateway.py` (`LiteLLMGatewayBackend`)
  is the FALLBACK local backend, used only when the `local-llm` CLI
  isn't on PATH; it talks to a LiteLLM gateway as an intermediate
  hop. The two local backends are mutually exclusive at registration
  time (see `__main__.py`). `ollama_cloud.py` (`OllamaCloudBackend`,
  `BackendKind="ollama_cloud"`) is an independent env-gated remote
  backend (`CALLOSUM_OLLAMA_CLOUD_ENABLED`, default off) for cloud
  models served by the local ollama daemon under `ollama signin`;
  callosum holds NO credential. It catalogs `/api/tags` filtered to
  `:cloud`-suffixed models, reads capabilities from `/api/show`, and
  dispatches via the daemon's `/v1/chat/completions`. Its Responses↔Chat
  translators and chat→Responses streaming generator live in the shared
  `backends/_responses_chat.py` (imported by both `litellm_gateway` and
  `ollama_cloud`; litellm re-exports the `_`-prefixed names for backward
  compatibility). `ollama_cloud` can optionally read real 5h-session /
  7d-weekly usage meters from credential proxy (credential-custody sibling on
  `127.0.0.1:7342`) via a credential-free loopback stand-in token —
  callosum still holds NO credential; this is env-gated and defaults OFF
  (see `docs/operations/runtime_deploy.md`).
- `cell_grid.py` — the (model, reasoning_effort) cell taxonomy and
  the merger that produces the live cell pool from backend
  `advertised_models` and per-backend `model_metadata`.
- `selectors.py` — client-driven routing selectors. A `callosum:`
  model id expresses routing intent for a single request: strategy
  selectors (`callosum:auto` / `callosum:local-only` /
  `callosum:remote-only`) set the routing engine, and concrete pins
  (`callosum:remote/<model>::<effort>`, `callosum:local/<model>`) name
  a specific model. `parse_selector()` runs on the dispatch hot path;
  a selector overrides the per-request routing value (and, for pins,
  narrows the cell pool and dispatch pool) **without mutating the
  global operator mode** — so concurrent clients sharing one proxy
  don't clobber each other. The canonical `/v1/models` catalog is
  built from these selectors plus the live remote/local catalogs
  (raw passthrough ids remain listed for back-compat).
- `codex_catalog.py` — projects that canonical catalog into a codex
  `/model` picker. codex's interactive picker is driven by a
  config-declared catalog file (`model_catalog_json = <path>` in
  `~/.codex/config.toml`), **not** by the provider's `/v1/models`, so
  `CodexCatalogReconciler` re-emits a codex-shaped `{models:[...]}`
  file from the same ids `/v1/models` serves — on startup, on
  catalog-hash change, and on a periodic safety interval (mirrors the
  `_PeriodicSmokeTester` start/stop lifecycle). Each lane inherits a
  rich codex `ModelInfo` template sourced live from `codex debug
  models` under a disposable `CODEX_HOME` (codex's bundled default
  catalog — avoids re-reading our own override, and keeps codex's
  prompt out of the repo). Strategy selectors lead the menu; remote
  pins restrict the offered reasoning effort to the one baked into the
  id; raw passthrough ids are excluded from the picker; operator-
  declared aspirational lanes (`declared_lanes`) appear even when no
  backend serves them yet — selecting one returns a clean
  `503 … not available yet: no backend currently serves model …` at
  dispatch. Off by default; enabled via `[codex_catalog]` in the
  server config. This lets a single `~/.codex/config.toml` (default
  `callosum:auto`) switch lanes in-session via `/model`, replacing the
  old per-lane `*.config.toml` + `codex -p <lane>` files.
- `operator_state.py` — SQLite-backed runtime state for operator
  decisions: per-cell inference-param overrides, cell denylist,
  routing mode (`auto` / `offline` / `local-only` / `remote-only`).
  This mode is the operator-set **default**; a per-request
  `callosum:` selector (see `selectors.py`) overrides it for that
  request only.
- `admin.py` — `/admin/*` HTTP surface, gated by an admin token under
  `~/.config/callosum/admin_token`. The unified `callosum` CLI's
  admin subcommands (`callosum status`, `callosum routing get|set`,
  etc.) call these endpoints to read and modify operator state
  without restarting the proxy.
- `cli.py` — the unified `callosum` CLI entry point. Hosts both
  `callosum serve` (which delegates to `__main__.serve_with_args`
  to start the daemon) and the admin subcommands (status, params,
  denylist, routing, autonomy, retention, self-assessment,
  probe-tools, auth-rotate). Bare `callosum` and missing-subcommand
  invocations print the relevant `--help` to stderr and exit 2.
  `callosum-ctl` is preserved as a backwards-compatible alias
  pointing at the same `main()` function.
- `usage_log.py` — SQLite request log. Every request lands here as
  a row in the `requests` table with request and response payloads,
  status, latency, and tokens. It also owns the first peer-quality
  reward substrate: `peer_quality_opinions` stores nonce-validated,
  per-session judge-to-subject model opinions separately from request
  rows so routing behavior can remain unchanged while the matrix
  accumulates. This is also the repo's Truth Log surface: it must
  preserve the exact upstream model-visible payload, the sanitized
  client-visible transcript, and enough transform metadata to
  reconstruct what Callosum actually sent.
- `peer_quality.py` — parser and fail-closed stripper for hidden
  in-band `<<qop ...>>` quality-opinion markers. It records only
  markers carrying the current request nonce; wrong-nonce markers are
  counted as echoes and malformed current-nonce markers are stripped
  without becoming training labels. The sending side is gated by
  `CALLOSUM_PEER_QUALITY_CAPTURE_RATE` (default `0`, disabled): when
  enabled for a sampled streamed request, `app.py` exact-matches recent
  same-session assistant outputs from the usage log, wraps only those
  prior non-self messages with model/effort provenance tags in the
  outbound upstream body, and adds a nonce-bearing audit instruction.
  Those injected tags/instructions are Hidden Model Payload: visible to
  the upstream model, hidden from Codex/UI by Callosum's veiling path.
  Injection is skipped when the projected outbound body would exceed
  the selected cell's known backend context window after the router
  safety margin; the added audit/provenance overhead is also capped
  independently. This is a conservative near-term gate, not the final
  tokenizer story: the long-term path should use provider-specific
  tokenizers and reserve output tokens explicitly. Tool/function-call
  requests and turns without exact text assistant outputs are skipped
  rather than forcing an opinion slot.
  The user's visible stream and persisted response body have qop
  markers stripped before storage/display. `peer_quality_opinions`
  stores valid opinions; `peer_quality_capture_metrics` stores
  request-level counters for valid opinions, wrong-nonce echoes, and
  malformed markers so the operator can measure task interference and
  echo behavior before any routing use.
- `routing_events.py` — `GET /events/routing` SSE stream of
  per-request routing decisions. External observability sidecars can consume
  this stream to display the model that actually served a request.
- `local.py` — wrapper around the external `local-llm` CLI
  used by `local_direct.py` for model discovery.
- `auth.py` / `auth_service.py` — API-key issuance and validation
  for the `/v1/*` surfaces.
- `sse_tee.py` — SSE stream collector used by every backend's
  `responses_stream` to capture the upstream stream while it flows
  to the client, so the dispatch layer can parse the terminal
  `response.completed` event for usage.

## State files (outside the repo tree)

- `~/.local/state/callosum/requests.sqlite` — usage log.
- `~/.local/state/callosum/operator_state.sqlite` — operator state.
- `~/.local/state/callosum/auth.sqlite` — API keys.
- `~/.config/callosum/admin_token` — admin-surface token.
- `~/.cache/callosum/` — (the embedding model cache that lived here was
  removed with the embedding/KNN subsystem; the dir may still exist
  empty).

State is preserved across restarts; recovery means restarting the
service.

## Decision records

Architecturally significant decisions are recorded as numbered ADRs
under `docs/adr/`. As of this writing the ADR directory is empty
because earlier work landed without ADR discipline; this is a known
gap and a follow-up to this compliance pass.

When a future change reshapes the cell taxonomy, the request
lifecycle, the auth model, the backend interface contract, or the
operator state schema, that change must come with an ADR.

## Diagrams

Version-controlled diagram source lives under `docs/architecture/`.
Each `.puml` file is a PlantUML source; the matching `.svg` next to
it is the rendered output, regenerated via `scripts/render_diagrams.sh`.

Current diagrams:

- `docs/architecture/request_lifecycle.puml` / `.svg` — sequence
  diagram for one request from inbound POST through routing decision,
  backend dispatch, upstream streaming, and usage-log write. Renders
  with PlantUML + Java alone (no Graphviz required).
- `docs/architecture/runtime_topology.puml` / `.svg` / `.png` —
  component diagram for the proxy, backend lanes, persisted state,
  capability harness, canary/failure path, and scheduled auto-dev
  loop.
- `docs/architecture/generated/call_graph_focus.dot` / `.svg` /
  `.json` — code-generated static call graph for the runtime roots
  (`_run_server`, `create_app`, request dispatch, router, and logging).
- `docs/architecture/generated/control_flow_focus.dot` / `.svg` /
  `.json` — code-generated statement-level control-flow sketches for
  the dispatch and router hot paths.

See `docs/architecture/README.md` for the full rendering procedure
and the tooling-choice rationale, plus
`docs/architecture/generated/README.md` for review-oriented drill-down
entry points and the full repository-wide architecture review target
inventory.

## Self-observation and self-adaptation

callosum is a router that observes its own behavior and can
(optionally, with safety gates) write code to adapt itself. Five
modules implement this:

### Capability harness (`src/callosum/capability/`)

A periodic in-process sweeper that probes every advertised local
cell against a registered set of capability dimensions
(`tool_call_shape`, `tool_call_at_scale`, ...). Each probe produces
a `DimensionFinding` with status (`pass` / `fail` / `error` /
`skipped`), summary, evidence, and — when status is `fail` — an
`adapter_hint` (free prose: what a transform would need to do) plus
the structured gap fields the shim-reduction P3 pass added so the dev
loop does not automation agently author a transform off free prose:
`gap_class` ("A" = generic protocol a substrate owns, "B" =
model-specific quirk no substrate owns) and `suggested_action` (a
`ContractAction`: `route_native` / `author_temporary_adapter` /
`remove_shim` / `verify_fix` / `quarantine_cell`). `suggested_action`
is the authoritative signal; `callosum.gap_triage.classify_gap` falls
back to inferring it from `adapter_hint` when the structured field is
absent (backward-compatible with pre-P3 findings).

Findings persist as JSON under `logs/capability_profiles/<cell>.json`.
The runner (`callosum.capability.runner.run_dimensions`) is the
single orchestrator both pytest (`tests/model_capability/`) and the
in-process scheduler (`callosum.capability.scheduler.PeriodicHarnessSweep`)
call into. Per-dimension TTL is one week; sweep cadence defaults to
six hours (env-overridable via `CALLOSUM_HARNESS_SWEEP_INTERVAL_SECONDS`).

### Weight identity (`src/callosum/capability/weight_identity.py`)

Two cells routing to identical model weights through different
transports (e.g. `model-a0a9` and
`model-a0a1`) appear in the cell grid as
unrelated rows. The weight-identity layer groups them: each cell's
profile carries a `WeightIdentity{source, runtime, quantization,
family}` so a downstream consumer can detect "same weights,
divergent findings → the failure is in the transport, not the model."

The layer is pluggable: `WeightIdentityProvider` is a protocol;
concrete providers stack via `CompositeWeightIdentityProvider`. The
default composite tries the local-llm CLI first, falls back to
naming-pattern heuristics, then a null backstop. Adding a new source
(manifest file, different CLI, HTTP endpoint) is one new class plus
one item in `build_default_provider()`; no caller in `app.py`
changes.

### Canary baseline + failure registry (`src/callosum/canary/`)

A configurable fraction (2-10%, default 10%) of inbound auto-mode
requests are deliberately redirected to the remote-only path,
regardless of what the router would have picked. The redirect
creates a continuous A/B: comparing rolling failure rate of `auto`
vs `canary_redirect` over real traffic produces a signal that
detects local-side regressions without any synthetic probe.

Quota-aware: scales down linearly when Codex weekly quota crosses
80% used; fully suspends above 95%. Override via
`CALLOSUM_CANARY_PERCENT` / `CALLOSUM_CANARY_FLOOR_PERCENT` etc.

Every failed request also writes a structured row to the
`failure_observations` table in the usage-log SQLite, with
`effective_mode`, `symptom`, and `responsible_layer` attribution.
The taxonomy is open-ended; we grow it from observed incidents
rather than pre-declaring failure classes.

`/status` exposes rolling per-mode failure rates over 1h / 6h / 24h
windows. The dev loop (below) uses the same data to detect
divergence.

### Transform substrate (`src/callosum/transforms/`)

Per-cell payload transforms live here. A `Transform` is a small
protocol with `name`, `applies_to(ctx) → bool`, `transform_request`,
and `transform_response`. The `TransformRegistry` walks registered
transforms middleware-style after cell selection.

Currently empty by default — shipping the substrate introduced zero
behavior change. Per-transform error isolation (a buggy transform
that raises is logged and skipped), name uniqueness enforcement at
registration, and middleware-style ordering are all properties of
the registry, tested.

The package is available for per-model payload adapters.


## Verification path

The canonical verification command is:

```
uv run ruff check src/ tests/
uv run mypy src/callosum
uv run pytest
```

All three must pass before any merge to `main`. As of 2026-06-01 the
verification path is fully green — no ruff errors, no mypy errors,
462 tests passing. The line-length cap is 120 (widened from 100 to
match modern defaults); test files have a per-file E501 ignore so
SSE-blob test fixtures can stay as single-line literals matching
real upstream payload shapes.
