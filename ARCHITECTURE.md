# Architecture

This document is the version-controlled architecture source for
callosum. It describes the runtime topology, the major modules, the
contracts between them, and where to find the source for each. It is
the artifact a future operator or contributor should be able to read
without tribal knowledge to understand what callosum does and why it
is shaped the way it is.

For the product's purpose and intended outcome, see `The Project Documentation`.

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
        ┌─────────────────┼─────────────────────────┐
        ▼                 ▼                         ▼
  codex_auth_vault  LocalModelRegistryBackend       LiteLLMGatewayBackend
   (remote: Codex)    (local: per-model         (local: LiteLLM
    Plus/Pro vaults     endpoints via              gateway via
    rotated by health,  local-llm CLI              local LLM gateway)
    quota, cooldown)     discovery)
        │                 │                         │
        ▼                 ▼                         ▼
   chatgpt.com       ollama / vllm / etc       LiteLLM proxy
   /backend-api      per-model endpoints       /v1/chat/completions
   /codex/responses
```

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
  `predictor/` predicts quality per cell (uniform prior today; k-NN
  on embeddings is the next-generation predictor). `selector/`
  picks one cell from the predicted-and-scored candidate set
  (cost-weighted today). `router.py` orchestrates the pipeline.
  `probe.py` provides a tool-call verification probe used by
  `callosum-ctl probe-tools`.
- `backends/` — one module per backend kind. `codex_auth_vault.py`
  rotates Codex Plus/Pro `auth.json` vaults by health and quota.
  `local_direct.py` (`LocalModelRegistryBackend`) discovers local models
  via the `local-llm` CLI and dispatches per-model. `litellm_gateway.py`
  (`LiteLLMGatewayBackend`) talks to a LiteLLM gateway as an
  intermediate hop. `credential_proxy.py` is a thin adapter for the
  legacy credential-proxy shape.
- `cell_grid.py` — the (model, reasoning_effort) cell taxonomy and
  the merger that produces the live cell pool from backend
  `advertised_models` and per-backend `model_metadata`.
- `operator_state.py` — SQLite-backed runtime state for operator
  decisions: per-cell inference-param overrides, cell denylist,
  routing mode (`auto` / `offline` / `local-only` / `remote-only`).
- `admin.py` — `/admin/*` HTTP surface, gated by an admin token under
  `~/.config/callosum/admin_token`. The `callosum-ctl` CLI talks to
  these endpoints to read and modify operator state without
  restarting the proxy.
- `cli.py` — `callosum-ctl` itself: a thin client over the admin
  endpoints (status, params, denylist, mode, probe-tools).
- `usage_log.py` — SQLite request log. Every request lands here as
  a row in the `requests` table with request and response payloads,
  status, latency, and tokens.
- `routing_events.py` — `GET /events/routing` SSE stream of
  per-request routing decisions. The `snorkel` sidecar HUD consumes
  this to display the actually-routed model in its bottom bar.
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
- `~/.cache/callosum/` — embedding model cache.

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

There is no diagram source yet. The ASCII topology above is the
current placeholder. The control plane requires diagrams generated from
version-controlled source; closing that gap is a follow-up.

## Verification path

The canonical verification command is:

```
uv run ruff check src/ tests/
uv run mypy src/callosum
uv run pytest
```

All three must pass before any merge to `main`. As of 2026-06-01,
ruff has 115 errors (47 auto-fixable) and mypy has 45 errors carried
over from before the verification path was treated as canonical;
these are tracked as a standing follow-up.
