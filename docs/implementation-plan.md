# Implementation Plan

Anchors off the ideation plan at `~/.claude/plans/i-m-going-to-golden-cupcake.md`. Locked decisions from that plan carry over unchanged. This document closes the gaps that plan punted to a follow-up.

## Scope of v1

- Local HTTP server on `127.0.0.1`.
- OpenAI-compatible `POST /v1/chat/completions` (streaming + non-streaming).
- `GET /health`, `GET /status`, `POST /control/pin`, `POST /control/unpin`.
- Credential-agnostic backend pool. v1 ships two backend types: `openai_api_key` and `azure_openai`.
- Stateless session policy only. Sticky modes are scaffolded at the protocol level but not wired.
- Consumer-auth (`codex_auth_vault`) backend type is out of v1. Protocol must leave room for it.

Deferred to later versions: `/v1/responses`, `codex_auth_vault` backend, sticky and sticky-replay session policies.

## Config schema

One TOML file, default path `~/.config/codex-proxy/config.toml`, override via `--config`. Example:

```toml
[server]
host = "127.0.0.1"
port = 8765
client_auth_token_env = "CODEX_PROXY_CLIENT_TOKEN"   # optional; clients send Bearer
control_auth_token_env = "CODEX_PROXY_CONTROL_TOKEN" # optional; required for /control/* if set

[policy]
default_mode = "stateless"           # stateless | sticky | sticky_replay
allow_consumer_auth_backends = false # explicit opt-in gate for codex_auth_vault backends

[[backends]]
id = "openai-primary"
type = "openai_api_key"
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"
models = ["model-a0f5", "model-a0f5-mini", "model-a0f7.1"]  # advertised on /v1/models

[[backends]]
id = "azure-main"
type = "azure_openai"
endpoint = "https://<resource>.openai.azure.com"
api_key_env = "AZURE_OPENAI_KEY"
api_version = "2024-10-01-preview"
deployments = { "model-a0f5" = "model-a0f5-prod" }  # model -> azure deployment name
```

Validation rules:

- Each `[[backends]]` entry must have unique `id`.
- Env vars named in `*_env` fields must resolve at startup; startup fails loud if missing.
- Any `type = "codex_auth_vault"` entry is rejected at load time unless `allow_consumer_auth_backends = true`.

## `Backend` protocol

```python
class Backend(Protocol):
    id: str
    kind: Literal["openai_api_key", "azure_openai", "codex_auth_vault"]
    advertised_models: frozenset[str]

    async def health(self) -> HealthStatus: ...
    async def usage_snapshot(self) -> UsageSnapshot: ...
    async def chat_completions(
        self, req: ChatCompletionRequest
    ) -> AsyncIterator[SSEEvent] | ChatCompletionResponse: ...

@dataclass
class HealthStatus:
    available: bool
    reason: Literal["ok", "rate_limited", "auth_invalid", "network", "unknown"]
    retry_after_s: float | None

@dataclass
class UsageSnapshot:
    remaining_fraction: float | None   # 0..1 if known, else None
    cooldown_until_ts: float | None
    weekly_exhausted: bool
    probed_at_ts: float
```

`chat_completions` returns a stream iterator when `stream=True`, a full response otherwise. Backend implementations own their own `httpx.AsyncClient` for connection pooling.

## Selector

Single function: `select(backends, request, policy, state) -> Backend`.

Inputs per backend:

- current `HealthStatus` (cached ≤ 10s or re-probed on demand).
- last `UsageSnapshot` (cached ≤ 60s).
- explicit pin for this session, if any.
- cooldown-until timestamp (if 429 was seen).

Ranking, in stateless mode:

1. Drop any backend with `available=False` or `cooldown_until_ts > now`.
2. Drop any backend whose `advertised_models` does not include the requested model.
3. Prefer `weekly_exhausted=False` over `True`.
4. Among ties, prefer higher `remaining_fraction` (None treated as 0.5).
5. Among ties, prefer more recent `probed_at_ts`.
6. Stable deterministic tiebreak on `id`.

Sticky modes reuse this ranking but only re-run it when the bound backend is unavailable.

## Error taxonomy → action

| Upstream signal           | Backend classification                | Proxy action                                    |
| ------------------------- | ------------------------------------- | ----------------------------------------------- |
| `200`                     | healthy                               | stream through                                   |
| `400`                     | client error                          | surface to client, no retry, no rotate           |
| `401` / `403`             | `auth_invalid`                        | mark unhealthy; rotate; if none left, `502` out  |
| `404` (unknown model)     | backend doesn't serve model           | rotate; if none left, `400` to client           |
| `429`                     | `rate_limited` + cooldown = retry-after or 60s default | rotate; if none left, pass `429` through with headers |
| `5xx` / timeout           | transient                             | retry once same backend → rotate → surface       |
| upstream closes mid-stream | transient                             | cannot retry; surface error frame to client     |

Rotation means re-run the selector excluding the failed backend for this request. Health cache is updated for future requests.

## Concurrency model

- FastAPI + `uvicorn` async.
- One `httpx.AsyncClient` per backend, configured with HTTP/2 and connection pool `max_connections=20`.
- No global locks for stateless mode. Sticky mode needs a per-session mutex at bind time.
- No client-side rate limiting in v1. The upstream 429 path is the source of truth for backpressure.

## Persistence

On disk, under `~/.local/state/codex-proxy/` by default:

- `usage/<backend-id>.json` — last `UsageSnapshot` persisted after each probe. Survives restart so the selector has a warm signal.
- `cooldowns/<backend-id>.json` — last cooldown decision; read on startup to avoid hammering a backend that was 429'd right before shutdown.
- `vaults/<backend-id>/auth.json` — reserved for `codex_auth_vault` (not in v1).

In memory only:

- streaming buffers
- session→backend binding (sticky mode, v2+)
- health cache

Startup reloads usage and cooldown state before accepting traffic.

## Control-surface auth

- Bind defaults to `127.0.0.1`. Do not expose on `0.0.0.0` by default.
- `/health` is always open.
- `/v1/*` requires `Authorization: Bearer <token>` iff `client_auth_token_env` is set in config.
- `/status` and `/control/*` require `Authorization: Bearer <token>` iff `control_auth_token_env` is set. If not set, they still require the request to originate from loopback; non-loopback requests get `403`.
- `X-Forwarded-For` is ignored for origin checks — we trust only the peer address.

## README framing

The repo's public `README.md` must make sense to a technically literate outsider without reading anything else. It should:

- describe what `codex-proxy` is in one paragraph: a local OpenAI-compatible endpoint that routes to a pool of upstream backends with health-aware selection.
- show a minimal quick start with a single OpenAI API key backend.
- document where the config lives and link to `docs/config.md`.
- document `/health`, `/status`, and `/v1/chat/completions` with a curl example.
- state the verification command used by CI and humans.

The README must not reference: , Relay, internal branches, polestar, consumer-auth specifics, or any workstation-local tooling.

## Verification

One canonical verification command runs locally and in CI:

```
make verify
```

Which runs, in order:

- `ruff check .`
- `ruff format --check .`
- `mypy src/codex_proxy`
- `pytest -q`

Test layout:

- `tests/unit/test_selector.py` — selector ranking on fixture backends.
- `tests/unit/test_error_taxonomy.py` — upstream-status → action mapping.
- `tests/unit/test_config.py` — config load + validation, including the consumer-auth gate.
- `tests/integration/test_rotation.py` — spin the app with two `httpx.MockTransport` backends, force 429 on one, assert second request lands on the other, both streaming and non-streaming.
- `tests/smoke/test_openai_sdk.py` — optional, skipped unless `OPENAI_API_KEY` is set; uses the official OpenAI Python SDK against `localhost`.

## Milestones

Order of work, each a small reviewable slice:

1. Bootstrap: `git init`, `pyproject.toml`, `src/codex_proxy/` layout, `Makefile`, FastAPI app with `/health`, `make verify` green on an empty test.
2. `Backend` protocol + in-memory fake backend + selector unit tests.
3. `openai_api_key` backend real implementation + `/v1/chat/completions` non-streaming.
4. Streaming SSE passthrough.
5. Error taxonomy + rotation integration test.
6. `/status` endpoint + usage/cooldown persistence.
7. `/control/pin` + `/control/unpin`.
8. `azure_openai` backend.
9. Public README + `docs/config.md`.
10. (Later) `codex_auth_vault` backend, gated by `allow_consumer_auth_backends`.
11. (Later) sticky session policy.
12. (Later) `/v1/responses`.

## Open items (small)

- Package manager: `uv` or `poetry` or plain `pip-tools`. Recommendation: `uv` for speed and single-file lock.
- Python version floor: 3.11 (for `tomllib`, structural-pattern-matching, asyncio ergonomics).
- Whether to expose `/v1/models` in v1 or defer. Recommendation: expose; clients often probe it.

These should be decided before milestone 1 ships but do not need their own planning round.
