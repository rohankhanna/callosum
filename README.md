# codex-proxy

A small local HTTP proxy that sits in front of multiple Codex Plus/Pro authentications and rotates across them. One endpoint on `127.0.0.1`, many `auth.json` vaults behind it. When one account is rate-limited or otherwise unavailable, the proxy sends the next request to another.

It is deliberately a less-than-intelligent router. It does not try to summarize, retry mid-stream, synthesize continuity, or do anything fancier than "pick an account that can serve this request, and if it fails, try the next one." The OpenAI-compatible routes (`/v1/responses` and `/v1/chat/completions`) exist so your normal Codex-speaking clients can point at this proxy without knowing anything changed.

## Quick start

Install (with [uv](https://docs.astral.sh/uv/)):

```
uv sync
```

Each account you want to rotate across needs its own `auth.json` — the file format the Codex CLI writes when it logs in. Give each one its own path, e.g. `~/.codex-proxy/vaults/account-a/auth.json`, `~/.codex-proxy/vaults/account-b/auth.json`.

Write a config at `~/.config/codex-proxy/config.toml`:

```toml
[[backends]]
id = "account-a"
vault_path = "/home/you/.codex-proxy/vaults/account-a/auth.json"
models = ["model-a0d0"]

[[backends]]
id = "account-b"
vault_path = "/home/you/.codex-proxy/vaults/account-b/auth.json"
models = ["model-a0d0"]
```

Start the server:

```
uv run python -m codex_proxy
```

It listens on `http://127.0.0.1:8765` by default. Point your Codex-speaking client at that URL.

Native Responses API:

```
curl http://127.0.0.1:8765/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-a0d0",
    "input": [{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}]
  }'
```

Chat-completions shape also works; requests are translated to the Responses API on the way out and back on the way in.

Streaming works on both routes with `"stream": true`. On `/v1/responses` the upstream SSE is forwarded byte-for-byte.

## How rotation works

On each request, the proxy picks the backend with the most headroom (non-exhausted, not in cooldown, advertises the requested model). If that backend returns a retryable error — `429`, `401/403`, `404`, transient `5xx` — it is excluded from the current request and the proxy re-selects. On `429`, the upstream's `Retry-After` (or 60 seconds if absent) becomes the cooldown; if `[state]` is configured, the cooldown is persisted so it survives a restart.

There is no cross-request "session" by default. Each call is independent. If a client wants the *same account* across a series of calls (e.g. to avoid switching accounts mid-conversation), it sends a `X-Codex-Session-Id: <anything>` header on each request and the proxy remembers the binding for that id. No header, no session — it stays a fresh pick every time.

## Operational endpoints

- `GET /health` — liveness probe.
- `GET /status` — backend pool view: each vault's id, advertised models, current health, usage snapshot, cooldown, plus the active pin and any session bindings.
- `POST /control/pin` with `{"backend_id":"<id>"}` — force all routing to one backend. Useful when you want to burn down one account on purpose.
- `POST /control/unpin` — clear the pin.

```
curl http://127.0.0.1:8765/status | jq
```

## Multi-tenant auth (optional)

The proxy ships single-operator by default — `127.0.0.1`, no auth on `/v1/*`. When `[auth] db = "..."` is configured, a small `/auth/*` surface goes live: register/login/issue-and-revoke-API-keys. With auth on, every `/v1/*` call requires `Authorization: Bearer <api-key>` and is attributed in the usage log to the issuing user. Argon2id for passwords, sha256-stored opaque keys. See [`docs/config.md`](docs/config.md#multi-tenant-auth-auth) for the endpoints and curl examples. Out of scope: per-key rate limits (comes after enough corpus to budget against), TLS (front with Caddy/nginx), and password reset (single-operator instance).

## Per-request usage log

When `[usage_log] path = "..."` is set in the config, every backend call is recorded as a row in a SQLite database with token counts, latency, and the upstream-reported per-account quota state (5-hour and weekly window) before and after the call. With `capture_bodies = true` (default) the request and response payloads are also stored, zlib-compressed. This is the data corpus for modeling how token-and-reasoning-effort inputs translate into Codex Plus/Pro quota consumption. See [`docs/config.md`](docs/config.md#usage-log-usage_log) for schema and example queries.

## Configuration

Full config reference: [`docs/config.md`](docs/config.md).

## Verify

```
make verify
```

Runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`.
