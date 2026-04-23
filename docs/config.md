# Configuration

`codex-proxy` reads a single TOML file. Default path: `~/.config/codex-proxy/config.toml`. Override with `--config /path/to/config.toml`.

## Sections

### `[server]`

```toml
[server]
host = "127.0.0.1"
port = 8765
client_auth_token_env  = "CODEX_PROXY_CLIENT_TOKEN"   # optional
control_auth_token_env = "CODEX_PROXY_CONTROL_TOKEN"  # optional
```

- `host` — bind address. Default `127.0.0.1`. Do not expose on `0.0.0.0` unless you understand the threat model.
- `port` — TCP port. Default `8765`.
- `client_auth_token_env` — if set, `/v1/*` requires `Authorization: Bearer <value of this env var>`.
- `control_auth_token_env` — if set, `/status` and `/control/*` require the same.

### `[policy]`

```toml
[policy]
default_mode = "stateless"
allow_consumer_auth_backends = false
```

- `default_mode` — session policy. Supported values: `"stateless"` (each request is selected fresh) and `"sticky"` (first-touch backend is remembered per session; see [Session policy](#session-policy) below). `"sticky_replay"` is reserved for a future release.
- `allow_consumer_auth_backends` — gate for the `codex_auth_vault` backend type. Must be `true` for that type to load. See the ToS note on that backend below before enabling.

### `[state]`

```toml
[state]
dir = "/home/you/.local/state/codex-proxy"
```

- `dir` — optional. If set, per-backend usage and cooldown snapshots are persisted here so they survive restart. If omitted, all state is memory-only.

### `[[backends]]`

One table per backend. Each backend needs a unique `id`.

#### `openai_api_key`

For OpenAI and any provider that accepts `Authorization: Bearer <key>`.

```toml
[[backends]]
id = "primary"
type = "openai_api_key"
api_key_env = "OPENAI_API_KEY"
base_url = "https://api.openai.com/v1"   # optional
models = ["model-a0f5", "model-a0f5-mini"]
```

- `api_key_env` — required. Name of the env var that holds the bearer token. Must be set at startup.
- `base_url` — optional. Defaults to OpenAI. Use this to point at OpenRouter or a local compatible server.
- `models` — advertised model list. The selector only considers this backend for requests whose `model` is in the list.

#### `azure_openai`

For Azure OpenAI resources. Azure uses per-deployment URLs and `api-key` header auth.

```toml
[[backends]]
id = "azure-main"
type = "azure_openai"
api_key_env = "AZURE_OPENAI_KEY"
endpoint = "https://<resource>.openai.azure.com"
api_version = "2024-10-01-preview"
deployments = { "model-a0f5" = "model-a0f5-prod", "model-a0f5-mini" = "model-a0f5-mini-prod" }
```

- `api_key_env` — required.
- `endpoint` — required. Your Azure OpenAI resource base URL.
- `api_version` — required. Azure API version.
- `deployments` — required. Map from model name (as clients request it) to Azure deployment name. The keys become the advertised model list for this backend.

#### `codex_auth_vault`

For Codex/ChatGPT consumer-plan credentials stored as `auth.json` (the format the Codex CLI's login flow writes). The backend reads tokens from disk, refreshes via OAuth when the access token nears expiry, and hits the ChatGPT backend Responses API on your behalf. Chat-completions requests are translated to the Responses API shape and back.

Gated behind `policy.allow_consumer_auth_backends = true`. Do not enable unless you have read OpenAI's consumer terms and decided you're comfortable with programmatic rotation across your own seats.

```toml
[policy]
allow_consumer_auth_backends = true

[[backends]]
id = "codex-account-a"
type = "codex_auth_vault"
vault_path = "/home/you/.local/state/codex-proxy/vaults/account-a/auth.json"
models = ["model-a0d0"]
codex_base_url = "https://chatgpt.com/backend-api/codex"   # optional
```

- `vault_path` — required. Absolute path to the account's `auth.json`. The file must contain a `tokens` object with `access_token` and `refresh_token`; `account_id` and `id_token` are used when present. The file is rewritten in place after each successful refresh.
- `models` — required. Advertised model list for this backend.
- `codex_base_url` — optional. Defaults to the ChatGPT backend base. Override for a staging or on-prem endpoint.

One backend entry per account. To rotate across multiple accounts, declare multiple `[[backends]]` entries with different `vault_path`s and the same advertised model.

## Selector behavior

Per request, the selector:

1. Drops any backend that does not advertise the requested model.
2. Drops any backend whose `available` is false or whose `cooldown_until_ts` is in the future.
3. Prefers `weekly_exhausted=false` over `true`.
4. Among ties, prefers higher `remaining_fraction` (unknown treated as `0.5`).
5. Among ties, prefers more recent `probed_at_ts`.
6. Final tiebreak: backend `id`, lexicographic.

If all backends for a model are exhausted, the terminal response mirrors the last error class: `rate_limited` → `429`, `auth_invalid` or `transient` → `502`, `unknown_model` → `400`, no error at all → `503`.

## Rotation and classification

| Upstream status | Classification | Action |
| --------------- | -------------- | ------ |
| `200`           | healthy        | pass through |
| `400`           | client_error   | surface to client, no rotation |
| `401` / `403`   | auth_invalid   | rotate; if none left, `502` |
| `404`           | unknown_model  | rotate; if none left, `400` |
| `429`           | rate_limited   | cooldown (retry-after or 60s default); rotate; if none left, `429` |
| `5xx` / network | transient      | rotate; if none left, `502` |

Rotation means: exclude the failing backend from this request and re-run the selector. Cooldowns are recorded on the backend and visible in `/status`.

## Streaming

Set `"stream": true` in the request body and the proxy returns `text/event-stream`. Upstream SSE chunks are forwarded byte-for-byte. Errors before the first chunk still trigger rotation; errors mid-stream cannot be retried and surface to the client.

## Endpoints

| Route                    | Request shape             | Backends considered                       |
| ------------------------ | ------------------------- | ----------------------------------------- |
| `POST /v1/chat/completions` | OpenAI Chat Completions | any backend that advertises the model     |
| `POST /v1/responses`       | OpenAI Responses API    | only backends with native Responses support |

Per-backend Responses support:

| Backend type        | `responses_supported` | Notes |
| ------------------- | --------------------- | ----- |
| `openai_api_key`    | yes                   | Body is forwarded to `{base_url}/responses`. |
| `codex_auth_vault`  | yes                   | Body is forwarded verbatim to the ChatGPT Responses endpoint with vault headers. `/v1/chat/completions` on this backend goes through a translation layer; `/v1/responses` is the native, streaming-friendly path. |
| `azure_openai`      | no                    | Azure's Responses API shape is not wired in this release. A request to `/v1/responses` that can only route to an Azure backend returns `503`. |

The selector's viability and rotation logic apply identically to both routes: the pool for `/v1/responses` is filtered to `responses_supported=True` backends first, then the usual ranking runs.

Sticky bindings are shared across routes. A session id bound on `/v1/chat/completions` is honored on `/v1/responses` and vice versa, so a client that mixes the two stays on one backend per session.

## Pinning

Use `/control/pin` to force routing to a specific backend id. While a pin is set, the selector considers only that backend — no fallback — so upstream errors surface directly to the client. `/control/unpin` clears the pin.

## Session policy

`policy.default_mode` controls how the proxy treats per-client session identity.

- `stateless` (default) — the selector runs fresh on every request. The `X-Codex-Session-Id` header is ignored. `/status.sessions` is always `{}`.
- `sticky` — when a request carries `X-Codex-Session-Id: <id>`, the proxy remembers which backend served it and prefers that backend on subsequent requests with the same id. If the bound backend is no longer viable (health, cooldown, excluded mid-retry), the selector falls back to normal ranking and the binding is updated to the backend that actually served the request. Requests without the header behave like stateless. Bindings are in-memory and process-local; they do not survive a restart.

A pin set via `/control/pin` overrides sticky bindings: while pinned, all traffic goes to the pinned backend regardless of the session header. Clearing the pin restores sticky behavior; existing bindings are preserved.

`/status` reports the active mode under `session_mode` and the current bindings under `sessions` (a map of `session_id → backend_id`, only populated when the mode is sticky and at least one binding exists).

## Verification

```
make verify
```
