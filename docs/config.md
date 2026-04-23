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

- `default_mode` — session policy. Only `"stateless"` is wired in v1 (every request is routed independently; no server-side session state).
- `allow_consumer_auth_backends` — gate for the `codex_auth_vault` backend type (not yet implemented). Must be `true` for that type to load.

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

## Pinning

Use `/control/pin` to force routing to a specific backend id. While a pin is set, the selector considers only that backend — no fallback — so upstream errors surface directly to the client. `/control/unpin` clears the pin.

## Verification

```
make verify
```
