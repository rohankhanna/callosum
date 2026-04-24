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

### `[state]`

```toml
[state]
dir = "/home/you/.local/state/codex-proxy"
```

- `dir` — optional. If set, per-backend usage and cooldown snapshots are persisted here so they survive restart. If omitted, all state is memory-only.

### `[[backends]]`

One table per Codex account. Each backend needs a unique `id` and its own `auth.json` vault.

```toml
[[backends]]
id = "account-a"
vault_path = "/home/you/.codex-proxy/vaults/account-a/auth.json"
models = ["model-a0d0"]
codex_base_url = "https://chatgpt.com/backend-api/codex"   # optional
```

- `id` — required. Used in `/status`, `/control/pin`, and session bindings.
- `vault_path` — required. Absolute path to the account's `auth.json` (the format the Codex CLI login flow writes). Must contain a `tokens` object with `access_token` and `refresh_token`; `account_id` and `id_token` are used when present. The file is rewritten in place after each successful OAuth refresh.
- `models` — required. Advertised model list for this account. The proxy only considers this backend for requests whose `model` is in the list.
- `type` — optional. Defaults to `"codex_auth_vault"` (the only supported type).
- `codex_base_url` — optional. Defaults to the ChatGPT backend base. Override for a staging endpoint.

To rotate across multiple accounts, declare multiple `[[backends]]` entries with different `vault_path`s and the same advertised model list.

## Endpoints

- `POST /v1/responses` — native OpenAI Responses API. Streaming SSE is forwarded upstream-to-client byte-for-byte.
- `POST /v1/chat/completions` — standard chat-completions shape. Requests are translated into Responses calls on the way out and the result is translated back.

Both routes use the same backend pool and the same selector.

## Selector behavior

Per request, the selector:

1. Drops any backend that does not advertise the requested model.
2. Drops any backend whose `available` is false or whose `cooldown_until_ts` is in the future.
3. Prefers `weekly_exhausted=false` over `true`.
4. Among ties, prefers higher `remaining_fraction` (unknown treated as `0.5`).
5. Among ties, prefers more recent `probed_at_ts`.
6. Final tiebreak: backend `id`, lexicographic.

If all accounts advertising the requested model are unavailable, the terminal response mirrors the last error class: `rate_limited` → `429`, `auth_invalid` or `transient` → `502`, `unknown_model` → `400`, no error at all → `503`.

## Rotation and classification

| Upstream status | Classification | Action |
| --------------- | -------------- | ------ |
| `200`           | healthy        | pass through |
| `400`           | client_error   | surface to client, no rotation |
| `401` / `403`   | auth_invalid   | rotate; if none left, `502` |
| `404`           | unknown_model  | rotate; if none left, `400` |
| `429`           | rate_limited   | cooldown (retry-after or 60s default); rotate; if none left, `429` |
| `5xx` / network | transient      | rotate; if none left, `502` |

Rotation means: exclude the failing backend from this request and re-run the selector. Cooldowns are recorded per backend and visible in `/status`.

## Streaming

Set `"stream": true` in the request body and the proxy returns `text/event-stream`. On `/v1/responses` the upstream SSE chunks are forwarded unchanged. On `/v1/chat/completions` the buffered upstream response is re-emitted as a small sequence of chat-completion chunks. Errors before the first chunk still trigger rotation; errors mid-stream cannot be retried and surface to the client.

## Session stickiness (client-opt-in)

By default every request is an independent selection. If a client wants to stay on the same account across a sequence of requests, it sends:

```
X-Codex-Session-Id: <any string>
```

The first request with that id binds it to whichever backend the selector picks. Subsequent requests with the same id route to that backend while it remains viable. If the bound backend becomes unavailable, the proxy falls back to the normal ranking and updates the binding to whichever backend actually served the request. Clients that never send the header get no stickiness — that is intentional.

Bindings are process-local and in-memory; they do not survive a restart.

## Pinning

Use `/control/pin` to force routing to a specific backend id. While a pin is set, the selector considers only that backend — no fallback — so upstream errors surface directly to the client. Pin overrides any session binding: while pinned, the header is ignored. `/control/unpin` clears the pin.

## Verification

```
make verify
```
