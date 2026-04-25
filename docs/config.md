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

## Usage log (`[usage_log]`)

```toml
[usage_log]
path = "/home/you/.local/state/codex-proxy/requests.sqlite"
capture_bodies = true
```

- `path` — optional. When set, every backend call is recorded as a row in this SQLite database. When unset, no logging happens.
- `capture_bodies` — default `true`. When on, the request body, the upstream response body (or full SSE blob for streams), and the upstream response headers are stored as zlib-compressed blobs in a companion `request_bodies` table. Turn off once a usage-prediction model is trained.

### Schema

`requests` (one row per backend attempt — both successes and rotated-from failures):

| Column | Notes |
| --- | --- |
| `id`, `ts_start`, `ts_end`, `latency_ms` | timing |
| `route` | `responses` or `chat_completions` |
| `stream` | `1` if the client asked for streaming |
| `session_id` | the `X-Codex-Session-Id` if the client sent one |
| `backend_id` | which vault served (or attempted) this call |
| `model`, `reasoning_effort` | extracted from the request body |
| `status`, `classification` | HTTP status returned + error class (`ok`, `rate_limited`, `auth_invalid`, `transient`, `unknown_model`, `client_error`) |
| `request_bytes`, `response_bytes` | sizes |
| `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens` | parsed from upstream `response.usage` (terminal `response.completed` event for streams) |
| `plan_type`, `active_limit` | from `x-codex-plan-type`, `x-codex-active-limit` |
| `primary_used_percent_before` / `_after` | the 5-hour-window quota state immediately before and after the call. `before` is null on the first call to a given backend. |
| `secondary_used_percent_before` / `_after` | same for the weekly window |
| `primary_reset_at`, `secondary_reset_at` | unix ts when each window resets |
| `primary_over_secondary_limit_percent` | upstream-reported overage signal |
| `credits_balance`, `credits_has_credits`, `credits_unlimited` | credits state |
| `quota_reset_crossover` | `1` if `_after < _before` (a window reset fired during the call); exclude these rows from training |

`request_bodies` (one row per `requests.id`, only present when `capture_bodies = true` and at least one of req/resp/headers had data):

| Column | Notes |
| --- | --- |
| `request_id` | foreign key to `requests.id` |
| `req_payload` | zlib-compressed client request JSON |
| `resp_payload` | zlib-compressed response (full JSON for non-stream; full SSE blob for streams) |
| `upstream_headers` | zlib-compressed JSON of all upstream response headers |

### Example: cost per request by model + reasoning effort

```sql
SELECT
  model,
  reasoning_effort,
  COUNT(*) AS calls,
  AVG(prompt_tokens) AS avg_prompt,
  AVG(completion_tokens) AS avg_completion,
  AVG(primary_used_percent_after - primary_used_percent_before) AS avg_d_primary_pct,
  AVG(secondary_used_percent_after - secondary_used_percent_before) AS avg_d_secondary_pct
FROM requests
WHERE status = 200 AND quota_reset_crossover = 0
  AND primary_used_percent_before IS NOT NULL
GROUP BY model, reasoning_effort;
```

### Operational notes

- **Privacy**: with `capture_bodies = true`, prompts and responses are stored on disk in cleartext (after zlib decompression). Treat the database file as sensitive. `chmod 600` is a sensible baseline.
- **Granularity**: `primary_used_percent` and `secondary_used_percent` are integer percentages reported by the upstream. Single small calls often show `Δ = 0`. Aggregate across many calls for a useful signal.
- **Auth-vault refresh-chain caveat**: the same Codex `auth.json` cannot be active in both this proxy and a normal `codex` CLI session at once — the refresh token is one-shot and the first refresh invalidates the other side's stored copy. Either dedicate an account to the proxy, or copy the latest `auth.json` into the vault path immediately before starting the proxy.
- **No rotation in v1**: the SQLite file grows append-only. Archive manually when it gets large.

## Multi-tenant auth (`[auth]`)

Optional. When enabled, other people can register, log in, mint their own API keys, and use those keys to call `/v1/responses` and `/v1/chat/completions`. Each call is attributed to a `(user_id, api_key_id)` pair in the usage log.

```toml
[auth]
db = "/home/you/.local/state/codex-proxy/auth.sqlite"
session_ttl_seconds = 1800   # default
```

- `db` — required to enable auth. Path to the SQLite database holding users, sessions, and API keys. Single-operator mode is the default (no `[auth]` block at all → `/v1/*` is open, the `/auth/*` routes do not exist).
- `session_ttl_seconds` — how long a session token returned by `/auth/login` stays valid. Default 30 minutes.

Passwords are stored as **argon2id** hashes. Session tokens and API keys are 32 bytes urandom each, returned to the user once and stored only as `sha256(plaintext)`.

### Endpoints

- `POST /auth/register` `{username, password}` → `201 {user_id, username}`
- `POST /auth/login` `{username, password}` → `200 {session_token, expires_at}`
- `POST /auth/logout` (Bearer session) → `204`
- `POST /auth/keys` (Bearer session) `{label?}` → `201 {id, api_key, prefix, label, created_at}`. **`api_key` is plaintext and shown ONCE — store it now.**
- `GET /auth/keys` (Bearer session) → `200 {keys: [{id, prefix, label, created_at, last_used_at, revoked_at}, ...]}`
- `DELETE /auth/keys/{id}` (Bearer session) → `200 {revoked: true}` or `404`

### Calling `/v1/*` with an API key (curl)

```
curl http://127.0.0.1:8765/v1/responses \
  -H "Authorization: Bearer cp-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  -H "Content-Type: application/json" \
  -d '{"model":"model-a0e7","input":[...]}'
```

Missing or revoked key → `401`. The `usage_log` `requests` table picks up `user_id` and `api_key_id` so a query like `SELECT user_id, SUM(total_tokens) FROM requests GROUP BY user_id` works as expected.

### Setting up the Codex CLI as a client

End users get an account from the operator (or self-register), log in to get a session token, mint an API key, and then point their **own** Codex CLI at the proxy URL with that key. The end user's Codex CLI never touches the operator's vault files; it only knows the proxy URL and its API key.

#### One-time bootstrap (per end user)

1. **Operator-side**: confirm the proxy is reachable from the user's machine and `[auth].db` is set.

2. **User-side: register and log in.** Substitute the proxy URL for the operator's host:port.

   ```
   curl -X POST http://your-proxy-host:8765/auth/register \
     -H "Content-Type: application/json" \
     -d '{"username":"alice","password":"<a-strong-password>"}'

   curl -X POST http://your-proxy-host:8765/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"alice","password":"<a-strong-password>"}'
   # → {"session_token":"<token>","expires_at":...}
   ```

3. **Mint an API key with the session token.** Save the returned plaintext immediately — it is shown only once.

   ```
   curl -X POST http://your-proxy-host:8765/auth/keys \
     -H "Authorization: Bearer <session-token>" \
     -H "Content-Type: application/json" \
     -d '{"label":"my-laptop"}'
   # → {"id":1,"api_key":"<plaintext-key>","prefix":"AbCd1234","label":"my-laptop",...}
   ```

#### Wiring the Codex CLI

Add a provider definition and a profile to the **client's** `~/.codex/config.toml`. (Or use a separate `CODEX_HOME` to keep this isolated from your normal Codex usage — recommended if you also use Codex against `chatgpt.com` directly, see the refresh-chain caveat below.)

```toml
[model_providers.codex_proxy]
name = "codex-proxy"
base_url = "http://your-proxy-host:8765/v1"
env_key = "CODEX_PROXY_TOKEN"
wire_api = "responses"

[profiles.via_proxy]
model = "model-a0e7"
model_provider = "codex_proxy"
```

`env_key` is the name of an env var holding the API key. Codex CLI insists on `env_key` being set to a non-empty value; the proxy uses it as the bearer token.

```
echo 'export CODEX_PROXY_TOKEN=<plaintext-key>' >> ~/.bashrc
source ~/.bashrc
```

#### Using the proxy via Codex CLI

```
# interactive
codex -p via_proxy

# non-interactive (great for scripts / automation)
codex exec -p via_proxy "summarise this file" < src/something.py

# without -p, the user's normal codex usage is unaffected
codex
```

#### Other clients

The proxy speaks standard OpenAI shapes on `/v1/responses` and `/v1/chat/completions`. Any client that lets you set a base URL and a bearer token will work — Aider, Cursor, Continue, Raycast AI, custom SDK scripts using `openai-python` with `base_url` set, etc. Use the same `Authorization: Bearer <api-key>` pattern.

### Operational notes

- **Auth.json refresh-chain caveat.** The proxy reads `auth.json` files directly. It does *not* run a Codex CLI subprocess — there is no Codex CLI inside the server. The CLI is only used (on the operator's machine, once) to log in and produce the file. Once the file is in a vault path, the proxy owns the OAuth refresh chain for that account: every successful refresh produces a new refresh token and writes it back to the file. If anything else (e.g. the operator's normal `codex` usage on the same account) refreshes against the same `auth.json` chain in parallel, whichever side rotates first invalidates the other side. **Either dedicate an account to the proxy, or route the operator's own Codex usage through the proxy too** (using the client setup above with a `localhost` `base_url`).
- **One Codex CLI, two roles.** The operator's CLI is used *once per account* to produce `auth.json`. The end user's CLI is used *all the time* to make actual requests, pointed at the proxy. They are the same binary playing different roles.

### Out of scope (today)

- Per-key rate limits / budgets — comes after enough usage data exists to set meaningful values.
- Email verification, password reset, MFA — single-operator instance; users are people you know personally.
- TLS / public exposure — keep the proxy on `127.0.0.1` and front it with Caddy/nginx if exposed.

## Daily upstream healthcheck (`GET /diagnose/upstream`)

The proxy doesn't have automated tests against the real Codex backend — a regression in the upstream contract (renamed model, removed header, changed SSE event shape) would only show up when a real request fails. `GET /diagnose/upstream` is a one-shot probe per backend that catches those regressions before they bite a user request.

Each invocation makes one tiny streaming request per non-cooldown'd backend (prompt: "say only: ok", a few hundred tokens of overhead total). It then evaluates:

| Check | Catches |
| --- | --- |
| `http_2xx` | URL or auth flow change (401, 404, 5xx from upstream) |
| `quota_headers_present` | `x-codex-*` headers removed or renamed |
| `primary_used_percent_present` | the 5-hour-window quota field disappeared |
| `secondary_used_percent_present` | the weekly quota field disappeared |
| `response_completed_event_present` | SSE terminal event renamed or restructured |
| `usage_block_present` | `response.completed.response.usage` shape changed |

Plus, an upstream `400` (e.g. "model not supported") surfaces as `stage = "upstream"` with the status code, which catches Codex retiring or renaming a model.

Response shape:

```json
{
  "ok": true,
  "backends": [
    {
      "id": "primary",
      "ok": true,
      "skipped": false,
      "stage": "evaluate",
      "model": "model-a0e7",
      "checks": {"http_2xx": true, "quota_headers_present": true, "...": true},
      "failed_checks": [],
      "upstream_status": 200
    }
  ]
}
```

Cooled-down backends are reported with `skipped: true` and **don't** flip the aggregate `ok` to false — the daily check shouldn't kick a backend that's already recovering.

When auth is enabled, the route requires a valid API key (same bearer middleware that gates `/v1/*`). Mint a dedicated key labeled "diag-cron" for this purpose so revoking it doesn't disturb other clients.

### Cron example

```
# Run at 09:00 every day; alert via mail or systemd-cat on any failure.
0 9 * * *   curl -s -H "Authorization: Bearer $CODEX_PROXY_DIAG_KEY" \
                 http://127.0.0.1:8765/diagnose/upstream \
            | jq -e '.ok' > /dev/null \
            || echo "codex-proxy upstream check failed at $(date)" \
                | mail -s "codex-proxy: upstream regression" you@example.com
```

For systemd users, prefer a `systemd.timer` over cron — easier to inspect via `systemctl list-timers` and to log with `journalctl -u codex-proxy-diag`.

## Verification

```
make verify
```
