# codex-proxy

A small local HTTP proxy that sits in front of multiple Codex Plus/Pro authentications and rotates across them. One endpoint on `127.0.0.1`, many `auth.json` vaults behind it. When one account is rate-limited or otherwise unavailable, the proxy sends the next request to another.

It is deliberately a less-than-intelligent router. It does not summarise, retry mid-stream, synthesise continuity, or do anything fancier than "pick an account that can serve this request, and if it fails, try the next one." The OpenAI-compatible routes (`/v1/responses` and `/v1/chat/completions`) exist so your normal Codex-speaking clients can point at this proxy without knowing anything changed.

## Contents

- [How it works (architecture)](#how-it-works-architecture)
- [Quick start (operator)](#quick-start-operator)
- [Bootstrap your daily-driver API key](#bootstrap-your-daily-driver-api-key)
- [Pointing clients at the proxy](#pointing-clients-at-the-proxy)
  - [Universal pattern (env-var override)](#universal-pattern-env-var-override)
  - [Codex CLI](#codex-cli)
  - [Hermes Agent](#hermes-agent)
  - [Aider](#aider)
  - [Cursor](#cursor)
  - [Continue (VS Code)](#continue-vs-code)
  - [OpenAI Python SDK](#openai-python-sdk)
  - [OpenAI JS / TypeScript SDK](#openai-js--typescript-sdk)
  - [curl](#curl)
  - [Generic: any OpenAI-compatible client](#generic-any-openai-compatible-client)
  - [Claude Code: requires Phase 3](#claude-code-requires-phase-3)
- [Configuration reference](#configuration-reference)
  - [`[server]`](#server)
  - [`[state]`](#state)
  - [`[usage_log]`](#usage_log)
  - [`[auth]`](#auth)
  - [`[[backends]]`](#backends)
- [Endpoints](#endpoints)
- [How rotation works](#how-rotation-works)
- [Sticky session header](#sticky-session-header-x-codex-session-id)
- [Pinning](#pinning-controlpin--controlunpin)
- [Per-request usage log](#per-request-usage-log)
- [Daily upstream healthcheck](#daily-upstream-healthcheck-getdiagnoseupstream)
- [Operational notes](#operational-notes)
- [Verify](#verify)

## How it works (architecture)

Three roles. Two of them involve a Codex CLI binary — easy to confuse.

```
                                                       ┌──────────────────────┐
[ operator's Codex CLI ]   one-off, only for login →  │  account-a/auth.json │
[ operator's Codex CLI ]   one-off, only for login →  │  account-b/auth.json │
                                                       └──────────┬───────────┘
                                                                  │ files on disk
                                                                  ▼
                                                       ┌──────────────────────┐
[ end user's Codex CLI ] ── HTTP /v1/responses ─────► │   codex-proxy server  │ ── HTTPS ──► chatgpt.com/backend-api/codex
[ another Codex CLI    ] ── HTTP /v1/responses ─────► │   (FastAPI process)   │           (with vault's access token +
[ Aider / Cursor / ... ] ── HTTP /v1/chat/completions │                       │            chatgpt-account-id header)
                                                       └──────────────────────┘
                                                                  ▲
                            Authorization: Bearer <api-key> ──────┘
                            (only when [auth] is enabled)
```

Two completely different auth flows are involved:

| Layer | What it secures | Credential | Where it lives |
| --- | --- | --- | --- |
| **Client → Proxy** | Your `codex` CLI authenticating to the proxy | API key minted via `/auth/keys` | `CODEX_PROXY_TOKEN` env var |
| **Proxy → ChatGPT** | The proxy authenticating to `chatgpt.com/backend-api/codex` | OAuth tokens from `auth.json` | `~/.codex-proxy/vaults/*/auth.json` |

The `auth.json` files are the proxy's **upstream** credentials. End users never see them. End users get their own short-lived API keys, completely unrelated to your Codex Plus accounts.

The proxy is just a FastAPI process. It does **not** shell out to a Codex CLI at request time — there is no Codex CLI inside the server. It reads `auth.json` directly, refreshes the OAuth access token over HTTPS when it nears expiry, and forwards Responses-API requests upstream itself. The operator's CLI is only used once per account, to log in and produce `auth.json`.

## Quick start (operator)

Install with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

Each Codex account you want to rotate across needs its own `auth.json`. The fastest way to get one: log into the account with the normal Codex CLI, then **move** (not copy) `~/.codex/auth.json` to a vault path:

```
mkdir -p ~/.codex-proxy/vaults/account-a
mv ~/.codex/auth.json ~/.codex-proxy/vaults/account-a/auth.json
chmod 600 ~/.codex-proxy/vaults/account-a/auth.json
```

Repeat for each account (log in to the CLI as that account, then `mv` its `auth.json` out). Moving rather than copying avoids the [refresh-chain caveat](#operational-notes).

Write `~/.config/codex-proxy/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8765

[state]
dir = "/home/you/.local/state/codex-proxy"

[usage_log]
path = "/home/you/.local/state/codex-proxy/requests.sqlite"
capture_bodies = true

[auth]
db = "/home/you/.local/state/codex-proxy/auth.sqlite"

[[backends]]
id = "account-a"
vault_path = "/home/you/.codex-proxy/vaults/account-a/auth.json"
models = ["model-a0e7"]

[[backends]]
id = "account-b"
vault_path = "/home/you/.codex-proxy/vaults/account-b/auth.json"
models = ["model-a0e7"]
```

Start the server:

```
uv run python -m codex_proxy
```

It listens on `http://127.0.0.1:8765` by default. Confirm it's up:

```
curl -s http://127.0.0.1:8765/health
# {"status":"ok","version":"0.1.0"}
```

## Bootstrap your daily-driver API key

If you set `[auth]` in your config (recommended), every `/v1/*` request needs a bearer token. Mint one for yourself once, store it in your shell rc, and forget about it.

**The easy way: web UI.** Open [`http://127.0.0.1:8765/ui/`](http://127.0.0.1:8765/ui/) in a browser. Click "Create account", pick a username and password, then "Mint key" with a label like `daily-driver`. The plaintext key is shown ONCE — copy it immediately into your shell rc:

```
echo 'export CODEX_PROXY_TOKEN=<the-plaintext-from-the-ui>' >> ~/.bashrc
source ~/.bashrc
```

**The curl way (same flow, scriptable):**

```
PROXY=http://127.0.0.1:8765

# 1. Register (once per user).
curl -s -X POST $PROXY/auth/register \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"<a-strong-password>"}'

# 2. Log in to get a 30-min session token.
SESSION=$(curl -s -X POST $PROXY/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"alice","password":"<a-strong-password>"}' \
  | jq -r .session_token)

# 3. Mint an API key. PLAINTEXT IS SHOWN ONLY ONCE — copy it now.
curl -s -X POST $PROXY/auth/keys \
  -H "Authorization: Bearer $SESSION" \
  -H 'Content-Type: application/json' \
  -d '{"label":"daily-driver"}' \
  | jq -r .api_key
```

Either way you end up with a plaintext key that goes into `CODEX_PROXY_TOKEN`. The proxy stores only `sha256(key)`, never the plaintext, so this is your only chance to see it.

If you don't want auth at all (single-operator localhost, no other people, no usage attribution): drop the `[auth]` block from your config. `/v1/*` is then open and the `/ui/` route returns 404.

## Pointing clients at the proxy

### Universal pattern (env-var override)

Almost every OpenAI-compatible client honours one or both of these env vars:

| Env var | Used by |
| --- | --- |
| `OPENAI_BASE_URL` | OpenAI Python/JS SDK, Hermes Agent, most modern clients |
| `OPENAI_API_BASE` | Aider, older clients |
| `OPENAI_API_KEY` | All of them. The proxy treats this as the bearer. |

The history-preserving pattern is to override **per command**, not in your shell rc:

```
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=$CODEX_PROXY_TOKEN \
your-tool ...
```

Persistent edits (rc files, app settings) work too; they just permanently re-target the tool.

> **Codex CLI does NOT honor `OPENAI_BASE_URL`.** It has its own typed provider system. Setting `OPENAI_BASE_URL=http://127.0.0.1:8765/v1 codex ...` will silently route to OpenAI's real API and present your codex-proxy key as if it were an OpenAI key — which OpenAI then rejects. Use the [Codex CLI section below](#codex-cli) instead.

### Codex CLI

The OpenAI Codex CLI (`codex`, `codex exec`, `codex resume`) uses a typed provider system. Two integration shapes — pick based on whether you want the proxy to be the default for all `codex` commands, or only when explicitly opted into.

#### Option A — `codex_proxy` as the global default (recommended for "single point of auth" setups)

Edit `~/.codex/config.toml` so global defaults stay at the top, **before any `[section]` header**.

> **TOML detail that bit me once.** Every `key = value` line after a `[section]` header belongs to that section until the next header. Putting `model_provider = "codex_proxy"` *after* a `[profiles.x]` block silently makes it part of that profile, not a global default. You'll then wonder why `-p via_proxy` is still required.

```toml
# Global defaults — MUST be above any [section] header.
model = "model-a0e7"
model_provider = "codex_proxy"

# (your other top-level settings: model_reasoning_effort, personality, etc.)

[model_providers.codex_proxy]
name = "codex-proxy"
base_url = "http://127.0.0.1:8765/v1"
env_key = "CODEX_PROXY_TOKEN"
wire_api = "responses"

# (your [projects.*], [features], etc. continue below)
```

Use it — no flags needed:

```
codex                  # routes through the proxy
codex exec "..."       # routes through the proxy
codex resume           # picker shows sessions tagged "codex_proxy"
```

**One-time migration of pre-existing sessions** so they appear under the new default. The `resume` picker filters by the saved `model_provider` field in each rollout. Migrate (with hardlink backup so it's safe and cheap):

```
# 1. Migrate the JSONL rollouts (the source of truth for replay).
cp -al ~/.codex/sessions ~/.codex/sessions.before-migration-$(date +%Y%m%d-%H%M%S)
find ~/.codex/sessions -name '*.jsonl' \
  -exec sed -i -E 's/"model_provider":\s*"openai"/"model_provider":"codex_proxy"/g' {} +

# 2. ALSO migrate the index DB the picker reads from. Without this, the
#    picker's WHERE model_provider = 'codex_proxy' filter returns 0 rows
#    even though the JSONLs are correctly tagged. Discovered the hard way.
cp ~/.codex/state_5.sqlite ~/.codex/state_5.sqlite.before-migration-$(date +%Y%m%d-%H%M%S)
sqlite3 ~/.codex/state_5.sqlite \
  "UPDATE threads SET model_provider = 'codex_proxy' WHERE model_provider = 'openai'"
```

The hardlink backup of `sessions/` costs near-zero disk space — `sed -i` rename-on-top breaks the hardlink for each modified file, leaving the backup pointing at the original inode. The index backup is a regular `cp` because SQLite is a single file.

After the migration, verify the picker actually works: `codex resume --all` should return your full history. If it returns "No sessions yet" with `--all`, the index didn't update — check `sqlite3 ~/.codex/state_5.sqlite "SELECT model_provider, COUNT(*) FROM threads GROUP BY model_provider"` and re-run step 2.

#### Option B — opt-in profile, default unchanged

Use this when you want plain `codex` to keep going to OpenAI directly and only `codex -p via_proxy` to route through the proxy.

```toml
# Global defaults stay whatever they were (OpenAI direct, etc.)

[model_providers.codex_proxy]
name = "codex-proxy"
base_url = "http://127.0.0.1:8765/v1"
env_key = "CODEX_PROXY_TOKEN"
wire_api = "responses"

[profiles.via_proxy]
model = "model-a0e7"
model_provider = "codex_proxy"
```

```
codex -p via_proxy
codex exec -p via_proxy "..."
codex resume -p via_proxy   # picker only shows sessions tagged "codex_proxy"
```

The `-p via_proxy` flag must appear on every invocation you want routed through the proxy.

### Hermes Agent

[Hermes Agent](https://github.com/nousresearch/hermes-agent) (Nous Research) honours standard OpenAI env vars and also exposes a `hermes config set` CLI that writes them to `~/.hermes/.env`.

Per-invocation (history-preserving):

```
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=$CODEX_PROXY_TOKEN \
hermes
```

Persistent (replaces whatever provider you had):

```
hermes config set OPENAI_BASE_URL http://127.0.0.1:8765/v1
hermes config set OPENAI_API_KEY  $CODEX_PROXY_TOKEN
```

### Aider

[Aider](https://aider.chat/) uses the older `OPENAI_API_BASE` name and supports both env vars and CLI flags.

```
OPENAI_API_BASE=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=$CODEX_PROXY_TOKEN \
aider --model model-a0e7
```

Or with explicit flags (also non-persistent):

```
aider --openai-api-base http://127.0.0.1:8765/v1 \
      --openai-api-key  $CODEX_PROXY_TOKEN \
      --model model-a0e7
```

Per-project chat history at `.aider.chat.history.md` is unaffected.

### Cursor

Cursor's "Override OpenAI Base URL" setting is **global and persistent** — there is no per-invocation override.

- **Settings → Models → Override OpenAI Base URL:** `http://127.0.0.1:8765/v1`
- **Settings → Models → API Key:** your codex-proxy key

### Continue (VS Code)

[Continue](https://www.continue.dev/) stores its config in `~/.continue/config.json`. **Add** an entry to its `models` array — your existing entries stay listed and selectable from the model picker.

```json
{
  "title": "via codex-proxy (model-a0e7)",
  "provider": "openai",
  "model": "model-a0e7",
  "apiBase": "http://127.0.0.1:8765/v1",
  "apiKey": "<your-codex-proxy-api-key>"
}
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="<your-codex-proxy-api-key>",
)

resp = client.responses.create(
    model="model-a0e7",
    input=[
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
    ],
)
print(resp.output[0].content[0].text)
```

Each `OpenAI(...)` instance is fully scoped — no global SDK state to mutate.

### OpenAI JS / TypeScript SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8765/v1",
  apiKey: "<your-codex-proxy-api-key>",
});

const resp = await client.responses.create({
  model: "model-a0e7",
  input: [
    { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
  ],
});
console.log(resp.output[0].content[0].text);
```

### curl

```
curl http://127.0.0.1:8765/v1/responses \
  -H "Authorization: Bearer $CODEX_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-a0e7",
    "input": [
      {"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}
    ]
  }'
```

For the chat-completions shape:

```
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer $CODEX_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"model-a0e7","messages":[{"role":"user","content":"hi"}]}'
```

For streaming, add `"stream": true` to either body.

### Generic: any OpenAI-compatible client

Look for one of these knobs in its config:

- "OpenAI base URL", "endpoint", "API base", "custom provider URL" — set to `http://127.0.0.1:8765/v1`.
- "API key" or "bearer token" — set to your codex-proxy API key.
- For OpenAI SDK-based clients, `OPENAI_BASE_URL` and `OPENAI_API_KEY` env vars usually override at startup.

### Claude Code: requires Phase 3

[Claude Code](https://github.com/anthropics/claude-code) and other Anthropic-shape clients speak `/v1/messages` with `claude-*` model names. The proxy currently only serves the OpenAI shapes (`/v1/responses`, `/v1/chat/completions`), so pointing Claude Code at the proxy returns `404`.

Routing Claude Code through this proxy means **Phase 3**, not yet built: a `/v1/messages` route plus an `anthropic_auth_vault` backend type that pools Claude Pro/Max session credentials with the same OAuth refresh discipline. Until then, run Claude Code against Anthropic directly.

## Configuration reference

`codex-proxy` reads a single TOML file. Default path: `~/.config/codex-proxy/config.toml`. Override with `--config /path/to/config.toml`.

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
- `client_auth_token_env` — if set, `/v1/*` requires `Authorization: Bearer <value of this env var>`. Ignored when `[auth]` is also set (the multi-tenant flow is the more flexible replacement).
- `control_auth_token_env` — if set, `/status` and `/control/*` require the same.

### `[state]`

```toml
[state]
dir = "/home/you/.local/state/codex-proxy"
```

- `dir` — optional. Per-backend usage and cooldown snapshots are persisted here so they survive restart. If omitted, all state is memory-only.

### `[usage_log]`

```toml
[usage_log]
path = "/home/you/.local/state/codex-proxy/requests.sqlite"
capture_bodies = true
```

- `path` — optional. When set, every backend call is recorded as a row. When unset, no logging happens.
- `capture_bodies` — default `true`. When on, the request body, the upstream response body (or full SSE blob for streams), and the upstream response headers are stored as zlib-compressed blobs in a companion `request_bodies` table. Turn off once a usage-prediction model is trained.

See [Per-request usage log](#per-request-usage-log) for the full schema.

### `[auth]`

```toml
[auth]
db = "/home/you/.local/state/codex-proxy/auth.sqlite"
session_ttl_seconds = 1800   # default
```

- `db` — required to enable auth. Path to the SQLite database holding users, sessions, and API keys. Without `[auth]`, `/v1/*` is open and the `/auth/*` and `/ui/` routes do not exist.
- `session_ttl_seconds` — how long a session token returned by `/auth/login` stays valid. Default 30 minutes.

Passwords are stored as **argon2id** hashes. Session tokens and API keys are 32 bytes urandom each, returned to the user once and stored only as `sha256(plaintext)`.

### `[[backends]]`

One table per Codex account. Each backend needs a unique `id` and its own `auth.json` vault.

```toml
[[backends]]
id = "account-a"
vault_path = "/home/you/.codex-proxy/vaults/account-a/auth.json"
models = ["model-a0e7"]
codex_base_url = "https://chatgpt.com/backend-api/codex"   # optional
```

- `id` — required. Used in `/status`, `/control/pin`, and session bindings.
- `vault_path` — required. Absolute path to the account's `auth.json` (the format the Codex CLI login flow writes). Must contain a `tokens` object with `access_token` and `refresh_token`; `account_id` and `id_token` are used when present. The file is rewritten in place after each successful OAuth refresh.
- `models` — required. Advertised model list for this account. The proxy only considers this backend for requests whose `model` is in the list.
- `type` — optional. Defaults to `"codex_auth_vault"` (the only supported type).
- `codex_base_url` — optional. Defaults to the ChatGPT backend base.

To rotate across multiple accounts, declare multiple `[[backends]]` entries with different `vault_path`s and the same advertised model list.

## Endpoints

| Route | Auth required | Notes |
| --- | --- | --- |
| `GET /health` | no | liveness probe |
| `GET /status` | no | backend pool view, pin, session bindings |
| `POST /v1/responses` | bearer when `[auth]` set | native OpenAI Responses API; SSE forwarded byte-for-byte |
| `POST /v1/chat/completions` | bearer when `[auth]` set | chat-completions shape; translated to/from Responses internally |
| `GET /diagnose/upstream` | bearer when `[auth]` set | daily contract check; see below |
| `POST /control/pin` | no | force routing to one backend |
| `POST /control/unpin` | no | clear the pin |
| `POST /auth/register` | no, only exists when `[auth]` set | `{username, password}` → `201 {user_id, username}` |
| `POST /auth/login` | no, only exists when `[auth]` set | `{username, password}` → `200 {session_token, expires_at}` |
| `POST /auth/logout` | session bearer | `204` |
| `POST /auth/keys` | session bearer | `{label?}` → `201 {id, api_key, prefix, label, created_at}`. **`api_key` is plaintext and shown ONCE.** |
| `GET /auth/keys` | session bearer | list this user's keys |
| `DELETE /auth/keys/{id}` | session bearer | revoke |
| `GET /ui/` | no, only exists when `[auth]` set | browser UI for register/login/keys |

## How rotation works

Per request, the selector:

1. Drops any backend that does not advertise the requested model.
2. Drops any backend whose `available` is false or whose `cooldown_until_ts` is in the future.
3. Prefers `weekly_exhausted=false` over `true`.
4. Among ties, prefers higher `remaining_fraction` (unknown treated as `0.5`).
5. Among ties, prefers more recent `probed_at_ts`.
6. Final tiebreak: backend `id`, lexicographic.

If the chosen backend returns a retryable error, it is excluded from this request and the selector re-runs.

| Upstream status | Classification | Action |
| --- | --- | --- |
| `200` | healthy | pass through |
| `400` | client_error | surface to client, no rotation |
| `401` / `403` | auth_invalid | rotate; if none left, `502` |
| `404` | unknown_model | rotate; if none left, `400` |
| `429` | rate_limited | cooldown (Retry-After or 60s default); rotate; if none left, `429` |
| `5xx` / network | transient | rotate; if none left, `502` |

Cooldowns are recorded per backend and visible in `/status`. If `[state] dir` is configured they survive restart.

## Sticky session header (`X-Codex-Session-Id`)

By default every request is an independent selection. If a client wants to stay on the same account across a sequence of requests, it sends:

```
X-Codex-Session-Id: <any string>
```

The first request with that id binds it to whichever backend the selector picks. Subsequent requests with the same id route to that backend while it remains viable. If the bound backend becomes unavailable, the proxy falls back to normal ranking and updates the binding to whichever backend actually served the request.

Bindings are process-local and in-memory; they do not survive a restart.

## Pinning (`/control/pin`, `/control/unpin`)

```
curl -X POST http://127.0.0.1:8765/control/pin -d '{"backend_id":"account-a"}' -H 'Content-Type: application/json'
curl -X POST http://127.0.0.1:8765/control/unpin
```

While pinned the selector considers only that backend (no fallback) — upstream errors surface directly. Pin overrides the session header.

## Per-request usage log

When `[usage_log] path = "..."` is set, every backend call is recorded as a row in a SQLite database. Combined with `capture_bodies = true` (default during the modeling phase), this is the data corpus for figuring out how `(model, reasoning_effort, token counts)` translate into the opaque "usage percent" Codex Plus accounts decrement against.

`requests` (one row per backend attempt — successes AND rotated-from failures):

| Column | Notes |
| --- | --- |
| `id`, `ts_start`, `ts_end`, `latency_ms` | timing |
| `route` | `responses` or `chat_completions` |
| `stream` | `1` if the client asked for streaming |
| `session_id` | the `X-Codex-Session-Id` if the client sent one |
| `user_id`, `api_key_id` | populated when `[auth]` is enabled |
| `backend_id` | which vault served (or attempted) this call |
| `model`, `reasoning_effort` | extracted from the request body |
| `status`, `classification` | HTTP status returned + error class |
| `request_bytes`, `response_bytes` | sizes |
| `prompt_tokens`, `completion_tokens`, `total_tokens`, `cached_tokens`, `reasoning_tokens` | parsed from upstream `response.usage` (terminal `response.completed` event for streams) |
| `plan_type`, `active_limit` | from `x-codex-plan-type`, `x-codex-active-limit` |
| `primary_used_percent_before` / `_after` | the 5-hour-window quota state immediately before and after the call. `before` is null on the first call to a given backend. |
| `secondary_used_percent_before` / `_after` | same for the weekly window |
| `primary_reset_at`, `secondary_reset_at` | unix ts when each window resets |
| `primary_over_secondary_limit_percent` | upstream-reported overage signal |
| `credits_balance`, `credits_has_credits`, `credits_unlimited` | credits state |
| `quota_reset_crossover` | `1` if `_after < _before` (a window reset fired during the call); exclude from training |

`request_bodies` (one row per `requests.id`, only when `capture_bodies = true`):

| Column | Notes |
| --- | --- |
| `request_id` | foreign key to `requests.id` |
| `req_payload` | zlib-compressed client request JSON |
| `resp_payload` | zlib-compressed response (full JSON for non-stream; full SSE blob for streams) |
| `upstream_headers` | zlib-compressed JSON of all upstream response headers |

Example aggregation — average quota Δ per request, by model and reasoning effort:

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

## Daily upstream healthcheck (`GET /diagnose/upstream`)

The proxy doesn't have automated tests against the real Codex backend — a regression in the upstream contract would only show up when a real request fails. `GET /diagnose/upstream` is a one-shot probe per backend that catches those regressions before they bite a user request.

Each invocation makes one tiny streaming request per non-cooldown backend (prompt: "say only: ok", a few hundred tokens of overhead total) and evaluates:

| Check | Catches |
| --- | --- |
| `http_2xx` | URL or auth flow change (401, 404, 5xx from upstream) |
| `quota_headers_present` | `x-codex-*` headers removed or renamed |
| `primary_used_percent_present` | the 5-hour-window quota field disappeared |
| `secondary_used_percent_present` | the weekly quota field disappeared |
| `response_completed_event_present` | SSE terminal event renamed or restructured |
| `usage_block_present` | `response.completed.response.usage` shape changed |

Plus, an upstream `400` (e.g. "model not supported") surfaces as `stage = "upstream"` with the status code, which catches Codex retiring or renaming a model.

Cooled-down backends are reported with `skipped: true` and **don't** flip the aggregate `ok` to false — the daily check shouldn't kick a backend that's already recovering.

When `[auth]` is enabled, the route requires a valid API key. Mint a dedicated key labeled `diag-cron` so revoking it doesn't disturb other clients.

Cron example:

```
# Run at 09:00 every day; alert via mail on any failure.
0 9 * * *   curl -s -H "Authorization: Bearer $CODEX_PROXY_DIAG_KEY" \
                 http://127.0.0.1:8765/diagnose/upstream \
            | jq -e '.ok' > /dev/null \
            || echo "codex-proxy upstream check failed at $(date)" \
                | mail -s "codex-proxy: upstream regression" you@example.com
```

For systemd users, prefer a `systemd.timer` over cron — easier to inspect via `systemctl list-timers` and to log with `journalctl`.

## Operational notes

- **Auth.json refresh-chain caveat.** The proxy reads `auth.json` directly and owns the OAuth refresh chain for that account. Every successful refresh produces a new refresh token and writes it back to the file. If anything else (e.g. your normal `codex` usage on the same account) refreshes against the same auth file in parallel, whichever side rotates first invalidates the other. **Either dedicate an account to the proxy, or route your own Codex usage through the proxy too** (using the [Codex CLI Option A](#option-a--codex_proxy-as-the-global-default-recommended-for-single-point-of-auth-setups) global-default setup).
- **Body capture is sensitive data.** With `capture_bodies = true`, prompts and responses are stored on disk in cleartext (after zlib decompression). Treat the database file as sensitive; `chmod 600` is a sensible baseline. Flip `capture_bodies = false` once you've collected enough corpus to model consumption; turn it back on whenever Codex updates its models.
- **Quota-percent granularity.** `primary_used_percent` and `secondary_used_percent` are integer percentages reported by the upstream. Single small calls often show `Δ = 0`. Aggregate across many calls for a useful signal.
- **No log rotation.** The usage-log SQLite file grows append-only. Archive manually when it gets large.
- **Localhost only.** The proxy binds `127.0.0.1`. If you need to expose it to other machines, front it with TLS (Caddy / nginx) and rely on `[auth]` for access control.
- **One worker.** uvicorn defaults to one worker; the SQLite databases are not safe across multiple worker processes. Don't increase `--workers`.

## Verify

```
make verify
```

Runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`.
