# API Proxy

A small local HTTP proxy that routes requests across multiple upstream accounts and manages request distribution. One endpoint on `127.0.0.1`, many upstream accounts behind it. When one account is rate-limited or otherwise unavailable, the proxy sends the next request to another. When *all* of them are exhausted, the proxy can optionally fall back to free-tier models so you keep working at a slower pace until limits reset.

It is deliberately a less-than-intelligent router. It does not summarise, retry mid-stream, synthesise continuity, or do anything fancier than "pick an account that can serve this request, and if it fails, try the next one." The OpenAI-compatible routes (`/v1/responses` and `/v1/chat/completions`) exist so any OpenAI-compatible client can point at this proxy without knowing anything changed.

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
- [Switching models and reasoning levels](#switching-models-and-reasoning-levels-model-reasoning)
- [Sticky session header](#sticky-session-header-x-codex-session-id)
- [Pinning](#pinning-controlpin--controlunpin)
- [Per-request usage log](#per-request-usage-log)
- [Daily upstream healthcheck](#daily-upstream-healthcheck-getdiagnoseupstream)
- [Operational notes](#operational-notes)
- [Verify](#verify)

## How it works (architecture)

The proxy sits between clients and upstream services, managing request distribution and credential handling.

```
[ Client 1 ]   HTTP /v1/responses ─────┐
[ Client 2 ]   HTTP /v1/chat/completions ├──► [ API Proxy ]  ─► [ Credential Custody ]  ─► [ Upstream Service ]
[ Client 3 ]   HTTP /v1/chat/completions │    (FastAPI)         (manages tokens)           (OpenAI-compatible)
                                        └─► (with bearer token)

                            Authorization: Bearer <api-key> (from proxy's /auth/keys)
```

Two auth flows are involved:

| Layer | What it secures | Credential | Where it lives |
| --- | --- | --- | --- |
| **Client → Proxy** | Client authenticating to the proxy | API key minted via `/auth/keys` | `PROXY_TOKEN` env var |
| **Proxy → Upstream** | The proxy authenticating to upstream services | Managed by credential custody service | Credential service (not proxy) |

Clients get their own short-lived API keys via the proxy's `/auth/keys` endpoint. The proxy never handles upstream credentials directly — a separate credential custody service manages them, issues stand-in tokens, and injects them only on final-hop requests.

The proxy is a FastAPI process that routes requests based on availability and load. It forwards OpenAI-compatible `/v1/responses` and `/v1/chat/completions` requests through the credential custody service, which handles token provisioning and upstream credential injection.

## Quick start

Install with [uv](https://docs.astral.sh/uv/):

```
uv sync
```

The proxy requires a credential custody service running on the local network. Configure it to point to your credential service:

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
id = "primary"
type = "credential_proxy"
proxy_url = "http://127.0.0.1:7342"
models = ["model-a0e7"]

[[backends]]
id = "secondary"
type = "credential_proxy"
proxy_url = "http://127.0.0.1:7342"
models = ["model-a0e7"]
```

Replace `proxy_url` with your credential service's URL. Ensure the credential service is running before starting the proxy. The service itself determines which accounts/credentials are used.

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

[Hermes Agent](https://github.com/nousresearch/hermes-agent) (Nous Research) needs three config keys to route through the proxy. Verified live against v0.10.0:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8765/v1
hermes config set model.api_mode codex_responses
hermes config set OPENAI_API_KEY "$CODEX_PROXY_TOKEN"
```

Three things worth knowing about the hermes setup that surprised me during integration:

- **The provider name is `custom`**, not `openai-compatible`. Hermes' `--provider` flag list omits it (and instead lists `auto`, `openai-codex`, `nous`, ...), but `model.provider = custom` is the value the OpenAI-compatible code path checks for.
- **`model.api_mode = codex_responses` is required**, not optional. Without it hermes sends `/v1/chat/completions` requests with hermes-shape bodies (system prompt as `developer` role, ~28 tool definitions). The proxy's chat-completions → Responses-API translator drops the `developer` role and the tools, and the upstream rejects the resulting minimal body with `400`. Setting `api_mode = codex_responses` makes hermes send native Responses-API requests at `/v1/responses`, which the proxy forwards verbatim — no translation gap. The 400 disappears.
- **Don't try to keep `model.provider = openai-codex` and just override `base_url`.** That provider path uses Codex's own OAuth token (read from `~/.codex/auth.json`) and a hardcoded base URL — it ignores your proxy entirely. The `custom` provider is the right one.

Verify the setup:

```bash
hermes chat -q "say only the words: hello via hermes"
sqlite3 ~/.local/state/codex-proxy/requests.sqlite \
  "SELECT id, route, user_id, api_key_id, status FROM requests ORDER BY id DESC LIMIT 1"
# Expect: route = "responses", status = 200, your user_id + api_key_id populated.

hermes chat -c -q "and again: hello again via hermes"
# Should resume the previous session and add another logged row.
```

Per-invocation alternative (no persistent config change):

```bash
OPENAI_API_KEY="$CODEX_PROXY_TOKEN" hermes chat -q "..." \
  -- /* you'd still need model.provider=custom, model.base_url=..., model.api_mode=codex_responses
        applied somehow; hermes doesn't accept those as flags. The persistent form above is the
        recommended setup for the "single point of auth" use case. */
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
startup_smoke_test = true                              # optional, default true
smoke_test_interval_seconds = 3600                     # optional, default 3600 (1h)
```

- `host` — bind address. Default `127.0.0.1`. Do not expose on `0.0.0.0` unless you understand the threat model.
- `port` — TCP port. Default `8765`.
- `client_auth_token_env` — if set, `/v1/*` requires `Authorization: Bearer <value of this env var>`. Ignored when `[auth]` is also set (the multi-tenant flow is the more flexible replacement).
- `control_auth_token_env` — if set, `/status` and `/control/*` require the same.
- `startup_smoke_test` — when true (default), the proxy probes each non-cooldown backend with one minimal upstream call at launch and logs OK/SKIPPED/FAILED so the operator sees auth and quota state immediately. Each probe burns a few hundred tokens of quota.
- `smoke_test_interval_seconds` — when > 0 (default 3600), the smoke test re-runs on this interval so the operator sees live state changes (weekly resets, auth refreshes, model catalog churn) without restarting the proxy. Set to 0 to disable the periodic re-run; the startup pass still happens.

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

One table per upstream account. Each backend needs a unique `id` and credentials at a credential service.

```toml
[[backends]]
id = "primary"
type = "credential_proxy"
proxy_url = "http://127.0.0.1:7342"
models = ["model-a0e7", "model-a0c3"]
```

- `id` — required. Used in `/status`, `/control/pin`, and session bindings. Must be unique across all backends.
- `type` — required. Must be `"credential_proxy"` to forward requests to a credential service.
- `proxy_url` — required. Base URL of the credential service (e.g., `http://127.0.0.1:7342` for a local service).
- `models` — required as a **cold-start fallback**. The proxy fetches the live model list from the service at startup (and refreshes hourly), and uses *that* dynamic list in preference to whatever's in the TOML. The TOML value is what's served until the first successful upstream refresh — so it must be non-empty, but it doesn't need to be exhaustive or up to date. Model listings change frequently, so this design means operators don't have to manually chase updates.

To distribute load across multiple backends, declare multiple `[[backends]]` entries pointing to the same or different credential services. The `models` values can be the same across them — each backend pulls its actual catalog at startup based on what the service provides.

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

## Switching models and reasoning levels (`/model`, `/reasoning`)

Codex CLI's `/model` and `/reasoning` slash commands work through the proxy with no code change — just a config tweak and an understanding of where the model list comes from.

**Model list source:** Codex CLI maintains `~/.codex/models_cache.json`, fetched from the upstream Codex backend. It contains every model your account has access to plus that model's supported reasoning levels. The `/model` picker shows that list. To see what your account has, run:

```
jq -r '.models[] | [.slug, .display_name, .default_reasoning_level, (.supported_reasoning_levels | map(.effort) | join(","))] | @tsv' ~/.codex/models_cache.json
```

For a Plus account the list typically looks like:

| slug | default reasoning | supported reasoning |
| --- | --- | --- |
| `model-a0e7` | xhigh | low, medium, high, xhigh |
| `model-a0c3` | medium | low, medium, high, xhigh |
| `model-a0b8` | medium | low, medium, high, xhigh |
| `model-a0e6` | medium | low, medium, high, xhigh |
| `codex-auto-review` | medium | low, medium, high, xhigh |

**To make `/model` actually route**, advertise every model you want to use in each backend's `models = [...]` list:

```toml
[[backends]]
id = "primary"
vault_path = "/home/you/.codex-proxy/vaults/primary/auth.json"
models = ["model-a0e7", "model-a0c3", "model-a0b8", "model-a0e6", "codex-auto-review"]
```

The selector only routes a request when at least one backend advertises the requested model. If the list omits the model the user just picked, they'll see a `503`.

**`/reasoning` works for free.** Codex CLI sends the chosen level as `reasoning.effort` in the request body. The proxy passes it through unchanged on `/v1/responses`. Picked levels show up in the usage log's `reasoning_effort` column for later analysis.

**Auto-routing virtual models (`auto-learning`, `auto-learning-synthetic`, `auto`).** Three virtual model names short-circuit the picker. Any client that can name a model — Codex CLI's `/model`, Hermes' `model.default`, Aider's `--model`, a curl with `"model": "..."` — can opt in.

- **`auto-learning`** — organic explorer. Each request is rewritten to the (model, reasoning_effort) cell with the fewest successful samples in the usage log so the corpus fills evenly across the 16-cell grid (4 models × 4 reasoning levels — see `src/codex_proxy/cell_grid.py`). Round-robin v1; ties break in cell-grid order. Stateless: every call rereads coverage, so concurrent calls converge.
- **`auto-learning-synthetic`** — synthetic background tier. Same round-robin algorithm, but uses an **independent coverage query** (only counts rows where `routing_mode='auto-learning-synthetic'`) so synthetics don't double-count organic samples and vice versa. Driven by a built-in worker (see `[auto_router]` config below) that fires bland prompts when the daily floor or pct-of-organic target hasn't been met. You can also send this name yourself for testing.
- **`auto`** — cost-optimal exploiter. Reserved for the router that picks the cell with the lowest expected Δquota per request. Currently returns `503 NotTrained` with an explanatory message until the cost model is fit on the explorer's corpus.

The rewrite happens before the selector, so backends do **not** need to advertise these virtual names — they only need to advertise the real grid models. Each request's `requested_model`, `requested_reasoning_effort`, and `routing_mode` are recorded in the usage log alongside the served `model` / `reasoning_effort`, so post-hoc analysis can separate router-driven samples from user-driven ones.

Inspect cell coverage (per tier):

```bash
sqlite3 ~/.local/state/codex-proxy/requests.sqlite \
  "SELECT routing_mode, model, reasoning_effort, COUNT(*) FROM requests
   WHERE routing_mode IN ('auto-learning', 'auto-learning-synthetic') AND status = 200
   GROUP BY routing_mode, model, reasoning_effort ORDER BY 1, 2, 3"
```

Enable organic auto-learning on each client:

```bash
# Codex CLI: edit ~/.codex/config.toml top-level
model = "auto-learning"

# Hermes
hermes config set model.default auto-learning
```

The synthetic worker has two controllers:

**Primary — weekly-exhaustion controller.** Per backend, every tick: read the latest `CodexQuotaSnapshot`, project human burn (last 7d organic rate × safety margin), and fire enough synthetics to land weekly at `weekly_target_pct` by reset. Honors the invariant that paid-monthly weekly capacity is never wasted. Pauses on an account when 5h is near-exhausted (would just 429-loop) or weekly is already past target.

**Fallback — cold-start.** Per backend with no quota snapshot yet (fresh deploy), uses the simple floor + pct-of-organic + hard-ceiling target. Same hard ceiling caps the weekly controller as a safety net.

Tune in the proxy's `config.toml`:

```toml
[auto_router]
# Cold-start fallback bounds (used when no per-account quota snapshot exists yet).
synthetic_floor_per_day = 50
synthetic_pct_of_organic = 0.05
synthetic_hard_ceiling_per_day = 200
synthetic_check_interval_seconds = 300

# Weekly-exhaustion controller knobs.
# Estimated cost of one synthetic, in weekly% points. Hand-set v1; learned
# by the cost model in v2.
pct_per_synthetic_estimate = 0.1
# How far back to look when projecting human burn (hours). 168 = 7 days.
prediction_window_hours = 168
# Multiplier on projected human burn — errs toward leaving the human room.
prediction_safety_margin = 1.20
# Per-tick cap on synthetics fired (spreads connection load).
max_synthetics_per_tick = 10
# Stop firing on an account once weekly_used_percent crosses this.
weekly_target_pct = 95.0
# Pause firing on an account when 5h-used crosses this.
five_hourly_pause_pct = 95.0
```

All `synthetic_*` defaults are 0, which keeps the cold-start fallback off. The weekly controller activates automatically once a backend has served at least one request and produced a quota snapshot — so on a fresh deploy with all-zero config, no synthetics fire until organic traffic establishes a quota baseline, then the weekly controller takes over.

## Auto-fallback to OpenRouter free tier

When all your Codex backends are weekly-exhausted, you don't have to stop working — set an `OPENROUTER_API_KEY` env var and the proxy will auto-register a free-tier OpenRouter backend that handles overflow at a "snail's pace." No TOML edits, no manual model selection.

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
uv run python -m codex_proxy
```

What you get:

- The proxy fetches OpenRouter's `/api/v1/models` catalog at startup (and refreshes hourly), filters to entries where both prompt and completion pricing are 0, and exposes them as a routable backend.
- The OpenRouter backend reports a deliberately tiny `remaining_fraction` (0.001) so the selector keeps preferring your Codex backends whenever they're viable. It only picks OpenRouter when every Codex backend is in cooldown after a 429.
- It also "shadow-advertises" every Codex model name from your `[[backends]]` config, so a request asking for `model-a0e7` still routes when Codex is exhausted — the OpenRouter backend internally substitutes the best free model that matches the request's needs (context length, tool support, code-friendliness).
- For `/v1/responses` requests, the OpenRouter backend translates to `/v1/chat/completions` on the way out and back, since OpenRouter doesn't natively support the Responses API.

**Per-request model selection** scores candidates dynamically from each catalog entry's metadata — no hardcoded family list. New free-tier models get scored on their own merits the moment they appear in OpenRouter's catalog.

Required filters:

1. Tool-call support — required if the request body has `tools` set.
2. Context length ≥ approximate input token budget.

Score components (all derived from per-entry features, applied to whatever's in the catalog right now):

- **Context length** (log-scale, capped at +40): bigger context = more room for codebase reading.
- **Parameter count** parsed from the model id, e.g. `model-a0c2` → 70B (log-scale, capped at +40): bigger model with diminishing returns.
- **Instruct-tuned bonus** (+20): presence of `architecture.instruct_type` indicates chat/instruction-tuning, not a base model.
- **Tool support** (+20): `tools` or `tool_choice` in `supported_parameters`.
- **Recency** (up to +30): newer `created` timestamp wins, decaying linearly to 0 over ~360 days.
- **Code-keyword bonus** (+30): generic substring match for `coder`, `code`, `starcoder`, `codellama` in id or display name — applies to any provider, not a specific list.

**Provider exclusion**: Chinese-origin cloud providers are filtered out of the catalog at parse time — Qwen, DeepSeek, Yi (01.AI), ChatGLM/GLM (Zhipu), InternLM, Doubao (ByteDance), Hunyuan (Tencent), MiniMax, Stepfun, Moonshot, Baichuan. Operator preference for cloud-routed models; if a separate local-models backend ships later, it can apply different rules. See `_BLOCKED_PROVIDER_PREFIXES` in `src/codex_proxy/backends/openrouter_free.py`.

The chosen model id appears in the response and in the usage log's `model` column, so you can see exactly what was used for each call. If you don't like the choice, set a more specific model in your client (the OpenRouter id, e.g. `model-a0g3/model-a0f3-coder-32b-instruct:free`); the proxy will pass it through unchanged.

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
| `five_hourly_used_percent_before` / `_after` | the 5-hour-window quota state immediately before and after the call. `before` is null on the first call to a given backend. (Upstream calls this "primary"; we name it for what it is.) |
| `weekly_used_percent_before` / `_after` | same for the weekly window. (Upstream calls this "secondary".) |
| `five_hourly_reset_at`, `weekly_reset_at` | unix ts when each window resets |
| `five_hourly_over_weekly_limit_percent` | upstream-reported overage signal |
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
  AVG(five_hourly_used_percent_after - five_hourly_used_percent_before) AS avg_d_5h_pct,
  AVG(weekly_used_percent_after - weekly_used_percent_before) AS avg_d_weekly_pct
FROM requests
WHERE status = 200 AND quota_reset_crossover = 0
  AND five_hourly_used_percent_before IS NOT NULL
GROUP BY model, reasoning_effort;
```

## Daily upstream healthcheck (`GET /diagnose/upstream`)

The proxy doesn't have automated tests against the real Codex backend — a regression in the upstream contract would only show up when a real request fails. `GET /diagnose/upstream` is a one-shot probe per backend that catches those regressions before they bite a user request.

Each invocation makes one tiny streaming request per non-cooldown backend (prompt: "say only: ok", a few hundred tokens of overhead total) and evaluates:

| Check | Catches |
| --- | --- |
| `http_2xx` | URL or auth flow change (401, 404, 5xx from upstream) |
| `quota_headers_present` | `x-codex-*` headers removed or renamed |
| `five_hourly_used_percent_present` | the 5-hour-window quota field disappeared |
| `weekly_used_percent_present` | the weekly quota field disappeared |
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

- **Background process interruption.** When running the service in the background with `&` (e.g., `uv run codex-proxy ... &`), the process is no longer in the terminal's foreground process group, so Ctrl+C won't reach it directly. Use `kill <pid>` or `killall codex-proxy` to stop background instances, or use a process manager (tmux, screen, systemd) for reliable lifecycle management. The service includes explicit signal handlers (SIGINT/SIGTERM) to ensure clean shutdown.
- **SystemManager dependency management.** On systems running system manager, codex-proxy may be configured as a managed dependency. Use `system manager dependencies list` to check status and `system manager dependencies stop codex-proxy` / `start codex-proxy` to control it. Do not run multiple instances on the same port (8765 by default)—only the first will bind successfully; subsequent instances fail with "address already in use" and requests will hit the original instance instead.
- **Auth.json refresh-chain caveat.** The proxy reads `auth.json` directly and owns the OAuth refresh chain for that account. Every successful refresh produces a new refresh token and writes it back to the file. If anything else (e.g. your normal `codex` usage on the same account) refreshes against the same auth file in parallel, whichever side rotates first invalidates the other. **Either dedicate an account to the proxy, or route your own Codex usage through the proxy too** (using the [Codex CLI Option A](#option-a--codex_proxy-as-the-global-default-recommended-for-single-point-of-auth-setups) global-default setup).
- **Body capture is sensitive data.** With `capture_bodies = true`, prompts and responses are stored on disk in cleartext (after zlib decompression). Treat the database file as sensitive; `chmod 600` is a sensible baseline. Flip `capture_bodies = false` once you've collected enough corpus to model consumption; turn it back on whenever Codex updates its models.
- **Quota-percent granularity.** `five_hourly_used_percent` and `weekly_used_percent` are integer percentages reported by the upstream (over the wire as `x-codex-primary-used-percent` and `x-codex-secondary-used-percent` respectively). Single small calls often show `Δ = 0`. Aggregate across many calls for a useful signal.
- **No log rotation.** The usage-log SQLite file grows append-only. Archive manually when it gets large.
- **Localhost only.** The proxy binds `127.0.0.1`. If you need to expose it to other machines, front it with TLS (Caddy / nginx) and rely on `[auth]` for access control.
- **One worker.** uvicorn defaults to one worker; the SQLite databases are not safe across multiple worker processes. Don't increase `--workers`.

## Verify

```
make verify
```

Runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`.
