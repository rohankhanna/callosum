# Callosum

A local, adaptive HTTP routing layer that decides which underlying model
should handle each prompt and forwards it there, behind one
OpenAI-compatible endpoint on `127.0.0.1`. Multiple upstream backends
(today: Codex Plus/Pro credential-proxy backends, local models via
local LLM gateway / LiteLLM-compatible OpenAI surfaces, and Ollama Cloud
models served by the local ollama daemon) sit behind one URL.
Routing decisions are made per request by an in-process pipeline:
extract prompt features, filter cells that cannot serve the request,
predict each compatible cell's chance of satisfying it, then select the
lowest-cost qualifying cell. Cell discovery, context windows, supported
reasoning levels, and strength rankings come from backend catalogs where
available — nothing is hardcoded against a particular model lineup. When
one cell is rate-limited, the router picks another; when all eligible
backends are exhausted, the proxy returns a self-diagnosing error
(per-backend cooldown, quota, and recovery ETA in the response body).

The OpenAI-compatible routes (`/v1/responses` and `/v1/chat/completions`)
exist so any OpenAI-compatible client can point at this endpoint without
knowing anything else is going on.

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
  - [Backends registered outside `[[backends]]`](#backends-registered-outside-backends)
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

Write `~/.config/callosum/config.toml`:

```toml
[server]
host = "127.0.0.1"
port = 8765

[state]
dir = "/home/you/.local/state/callosum"

[usage_log]
path = "/home/you/.local/state/callosum/requests.sqlite"
capture_bodies = true

[auth]
db = "/home/you/.local/state/callosum/auth.sqlite"

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
uv run python -m callosum
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
echo 'export CALLOSUM_TOKEN=<the-plaintext-from-the-ui>' >> ~/.bashrc
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

Either way you end up with a plaintext key that goes into `CALLOSUM_TOKEN`. The proxy stores only `sha256(key)`, never the plaintext, so this is your only chance to see it.

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
OPENAI_API_KEY=$CALLOSUM_TOKEN \
your-tool ...
```

Persistent edits (rc files, app settings) work too; they just permanently re-target the tool.

> **Codex CLI does NOT honor `OPENAI_BASE_URL`.** It has its own typed provider system. Setting `OPENAI_BASE_URL=http://127.0.0.1:8765/v1 codex ...` will silently route to OpenAI's real API and present your callosum key as if it were an OpenAI key — which OpenAI then rejects. Use the [Codex CLI section below](#codex-cli) instead.

### Codex CLI

The OpenAI Codex CLI (`codex`, `codex exec`, `codex resume`) uses a typed provider system. Two integration shapes — pick based on whether you want the proxy to be the default for all `codex` commands, or only when explicitly opted into.

#### Option A — `callosum` as the global default (recommended for "single point of auth" setups)

Edit `~/.codex/config.toml` so global defaults stay at the top, **before any `[section]` header**.

> **TOML detail that bit me once.** Every `key = value` line after a `[section]` header belongs to that section until the next header. Putting `model_provider = "callosum"` *after* a `[profiles.x]` block silently makes it part of that profile, not a global default. You'll then wonder why `-p via_proxy` is still required.

```toml
# Global defaults — MUST be above any [section] header.
model = "model-a0e7"
model_provider = "callosum"

# (your other top-level settings: model_reasoning_effort, personality, etc.)

[model_providers.callosum]
name = "callosum"
base_url = "http://127.0.0.1:8765/v1"
env_key = "CALLOSUM_TOKEN"
wire_api = "responses"

# (your [projects.*], [features], etc. continue below)
```

Use it — no flags needed:

```
codex                  # routes through the proxy
codex exec "..."       # routes through the proxy
codex resume           # picker shows sessions tagged "callosum"
```

**One-time migration of pre-existing sessions** so they appear under the new default. The `resume` picker filters by the saved `model_provider` field in each rollout. Migrate (with hardlink backup so it's safe and cheap):

```
# 1. Migrate the JSONL rollouts (the source of truth for replay).
cp -al ~/.codex/sessions ~/.codex/sessions.before-migration-$(date +%Y%m%d-%H%M%S)
find ~/.codex/sessions -name '*.jsonl' \
  -exec sed -i -E 's/"model_provider":\s*"openai"/"model_provider":"callosum"/g' {} +

# 2. ALSO migrate the index DB the picker reads from. Without this, the
#    picker's WHERE model_provider = 'callosum' filter returns 0 rows
#    even though the JSONLs are correctly tagged. Discovered the hard way.
cp ~/.codex/state_5.sqlite ~/.codex/state_5.sqlite.before-migration-$(date +%Y%m%d-%H%M%S)
sqlite3 ~/.codex/state_5.sqlite \
  "UPDATE threads SET model_provider = 'callosum' WHERE model_provider = 'openai'"
```

The hardlink backup of `sessions/` costs near-zero disk space — `sed -i` rename-on-top breaks the hardlink for each modified file, leaving the backup pointing at the original inode. The index backup is a regular `cp` because SQLite is a single file.

After the migration, verify the picker actually works: `codex resume --all` should return your full history. If it returns "No sessions yet" with `--all`, the index didn't update — check `sqlite3 ~/.codex/state_5.sqlite "SELECT model_provider, COUNT(*) FROM threads GROUP BY model_provider"` and re-run step 2.

#### Option B — opt-in profile, default unchanged

Use this when you want plain `codex` to keep going to OpenAI directly and only `codex -p via_proxy` to route through the proxy.

```toml
# Global defaults stay whatever they were (OpenAI direct, etc.)

[model_providers.callosum]
name = "callosum"
base_url = "http://127.0.0.1:8765/v1"
env_key = "CALLOSUM_TOKEN"
wire_api = "responses"

[profiles.via_proxy]
model = "model-a0e7"
model_provider = "callosum"
```

```
codex -p via_proxy
codex exec -p via_proxy "..."
codex resume -p via_proxy   # picker only shows sessions tagged "callosum"
```

The `-p via_proxy` flag must appear on every invocation you want routed through the proxy.

### Hermes Agent

[Hermes Agent](https://github.com/nousresearch/hermes-agent) (Nous Research) needs three config keys to route through the proxy. Verified live against v0.10.0:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8765/v1
hermes config set model.api_mode codex_responses
hermes config set OPENAI_API_KEY "$CALLOSUM_TOKEN"
```

Three things worth knowing about the hermes setup that surprised me during integration:

- **The provider name is `custom`**, not `openai-compatible`. Hermes' `--provider` flag list omits it (and instead lists `auto`, `openai-codex`, `nous`, ...), but `model.provider = custom` is the value the OpenAI-compatible code path checks for.
- **`model.api_mode = codex_responses` is required**, not optional. Without it hermes sends `/v1/chat/completions` requests with hermes-shape bodies (system prompt as `developer` role, ~28 tool definitions). The proxy's chat-completions → Responses-API translator drops the `developer` role and the tools, and the upstream rejects the resulting minimal body with `400`. Setting `api_mode = codex_responses` makes hermes send native Responses-API requests at `/v1/responses`, which the proxy forwards verbatim — no translation gap. The 400 disappears.
- **Don't try to keep `model.provider = openai-codex` and just override `base_url`.** That provider path uses Codex's own OAuth token (read from `~/.codex/auth.json`) and a hardcoded base URL — it ignores your proxy entirely. The `custom` provider is the right one.

Verify the setup:

```bash
hermes chat -q "say only the words: hello via hermes"
sqlite3 ~/.local/state/callosum/requests.sqlite \
  "SELECT id, route, user_id, api_key_id, status FROM requests ORDER BY id DESC LIMIT 1"
# Expect: route = "responses", status = 200, your user_id + api_key_id populated.

hermes chat -c -q "and again: hello again via hermes"
# Should resume the previous session and add another logged row.
```

Per-invocation alternative (no persistent config change):

```bash
OPENAI_API_KEY="$CALLOSUM_TOKEN" hermes chat -q "..." \
  -- /* you'd still need model.provider=custom, model.base_url=..., model.api_mode=codex_responses
        applied somehow; hermes doesn't accept those as flags. The persistent form above is the
        recommended setup for the "single point of auth" use case. */
```

### Aider

[Aider](https://aider.chat/) uses the older `OPENAI_API_BASE` name and supports both env vars and CLI flags.

```
OPENAI_API_BASE=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=$CALLOSUM_TOKEN \
aider --model model-a0e7
```

Or with explicit flags (also non-persistent):

```
aider --openai-api-base http://127.0.0.1:8765/v1 \
      --openai-api-key  $CALLOSUM_TOKEN \
      --model model-a0e7
```

Per-project chat history at `.aider.chat.history.md` is unaffected.

### Cursor

Cursor's "Override OpenAI Base URL" setting is **global and persistent** — there is no per-invocation override.

- **Settings → Models → Override OpenAI Base URL:** `http://127.0.0.1:8765/v1`
- **Settings → Models → API Key:** your callosum key

### Continue (VS Code)

[Continue](https://www.continue.dev/) stores its config in `~/.continue/config.json`. **Add** an entry to its `models` array — your existing entries stay listed and selectable from the model picker.

```json
{
  "title": "via callosum (model-a0e7)",
  "provider": "openai",
  "model": "model-a0e7",
  "apiBase": "http://127.0.0.1:8765/v1",
  "apiKey": "<your-callosum-api-key>"
}
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="<your-callosum-api-key>",
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
  apiKey: "<your-callosum-api-key>",
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
  -H "Authorization: Bearer $CALLOSUM_TOKEN" \
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
  -H "Authorization: Bearer $CALLOSUM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"model-a0e7","messages":[{"role":"user","content":"hi"}]}'
```

For streaming, add `"stream": true` to either body.

### Generic: any OpenAI-compatible client

Look for one of these knobs in its config:

- "OpenAI base URL", "endpoint", "API base", "custom provider URL" — set to `http://127.0.0.1:8765/v1`.
- "API key" or "bearer token" — set to your callosum API key.
- For OpenAI SDK-based clients, `OPENAI_BASE_URL` and `OPENAI_API_KEY` env vars usually override at startup.

### Claude Code: requires Phase 3

[Claude Code](https://github.com/anthropics/claude-code) and other Anthropic-shape clients speak `/v1/messages` with `claude-*` model names. The proxy currently only serves the OpenAI shapes (`/v1/responses`, `/v1/chat/completions`), so pointing Claude Code at the proxy returns `404`.

Routing Claude Code through this proxy means **Phase 3**, not yet built: a `/v1/messages` route plus an `anthropic_auth_vault` backend type that pools Claude Pro/Max session credentials with the same OAuth refresh discipline. Until then, run Claude Code against Anthropic directly.

## Configuration reference

`callosum` reads a single TOML file. Default path: `~/.config/callosum/config.toml`. Override with `--config /path/to/config.toml`.

### `[server]`

```toml
[server]
host = "127.0.0.1"
port = 8765
client_auth_token_env  = "CALLOSUM_CLIENT_TOKEN"   # optional
control_auth_token_env = "CALLOSUM_CONTROL_TOKEN"  # optional
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
dir = "/home/you/.local/state/callosum"
```

- `dir` — optional. Per-backend usage and cooldown snapshots are persisted here so they survive restart. If omitted, all state is memory-only.

### `[usage_log]`

```toml
[usage_log]
path = "/home/you/.local/state/callosum/requests.sqlite"
capture_bodies = true
```

- `path` — optional. When set, every backend call is recorded as a row. When unset, no logging happens.
- `capture_bodies` — default `true`. When on, the request body, the upstream response body (or full SSE blob for streams), and the upstream response headers are stored as zlib-compressed blobs in a companion `request_bodies` table. Turn off once a usage-prediction model is trained.

See [Per-request usage log](#per-request-usage-log) for the full schema.

### `[auth]`

```toml
[auth]
db = "/home/you/.local/state/callosum/auth.sqlite"
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

### Backends registered outside `[[backends]]`

Not every backend is a `[[backends]]` TOML entry. Two local-process backends and one remote-band backend are registered programmatically in `__main__.build_runtime_backends` and are NOT configured via `[[backends]]`:

- **Local lane** (`LocalModelRegistryBackend`, preferred; `LiteLLMGatewayBackend`, fallback) — local models served by local LLM gateway / a LiteLLM gateway. The two are mutually exclusive; the preferred one is registered whenever the `local-llm` CLI is on PATH. See `ARCHITECTURE.md`.
- **Ollama Cloud** (`OllamaCloudBackend`, `BackendKind="ollama_cloud"`) — cloud models served by ollama.com through credential proxy's boundary-native proxy custody. Callosum mints a short-TTL `ollama-cloud`-scoped stand-in token at credential proxy's `POST /v1/standin` and POSTs every ollama.com call (chat, catalog, capabilities) through credential proxy's `POST /v1/proxy` (buffered) or `POST /v1/proxy/stream` (streaming) with the stand-in as bearer; credential proxy reads the real ollama.com API key from its `pass` store and injects it on the final hop, so the real key NEVER enters callosum's process — callosum holds only the revocable stand-in. This retires the prior local-ollama-daemon `ollama signin` chat path. It is registered only when `CALLOSUM_OLLAMA_CLOUD_ENABLED=1` (default off), catalogs `/api/tags` (via the credential proxy proxy) with a `:cloud`-suffix name filter, and dispatches to ollama.com's OpenAI-compatible `/v1/chat/completions` (via the credential proxy proxy). It sits in the remote band (`CLOUD_PRIORITY_OFFSET=1_000`). Because it is a distinct `BackendKind` (not a `litellm_gateway` model-name predicate), cloud cells land in remote lanes and stay out of local-only/probing paths by omission. Its `usage_snapshot()` is honest-advisory by default; advisory usage metering can be read from credential proxy (the credential-custody sibling) via its `/v1/ollama/usage` loopback when operator-gated via `CALLOSUM_OLLAMA_CLOUD_USAGE_*` (defaults OFF). Enable it durably via a systemd drop-in — see `docs/operations/runtime_deploy.md`.

## Endpoints

| Route | Auth required | Notes |
| --- | --- | --- |
| `GET /health` | no | liveness probe |
| `GET /status` | no | backend pool view, pin, session bindings |
| `POST /v1/responses` | bearer when `[auth]` set | native OpenAI Responses API; SSE forwarded byte-for-byte |
| `POST /v1/chat/completions` | bearer when `[auth]` set | chat-completions shape; translated to/from Responses internally |
| `POST /v1/eta` | bearer when `[auth]` set | approximate pre-flight p50-p95 latency ranges per cell |
| `GET /v1/usage` | bearer when `[auth]` set | empirical five-hourly and weekly quota token-rate tables per cell |
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
3. Drops any backend whose observed five-hourly or weekly Codex quota meter is exhausted.
4. When a forward cost estimate is available, ranks candidates by the worst normalized pressure across both independent ledgers: estimated five-hourly burn against five-hourly headroom and estimated weekly burn against weekly headroom.
5. When measured data is insufficient, falls back to higher `remaining_fraction` (unknown treated as `0.5`), then more recent `probed_at_ts`.
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
vault_path = "/home/you/.callosum/vaults/primary/auth.json"
models = ["model-a0e7", "model-a0c3", "model-a0b8", "model-a0e6", "codex-auto-review"]
```

The selector only routes a request when at least one backend advertises the requested model. If the list omits the model the user just picked, they'll see a `503`.

**`/reasoning` works for free.** Codex CLI sends the chosen level as `reasoning.effort` in the request body. The proxy passes it through unchanged on `/v1/responses`. Picked levels show up in the usage log's `reasoning_effort` column for later analysis.

**Auto-routing virtual models (`auto`, `auto-learning`).** Two virtual model names invoke the learning router instead of pinning a concrete model. Any client that can name a model — Codex CLI's `/model`, Hermes' `model.default`, Aider's `--model`, a curl with `"model": "..."` — can opt in. `auto` is the primary name; `auto-learning` is a backward-compat alias. Both route through the same recommender pipeline (`src/callosum/routing/`): extract prompt features → capability filter → quality predict → cost-weighted select. There is no longer an "explorer" vs "exploiter" split, and no background synthetic tier (the former `auto-learning-synthetic` worker was removed).

The learning router is **adaptive, not statically configured**:

- Cold-start (no labeled corpus yet) uses a uniform predictor plus catalog-priority / measured-cost ordering — local-first, cost-ordered routing with no ML deps.
- As labeled outcomes (peer-quality labels and failure labels) and measured per-cell quota burn accumulate, the predictor and a dynamic per-model `cost_rank` (fit from measured `weekly_used_percent` deltas in the request log) take over. The learned predictor is `cell_majority_prior` (a per-cell majority-baseline prior that ignores prompt embeddings — the BGE prompt-embedding + KNN predictor was evaluated and removed). Catalog priority is the cold-start prior; measured overrides win; operator overrides win outright.

The rewrite happens before the selector, so backends do **not** need to advertise these virtual names — they only need to advertise the real grid models. Each request's `requested_model`, `requested_reasoning_effort`, and `routing_mode` are recorded in the usage log alongside the served `model` / `reasoning_effort`, so post-hoc analysis can separate router-driven samples from user-driven ones.

Inspect cell coverage:

```bash
sqlite3 ~/.local/state/callosum/requests.sqlite \
  "SELECT routing_mode, model, reasoning_effort, COUNT(*) FROM requests
   WHERE routing_mode IN ('auto', 'auto-learning') AND status = 200
   GROUP BY routing_mode, model, reasoning_effort ORDER BY 1, 2, 3"
```

Enable the learning router on each client:

```bash
# Codex CLI: edit ~/.codex/config.toml top-level
model = "auto"

# Hermes
hermes config set model.default auto
```

Tune the learning router in the proxy's `config.toml` (selected knobs — see `AutoRouterConfig` in `src/callosum/config.py` for the full set):

```toml
[auto_router]
# Cooldown self-healing prober: re-probe backends whose persisted cooldown is
# still in the future; clear the cooldown if the probe succeeds. 0 disables.
cooldown_probe_interval_seconds = 3600

# Per-cell minimum-coverage quota (deterministic, default off).
min_coverage_quota_enabled = false
min_coverage_budget_pct = 0.10          # even split across a lane's candidate cells
min_coverage_floor_pct = 0.0            # optional absolute per-cell floor; 0 = pure even split
min_coverage_window_seconds = 2592000   # 30 days
min_coverage_feasibility_enabled = true # forced cell must fit the stall-guard budget
min_coverage_cooldown_enabled = true     # skip a cell briefly after a forced-turn timeout

# Dynamic per-model cost rank from MEASURED weekly-quota burn (replaces a flat
# remote constant). Catalog priority is the cold-start prior; overrides win.
cost_rank_dynamic_enabled = true
cost_rank_min_nonzero_samples = 10
cost_rank_window_seconds = 2592000       # 30 days
cost_rank_base = 10                      # cheapest measured remote model starts here; local = 0
cost_rank_overrides = {}                 # {model_slug: cost_rank}

# Forward cost estimator: per-request predicted weekly_used_percent burn.
cost_estimate_enabled = true
cost_estimate_fallback_rate = 5e-6       # weekly-% points per token when no measured signal
cost_estimate_overrides = {}             # {model_slug: pct_per_token}

# Forward time estimator: per-request predicted wall-clock latency (ms), fit
# per cell from the request log. Local is NOT zero — it is often the slow path.
time_estimate_enabled = true
time_estimate_fallback_ms_per_token = 12.0
time_estimate_fallback_base_ms = 500.0
time_estimate_local_slowdown = 4.0       # cold-start tilt when a remote prior is reused for a local cell
time_estimate_overrides = {}             # {model_slug: [ms_per_token, base_ms]}

# Guardrail: cap the top-N canonical reasoning tiers (by severity rank, not by
# name) so auto-routing keeps the expensive levels rare. Each capped tier is
# dropped from automatic routing once its share of recent successful traffic
# exceeds this fraction (while cheaper alternatives exist). ollama-cloud models
# are immune — their reasoning levels are not subject to this regulation.
effort_cap_enabled = true
effort_cap_pct = 0.01
effort_cap_top_n = 2
effort_cap_window_seconds = 604800       # 7 days, aligned to the weekly quota window
```


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

## Pre-flight ETA ranges

`POST /v1/eta` exposes the request-log-backed time estimator as approximate
operator guidance. Send `input_tokens` plus optional `model` and
`reasoning_effort` filters; the response returns one entry per matching cell
with `eta.p50_ms`, `eta.high_ms`, `eta.range = "approximate_p50_to_p95"`,
`eta.exact = false`, and metadata for source, sample counts, confidence, and
verifiability. When usage logging or the estimator is unavailable, the route
returns `available: false` instead of inventing a prediction.

## Empirical usage-rate table

`GET /v1/usage` exposes the request-log-backed quota-rate analysis used for
operator observability. The backward-compatible top-level `unit`,
`metadata`, and `rates[]` fields still report the weekly meter. New clients
should read `meters.five_hourly` and `meters.weekly`, each with its own
`unit`, `metadata`, and `rates[]` table keyed by served `model` and
`reasoning_effort`.

The two remote Codex quota ledgers are measured independently:

- `five_hourly_used_percent`, reset by `five_hourly_reset_at`
- `weekly_used_percent`, reset by `weekly_reset_at`

The same upstream request consumes both ledgers, but not necessarily at the
same rate. Either ledger can independently block access, so weekly-only
measurement is not enough for quota-aware routing decisions. The endpoint also
returns `relationships[]` rows per cell with empirical
`five_hourly_per_weekly_ratio` and `weekly_per_five_hourly_ratio` when both
meters have enough closed tick windows.

Routing uses the same quantized meter-window substrate for composite
quota-aware backend selection. It does not average the ledgers: a candidate
whose five-hourly estimate would consume too much of the remaining five-hourly
headroom loses even if weekly headroom is ample, and the reverse is true when
weekly quota is the tighter constraint. `/status` and no-viable-backend
diagnostics expose `blocking_meters` and the currently `constraining_meter`
when quota snapshots are available.

Each rate row includes `samples.total`, `samples.usable`, component entries
for `input_uncached`, `input_cached`, `output`, and `reasoning`, a
`cache_effect`, `source`, `confidence`, and `updated_at`. A component is
reported with `available: false` when the request log cannot identify that
coefficient independently, for example because cached and uncached token
counts are collinear or there are too few positive quota-delta samples. Local
cells are reported as `local_zero` because they do not consume ChatGPT/Codex
plan quota.

High-confidence quota rates are fit from serialized integer-meter windows, not
single request rows. Because both percent meters are integer-resolution, a row
with no visible movement for that meter is treated as pending burn evidence.
The estimator accumulates serialized same-backend rows while the selected
meter is unchanged and emits one aggregate training sample only when a later
request advances that same meter. If that pending window crosses a reset, a
credit/top-up change, a concurrent same-backend request, or multiple served
cells, it is excluded rather than guessed. The current request log does not
persist a non-secret upstream credential/account identifier, so this is
backend-level serialization rather than true credential-level serialization.
Each meter's metadata under `cost_label_attribution` reports the attribution
key, included serialized count, and excluded counts such as `overlapped`,
`scheduled_reset`, `credit_or_topup_change`, `zero_quantized_or_unobservable`,
`unknown_negative_delta`, `reset_crossover`, `non_success`, `missing_quota`,
and `nonpositive_delta`.

Call it directly:

```bash
curl -s http://127.0.0.1:8765/v1/usage \
  -H "Authorization: Bearer $CALLOSUM_TOKEN" | jq .
```

The units are integer-resolution plan-quota counters persisted from upstream
quota snapshots. They are not OpenAI API dollar pricing, and the endpoint
intentionally does not import published API token prices or prompt-cache
discounts. Cached-vs-uncached differences are only "measured from this proxy's
request log" when the logged quota deltas make that split identifiable.
Upstream prompt-cache prefix matching, partial-match behavior, ordering
sensitivity, and expiry are upstream behavior and are not verified by this
quota data.

## Per-request usage log

When `[usage_log] path = "..."` is set, every backend call is recorded as a row in a SQLite database. Combined with `capture_bodies = true` (default during the modeling phase), this is the data corpus for figuring out how `(model, reasoning_effort, token counts)` translate into the opaque "usage percent" Codex Plus accounts decrement against.

Peer-quality capture is stored separately from live routing decisions.
Nonce-validated `<<qop ...>>` opinions are persisted in
`peer_quality_opinions`; new rows include an exact subject request id
when the judging model returns it, while older rows fall back to the
latest prior same-session subject-cell match. The shadow label job can
write conservative `quality_score` candidates with
`quality_label_method='peer_quality_v1'`. These labels feed the
configured quality predictor (currently `cell_majority_prior`); the
job does not switch live routing by itself.

Capture is opt-in. Set `CALLOSUM_PEER_QUALITY_CAPTURE_RATE` to a value
above `0` before expecting new peer opinions or capture metrics. `/status`
reports the current setting under `router.peer_quality_capture`. For a
managed service, use the reversible capture-window flow in
`docs/operations/dispatch.md` rather than editing the unit file directly.

Apply peer-derived labels manually:

```bash
python -m callosum.jobs.apply_peer_quality_labels \
  --db-path /home/<user>/.local/state/callosum/requests.sqlite \
  --checkpoint-path /home/<user>/.local/state/callosum/apply_peer_quality_labels.ckpt \
  --batch-size 200 \
  --dry-run
```

Drop `--dry-run` only after the preview shows candidate labels worth
writing. Dry-run mode resolves candidates through the same operator path
but does not update request rows or checkpoint state.

Inspect the shadow report without changing routing:

```bash
curl -s http://127.0.0.1:8765/status \
  -H "Authorization: Bearer $CALLOSUM_TOKEN" \
  | jq .router.peer_quality_shadow
```

If the service is stopped, inspect the same shadow report
directly from the usage log:

```bash
python -m callosum.jobs.peer_quality_shadow_report \
  --db-path /home/<user>/.local/state/callosum/requests.sqlite
```

`peer_quality_shadow` reports captured opinion counts, the capture
funnel (sampled → injected → opinion, via `peer_quality_capture`), the
sidecar judging-cost breakdown, whether the label job has pending
candidates, `peer_quality_v1` label counts, and per-cell label
coverage. Treat that block as diagnostic only; live routing stays on
the configured predictor (`cell_majority_prior`) until the operator
approves a rollout.

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
0 9 * * *   curl -s -H "Authorization: Bearer $CALLOSUM_DIAG_KEY" \
                 http://127.0.0.1:8765/diagnose/upstream \
            | jq -e '.ok' > /dev/null \
            || echo "callosum upstream check failed at $(date)" \
                | mail -s "callosum: upstream regression" you@example.com
```

For systemd users, prefer a `systemd.timer` over cron — easier to inspect via `systemctl list-timers` and to log with `journalctl`.

## Self-observation and self-implementation (the dev loop)

callosum observes its own routing behavior and, optionally, writes
per-model adapter code for itself. The mechanism is gated by
deterministic pre-filters and LLM judgment; it cannot push to a
remote, merge to main, or run `git revert` autonomously.

Components, briefly:

- **Capability harness** — periodic in-process sweeper that probes
  every local cell for tool-call behavior at small and at realistic
  context sizes, writing structured findings (with adapter hints)
  to `logs/capability_profiles/`. See `src/callosum/capability/`.
- **Weight identity** — groups cells routing to the same underlying
  weights so divergent findings are attributable to the transport,
  not the model. Pluggable via `WeightIdentityProvider` protocol.
- **Canary baseline** — 2–10% of `auto`-mode requests are redirected
  to remote-only, giving a continuous A/B for regression detection.
  Quota-aware. Surfaced on `/status`.
- **Failure registry** — per-request structured failure observations
  in the usage-log SQLite, with symptom and responsible-layer
  attribution.
- **Transform substrate** — `src/callosum/transforms/`, the package
  where per-model payload adapters live. Empty by default; the dev
  loop's only scope for writes.
- **Dev loop** — `callosum-dev-loop` CLI. Reads the above as a
  perception snapshot, runs a cheap pre-filter, only invokes the
  agent when there's actionable signal. When invoked, the agent
  writes (if anything) only inside the transform substrate; commits
  land on `auto/dev-loop-*` branches for manual review.


```bash
bash deploy/systemd/install.sh --enable
```

## Operational notes

- **Background process interruption.** When running the service in the background with `&` (e.g., `uv run callosum ... &`), the process is no longer in the terminal's foreground process group, so Ctrl+C won't reach it directly. Use `kill <pid>` or `killall callosum` to stop background instances, or use a process manager (tmux, screen, systemd) for reliable lifecycle management. The service includes explicit signal handlers (SIGINT/SIGTERM) to ensure clean shutdown.
- **Managed service control.** On systems running Callosum as a systemd user service, use `callosum service status`, `callosum service logs --follow`, `callosum service logs -n 100`, `callosum service restart`, `callosum service stop`, and `callosum service start`. Do not run multiple instances on the same port (8765 by default)—only the first will bind successfully; subsequent instances fail with "address already in use" and requests will hit the original instance instead.
- **Artifact-backed managed service.** The managed service should not run `uv run` from the checkout. Use `callosum build` to rebuild and reinstall the runtime venv, then point the user service at `~/.local/share/callosum/runtime/venv/bin/callosum serve`. Use that same installed binary for operator commands too: `~/.local/share/callosum/runtime/venv/bin/callosum service ...`. For development-only source serving, use `callosum serve --source` or `callosum service restart --source`. See `docs/operations/runtime_deploy.md`.
- **Auth.json refresh-chain caveat.** The proxy reads `auth.json` directly and owns the OAuth refresh chain for that account. Every successful refresh produces a new refresh token and writes it back to the file. If anything else (e.g. your normal `codex` usage on the same account) refreshes against the same auth file in parallel, whichever side rotates first invalidates the other. **Either dedicate an account to the proxy, or route your own Codex usage through the proxy too** (using the [Codex CLI Option A](#option-a--callosum-as-the-global-default-recommended-for-single-point-of-auth-setups) global-default setup).
- **Body capture is sensitive data.** With `capture_bodies = true`, prompts and responses are stored on disk in cleartext (after zlib decompression). Treat the database file as sensitive; `chmod 600` is a sensible baseline. Flip `capture_bodies = false` once you've collected enough corpus to model consumption; turn it back on whenever Codex updates its models.
- **Quota-percent granularity.** `five_hourly_used_percent` and `weekly_used_percent` are integer percentages reported by the upstream (over the wire as `x-codex-primary-used-percent` and `x-codex-secondary-used-percent` respectively). Single small calls often show `Δ = 0`; those rows are pending burn evidence, not free requests. The rate fitter aggregates serialized rows until the meter ticks and uses that closed window as the training label.
- **No log rotation.** The usage-log SQLite file grows append-only. Archive manually when it gets large.
- **Localhost only.** The proxy binds `127.0.0.1`. If you need to expose it to other machines, front it with TLS (Caddy / nginx) and rely on `[auth]` for access control.
- **One worker.** uvicorn defaults to one worker; the SQLite databases are not safe across multiple worker processes. Don't increase `--workers`.

## Verify

```
make verify
```

Runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`.
