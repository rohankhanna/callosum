# Client Integration Cookbook

`codex-proxy` speaks the standard OpenAI shapes (`/v1/responses` and `/v1/chat/completions`). Any client that lets you set a base URL and a bearer token works. This document gives the exact knobs for the most common tools, with a strong bias for **per-invocation overrides** over persistent config edits — so your existing tool histories, profiles, and credentials stay intact.

The universal recipe:

```
base URL: http://127.0.0.1:8765/v1
bearer:   Authorization: Bearer <api-key>
```

`<api-key>` is whatever you minted from `POST /auth/keys`. In single-operator mode (no `[auth]` block), most clients still want a non-empty value in their bearer slot — set it to literally any non-empty string; the proxy ignores it.

## Universal pattern: env-var override

Almost every OpenAI-compatible client honours one or both of these env vars:

| Env var | Notes |
| --- | --- |
| `OPENAI_BASE_URL` | New OpenAI SDK convention. Used by Hermes Agent, OpenAI Python/JS SDKs, most modern clients. |
| `OPENAI_API_BASE` | Older convention, still used by Aider and a few others. |
| `OPENAI_API_KEY` | Standard everywhere. The proxy treats this as the bearer. |

The history-preserving pattern is to override these **per command**, not in your shell rc:

```
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=<your-codex-proxy-api-key> \
your-tool ...
```

Persistent edits (rc files, app settings) work too; they just permanently re-target the tool, which is exactly what some users want. Pick whichever matches your intent.

> **Codex CLI does not honor `OPENAI_BASE_URL`.** It has its own typed provider system. Setting `OPENAI_BASE_URL=http://127.0.0.1:8765/v1 codex ...` will silently route to OpenAI's real API and present your codex-proxy key as if it were an OpenAI key — which OpenAI then rejects as invalid. Use the profile pattern in the next section instead.

## Codex CLI

The OpenAI Codex CLI (`codex`, `codex exec`, `codex resume`). It supports a typed provider system. **The universal `OPENAI_BASE_URL` env var has no effect on this client** — only the typed provider config does.

You have two integration shapes. Pick based on whether you want the proxy to be the default for all `codex` commands, or only when explicitly opted into.

### Option A — `codex_proxy` as the global default (recommended for "single point of auth" setups)

Edit `~/.codex/config.toml` so global defaults stay at the top, before any `[section]` header. **TOML detail that bit me once:** every `key = value` line after a `[section]` header belongs to that section until the next header. Putting `model_provider = "codex_proxy"` *after* a `[profiles.x]` block silently makes it part of that profile, not a global default.

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

Set the API key in your shell rc:

```
echo 'export CODEX_PROXY_TOKEN=<your-codex-proxy-api-key>' >> ~/.bashrc
source ~/.bashrc
```

Use it — no flags needed:

```
codex                  # routes through the proxy
codex exec "..."       # routes through the proxy
codex resume           # picker shows sessions whose model_provider matches "codex_proxy"
```

**One-time migration of pre-existing sessions** so they appear under the new default. The `resume` picker filters by the saved `model_provider` field in each rollout. Migrate (with hardlink backup so it's safe and cheap):

```
cp -al ~/.codex/sessions ~/.codex/sessions.before-migration-$(date +%Y%m%d-%H%M%S)
find ~/.codex/sessions -name '*.jsonl' \
  -exec sed -i -E 's/"model_provider":\s*"openai"/"model_provider":"codex_proxy"/g' {} +
```

The hardlink backup costs near-zero disk space — `sed -i` rename-on-top breaks the hardlink for each modified file, leaving the backup pointing at the original inode.

### Option B — opt-in profile, default unchanged

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

The `-p via_proxy` flag must appear on every invocation you want routed through the proxy. Sessions resumed without `-p` replay against the default provider — `OPENAI_BASE_URL` does not change this.

**Stronger isolation** (if you don't want the proxy provider definition in your real config at all): use a separate `CODEX_HOME` for proxy-routed runs:

```
mkdir -p ~/.codex-proxy-home
# write the [model_providers.codex_proxy] + [profiles.via_proxy] block into ~/.codex-proxy-home/config.toml
CODEX_HOME=~/.codex-proxy-home codex exec -p via_proxy "..."
```

## Hermes Agent

[Hermes Agent](https://github.com/nousresearch/hermes-agent) (Nous Research) honours standard OpenAI env vars for custom endpoints, and also exposes a `hermes config set` CLI that writes them to `~/.hermes/.env`.

**Per-invocation (history-preserving):**

```
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=<your-codex-proxy-api-key> \
hermes
```

**Persistent:**

```
hermes config set OPENAI_BASE_URL http://127.0.0.1:8765/v1
hermes config set OPENAI_API_KEY <your-codex-proxy-api-key>
```

The persistent form replaces whatever provider you had configured before (Nous Portal, OpenRouter, etc.). To keep both, prefer the per-invocation form, or run Hermes from a separate `$HOME` for proxy-routed sessions.

## Aider

[Aider](https://aider.chat/) supports both env vars and CLI flags. It uses the older `OPENAI_API_BASE` name.

**Per-invocation:**

```
OPENAI_API_BASE=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY=<your-codex-proxy-api-key> \
aider --model model-a0e7
```

Or with explicit flags (also non-persistent):

```
aider --openai-api-base http://127.0.0.1:8765/v1 \
      --openai-api-key  <your-codex-proxy-api-key> \
      --model model-a0e7
```

Per-project chat history at `.aider.chat.history.md` is unaffected by either form.

## Cursor

Cursor's "Override OpenAI Base URL" setting is **global and persistent** — there is no per-invocation override. The cleanest pattern is to leave Cursor pointed at OpenAI for normal use and switch it temporarily when you want to route through the proxy.

**Settings → Models → Override OpenAI Base URL:**

```
http://127.0.0.1:8765/v1
```

**Settings → Models → API Key:**

```
<your-codex-proxy-api-key>
```

If you want to keep both routings switchable, the practical workaround is two Cursor profiles (one configured for direct OpenAI, one for the proxy) — but Cursor doesn't have first-class profile support, so this is fiddly. Most users either commit to the proxy or don't.

## Continue (VS Code extension)

[Continue](https://www.continue.dev/) stores its config in `~/.continue/config.json` under a `models` array. You can **add** a new entry alongside your existing ones — your previous models stay listed and selectable from the model picker.

**Append to `models` in `~/.continue/config.json`:**

```json
{
  "title": "via codex-proxy (model-a0e7)",
  "provider": "openai",
  "model": "model-a0e7",
  "apiBase": "http://127.0.0.1:8765/v1",
  "apiKey": "<your-codex-proxy-api-key>"
}
```

Switch between models via Continue's model dropdown in the chat panel. Existing entries (Anthropic, local Ollama, etc.) keep working.

## OpenAI Python SDK

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

Each `OpenAI(...)` instantiation is fully scoped — no global SDK state to mutate. Use a different `client` for direct OpenAI calls in the same script.

## OpenAI JS / TypeScript SDK

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

Same isolation guarantee as the Python SDK — per-instance config, no globals.

## curl

```
curl http://127.0.0.1:8765/v1/responses \
  -H "Authorization: Bearer <your-codex-proxy-api-key>" \
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
  -H "Authorization: Bearer <your-codex-proxy-api-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-a0e7",
    "messages": [{"role":"user","content":"hi"}]
  }'
```

For streaming, add `"stream": true` to either body and read the response as `text/event-stream`.

## Generic: any OpenAI-compatible client

If a client doesn't appear above, look for one of these knobs in its config:

- An "OpenAI base URL", "endpoint", "API base", or "custom provider URL" field — set it to `http://127.0.0.1:8765/v1`.
- An "API key" or "bearer token" field — set it to your codex-proxy API key.
- For OpenAI SDK-based clients, the `OPENAI_BASE_URL` and `OPENAI_API_KEY` env vars usually override at startup.

Sticky-session header (optional): add `X-Codex-Session-Id: <any-string>` if the client lets you set custom headers and you want a sequence of requests to bind to the same backend account.

## Claude Code (and other Anthropic-shape clients): requires Phase 3

[Claude Code](https://github.com/anthropics/claude-code) and Anthropic's SDK speak `/v1/messages` and the `claude-*` model family. The proxy currently only serves the OpenAI shapes (`/v1/responses` and `/v1/chat/completions`), so pointing Claude Code's `ANTHROPIC_BASE_URL` at the proxy will return `404`.

Routing Claude Code through this proxy means a planned **Phase 3** with two pieces:

1. **Anthropic-shape route** — implement `POST /v1/messages` (and the streaming SSE format Anthropic uses) in the proxy.
2. **`anthropic_auth_vault` backend type** — pool Claude Pro / Max session credentials the same way `codex_auth_vault` pools Codex Plus auths, with the same OAuth refresh discipline.

When that ships, the integration recipe will look identical to the others above — set `ANTHROPIC_BASE_URL=http://127.0.0.1:8765` and `ANTHROPIC_AUTH_TOKEN=<your-codex-proxy-api-key>`. Until then, run Claude Code against Anthropic directly.

## Operational notes

- **The same auth.json cannot be shared with the proxy and a normal Codex CLI session at the same time.** Refresh tokens are one-shot — whichever side rotates first invalidates the other side's stored copy. Either dedicate a Codex account to the proxy, or route the operator's own Codex usage through the proxy too. See [`config.md` → Operational notes](config.md#operational-notes).
- **Localhost only by default.** The proxy binds `127.0.0.1`. If you need to expose it to other machines, front it with TLS (Caddy / nginx) and restrict by IP allowlist or by relying on the proxy's own `[auth]` enforcement.
- **The X-Codex-Session-Id header is opt-in.** Without it every request is an independent backend selection. Set it if your client makes a sequence of related requests and you want them all to land on the same backend account (better prompt-cache hit rate, consistent quota attribution).
