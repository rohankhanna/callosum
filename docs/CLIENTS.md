# Client integrations

Callosum exposes OpenAI-compatible `/v1/responses` and
`/v1/chat/completions` endpoints. Any OpenAI-compatible client can point at
the proxy with the standard base URL and API key settings.

## Universal pattern

Almost every OpenAI-compatible client honors one or both of these environment variables:

| Variable | Used by |
| --- | --- |
| `OPENAI_BASE_URL` | OpenAI Python/JS SDK, Hermes Agent, most modern clients |
| `OPENAI_API_BASE` | Aider, older clients |
| `OPENAI_API_KEY` | All of them. The proxy treats this as the bearer token. |

The history-preserving pattern is to override these per command:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY="$CALLOSUM_TOKEN" \
your-tool ...
```

Persistent edits in shell rc files or application settings also work, but they permanently retarget that tool.

## Codex CLI

Codex CLI uses its own typed provider configuration and does not honor
`OPENAI_BASE_URL`.

### Global default

Edit `~/.codex/config.toml` so global defaults appear before any section
headers:

```toml
model = "model-a0e7"
model_provider = "callosum"

[model_providers.callosum]
name = "callosum"
base_url = "http://127.0.0.1:8765/v1"
env_key = "CALLOSUM_TOKEN"
wire_api = "responses"
```

Use it without flags:

```bash
codex
codex exec "Say hello"
codex resume
```

### Opt-in profile

Use this when you want Codex CLI to keep its existing default provider and
only route specific commands through Callosum:

```toml
[model_providers.callosum]
name = "callosum"
base_url = "http://127.0.0.1:8765/v1"
env_key = "CALLOSUM_TOKEN"
wire_api = "responses"

[profiles.via_proxy]
model = "model-a0e7"
model_provider = "callosum"
```

Invoke it with:

```bash
codex -p via_proxy
codex exec -p via_proxy "Say hello"
codex resume -p via_proxy
```

## Hermes Agent

[Hermes Agent](https://github.com/nousresearch/hermes-agent) needs a custom
provider and a native Responses API mode:

```bash
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8765/v1
hermes config set model.api_mode codex_responses
hermes config set OPENAI_API_KEY "$CALLOSUM_TOKEN"
```

Verify a live call:

```bash
hermes chat -q "Say only: hello via Hermes"
sqlite3 ~/.local/state/callosum/requests.sqlite \
  "SELECT id, route, user_id, api_key_id, status FROM requests ORDER BY id DESC LIMIT 1"
```

## Aider

[Aider](https://aider.chat/) uses the older `OPENAI_API_BASE` variable:

```bash
OPENAI_API_BASE=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY="$CALLOSUM_TOKEN" \
aider --model model-a0e7
```

Equivalent flags:

```bash
aider --openai-api-base http://127.0.0.1:8765/v1 \
      --openai-api-key "$CALLOSUM_TOKEN" \
      --model model-a0e7
```

## Cursor

Cursor’s OpenAI base URL override is global and persistent:

- **Settings → Models → Override OpenAI Base URL:** `http://127.0.0.1:8765/v1`
- **Settings → Models → API Key:** your Callosum API key

## Continue for VS Code

[Continue](https://www.continue.dev/) stores model definitions in
`~/.continue/config.json`. Add an entry to its `models` array:

```json
{
  "title": "via Callosum",
  "provider": "openai",
  "model": "model-a0e7",
  "apiBase": "http://127.0.0.1:8765/v1",
  "apiKey": "REPLACE_WITH_CALLOSUM_API_KEY"
}
```

## OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8765/v1",
    api_key="REPLACE_WITH_CALLOSUM_API_KEY",
)

response = client.responses.create(
    model="model-a0e7",
    input=[
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "hi"}],
        }
    ],
)
print(response.output[0].content[0].text)
```

## OpenAI JavaScript/TypeScript SDK

```typescript
import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "http://127.0.0.1:8765/v1",
  apiKey: "REPLACE_WITH_CALLOSUM_API_KEY",
});

const response = await client.responses.create({
  model: "model-a0e7",
  input: [
    { type: "message", role: "user", content: [{ type: "input_text", text: "hi" }] },
  ],
});
console.log(response.output[0].content[0].text);
```

## curl

Responses API:

```bash
curl http://127.0.0.1:8765/v1/responses \
  -H "Authorization: Bearer $CALLOSUM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model-a0e7",
    "input": [
      {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
  }'
```

Chat Completions:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer $CALLOSUM_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model":"model-a0e7","messages":[{"role":"user","content":"hi"}]}'
```

For streaming, add `"stream": true` to the request body.

## Generic OpenAI-compatible clients

Look for these settings in your client:

- Base URL, endpoint, or API base: `http://127.0.0.1:8765/v1`
- API key or bearer token: your Callosum API key

OpenAI SDK-based clients usually honor `OPENAI_BASE_URL` and
`OPENAI_API_KEY`.

## Claude Code

Claude Code uses Anthropic’s `/v1/messages` API shape and is not currently
supported by Callosum. Point it directly at Anthropic until an Anthropic
backend is added.
