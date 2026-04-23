# codex-proxy

A local, OpenAI-compatible HTTP endpoint that fronts a pool of upstream model backends and picks one per request based on health, cooldowns, and advertised models. Clients point at `codex-proxy` instead of a single provider, and the proxy rotates across the pool when one backend is rate-limited, returns an auth error, or flakes transiently.

Intended to run on `127.0.0.1` alongside the clients that call it — editor plugins, chat UIs, SDK scripts. Stateless per-request: each request carries its own full history, so backend rotation between turns has no continuity problem.

## Quick start

Install (with [uv](https://docs.astral.sh/uv/)):

```
uv sync
```

Write a config at `~/.config/codex-proxy/config.toml`:

```toml
[[backends]]
id = "primary"
type = "openai_api_key"
api_key_env = "OPENAI_API_KEY"
models = ["model-a0f5-mini"]
```

Export your key and start the server:

```
export OPENAI_API_KEY=sk-...
uv run python -m codex_proxy
```

The server listens on `http://127.0.0.1:8765` by default.

Call it as if it were OpenAI:

```
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"model-a0f5-mini","messages":[{"role":"user","content":"hi"}]}'
```

Streaming works with `"stream": true` — SSE chunks are forwarded unmodified.

## Backends

Two backend types ship today:

- `openai_api_key` — any OpenAI-compatible endpoint that accepts `Authorization: Bearer <key>` (OpenAI itself, OpenRouter, local LLM servers that expose the same shape).
- `azure_openai` — Azure OpenAI resources, with per-model deployment mapping.

You can mix and match. When one backend returns `429`, `401/403`, `404`, or a transient 5xx, the proxy excludes it from the pool and re-selects. On `429`, the upstream's `Retry-After` becomes the cooldown window (60s default); the cooldown is persisted to disk if `[state]` is configured, so it survives restart.

## Operational endpoints

- `GET /health` — liveness probe. Returns `{"status":"ok","version":"..."}`.
- `GET /status` — pool state: each backend's id, kind, advertised models, current health, usage snapshot, and the active pin (if any).
- `POST /control/pin` with `{"backend_id":"<id>"}` — force all routing to one backend.
- `POST /control/unpin` — clear the pin.

```
curl http://127.0.0.1:8765/status | jq
curl -X POST http://127.0.0.1:8765/control/pin \
  -H "Content-Type: application/json" \
  -d '{"backend_id":"primary"}'
```

## Configuration

Full config reference: [`docs/config.md`](docs/config.md).

## Verify

```
make verify
```

Runs `ruff check`, `ruff format --check`, `mypy --strict`, and `pytest`.
