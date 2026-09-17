# Callosum

Callosum is a local, adaptive HTTP routing layer for OpenAI-compatible model
backends. It accepts `/v1/responses` and `/v1/chat/completions` requests,
chooses the best available backend for each request, and forwards the call
transparently.

It is designed for a single operator on one machine. Callosum is not a
multi-tenant service.

## Contents

- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Client integrations](#client-integrations)
- [Architecture](#architecture)
- [Operations](#operations)
- [Extending Callosum](#extending-callosum)
- [API stability](#api-stability)
- [Terms of service](#terms-of-service)
- [Verify](#verify)

## Quick start

Install dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

Create a configuration file:

```toml
# ~/.config/callosum/config.toml
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
type = "codex_gateway"
base_url = "http://127.0.0.1:7342"
api_key = "REPLACE_WITH_UPSTREAM_API_KEY"
models = ["model-a0e7"]
```

Start the server:

```bash
uv run python -m callosum
```

Verify it is running:

```bash
curl -s http://127.0.0.1:8765/health
```

Expected response:

```json
{"status": "ok", "version": "0.1.0"}
```

## Configuration

Callosum reads a single TOML file from `~/.config/callosum/config.toml` by
default.

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for:

- server options
- persistent state paths
- request logging
- authentication
- backend definitions
- programmatically registered backends

## Client integrations

Any OpenAI-compatible client can point at Callosum.

The standard environment override is:

```bash
OPENAI_BASE_URL=http://127.0.0.1:8765/v1 \
OPENAI_API_KEY="$CALLOSUM_TOKEN" \
your-tool
```

Detailed setup guides for Codex CLI, Hermes, Aider, Cursor, Continue, the
OpenAI SDKs, curl, and generic clients are in
[`docs/CLIENTS.md`](docs/CLIENTS.md).

## Architecture

Callosum has three main layers:

1. HTTP surface — FastAPI routes and request/response translation
2. Routing pipeline — feature extraction, capability filtering, prediction,
   and backend selection
3. Backends — Codex-compatible, local, and Ollama-compatible upstreams

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the full architecture and
generated diagrams.

## Operations

Callosum exposes a small operator surface:

- `GET /status`
- `POST /control/pin`
- `POST /control/unpin`
- `GET /diagnose/upstream`
- `GET /v1/usage`
- `POST /v1/eta`

See [`docs/OPERATIONS.md`](docs/OPERATIONS.md) for routing behavior, sticky
sessions, request logging, usage rates, health checks, and operational notes.

## Extending Callosum

Callosum is built around two extension points:

1. Backends — add an upstream model endpoint
2. Routing components — add or replace a predictor, selector, or capability
   provider

See [`docs/EXTENDING.md`](docs/EXTENDING.md).

## API stability

Callosum follows a rolling `main` release model.

See [`docs/API_STABILITY.md`](docs/API_STABILITY.md) for the compatibility
policy.

## Terms of service

Callosum is a routing proxy. It does not bypass provider authentication or
quota enforcement.

See [`docs/TERMS.md`](docs/TERMS.md).

## Verify

Run the canonical verification path:

```bash
make verify
```

This runs:

- `ruff check .`
- `ruff format --check .`
- `mypy src/callosum`
- `pytest`
