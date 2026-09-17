# Configuration reference

Callosum reads a single TOML file.

Default path:

```bash
~/.config/callosum/config.toml
```

Override it with:

```bash
callosum serve --config /path/to/config.toml
```

## `[server]`

```toml
[server]
host = "127.0.0.1"
port = 8765
client_auth_token_env = "CALLOSUM_CLIENT_TOKEN"
control_auth_token_env = "CALLOSUM_CONTROL_TOKEN"
startup_smoke_test = true
smoke_test_interval_seconds = 3600
```

- `host` — bind address. Default: `127.0.0.1`.
- `port` — TCP port. Default: `8765`.
- `client_auth_token_env` — environment variable containing the client bearer token.
- `control_auth_token_env` — environment variable containing the control bearer token.
- `startup_smoke_test` — probe each backend at startup. Default: `true`.
- `smoke_test_interval_seconds` — periodic smoke-test interval. Default: `3600`.

## `[state]`

```toml
[state]
dir = "/home/you/.local/state/callosum"
```

Persistent state is stored under this directory.

## `[usage_log]`

```toml
[usage_log]
path = "/home/you/.local/state/callosum/requests.sqlite"
capture_bodies = true
```

- `path` — SQLite request-log path.
- `capture_bodies` — store compressed request and response bodies. Default: `true`.

## `[auth]`

```toml
[auth]
db = "/home/you/.local/state/callosum/auth.sqlite"
session_ttl_seconds = 1800
```

- `db` — SQLite path for users, sessions, and API keys.
- `session_ttl_seconds` — session lifetime. Default: `1800`.

Passwords are stored as Argon2id hashes. API keys are shown once and stored only as SHA-256 hashes.

## `[[backends]]`

```toml
[[backends]]
id = "primary"
type = "codex_gateway"
base_url = "http://127.0.0.1:7342"
api_key = "REPLACE_WITH_UPSTREAM_API_KEY"
models = ["model-a0e7"]
```

- `id` — unique backend identifier.
- `type` — backend implementation. Public builds support `codex_gateway`.
- `base_url` — upstream endpoint URL.
- `api_key` — upstream bearer token.
- `models` — cold-start model list; live discovery replaces it when available.

Repeat `[[backends]]` for each endpoint.

## Programmatic backends

Some backends are registered automatically instead of through `[[backends]]`:

- **Local lane** — registered when the `local-llm` CLI is available.
- **Ollama Cloud** — registered when `CALLOSUM_OLLAMA_CLOUD_ENABLED=1`.
- **OpenRouter** — registered when `CALLOSUM_OPENROUTER_ENABLED=1`.

See `src/callosum/__main__.py` for the exact registration rules.
