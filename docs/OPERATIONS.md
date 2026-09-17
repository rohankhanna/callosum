# Operations

## Endpoints

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /health` | none | Liveness probe |
| `GET /status` | none | Backend pool, cooldowns, pin state, session bindings |
| `POST /v1/responses` | bearer when `[auth]` is set | OpenAI Responses API |
| `POST /v1/chat/completions` | bearer when `[auth]` is set | OpenAI Chat Completions API |
| `POST /v1/eta` | bearer when `[auth]` is set | Approximate pre-flight latency ranges |
| `GET /v1/usage` | bearer when `[auth]` is set | Empirical usage-rate table |
| `GET /diagnose/upstream` | bearer when `[auth]` is set | Upstream health check |
| `POST /control/pin` | control token when configured | Pin routing to one backend |
| `POST /control/unpin` | control token when configured | Clear the pin |

## Routing behavior

For each request, Callosum:

1. Drops backends that do not advertise the requested model.
2. Drops unavailable backends and backends in cooldown.
3. Drops backends whose quota is exhausted.
4. Ranks candidates by predicted cost and quality.
5. Selects the best available backend and retries on retryable failures.

If no backend can serve the request, Callosum returns a self-diagnosing error
with cooldown, quota, and recovery information.

## Model and reasoning levels

Clients can use concrete model IDs or the virtual model names `auto` and
`auto-learning`. Concrete IDs route to the matching backend. Virtual names use
the adaptive router.

Reasoning effort is passed through unchanged when the upstream supports it.
Clients can also use selector IDs such as:

```text
callosum:auto
callosum:local-only
callosum:remote-only
callosum:remote/model-id::high
callosum:local/model-id::low
```

Use `::` as the explicit effort delimiter when the model ID itself contains
colons.

## Sticky sessions

Send the following header to keep a sequence of requests on the same backend:

```text
X-Codex-Session-Id: session-id
```

Bindings are process-local and do not survive restarts.

## Pinning

Pin all routing to one backend:

```bash
curl -X POST http://127.0.0.1:8765/control/pin \
  -H 'Content-Type: application/json' \
  -d '{"backend_id":"primary"}'
```

Clear the pin:

```bash
curl -X POST http://127.0.0.1:8765/control/unpin
```

## Pre-flight ETA

`POST /v1/eta` returns approximate p50-to-p95 latency ranges per compatible
cell. Send:

```json
{
  "input_tokens": 1000,
  "model": "auto",
  "reasoning_effort": "medium"
}
```

When there is not enough data, the route returns `available: false`.

## Usage rates

`GET /v1/usage` returns empirical per-model, per-reasoning-effort quota
rates. It reports five-hourly and weekly meters independently and exposes the
same data used by the routing cost estimator.

## Request log

When `[usage_log].path` is configured, each backend attempt is recorded in
SQLite.

Useful columns include:

- `backend_id`
- `model`
- `reasoning_effort`
- `status`
- `classification`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `five_hourly_used_percent_before`
- `five_hourly_used_percent_after`
- `weekly_used_percent_before`
- `weekly_used_percent_after`

When `capture_bodies` is true, Callosum also stores compressed request and
response payloads.

## Daily upstream health check

`GET /diagnose/upstream` sends a tiny request to each non-cooldown backend and
checks:

- HTTP status
- quota headers
- Responses API terminal event
- usage block

Example:

```bash
curl -s -H "Authorization: Bearer $CALLOSUM_DIAG_KEY" \
  http://127.0.0.1:8765/diagnose/upstream | jq .
```

## Dev loop

Callosum includes an optional dev loop that observes routing behavior and can
write per-model adapter code. It is gated by pre-filters and cannot push to a
remote, merge to `main`, or revert history autonomously.

See `docs/operations/dev_loop.md` when present.

## Operational notes

- Keep Callosum bound to `127.0.0.1` unless you also add TLS and authentication.
- Use one worker; the SQLite databases are not safe across multiple workers.
- `capture_bodies = true` stores sensitive prompt and response data.
- Request-log databases grow append-only; archive them manually.
- Backends with OAuth refresh chains should be dedicated to Callosum or routed
  through it consistently to avoid refresh-token races.
