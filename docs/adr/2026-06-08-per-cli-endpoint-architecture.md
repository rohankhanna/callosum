# ADR: One HTTP endpoint per CLI (organized by client, not by API shape)

Date: 2026-06-08

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-06-08 (introducing commit `1a3931d`). It records a past decision;
> the implementation landed prior to this document. The date above is
> the decision date.

## Context

Callosum fronts several LLM CLIs (codex first; hermes and claude-code
deferred) plus generic OpenAI-compatible clients. Each CLI sends
subtly different request bodies and parses responses with subtly
different expectations — codex sends Responses-API bodies with
`instructions`, `store: false`, and specific reasoning-hint shapes;
claude-code speaks Anthropic `/v1/messages`. The HTTP surface has to
let CLI-specific translation happen without forcing that translation
on generic clients.

The previous design served every client from the generic `/v1/*`
endpoints and branched on request headers to detect which CLI was
calling. That produced **silent shape-translation bugs**: a transform
meant for one CLI's contract would bleed into another's traffic (or
fail to fire for the client that needed it) because the routing key was
a fragile header heuristic, not a structural fact the system carried
end-to-end.

## Decision

Organize the HTTP surface **by client, not by API shape**: one
dedicated endpoint per CLI callosum serves, sharing a single dispatch
core but tagged with the calling CLI (`1a3931d`,
`src/callosum/app.py`):

- `POST /codex` — the codex CLI's endpoint. It calls the same
  `_dispatch_route` core as `POST /v1/responses` but passes
  `client_endpoint="codex"`, so codex-specific transforms can scope
  themselves to this endpoint without touching generic-OpenAI traffic.
- `POST /v1/responses` and `POST /v1/chat/completions` remain for
  generic OpenAI-compatible clients (no `client_endpoint` tag —
  `None`/generic in the transform context).
- Adding a new CLI (hermes, claude-code, future) is a fixed pattern:
  a `transforms/<cli-name>/` module carrying that CLI's concrete
  transforms, plus an endpoint declaration in `app.py` that tags
  `client_endpoint="<cli-name>"`.

The CLI identity is carried structurally through the
`TransformContext.endpoint` field
(`src/callosum/transforms/protocol.py`), not inferred from headers at
transform time. Each transform's `applies_to(ctx)` checks
`ctx.endpoint` to decide whether it fires, so CLI-specific behavior is
opt-in per transform and cannot leak across endpoints. The same
`1a3931d` wired response-side transforms on the non-stream path (the
substrate's `apply_response` was previously dead code with a TODO);
streaming response transforms remain deferred until a concrete
streaming-specific transform needs them.

## Consequences

**Positive:**

- CLI-specific translation is scoped by a structural endpoint tag, not
  a header heuristic, so the silent cross-CLI shape-translation bleed
  of the previous design cannot recur — a transform fires for a CLI
  only when `applies_to(ctx)` sees that CLI's `endpoint`.
- Generic OpenAI clients are untouched: they keep using `/v1/*` with
  no CLI-specific transforms applied.
- Adding a CLI is a fixed, small pattern (endpoint + transform module)
  rather than a header-branching change that risks every other client.

**Negative / accepted:**

- No concrete per-CLI transform module ships yet: the `transforms/
  <cli-name>/` extension pattern is the agreed shape, but
  `1a3931d` added only the `/codex` endpoint and the `endpoint` field
  in `TransformContext` — the substrate is ready, concrete codex
  transforms are deferred until a concrete need. So the decision's
  value is structural (the routing key is sound) more than
  behavioral (no codex-only transform is exercised today).
- Each CLI gets its own endpoint declaration, so the endpoint count
  grows with the number of CLIs; that is the deliberate trade for
  not branching on headers inside a shared endpoint.
- Streaming response transforms are deferred (the non-stream path
  buffers the full response, so per-CLI output-shape translation is
  addressable there first); a streaming-specific transform will need
  careful SSE chunk-boundary handling when one is needed.

**Reversibility:** normal git-history change. The dedicated endpoints
can coexist indefinitely with the generic `/v1/*` endpoints; the
`TransformContext.endpoint` field is additive (None/generic for
untagged endpoints). This ADR records the one-endpoint-per-CLI,
client-organized, structural-endpoint-tag shape as the agreed one, and
records that the concrete per-CLI transform modules are the deferred
extension, not the landed behavior.