# ADR: Translate local Responses SSE to chat-completions SSE

Date: 2026-06-13

## Status

Accepted

## Context

Local model entries discovered through local LLM gateway may advertise both
`responses` and `chat` API surfaces even when the per-model runtime endpoint
is Responses-native. Non-streaming chat requests already translate through
`/v1/responses`, but streaming chat-completions requests previously could not
be served directly by `LocalModelRegistryBackend`.

That mismatch made a routed streaming chat request depend on retry behavior
instead of a clear backend contract.

## Decision

`LocalModelRegistryBackend.chat_completions_stream` will serve Responses-native local
entries by:

- translating the inbound chat-completions request body to a Responses request,
- posting to the entry's `/v1/responses` endpoint with `stream: true`,
- converting Responses SSE events into OpenAI chat-completions SSE chunks as
  they arrive,
- preserving text deltas, function-call argument deltas, terminal completion,
  and usage when present.

Chat-only local entries are unchanged: they continue to stream through their
native `/v1/chat/completions` endpoint.

## Consequences

Streaming chat-completions traffic can now use Responses-native local cells
without waiting for a timeout or relying on dispatch fallback. The translator is
intentionally narrow and does not add mid-stream failover; a stream that starts
on one backend still finishes there or fails.
