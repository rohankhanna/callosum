# ADR: stall_guarded streaming wrap (first-byte + idle-gap deadlines replacing the size-based pre-flight cap)

Date: 2026-06-15

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-06-15 (introducing commit `7291538`). It records a past decision;
> the implementation landed prior to this document. The date above is the
> decision date.

## Context

Callosum routes to local and cloud LLM backends over streaming HTTP. The
local lane previously enforced a **50KB pre-flight tool-request byte
cap**: a request whose serialized body exceeded the cap was rejected
before it was ever sent to the upstream. The cap was coarse — it rejected
legitimate-large contexts the model could actually serve (a long context
that streams fine is indistinguishable from one that hangs under a
size-only rule), and it did not catch a model that accepts the request
then goes silent mid-stream. The failure mode it guarded against
(upstream hang) is a *behavioral* signal, not a *size* signal: a working
model emits tokens milliseconds apart; a stalled one emits nothing.

Every streaming backend call — local (`litellm_gateway`,
`local_direct`) and cloud (`ollama_cloud`) — needed a uniform contract
for "cut a stalled stream short and retry the next cell," instead of
blocking to the full transport timeout.

## Decision

Replace the size-based pre-flight cap with a behavioral stall guard —
the `stall_guarded` async iterator wrapper
(`src/callosum/backends/_http.py`), enforcing two deadlines on every
streaming backend call:

- **`first_item_timeout_s`** — max wait for the FIRST item, sized to
  cover a local model's cold weight-load plus prefill of a large prompt
  (env `CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S`, default 180s).
- **`idle_timeout_s`** — max gap between subsequent items once data is
  flowing (env `CALLOSUM_LOCAL_STREAM_IDLE_TIMEOUT_S`, default 45s). A
  long gap means the model has stalled.

On either deadline, `stall_guarded` raises
`BackendError(classification="transient")` so the dispatch layer treats
it like any other transient upstream failure and retries the next
candidate cell (or surfaces a clean 5xx), instead of blocking to the
full transport timeout. Cancelling the in-flight `__anext__` (what
`wait_for` does on timeout) propagates out through the caller's
`async with client.stream(...)` block, which closes the upstream socket
— so the stalled runtime sees the disconnect and can stop generating.

Admission stays cheap (just the context-aware router); the failure mode
is now "upstream silent too long," not "we guessed it was too big." A
request the model can actually serve streams through untouched; only a
genuine hang is cut short. The size-based pre-flight cap is removed.

When a `CallHandle` is passed (`handle=...`), the longest inter-chunk
wait — the time spent inside `wait_for` for a chunk AFTER the first — is
recorded as `handle.max_idle_gap_s`. That wait is the pure upstream
idle (consumer pull-time falls outside `wait_for`), so it is the right
signal for tuning `idle_timeout_s`. First-byte wait is excluded (that is
the TTFB signal, captured separately by the dispatch layer).

## Consequences

**Positive:**

- A stalled stream retries the next cell instead of hanging to the
  transport timeout: every local/cloud streaming backend depends on
  this contract (`litellm_gateway`, `local_direct` chat + responses
  paths, `_responses_chat`, `ollama_cloud`).
- Legitimate-large contexts that stream fine are no longer rejected by
  a size-only rule; the guard measures behavior, not size.
- `idle_gap_ms` is captured for observability (landed `235212d`,
  2026-08-02) alongside `ttfb_ms` (`93d0590`, 2026-08-02), giving
  data-driven stall-guard tuning signal.

**Negative / accepted:**

- The deadlines are env-tunable and were calibrated for the local lane;
  cloud lanes reuse the same wrapper. `first_item_timeout_s=180s` is
  generous by design (cold local weight-load + large prefill); a tighter
  cloud budget would need a per-lane override, not currently wired.
- The guard only fires on streaming calls; a non-streaming (buffered)
  backend call has no inter-byte signal and falls back to the transport
  timeout.
- A later follow-up (`d295876`, 2026-08-04) wrapped the
  `local_direct` chat-native-responses stream that was missed by the
  original introduction — that branch iterated `response.aiter_lines()`
  through `_responses_sse_to_chat_sse` with no `stall_guarded` wrap, so
  `idle_gap_ms` stayed NULL for that sub-path. The follow-up is a
  coverage fix, not a change to the decision; it confirms every
  streaming sub-path must wrap.
- The idle-gap signal needs real streamed local traffic to accrue; in
  the live DB the rows observed so far are remote-lane (which never set
  `idle_gap_ms` by design), so the tuning signal is a data-accrual gap,
  not a bug.

**Reversibility:** normal git-history change. Reverting to a size-based
cap is discouraged by this ADR's behavioral rationale; the
first-byte + idle-deadline model is the agreed shape. The deadlines are
env-tunable without a code change.