from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, TypeVar

import httpx

from callosum.errors import BackendError, classify_http_status

if TYPE_CHECKING:
    from callosum.backend import CallHandle

DEFAULT_COOLDOWN_S = 60.0

_T = TypeVar("_T")


async def stall_guarded(
    source: AsyncIterator[_T],
    *,
    first_item_timeout_s: float,
    idle_timeout_s: float,
    what: str = "local upstream",
    handle: CallHandle | None = None,
) -> AsyncIterator[_T]:
    """Re-yield items from a streaming upstream, failing fast if it stalls.

    Wraps any async byte/line iterator (e.g. response.aiter_bytes()) and
    enforces two deadlines:

      * first_item_timeout_s — max wait for the FIRST item. Sized to cover
        a local model's cold weight-load plus prefill of a large prompt.
      * idle_timeout_s — max gap between subsequent items once data is
        flowing. A working model emits tokens milliseconds apart; a long gap
        means it has stalled.

    On either deadline this raises BackendError(classification="transient")
    so the dispatch layer treats it like any other transient upstream failure
    and retries the next candidate cell (or surfaces a clean 5xx), instead of
    blocking to the full transport timeout. This is the behavioral replacement
    for the old size-based pre-flight cap: a request the model can actually
    serve streams through untouched; only a genuine hang is cut short.

    Cancelling the in-flight __anext__ (what wait_for does on timeout)
    propagates out through the caller's async with client.stream(...)
    block, which closes the upstream socket — so the stalled runtime sees the
    disconnect and can stop generating.

    When handle is given, the longest inter-chunk wait (the time spent
    inside wait_for for a chunk AFTER the first) is recorded as
    handle.max_idle_gap_s. That wait is the pure upstream idle — consumer
    pull-time falls outside wait_for — so it is the right signal for
    tuning idle_timeout_s. First-byte wait is excluded (that is the
    TTFB signal, captured separately by the dispatch layer).
    """
    iterator = source.__aiter__()
    budget = first_item_timeout_s
    seen_first = False
    max_gap_s = 0.0
    while True:
        t0 = time.monotonic()
        try:
            item = await asyncio.wait_for(iterator.__anext__(), timeout=budget)
        except StopAsyncIteration:
            if handle is not None and max_gap_s > 0.0:
                handle.max_idle_gap_s = max_gap_s
            return
        except TimeoutError as exc:
            if handle is not None and max_gap_s > 0.0:
                handle.max_idle_gap_s = max_gap_s
            phase = "before first byte" if not seen_first else "mid-stream"
            raise BackendError(
                classification="transient",
                message=(
                    f"{what} stalled {phase}: no data for {budget:.0f}s — treating as a hang; routing will fall back"
                ),
            ) from exc
        if seen_first:
            # Subsequent chunk: this wait IS the inter-chunk idle gap. Consumer
            # processing time happens between yields, outside this wait_for.
            max_gap_s = max(max_gap_s, time.monotonic() - t0)
        seen_first = True
        budget = idle_timeout_s
        yield item


def error_from_response(response: httpx.Response) -> BackendError:
    """Build a classified BackendError from an upstream non-2xx response.

    Includes a truncated copy of the upstream response body in the message
    so operators can distinguish "model retired" from "rate limited" from
    "auth invalid" etc. without having to enable verbose logging.
    """
    classification = classify_http_status(response.status_code)
    retry_after = response.headers.get("retry-after")
    retry_after_s: float | None = None
    if retry_after is not None:
        try:
            retry_after_s = float(retry_after)
        except ValueError:
            retry_after_s = None
    detail = _extract_response_detail(response)
    message = f"upstream {response.status_code}"
    if detail:
        message = f"{message}: {detail}"
    return BackendError(
        classification=classification,
        status_code=response.status_code,
        retry_after_s=retry_after_s,
        message=message,
    )


def _extract_response_detail(response: httpx.Response) -> str:
    """Pull a human-readable message out of an upstream error body.

    Tries JSON first (most upstreams return `{"error": {"message": "..."}}`
    or a flat `{"message": "..."}`), falls back to the raw text. Capped at
    400 chars to keep log lines manageable.
    """
    try:
        body = response.text
    except Exception:
        return ""
    if not body:
        return ""
    try:
        import json

        parsed = json.loads(body)
    except (ValueError, TypeError):
        return body[:400]
    if isinstance(parsed, dict):
        # Common shapes: {error: {message: "..."}}, {error: "..."}, {message: "..."}
        err = parsed.get("error")
        if isinstance(err, dict):
            msg = err.get("message")
            if isinstance(msg, str) and msg:
                return msg[:400]
            code = err.get("code")
            if isinstance(code, str) and code:
                return code[:400]
        if isinstance(err, str) and err:
            return err[:400]
        msg = parsed.get("message")
        if isinstance(msg, str) and msg:
            return msg[:400]
        detail = parsed.get("detail")
        if isinstance(detail, str) and detail:
            return detail[:400]
    return body[:400]
