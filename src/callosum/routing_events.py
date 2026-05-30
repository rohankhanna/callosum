"""SSE broadcaster for routing-decision events.

Exposes `GET /events/routing` as a Server-Sent Events stream. One event
fires per recorded request — the payload tells external observers what
the proxy actually did, in a shape that's stable for sidecar UIs to
consume without poking at the SQLite schema:

    {
      "request_id": int,
      "ts": float,                    # unix seconds, when the request started
      "requested_model": str | null,  # what the client asked the proxy for
      "served_cell": str,             # which cell actually handled it
      "served_context_window": int | null,  # the served cell's REAL window
      "status": int,                  # HTTP status returned to the client
      "latency_ms": int,
      "prompt_tokens": int | null,
      "completion_tokens": int | null,
      "retry_count": int,             # cell-retries (0 if primary succeeded)
      "mode": str                     # "auto" | "remote-only" | etc.
    }

Subscribers receive events starting from when they connect; there is no
backfill. A `: keepalive` comment fires every 30s of idle so proxies
and load-balancers don't close the connection between requests.

Design notes for consumers (e.g. the `snorkel` sidecar HUD):
  * `requested_model` is what the client (Codex CLI) put in the request;
    `served_cell` is what callosum's router chose. They often differ.
    The lying-CLI-footer problem this endpoint exists to solve is
    precisely the gap between these two fields.
  * `served_context_window` is the served cell's REAL advertised window.
    Use this — not the client's assumed window — when computing "how
    much of the context have I used?"
  * Slow consumers drop events (per-subscriber queue has a maxsize and
    QueueFull is silently swallowed). Don't rely on this stream for
    accounting; use the SQLite log for that.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from callosum.cell_grid import Cell
    from callosum.operator_state import OperatorState
    from callosum.routing.protocols import CellCapabilities
    from callosum.usage_log import UsageLog


class _RoutingEventBroadcaster:
    """Fan-out of routing-event payloads to all subscribed queues.

    Each new subscriber gets its own bounded queue. Slow consumers
    drop events (QueueFull is swallowed) rather than backpressure the
    request-handling thread that's calling notify().
    """

    def __init__(
        self,
        *,
        usage_log: "UsageLog",
        operator_state: "OperatorState | None",
        capabilities_of: "Callable[[Cell], CellCapabilities] | None",
    ) -> None:
        self._usage_log = usage_log
        self._operator_state = operator_state
        self._capabilities_of = capabilities_of
        self._queues: list[asyncio.Queue[dict[str, Any]]] = []

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    def notify(self, request_id: int) -> None:
        """Called from sync context after usage_log.record() inserts a row.

        Builds the rich payload, fans out to subscribers. Defensive: any
        exception is swallowed so the request-handling thread is never
        impacted by an observability bug.
        """
        try:
            payload = self._build_payload(request_id)
        except Exception:
            return
        if payload is None:
            return
        for q in list(self._queues):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                # Slow consumer; we'd rather drop an observability event
                # than backpressure the proxy.
                pass

    def _build_payload(self, request_id: int) -> dict[str, Any] | None:
        # Fresh connection so we don't contend with usage_log's writer
        # connection. WAL mode makes concurrent readers near-free.
        conn = sqlite3.connect(str(self._usage_log.path), timeout=5.0)
        try:
            row = conn.execute(
                "SELECT ts_start, model, reasoning_effort, requested_model, "
                "status, latency_ms, prompt_tokens, completion_tokens "
                "FROM requests WHERE id = ?",
                (request_id,),
            ).fetchone()
            if row is None:
                return None
            (
                ts_start,
                model,
                effort,
                requested_model,
                status,
                latency_ms,
                prompt_tokens,
                completion_tokens,
            ) = row
            # Cell-retry attempts are persisted only for multi-attempt
            # requests. A missing row means single-attempt → retry_count
            # is 0. When present, the row count includes the final
            # successful attempt, so retry_count = total - 1.
            attempts_row = conn.execute(
                "SELECT COUNT(*) FROM request_routing_attempts "
                "WHERE request_id = ?",
                (request_id,),
            ).fetchone()
            total_attempts = int(attempts_row[0]) if attempts_row else 0
            retry_count = max(0, total_attempts - 1)
        finally:
            conn.close()

        served_context_window: int | None = None
        if self._capabilities_of is not None and model:
            # Cell lookup is best-effort; if the served cell is no
            # longer in the live pool (rare — would mean the model
            # was deadvertised between routing and event emission),
            # we still emit the event with a null window.
            try:
                from callosum.cell_grid import Cell

                cell = Cell(model=model, reasoning_effort=effort or "default")
                caps = self._capabilities_of(cell)
                if caps is not None:
                    served_context_window = caps.context_window
            except Exception:
                pass

        mode = (
            self._operator_state.get_mode()
            if self._operator_state is not None
            else "auto"
        )

        return {
            "request_id": int(request_id),
            "ts": float(ts_start) if ts_start is not None else 0.0,
            "requested_model": requested_model,
            "served_cell": model,
            "served_context_window": served_context_window,
            "status": int(status) if status is not None else 0,
            "latency_ms": int(latency_ms) if latency_ms is not None else 0,
            "prompt_tokens": (
                int(prompt_tokens) if prompt_tokens is not None else None
            ),
            "completion_tokens": (
                int(completion_tokens)
                if completion_tokens is not None
                else None
            ),
            "retry_count": retry_count,
            "mode": mode,
        }


def install_routing_events(
    app: FastAPI,
    *,
    usage_log: "UsageLog",
    operator_state: "OperatorState | None",
    capabilities_of: "Callable[[Cell], CellCapabilities] | None",
) -> None:
    """Mount `GET /events/routing` and wire the broadcaster to usage_log.

    Safe to call once at startup. After this, every recorded request
    produces one SSE event fanned out to every connected client.
    """
    broadcaster = _RoutingEventBroadcaster(
        usage_log=usage_log,
        operator_state=operator_state,
        capabilities_of=capabilities_of,
    )

    @app.get("/events/routing", include_in_schema=False)
    async def routing_events_stream() -> StreamingResponse:
        q = broadcaster.subscribe()

        async def generate():
            try:
                while True:
                    try:
                        payload = await asyncio.wait_for(q.get(), timeout=30.0)
                        yield f"data: {json.dumps(payload)}\n\n".encode()
                    except asyncio.TimeoutError:
                        # SSE comment line keeps proxies/load-balancers
                        # from closing the connection during idle.
                        yield b": keepalive\n\n"
            finally:
                broadcaster.unsubscribe(q)

        return StreamingResponse(
            generate(), media_type="text/event-stream"
        )

    usage_log.add_new_request_callback(broadcaster.notify)
