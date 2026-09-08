"""Tests for the routing-events SSE broadcaster.

Covers payload construction (the contract the snorkel sidecar consumes),
graceful degradation when components are missing, and slow-subscriber
backpressure (events drop rather than block the request thread).
"""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pytest

from callosum.cell_grid import Cell
from callosum.routing.protocols import CellCapabilities
from callosum.routing_events import _RoutingEventBroadcaster


class _FakeUsageLog:
    """Minimal UsageLog stand-in: just owns a path to a SQLite file
    populated by the test. The broadcaster reads from this path."""

    def __init__(self, path: Path) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._new_request_callbacks: list[Any] = []

    def add_new_request_callback(self, cb: Any) -> None:  # pragma: no cover
        self._new_request_callbacks.append(cb)


class _FakeOperatorState:
    def __init__(self, routing: str = "auto") -> None:
        self._routing = routing

    def get_routing(self) -> str:
        return self._routing


def _make_db(tmp_path: Path, *, session_id: str | None = "sess-abc123") -> Path:
    """Create a tiny requests schema and seed one row."""
    db = tmp_path / "requests.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start REAL NOT NULL,
            model TEXT,
            reasoning_effort TEXT,
            requested_model TEXT,
            session_id TEXT,
            status INTEGER NOT NULL,
            latency_ms INTEGER NOT NULL,
            prompt_tokens INTEGER,
            completion_tokens INTEGER
        );
        CREATE TABLE request_routing_attempts (
            request_id INTEGER NOT NULL,
            attempt_idx INTEGER NOT NULL,
            PRIMARY KEY (request_id, attempt_idx)
        );
        """
    )
    conn.execute(
        "INSERT INTO requests (id, ts_start, model, reasoning_effort, "
        "requested_model, session_id, status, latency_ms, prompt_tokens, "
        "completion_tokens) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)",
        (42, 1780000000.5, "model-a0a9", "default", "model-a0e8", session_id, 200, 38267, 8421, 270),
    )
    conn.commit()
    conn.close()
    return db


def _caps_of_gemma(cell: Cell) -> CellCapabilities:
    """Capabilities resolver that returns model-a0d5's window for the model-a0d5 cell."""
    if cell.model.startswith("model-a0d5"):
        return CellCapabilities(
            context_window=128_000,
            modalities=frozenset({"text"}),
            supports_tools=True,
            cost_rank=0,
        )
    return CellCapabilities(
        context_window=256_000,
        modalities=frozenset({"text"}),
        supports_tools=True,
        cost_rank=10,
    )


def test_builds_payload_with_all_fields(tmp_path: Path) -> None:
    """The payload contract — every field documented in the module
    docstring must appear, with correct values. snorkel and any other
    consumer keys off this shape."""
    db = _make_db(tmp_path)
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=_FakeOperatorState("auto"),
        capabilities_of=_caps_of_gemma,
    )
    payload = bcast._build_payload(42)
    assert payload is not None
    assert payload["request_id"] == 42
    assert payload["ts"] == 1780000000.5
    assert payload["session_id"] == "sess-abc123"
    assert payload["requested_model"] == "model-a0e8"
    assert payload["served_cell"] == "model-a0a9"
    assert payload["served_context_window"] is not None
    assert payload["status"] == 200
    assert payload["latency_ms"] == 38267
    assert payload["prompt_tokens"] == 8421
    assert payload["completion_tokens"] == 270
    assert payload["retry_count"] == 0
    assert payload["routing"] == "auto"


def test_filter_forwards_only_matching_session(tmp_path: Path) -> None:
    """When a subscriber passes a session_id filter, only events whose
    payload session_id matches are forwarded. Other events go to
    unfiltered subscribers but not this one. This is what lets
    per-instance sidecars (snorkel) avoid showing other instances'
    routing decisions."""
    db = _make_db(tmp_path, session_id="sess-A")
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )

    async def _scenario() -> None:
        filtered_q = bcast.subscribe(session_id="sess-A")
        other_q = bcast.subscribe(session_id="sess-B")
        unfiltered_q = bcast.subscribe()

        # Push the row that has session_id="sess-A".
        bcast.notify(42)

        # filtered_q sees it (filter matches).
        assert filtered_q.qsize() == 1
        payload = filtered_q.get_nowait()
        assert payload["session_id"] == "sess-A"

        # other_q does NOT see it (filter mismatch).
        assert other_q.qsize() == 0

        # unfiltered_q sees it (no filter).
        assert unfiltered_q.qsize() == 1

    asyncio.run(_scenario())


def test_filter_handles_null_session_id_event(tmp_path: Path) -> None:
    """Events without a session_id (the client didn't send the header)
    only reach unfiltered subscribers. A subscriber with any filter
    won't match a null session_id."""
    db = _make_db(tmp_path, session_id=None)
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )

    async def _scenario() -> None:
        filtered_q = bcast.subscribe(session_id="sess-A")
        unfiltered_q = bcast.subscribe()
        bcast.notify(42)
        assert filtered_q.qsize() == 0  # filter "sess-A" != None
        assert unfiltered_q.qsize() == 1
        payload = unfiltered_q.get_nowait()
        assert payload["session_id"] is None

    asyncio.run(_scenario())


def test_retry_count_reflects_cell_retry_history(tmp_path: Path) -> None:
    """When the request went through multiple cell-retry attempts,
    retry_count is (total_attempts - 1) — i.e. the retries before
    the final attempt that succeeded."""
    db = _make_db(tmp_path)
    conn = sqlite3.connect(str(db))
    # Three attempts: 0 (failed), 1 (failed), 2 (succeeded). retry_count=2.
    conn.executemany(
        "INSERT INTO request_routing_attempts (request_id, attempt_idx) VALUES (?,?)",
        [(42, 0), (42, 1), (42, 2)],
    )
    conn.commit()
    conn.close()
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )
    payload = bcast._build_payload(42)
    assert payload is not None
    assert payload["retry_count"] == 2


def test_missing_request_returns_none(tmp_path: Path) -> None:
    """Defensive: if the row was deleted/never written, no event is
    fanned out. Avoids surfacing phantom events to subscribers."""
    db = _make_db(tmp_path)
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )
    assert bcast._build_payload(999) is None


def test_missing_capabilities_resolver_yields_null_window(tmp_path: Path) -> None:
    """When capabilities_of is None (no backends loaded), the payload
    still emits — just without a real context window. snorkel handles
    null gracefully in its renderer."""
    db = _make_db(tmp_path)
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )
    payload = bcast._build_payload(42)
    assert payload is not None
    assert payload["served_context_window"] is None


def test_missing_operator_state_defaults_to_auto(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )
    payload = bcast._build_payload(42)
    assert payload is not None
    assert payload["routing"] == "auto"


def test_notify_swallows_exceptions(tmp_path: Path) -> None:
    """The notify() callback fires from the request-handling thread.
    A bug in payload construction must NEVER propagate back — the
    proxy keeps serving even if observability is broken."""
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(Path("/nonexistent/path.sqlite")),
        operator_state=None,
        capabilities_of=None,
    )
    # Must not raise.
    bcast.notify(42)


def test_slow_subscriber_drops_events_not_block(tmp_path: Path) -> None:
    """When a subscriber's queue is full, notify() drops the event
    instead of blocking. Otherwise a hung sidecar could backpressure
    the entire proxy."""
    db = _make_db(tmp_path)
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )

    async def _scenario() -> None:
        q = bcast.subscribe()
        # Fill the queue (maxsize=100).
        for _ in range(100):
            q.put_nowait({"filler": True})
        # Now notify — there's a real row at id=42; the broadcaster
        # would normally enqueue, but the queue is full. Must not raise.
        bcast.notify(42)
        # Queue still has 100 items (the dropped event was silently lost).
        assert q.qsize() == 100

    asyncio.run(_scenario())


@pytest.mark.parametrize("status,expected", [(200, 200), (500, 500), (None, 0)])
def test_status_defaults_to_zero_when_null(tmp_path: Path, status: int | None, expected: int) -> None:
    """status is NOT NULL in the real schema, but defensive coercion
    keeps the payload contract intact even if the column ever changes."""
    db = tmp_path / "requests.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY,
            ts_start REAL,
            model TEXT,
            reasoning_effort TEXT,
            requested_model TEXT,
            session_id TEXT,
            status INTEGER,
            latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER
        );
        CREATE TABLE request_routing_attempts (
            request_id INTEGER NOT NULL,
            attempt_idx INTEGER NOT NULL,
            PRIMARY KEY (request_id, attempt_idx)
        );
        """
    )
    conn.execute(
        "INSERT INTO requests (id, status, latency_ms) VALUES (?,?,?)",
        (1, status, 100),
    )
    conn.commit()
    conn.close()
    bcast = _RoutingEventBroadcaster(
        usage_log=_FakeUsageLog(db),
        operator_state=None,
        capabilities_of=None,
    )
    payload = bcast._build_payload(1)
    assert payload is not None
    assert payload["status"] == expected
