"""Tests for the cell-level retry wrapper in app.py.

Phase 4b.3: when the recommender's primary cell fails with a retryable
error (5xx from its backend pool), the dispatch layer rewrites the body
to the next-best candidate cell and retries. Per-cell attempt history
lands in request_routing_attempts so future training can learn from the
reroute pattern.

The wrapper is tested by monkey-patching its inner `_dispatch_nonstream`
to a stand-in that records the body it saw and returns a canned result
(or raises) per cell. This decouples the wrapper's branching logic from
the inner dispatch's many code paths.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi import HTTPException

from callosum import app as app_module
from callosum.app import (
    MAX_CELL_ATTEMPTS,
    _dispatch_nonstream,
    _dispatch_nonstream_with_cell_retry,
    _DispatchRetryBudget,
    _request_id_context,
    _retry_budget_http,
)
from callosum.backend import CallHandle
from callosum.cell_grid import Cell
from callosum.errors import BackendError
from callosum.fakes import InMemoryFakeBackend
from callosum.session import SessionRegistry
from callosum.usage_log import UsageLog, UsageLogEntry

CELLS = [
    Cell(model="model-a", reasoning_effort="high", context_window=128_000),
    Cell(model="model-b", reasoning_effort="medium", context_window=128_000),
    Cell(model="model-c", reasoning_effort="low", context_window=128_000),
    Cell(model="model-d", reasoning_effort="low", context_window=128_000),
]


def _write_minimal_request_row(log: UsageLog) -> int:
    """Insert a minimal requests row so the FK on request_routing_attempts
    can resolve. Returns the new request_id."""
    return log.record(
        UsageLogEntry(
            ts_start=0.0,
            ts_end=0.0,
            route="/v1/responses",
            stream=False,
            session_id=None,
            backend_id="primary",
            model="model-a",
            reasoning_effort="high",
            status=200,
            classification="ok",
        )
    )


def _install_inner_stub(
    monkeypatch: pytest.MonkeyPatch,
    *,
    log: UsageLog | None,
    behavior: dict[str, Any],
) -> list[dict[str, Any]]:
    """Replace _dispatch_nonstream with a stub that:
    - Records each call's body['model'] + reasoning effort.
    - Writes a minimal requests row so _request_id_context behaves like
      the real dispatch (which always logs at least once per attempt).
    - Returns the canned result OR raises HTTPException per `behavior`.
    """
    seen: list[dict[str, Any]] = []

    async def _stub(body, **kwargs):
        seen.append(
            {
                "model": body.get("model"),
                "effort": (body.get("reasoning") or {}).get("effort"),
            }
        )
        if log is not None:
            rid = _write_minimal_request_row(log)
            _request_id_context.set(rid)
        b = behavior.get(body.get("model"))
        if isinstance(b, HTTPException):
            raise b
        return b

    monkeypatch.setattr(app_module, "_dispatch_nonstream", _stub)
    return seen


# ---------- empty / pass-through path --------------------------------------


def test_no_candidates_invokes_inner_once_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """candidates=() is the pass-through hatch — wrapper calls inner once
    and returns its result without any cell-retry bookkeeping."""
    seen = _install_inner_stub(monkeypatch, log=None, behavior={"original-model": {"ok": True}})
    body = {"model": "original-model"}
    out = asyncio.run(
        _dispatch_nonstream_with_cell_retry(
            body,
            candidates=(),
            usage_log=None,
            model="original-model",
            route_name="/v1/responses",
            backends_list=[],
            preferred_id=None,
            session_id=None,
            session_registry=object(),
            call=None,
        )
    )
    assert out == {"ok": True}
    # The wrapper did NOT mutate body; the inner was called with the body
    # exactly as passed.
    assert len(seen) == 1
    assert seen[0]["model"] == "original-model"


# ---------- success on first cell ------------------------------------------


def test_success_on_first_cell_does_not_record_routing_attempts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Single-attempt success is the uninteresting common case; keep the
    sibling table to the reroute cohort so "how often did we reroute?"
    stays a one-line query.
    """
    log = UsageLog(tmp_path / "u.sqlite")
    seen = _install_inner_stub(monkeypatch, log=log, behavior={"model-a": {"served": "a"}})
    body = {"model": "auto"}
    out = asyncio.run(
        _dispatch_nonstream_with_cell_retry(
            body,
            candidates=(CELLS[0], CELLS[1], CELLS[2]),
            usage_log=log,
            model="auto",
            route_name="/v1/responses",
            backends_list=[],
            preferred_id=None,
            session_id=None,
            session_registry=object(),
            call=None,
        )
    )
    assert out == {"served": "a"}
    assert seen == [{"model": "model-a", "effort": "high"}]
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    n = conn.execute("SELECT COUNT(*) FROM request_routing_attempts").fetchone()[0]
    assert n == 0  # single-attempt → no sibling row


# ---------- 5xx → reroute ---------------------------------------------------


def test_reroutes_to_next_cell_on_5xx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Primary cell's backend pool errors with 503 → wrapper rewrites body
    to candidates[1] and retries. Both attempts logged to sibling table.
    """
    log = UsageLog(tmp_path / "u.sqlite")
    seen = _install_inner_stub(
        monkeypatch,
        log=log,
        behavior={
            "model-a": HTTPException(status_code=503, detail="all backends exhausted"),
            "model-b": {"served": "b"},
        },
    )
    body = {"model": "auto"}
    out = asyncio.run(
        _dispatch_nonstream_with_cell_retry(
            body,
            candidates=(CELLS[0], CELLS[1], CELLS[2]),
            usage_log=log,
            model="auto",
            route_name="/v1/responses",
            backends_list=[],
            preferred_id=None,
            session_id=None,
            session_registry=object(),
            call=None,
        )
    )
    assert out == {"served": "b"}
    assert [s["model"] for s in seen] == ["model-a", "model-b"]
    assert [s["effort"] for s in seen] == ["high", "medium"]
    # Both attempts in sibling table — the success row for cell-1 plus
    # the failure row for cell-0.
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    rows = conn.execute(
        "SELECT attempt_idx, model, reasoning_effort, status, classification"
        " FROM request_routing_attempts ORDER BY attempt_idx"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0] == (0, "model-a", "high", 503, "retried_next_cell")
    assert rows[1] == (1, "model-b", "medium", 200, "ok")


# ---------- 4xx → propagate without retry ----------------------------------


def test_4xx_propagates_without_trying_next_cell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Non-retryable errors (auth failure, malformed input) are not a
    routing problem; retrying with a different model won't help."""
    log = UsageLog(tmp_path / "u.sqlite")
    seen = _install_inner_stub(
        monkeypatch,
        log=log,
        behavior={
            "model-a": HTTPException(status_code=401, detail="auth failed"),
            "model-b": {"served": "b"},  # would succeed if tried, but mustn't be
        },
    )
    body = {"model": "auto"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream_with_cell_retry(
                body,
                candidates=(CELLS[0], CELLS[1]),
                usage_log=log,
                model="auto",
                route_name="/v1/responses",
                backends_list=[],
                preferred_id=None,
                session_id=None,
                session_registry=object(),
                call=None,
            )
        )
    assert exc_info.value.status_code == 401
    # Only the failing cell was tried; second cell never reached.
    assert [s["model"] for s in seen] == ["model-a"]
    # Persistence still happens — failures with cell-level history are
    # worth keeping even on early bail-out.
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    rows = conn.execute("SELECT attempt_idx, status, classification FROM request_routing_attempts").fetchall()
    assert rows == [(0, 401, "failed")]


def test_413_payload_too_large_propagates_without_trying_next_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 413 from a backend is a client/request-shape problem, not a
    transient routing problem (e.g. an upstream that rejects an over-context
    payload). Cell-level retry must not walk to another cell and re-burn the
    same oversized request."""
    log = UsageLog(tmp_path / "u.sqlite")
    seen = _install_inner_stub(
        monkeypatch,
        log=log,
        behavior={
            "model-a": HTTPException(
                status_code=413,
                detail="upstream rejected request: payload too large",
            ),
            "model-b": {"served": "b"},
        },
    )
    body = {"model": "auto"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream_with_cell_retry(
                body,
                candidates=(CELLS[0], CELLS[1]),
                usage_log=log,
                model="auto",
                route_name="/v1/responses",
                backends_list=[],
                preferred_id=None,
                session_id=None,
                session_registry=object(),
                call=None,
            )
        )
    assert exc_info.value.status_code == 413
    assert [s["model"] for s in seen] == ["model-a"]
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    rows = conn.execute(
        "SELECT attempt_idx, status, classification, error_message FROM request_routing_attempts"
    ).fetchall()
    assert rows == [
        (
            0,
            413,
            "failed",
            "upstream rejected request: payload too large",
        )
    ]


# ---------- cap at MAX_CELL_ATTEMPTS ---------------------------------------


def test_caps_at_max_cell_attempts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Even if the recommender supplies more candidates, we walk at most
    MAX_CELL_ATTEMPTS of them. Three cells failing in a row → 5xx
    propagated; the 4th candidate is never touched."""
    log = UsageLog(tmp_path / "u.sqlite")
    all_503 = HTTPException(status_code=503, detail="exhausted")
    seen = _install_inner_stub(
        monkeypatch,
        log=log,
        behavior={
            "model-a": all_503,
            "model-b": all_503,
            "model-c": all_503,
            "model-d": {"unreachable": True},
        },
    )
    body = {"model": "auto"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream_with_cell_retry(
                body,
                candidates=tuple(CELLS),  # 4 cells provided, only 3 should run
                usage_log=log,
                model="auto",
                route_name="/v1/responses",
                backends_list=[],
                preferred_id=None,
                session_id=None,
                session_registry=object(),
                call=None,
            )
        )
    assert exc_info.value.status_code == 503
    assert len(seen) == MAX_CELL_ATTEMPTS == 3
    assert [s["model"] for s in seen] == ["model-a", "model-b", "model-c"]
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    rows = conn.execute(
        "SELECT attempt_idx, model, classification FROM request_routing_attempts ORDER BY attempt_idx"
    ).fetchall()
    assert len(rows) == 3
    # First two were rerouted, last one was the terminal failure.
    assert rows[0][2] == "retried_next_cell"
    assert rows[1][2] == "retried_next_cell"
    assert rows[2][2] == "failed"


def test_retry_budget_http_does_not_reroute_to_next_cell(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A budget-exhausted 503 is terminal for the whole request, not another
    retryable cell failure to walk past."""
    log = UsageLog(tmp_path / "u.sqlite")
    seen = _install_inner_stub(
        monkeypatch,
        log=log,
        behavior={
            "model-a": _retry_budget_http("wall-clock budget exhausted"),
            "model-b": {"served": "b"},
        },
    )
    body = {"model": "auto"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream_with_cell_retry(
                body,
                candidates=(CELLS[0], CELLS[1]),
                usage_log=log,
                model="auto",
                route_name="/v1/responses",
                backends_list=[],
                preferred_id=None,
                session_id=None,
                session_registry=object(),
                call=None,
                dispatch_budget=_DispatchRetryBudget.from_config(seconds=360.0, max_backend_attempts=4),
            )
        )
    assert exc_info.value.status_code == 503
    assert "dispatch retry budget exhausted" in str(exc_info.value.detail)
    assert [s["model"] for s in seen] == ["model-a"]


def test_expired_retry_budget_short_circuits_before_any_cell_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An already-expired request budget should fail closed before burning
    another upstream attempt."""
    seen = _install_inner_stub(monkeypatch, log=None, behavior={"model-a": {"served": "a"}})
    budget = _DispatchRetryBudget(deadline_ts=time.monotonic() - 1.0, max_backend_attempts=4)
    body = {"model": "auto"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream_with_cell_retry(
                body,
                candidates=(CELLS[0],),
                usage_log=None,
                model="auto",
                route_name="/v1/responses",
                backends_list=[],
                preferred_id=None,
                session_id=None,
                session_registry=object(),
                call=None,
                dispatch_budget=budget,
            )
        )
    assert exc_info.value.status_code == 503
    assert seen == []


def test_inner_dispatch_wall_clock_budget_cancels_hanging_backend(tmp_path: Path) -> None:
    """The inner backend loop must be bounded by the shared wall-clock budget;
    otherwise one slow local backend can consume its whole transport timeout
    before the caller sees a response."""
    log = UsageLog(tmp_path / "u.sqlite")
    backend = InMemoryFakeBackend(id="local", advertised_models=frozenset({"model-a"}))
    started = False

    async def slow_call(b: InMemoryFakeBackend, body: dict[str, Any], handle: CallHandle) -> dict[str, Any]:
        nonlocal started
        del b, body, handle
        started = True
        await asyncio.sleep(1.0)
        return {"unreachable": True}

    body = {"model": "model-a"}
    before = time.monotonic()
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream(
                body,
                model="model-a",
                route_name="/v1/responses",
                backends_list=[backend],
                preferred_id=None,
                session_id=None,
                session_registry=SessionRegistry(),
                usage_log=log,
                call=slow_call,
                dispatch_budget=_DispatchRetryBudget.from_config(seconds=0.1, max_backend_attempts=4),
            )
        )
    elapsed = time.monotonic() - before
    assert started is True
    assert elapsed < 0.5
    assert exc_info.value.status_code == 503
    assert "dispatch retry budget exhausted" in str(exc_info.value.detail)
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute("SELECT backend_id, status, classification FROM requests").fetchone()
    assert row == ("local", 503, "transient")


def test_inner_dispatch_backend_attempt_cap_stops_before_second_backend(
    tmp_path: Path,
) -> None:
    """The max-backend-attempt cap is shared across the backend failover loop,
    so retryable errors cannot walk an unbounded backend list."""
    log = UsageLog(tmp_path / "u.sqlite")
    first = InMemoryFakeBackend(id="first", advertised_models=frozenset({"model-a"}))
    second = InMemoryFakeBackend(id="second", advertised_models=frozenset({"model-a"}))
    calls: list[str] = []

    async def failing_call(backend: InMemoryFakeBackend, body: dict[str, Any], handle: CallHandle) -> dict[str, Any]:
        del body, handle
        calls.append(backend.id)
        raise BackendError(classification="transient", message=f"{backend.id} failed")

    body = {"model": "model-a"}
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(
            _dispatch_nonstream(
                body,
                model="model-a",
                route_name="/v1/responses",
                backends_list=[first, second],
                preferred_id=None,
                session_id=None,
                session_registry=SessionRegistry(),
                usage_log=log,
                call=failing_call,
                dispatch_budget=_DispatchRetryBudget.from_config(seconds=360.0, max_backend_attempts=1),
            )
        )
    assert calls == ["first"]
    assert exc_info.value.status_code == 503
    assert "backend attempt cap reached" in str(exc_info.value.detail)
