from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from callosum import app as app_module
from callosum.app import create_app
from callosum.cell_grid import DEFAULT_MODELS, REASONING_LEVELS, build_cells
from callosum.fakes import InMemoryFakeBackend
from callosum.usage_log import UsageLog


def _backend() -> InMemoryFakeBackend:
    # The router rewrites the body BEFORE the selector runs, so the backend
    # must advertise the real cell-grid models (not "auto-learning").
    return InMemoryFakeBackend(
        id="primary",
        advertised_models=frozenset(DEFAULT_MODELS),
    )


@asynccontextmanager
async def _client(**kwargs: object) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=create_app(**kwargs))
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


@pytest.mark.asyncio
async def test_auto_learning_rewrites_body_and_logs_router_columns(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "auto-learning", "input": []},
        )
    assert response.status_code == 200
    served_model = response.json()["model"]
    # The router picks the least-sampled cell. Whatever it picks must be a real
    # cell from the grid — never the virtual model name.
    cells = build_cells()
    assert served_model in {c.model for c in cells}
    assert served_model != "auto-learning"

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT requested_model, requested_reasoning_effort, routing_mode,"
        " model, reasoning_effort, status FROM requests"
    ).fetchone()
    assert row is not None
    requested_model, requested_reasoning, routing_mode, model, reasoning, status = row
    assert requested_model == "auto-learning"
    assert requested_reasoning is None  # caller didn't specify one
    assert routing_mode == "auto-learning"
    assert status == 200
    # The served (model, reasoning) must be a real grid cell.
    assert model in DEFAULT_MODELS
    assert reasoning in REASONING_LEVELS


@pytest.mark.asyncio
async def test_explicit_model_request_routes_through_router(tmp_path: Path) -> None:
    """All requests route through the learning router now — explicit-model
    requests included. routing_mode records the originally-requested name
    (for provenance); model/reasoning_effort record what the router
    actually chose. With a single fake backend serving model-a0e7, the
    capability filter leaves model-a0e7 cells, cost selector picks one."""
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        response = await client.post(
            "/v1/responses",
            json={"model": "model-a0e7", "input": [], "reasoning": {"effort": "high"}},
        )
    assert response.status_code == 200

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT requested_model, requested_reasoning_effort, routing_mode, model, reasoning_effort FROM requests"
    ).fetchone()
    requested_model, requested_reasoning, routing_mode, model, reasoning = row
    assert requested_model == "model-a0e7"
    assert requested_reasoning == "high"
    assert routing_mode == "model-a0e7"
    # Router picked a real cell from the grid. The exact (model, effort)
    # depends on cost ordering — we just assert it IS a grid cell.
    assert model in DEFAULT_MODELS
    assert reasoning in REASONING_LEVELS


# ---------- arm-level exploration -----------------------------------------


def _served_cells(db: Path, routing_mode: str) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db)
    try:
        return conn.execute(
            "SELECT model, reasoning_effort FROM requests WHERE routing_mode = ? ORDER BY id",
            (routing_mode,),
        ).fetchall()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_organic_cold_start_explores_across_cells(tmp_path: Path) -> None:
    """Cold start has no learned quality signal, so organic auto traffic
    EXPLORES — it spreads randomly across compatible cells instead of
    collapsing onto one. This replaces the old cost-steered single-cell
    collapse: 'cheapest capable' selection is deferred to the exploit phase
    once a quality model exists. (Effort/cost are no longer guessed cold.)"""
    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        for _ in range(30):
            r = await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning")
    assert len(served) == 30
    # Random cold-start exploration spreads across the grid, not a single cell.
    assert len(set(served)) > 1


def _effective_modes(db: Path) -> list[str]:
    conn = sqlite3.connect(db)
    try:
        return [r[0] for r in conn.execute("SELECT effective_routing_mode FROM requests ORDER BY id").fetchall()]
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_xhigh_cap_removes_xhigh_from_auto_candidates(tmp_path: Path) -> None:
    """The temporary xhigh guardrail caps automatic routing once recent
    successful xhigh traffic is already at 1%, while leaving non-xhigh cells
    available for the router and quota layer."""
    from callosum.config import AutoRouterConfig

    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    conn = sqlite3.connect(db)
    import time as _time

    now = _time.time()
    try:
        conn.executemany(
            "INSERT INTO requests"
            " (ts_start, ts_end, latency_ms, route, stream, backend_id, status, model, reasoning_effort)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    now + i,
                    now + i + 0.1,
                    200,
                    "/v1/responses",
                    0,
                    "seed",
                    200,
                    "model-a0e7",
                    "low",
                )
                for i in range(100)
            ]
            + [
                (
                    now + 300,
                    now + 300.1,
                    100,
                    "/v1/responses",
                    0,
                    "seed",
                    200,
                    "model-a0e7",
                    "xhigh",
                )
            ],
        )
        conn.commit()
    finally:
        conn.close()
    cfg = AutoRouterConfig(xhigh_cap_enabled=True, xhigh_cap_pct=0.01, xhigh_cap_window_seconds=604_800)
    async with _client(backends=[backend], usage_log=log, auto_router_config=cfg) as client:
        r = await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
        assert r.status_code == 200
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT model, reasoning_effort FROM requests WHERE routing_mode = 'auto-learning' ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[1] != "xhigh"


@pytest.mark.asyncio
async def test_quota_enabled_spreads_and_stamps_provenance(tmp_path: Path) -> None:
    """With the exploration quota on, a burst of eligible (tool-less, easy)
    turns deficit-fills across the grid instead of collapsing, and forced turns
    are stamped effective_routing_mode='quota_explore' ()."""
    from callosum.config import AutoRouterConfig

    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    cfg = AutoRouterConfig(exploration_quota_enabled=True)
    async with _client(backends=[backend], usage_log=log, auto_router_config=cfg) as client:
        for _ in range(16):
            r = await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning")
    assert len(served) == 16
    # Deficit-fill from zero coverage => broad spread, not a single-cell collapse.
    assert len(set(served)) >= 8
    # At least the forced turns carry the quota provenance marker.
    assert _effective_modes(db).count("quota_explore") >= 1


@pytest.mark.asyncio
async def test_quota_forces_tool_turns_too(tmp_path: Path) -> None:
    """The quota has no tools exception: tool-bearing turns (all real Codex
    traffic) are forced toward under-floor cells like any other, so they spread
    and carry the quota_explore marker ()."""
    from callosum.config import AutoRouterConfig

    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    cfg = AutoRouterConfig(exploration_quota_enabled=True)
    tools = [{"type": "function", "name": "noop", "parameters": {"type": "object", "properties": {}}}]
    async with _client(backends=[backend], usage_log=log, auto_router_config=cfg) as client:
        for _ in range(16):
            r = await client.post("/v1/responses", json={"model": "auto-learning", "input": [], "tools": tools})
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning")
    assert len(served) == 16
    assert len(set(served)) >= 8  # forced despite tools => spreads
    assert _effective_modes(db).count("quota_explore") >= 1


@pytest.mark.asyncio
async def test_quota_skips_infeasible_under_floor_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Quota forcing should not redirect onto an under-floor cell the latency
    guard already considers structurally too slow for this request."""
    from callosum.config import AutoRouterConfig

    def _fake_feasible(body: dict[str, object], cell) -> bool:
        return cell.model != "model-a0c3"

    monkeypatch.setattr(app_module, "_cell_is_latency_feasible_for_body", _fake_feasible)

    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            "INSERT INTO requests "
            "(ts_start, ts_end, latency_ms, route, stream, backend_id, model, reasoning_effort, status, classification, routing_mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (0.0, 0.1, 100, "/v1/responses", 0, "seed", "model-a0e7", "low", 200, "ok", "auto-learning"),
                (0.2, 0.3, 100, "/v1/responses", 0, "seed", "model-a0e7", "medium", 200, "ok", "auto-learning"),
            ],
        )
        conn.commit()
    finally:
        conn.close()

    cfg = AutoRouterConfig(exploration_quota_enabled=True)
    async with _client(backends=[backend], usage_log=log, auto_router_config=cfg) as client:
        response = await client.post("/v1/responses", json={"model": "model-a0e7", "input": []})
    assert response.status_code == 200

    conn = sqlite3.connect(db)
    try:
        rows = conn.execute(
            "SELECT effective_routing_mode, model, reasoning_effort FROM requests ORDER BY id"
        ).fetchall()
    finally:
        conn.close()
    assert rows
    quota_rows = [(model, reasoning) for effective_mode, model, reasoning in rows if effective_mode == "quota_explore"]
    assert all(model != "model-a0c3" for model, _reasoning in quota_rows)


# ---------- measured dynamic cost_rank ------------------------------------


def _seed_quota_rows(db: Path, model: str, n: int, *, before: int, after: int) -> None:
    import time as _time

    now = _time.time()
    delta = after - before
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            "INSERT INTO requests"
            " (ts_start, ts_end, latency_ms, route, stream, backend_id, status,"
            "  model, quota_reset_crossover, weekly_used_percent_before,"
            "  weekly_used_percent_after, prompt_tokens, completion_tokens, total_tokens)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    now - 100 + i * 2,
                    now - 99 + i * 2,
                    1000,
                    "/v1/responses",
                    0,
                    f"seed-{model}",
                    200,
                    model,
                    0,
                    before + i * delta,
                    before + (i + 1) * delta,
                    900,
                    100,
                    1000,
                )
                for i in range(n)
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_cost_rank_does_not_steer_cold_start_exploration(tmp_path: Path) -> None:
    """Measured cost rank steers the EXPLOIT phase (cheapest capable cell once
    a quality model exists), NOT cold start. Even with one model demonstrably
    cheaper, cold-start organic traffic still explores randomly rather than
    collapsing onto the cheapest. (Cost-steering in exploit is covered by the
    cost-model unit tests; an exploit-mode integration test lands with P3.)"""
    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    # model-a0c3 burns +1% per call; model-a0e6 burns +5%. Both clear the
    # min-nonzero-samples threshold; the rest of the grid stays unmeasured.
    _seed_quota_rows(db, "model-a0c3", 12, before=10, after=11)
    _seed_quota_rows(db, "model-a0e6", 12, before=22, after=27)
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        for _ in range(30):
            r = await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning")
    # Cold start explores; it does NOT collapse onto the cheapest measured model.
    assert len(set(served)) > 1
