from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

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
async def test_synthetic_virtual_model_tags_rows_separately(tmp_path: Path) -> None:
    # `auto-learning-synthetic` should rewrite to a real cell AND log
    # routing_mode='auto-learning-synthetic' so synthetic rows don't double-
    # count organic coverage.
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        r = await client.post(
            "/v1/responses",
            json={"model": "auto-learning-synthetic", "input": []},
        )
    assert r.status_code == 200

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute("SELECT requested_model, routing_mode, model, reasoning_effort FROM requests").fetchone()
    assert row is not None
    requested_model, routing_mode, model, reasoning = row
    assert requested_model == "auto-learning-synthetic"
    assert routing_mode == "auto-learning-synthetic"
    assert model in DEFAULT_MODELS
    assert reasoning in REASONING_LEVELS


@pytest.mark.asyncio
async def test_synthetic_and_organic_are_logged_as_distinct_routing_modes(tmp_path: Path) -> None:
    # Fire one organic + one synthetic. Both go through the same recommender,
    # but the request log must preserve which virtual-model name the client
    # sent so synthetic-tier velocity can be measured separately from organic
    # traffic. With no recommender configured, both fall back to the same
    # cheap cell — so they'll match on (model, effort) but differ in routing_mode.
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
        await client.post("/v1/responses", json={"model": "auto-learning-synthetic", "input": []})

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    rows = conn.execute("SELECT routing_mode, model, reasoning_effort FROM requests ORDER BY id").fetchall()
    assert len(rows) == 2
    organic = [r for r in rows if r[0] == "auto-learning"]
    synthetic = [r for r in rows if r[0] == "auto-learning-synthetic"]
    assert len(organic) == 1 and len(synthetic) == 1
    # No recommender → both fall back to the same cheap cell.
    assert organic[0][1:] == synthetic[0][1:]


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
async def test_synthetic_exploration_spreads_across_cells(tmp_path: Path) -> None:
    """Synthetic traffic must sample the least-covered cell each time, so a
    burst of synthetics spreads across the grid instead of collapsing to the
    single cost-cheapest cell."""
    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        for _ in range(16):
            r = await client.post(
                "/v1/responses",
                json={"model": "auto-learning-synthetic", "input": []},
            )
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning-synthetic")
    assert len(served) == 16
    # Round-robin over under-sampled arms => broad coverage, not a collapse.
    distinct = set(served)
    assert len(distinct) >= 10


@pytest.mark.asyncio
async def test_organic_traffic_does_not_explore(tmp_path: Path) -> None:
    """Organic auto traffic keeps the cost-optimal pick — it must NOT spread
    across cells the way synthetic exploration does."""
    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        for _ in range(8):
            r = await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning")
    assert len(served) == 8
    # No exploration => every organic request collapses to the same cell.
    assert len(set(served)) == 1


@pytest.mark.asyncio
async def test_exploration_can_be_disabled(tmp_path: Path) -> None:
    """With exploration_enabled=False, even synthetic traffic collapses to the
    cost-cheapest cell (legacy behavior)."""
    from callosum.config import AutoRouterConfig

    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    backend = _backend()
    cfg = AutoRouterConfig(exploration_enabled=False)
    async with _client(backends=[backend], usage_log=log, auto_router_config=cfg) as client:
        for _ in range(8):
            r = await client.post(
                "/v1/responses",
                json={"model": "auto-learning-synthetic", "input": []},
            )
            assert r.status_code == 200
    served = _served_cells(db, "auto-learning-synthetic")
    assert len(served) == 8
    assert len(set(served)) == 1


# ---------- measured dynamic cost_rank ------------------------------------


def _seed_quota_rows(db: Path, model: str, n: int, *, before: int, after: int) -> None:
    import time as _time

    now = _time.time()
    conn = sqlite3.connect(db)
    try:
        conn.executemany(
            "INSERT INTO requests"
            " (ts_start, ts_end, latency_ms, route, stream, backend_id, status,"
            "  model, quota_reset_crossover, weekly_used_percent_before,"
            "  weekly_used_percent_after)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (now - 100, now - 99, 1000, "/v1/responses", 0, "primary", 200, model, 0, before, after)
                for _ in range(n)
            ],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_measured_cost_rank_steers_organic_routing(tmp_path: Path) -> None:
    """Organic routing should prefer the model that MEASURABLY burns less
    weekly quota, not a flat-cost arbitrary pick. Seed the request log so one
    model is demonstrably cheaper, then confirm organic traffic lands there."""
    db = tmp_path / "u.sqlite"
    log = UsageLog(db)
    # model-a0c3 burns +1% per call; model-a0e6 burns +5%. Both clear the
    # min-nonzero-samples threshold; the rest of the grid stays unmeasured.
    _seed_quota_rows(db, "model-a0c3", 12, before=10, after=11)
    _seed_quota_rows(db, "model-a0e6", 12, before=10, after=15)
    backend = _backend()
    async with _client(backends=[backend], usage_log=log) as client:
        r = await client.post("/v1/responses", json={"model": "auto-learning", "input": []})
        assert r.status_code == 200
        served = r.json()["model"]
    # Cheapest measured burn wins under the cold-start (uniform) predictor.
    assert served == "model-a0c3"

