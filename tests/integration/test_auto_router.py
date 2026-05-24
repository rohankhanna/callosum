from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

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


def test_auto_learning_rewrites_body_and_logs_router_columns(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    with TestClient(create_app(backends=[backend], usage_log=log)) as client:
        response = client.post(
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


def test_auto_learning_distributes_across_cells(tmp_path: Path) -> None:
    # Round-robin: four sequential calls into a fresh log should hit four
    # different cells (the explorer picks min-coverage; coverage updates after
    # each call lands in the log).
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    served: list[tuple[str, str]] = []
    with TestClient(create_app(backends=[backend], usage_log=log)) as client:
        for _ in range(4):
            r = client.post("/v1/responses", json={"model": "auto-learning", "input": []})
            assert r.status_code == 200

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    served = conn.execute("SELECT model, reasoning_effort FROM requests ORDER BY id").fetchall()
    # Four distinct cells visited.
    assert len(set(served)) == 4


def test_auto_returns_503_with_not_trained_message() -> None:
    backend = _backend()
    with TestClient(create_app(backends=[backend])) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "auto", "input": []},
        )
    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "not ready" in detail.lower() or "not trained" in detail.lower()


def test_synthetic_virtual_model_tags_rows_separately(tmp_path: Path) -> None:
    # `auto-learning-synthetic` should rewrite to a real cell AND log
    # routing_mode='auto-learning-synthetic' so synthetic rows don't double-
    # count organic coverage.
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    with TestClient(create_app(backends=[backend], usage_log=log)) as client:
        r = client.post(
            "/v1/responses",
            json={"model": "auto-learning-synthetic", "input": []},
        )
    assert r.status_code == 200

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT requested_model, routing_mode, model, reasoning_effort FROM requests"
    ).fetchone()
    assert row is not None
    requested_model, routing_mode, model, reasoning = row
    assert requested_model == "auto-learning-synthetic"
    assert routing_mode == "auto-learning-synthetic"
    assert model in DEFAULT_MODELS
    assert reasoning in REASONING_LEVELS


def test_synthetic_and_organic_have_independent_coverage(tmp_path: Path) -> None:
    # Fire one organic + one synthetic. The synthetic explorer's coverage
    # should NOT count the organic call (and vice versa). After both calls,
    # each tier should have advanced exactly one cell.
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    with TestClient(create_app(backends=[backend], usage_log=log)) as client:
        client.post("/v1/responses", json={"model": "auto-learning", "input": []})
        client.post("/v1/responses", json={"model": "auto-learning-synthetic", "input": []})

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    rows = conn.execute(
        "SELECT routing_mode, model, reasoning_effort FROM requests ORDER BY id"
    ).fetchall()
    assert len(rows) == 2
    organic = [r for r in rows if r[0] == "auto-learning"]
    synthetic = [r for r in rows if r[0] == "auto-learning-synthetic"]
    # Both tiers picked the first cell (since both started from zero coverage).
    assert organic[0][1:] == synthetic[0][1:]


def test_pass_through_request_records_routing_mode(tmp_path: Path) -> None:
    # Sanity: a normal model name should record routing_mode='pass-through'
    # and leave requested_* equal to served_*.
    log = UsageLog(tmp_path / "u.sqlite")
    backend = _backend()
    with TestClient(create_app(backends=[backend], usage_log=log)) as client:
        response = client.post(
            "/v1/responses",
            json={"model": "model-a0e7", "input": [], "reasoning": {"effort": "high"}},
        )
    assert response.status_code == 200

    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT requested_model, requested_reasoning_effort, routing_mode,"
        " model, reasoning_effort FROM requests"
    ).fetchone()
    assert row == ("model-a0e7", "high", "pass-through", "model-a0e7", "high")
