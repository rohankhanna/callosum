"""Tests for LearnedModelRouter and EfficiencyModel."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from codex_proxy.cell_grid import Cell, build_cells
from codex_proxy.efficiency_model import EfficiencyModel
from codex_proxy.router import LearnedModelRouter


def _seed_usage_log(db: Path, rows: list[tuple[str, str, int, str, int]]) -> None:
    """Seed a usage log with requests table rows.

    Each row: (model, reasoning_effort, total_tokens, routing_mode, status)
    """
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT, reasoning_effort TEXT, total_tokens INTEGER,
            routing_mode TEXT, status INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO requests (model, reasoning_effort, total_tokens, routing_mode, status)"
        " VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_efficiency_model_not_ready_with_no_data(tmp_path: Path) -> None:
    """With no database or no rows, is_ready = False."""
    db = tmp_path / "no_data.sqlite"
    cells = build_cells()
    model = EfficiencyModel.from_db(db, cells)
    assert not model.is_ready
    assert len(model.scores) == 0


def test_efficiency_model_ready_when_min_samples_met(tmp_path: Path) -> None:
    """When all cells have >= min_samples, is_ready = True."""
    db = tmp_path / "ready.sqlite"
    cells = build_cells()
    # Seed every cell with exactly min_samples
    min_samples = 30
    rows = [
        (c.model, c.reasoning_effort, 1000, "auto-learning", 200)
        for c in cells for _ in range(min_samples)
    ]
    _seed_usage_log(db, rows)

    model = EfficiencyModel.from_db(db, cells, min_samples_per_cell=min_samples)
    assert model.is_ready
    assert len(model.scores) == len(cells)


def test_efficiency_model_not_ready_when_partial_data(tmp_path: Path) -> None:
    """If some cells lack min_samples, is_ready = False."""
    db = tmp_path / "partial.sqlite"
    cells = build_cells()
    min_samples = 30
    # Seed only the first cell
    rows = [
        (cells[0].model, cells[0].reasoning_effort, 1000, "auto-learning", 200)
        for _ in range(min_samples)
    ]
    _seed_usage_log(db, rows)

    model = EfficiencyModel.from_db(db, cells, min_samples_per_cell=min_samples)
    assert not model.is_ready


def test_efficiency_model_picks_cheapest_cell(tmp_path: Path) -> None:
    """best_cell() returns the cell with lowest avg_tokens from candidates."""
    db = tmp_path / "cheapest.sqlite"
    cells = build_cells()
    min_samples = 5

    # Seed all cells so is_ready=True, but with different token costs
    cheap_cell = cells[0]
    expensive_cell = cells[1]
    rows = [
        (c.model, c.reasoning_effort, 100 if c == cheap_cell else 500 if c == expensive_cell else 250,
         "auto-learning", 200)
        for c in cells for _ in range(min_samples)
    ]
    _seed_usage_log(db, rows)

    model = EfficiencyModel.from_db(db, cells, min_samples_per_cell=min_samples)
    assert model.is_ready
    best = model.best_cell(cells)
    assert best == cheap_cell


def test_efficiency_model_ignores_failed_requests(tmp_path: Path) -> None:
    """Requests with status != 200 are excluded."""
    db = tmp_path / "ignore_failed.sqlite"
    cells = build_cells()
    min_samples = 5

    cell = cells[0]
    rows = [
        # Successful requests with 100 tokens
        (cell.model, cell.reasoning_effort, 100, "auto-learning", 200)
        for _ in range(min_samples)
    ] + [
        # Failed requests with 1000 tokens — should be ignored
        (cell.model, cell.reasoning_effort, 1000, "auto-learning", 429)
        for _ in range(10)
    ]
    _seed_usage_log(db, rows)

    model = EfficiencyModel.from_db(db, cells, min_samples_per_cell=min_samples)
    # avg_tokens should be ~100 (only the 200 status rows count)
    avg = model.scores.get((cell.model, cell.reasoning_effort), 0)
    assert 90 < avg < 110


def test_exploiter_not_trained_when_no_data(tmp_path: Path) -> None:
    """choose() raises NotTrained when model is not ready."""
    db = tmp_path / "empty.sqlite"
    router = LearnedModelRouter(usage_log_path=db)

    with pytest.raises(LearnedModelRouter.NotTrained):
        router.choose()


def test_exploiter_transitions_to_ready_after_fit(tmp_path: Path) -> None:
    """After fit() with sufficient data, choose() no longer raises."""
    db = tmp_path / "train.sqlite"
    cells = build_cells()
    min_samples = 30

    # Seed with sufficient data
    rows = [
        (c.model, c.reasoning_effort, 1000, "auto-learning", 200)
        for c in cells for _ in range(min_samples)
    ]
    _seed_usage_log(db, rows)

    router = LearnedModelRouter(usage_log_path=db)
    assert router._model is None

    # fit() should populate the model
    router.fit(min_samples_per_cell=min_samples)
    assert router._model is not None
    assert router._model.is_ready

    # Now choose() should work
    decision = router.choose()
    assert decision.cell in cells
    assert "cheapest" in decision.reason


def test_exploiter_context_window_filtering(tmp_path: Path) -> None:
    """best_cell filters out cells with insufficient context window."""
    db = tmp_path / "context.sqlite"

    # Create cells with different context windows manually
    cell_small = Cell(model="model-a0e6", reasoning_effort="low", context_window=50000)
    cell_large = Cell(model="model-a0e7", reasoning_effort="low", context_window=200000)
    cells = [cell_small, cell_large]
    min_samples = 5

    rows = [
        (cell_small.model, cell_small.reasoning_effort, 100, "auto-learning", 200)
        for _ in range(min_samples)
    ] + [
        (cell_large.model, cell_large.reasoning_effort, 150, "auto-learning", 200)
        for _ in range(min_samples)
    ]
    _seed_usage_log(db, rows)

    model = EfficiencyModel.from_db(db, cells, min_samples_per_cell=min_samples)

    # With large session size, small-context cell should be filtered
    session_tokens = 60000  # Larger than cell_small's context window
    safety_margin = 8192
    best = model.best_cell(cells, session_prompt_tokens=session_tokens,
                          router_context_safety_margin=safety_margin)
    assert best == cell_large


def test_exploiter_respects_cells_fn(tmp_path: Path) -> None:
    """LearnedModelRouter uses cells_fn to get live cell grid."""
    db = tmp_path / "dynamic.sqlite"

    # Use a dynamic cells function
    pool: list[str] = ["model-a0e7"]
    def cells_fn():
        from codex_proxy.cell_grid import REASONING_LEVELS
        return [Cell(model=m, reasoning_effort=r) for m in pool for r in REASONING_LEVELS]

    router = LearnedModelRouter(usage_log_path=db, cells_fn=cells_fn)

    # With one model, fit should see 4 cells (1 model × 4 reasoning levels)
    cells = router._cells_fn()
    assert len(cells) == 4
