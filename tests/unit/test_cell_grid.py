from __future__ import annotations

import sqlite3
from pathlib import Path

from codex_proxy.cell_grid import (
    DEFAULT_MODELS,
    REASONING_LEVELS,
    Cell,
    CellCoverage,
    build_cells,
    coverage_from_db,
)


def test_build_cells_is_cross_product() -> None:
    cells = build_cells()
    assert len(cells) == len(DEFAULT_MODELS) * len(REASONING_LEVELS)
    assert len(set(cells)) == len(cells)
    # Models-major ordering: first 4 cells share the first model.
    assert all(c.model == DEFAULT_MODELS[0] for c in cells[: len(REASONING_LEVELS)])


def test_least_sampled_picks_lowest_count() -> None:
    cells = build_cells()
    counts = {c: 5 for c in cells}
    counts[cells[3]] = 0
    counts[cells[7]] = 1
    cov = CellCoverage(counts=counts)
    assert cov.least_sampled(cells) == cells[3]


def test_least_sampled_breaks_ties_by_input_order() -> None:
    cells = build_cells()
    counts = {c: 0 for c in cells}
    cov = CellCoverage(counts=counts)
    # All zero — first cell wins by stable order.
    assert cov.least_sampled(cells) == cells[0]


def test_cells_below_target_returns_only_under() -> None:
    cells = build_cells()
    counts = {c: (50 if i % 2 == 0 else 10) for i, c in enumerate(cells)}
    cov = CellCoverage(counts=counts)
    under = cov.cells_below(target=20, cells=cells)
    assert under == [c for i, c in enumerate(cells) if i % 2 == 1]


def test_coverage_from_missing_db_is_all_zero(tmp_path: Path) -> None:
    cells = build_cells()
    cov = coverage_from_db(tmp_path / "nonexistent.sqlite", cells)
    assert all(cov.counts[c] == 0 for c in cells)


def test_coverage_only_counts_auto_learning_successes(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT, reasoning_effort TEXT, routing_mode TEXT, status INTEGER
        )
        """
    )
    conn.executemany(
        "INSERT INTO requests (model, reasoning_effort, routing_mode, status) VALUES (?, ?, ?, ?)",
        [
            # 3 auto-learning successes for (model-a0c3, low)
            ("model-a0c3", "low", "auto-learning", 200),
            ("model-a0c3", "low", "auto-learning", 200),
            ("model-a0c3", "low", "auto-learning", 200),
            # 1 auto-learning failure (excluded by status = 200 filter)
            ("model-a0c3", "low", "auto-learning", 429),
            # 1 pass-through success (excluded — not exploration data)
            ("model-a0c3", "low", "pass-through", 200),
            # 2 different cell
            ("model-a0e7", "xhigh", "auto-learning", 200),
            ("model-a0e7", "xhigh", "auto-learning", 200),
            # 1 unknown cell (model not in grid) — should be silently ignored
            ("totally-made-up-model", "medium", "auto-learning", 200),
        ],
    )
    conn.commit()
    conn.close()
    cells = build_cells()
    cov = coverage_from_db(db, cells)
    assert cov.counts[Cell(model="model-a0c3", reasoning_effort="low")] == 3
    assert cov.counts[Cell(model="model-a0e7", reasoning_effort="xhigh")] == 2
    # Cells not appearing in the data are zero, not missing.
    assert cov.counts[Cell(model="model-a0e6", reasoning_effort="medium")] == 0
    assert cov.total_samples() == 5
