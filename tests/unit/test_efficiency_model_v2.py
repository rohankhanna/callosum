"""Tests for the v2 (per-complexity) cost model in EfficiencyModel.

When training data contains prompt_complexity_class, EfficiencyModel keeps
both a v1 per-cell average and a v2 per-(complexity, cell) average. The
router consults v2 when it can and falls back to v1 per-cell when it can't.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codex_proxy.cell_grid import Cell
from codex_proxy.efficiency_model import EfficiencyModel


def _make_db(path: Path, rows: list[tuple]) -> None:
    """rows = [(model, effort, total_tokens, prompt_complexity_class_or_None), ...]"""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT, reasoning_effort TEXT,
            total_tokens INTEGER, status INTEGER,
            routing_mode TEXT,
            prompt_complexity_class INTEGER
        )
        """
    )
    for model, effort, tokens, complexity in rows:
        conn.execute(
            "INSERT INTO requests(model, reasoning_effort, total_tokens, status, "
            "routing_mode, prompt_complexity_class) VALUES (?,?,?,?,?,?)",
            (model, effort, tokens, 200, "auto-learning", complexity),
        )
    conn.commit()
    conn.close()


def test_v2_routes_to_per_complexity_cheapest_cell(tmp_path: Path) -> None:
    """For complexity=1, the cheapest cell should differ from v1's overall cheapest."""
    db = tmp_path / "u.db"
    rows: list[tuple] = []
    # cell A: cheap on simple (1) but average overall
    rows += [("model-a", "low", 100, 1) for _ in range(30)]
    rows += [("model-a", "low", 800, 2) for _ in range(30)]
    rows += [("model-a", "low", 2000, 3) for _ in range(30)]
    # cell B: expensive on simple but cheaper overall (lots of cheap complex rows)
    rows += [("model-b", "low", 500, 1) for _ in range(30)]
    rows += [("model-b", "low", 200, 2) for _ in range(30)]
    rows += [("model-b", "low", 150, 3) for _ in range(30)]
    _make_db(db, rows)

    cells = [
        Cell(model="model-a", reasoning_effort="low", context_window=None),
        Cell(model="model-b", reasoning_effort="low", context_window=None),
    ]
    em = EfficiencyModel.from_db(db, cells, min_samples_per_cell=30)

    # v1: model-b is cheaper on average (283 vs 967).
    assert em.is_ready
    assert em.best_cell(cells).model == "model-b"

    # v2 complexity=1: model-a is cheaper (100 vs 500).
    assert em.has_complexity_data(1)
    assert em.best_cell(cells, complexity=1).model == "model-a"

    # v2 complexity=3: model-b is cheaper (150 vs 2000).
    assert em.best_cell(cells, complexity=3).model == "model-b"


def test_v2_falls_back_to_v1_when_bucket_is_empty(tmp_path: Path) -> None:
    """If no rows exist for a given complexity yet, best_cell() falls back to v1 score."""
    db = tmp_path / "u.db"
    rows: list[tuple] = []
    # Plenty of complexity-NULL rows (legacy / pre-fix data) and a couple labeled.
    rows += [("model-a", "low", 1000, None) for _ in range(30)]
    rows += [("model-b", "low", 200, None) for _ in range(30)]
    # Only complexity=2 has labeled data; nothing for complexity=1 or 3.
    rows += [("model-a", "low", 100, 2) for _ in range(5)]
    rows += [("model-b", "low", 500, 2) for _ in range(5)]
    _make_db(db, rows)

    cells = [
        Cell(model="model-a", reasoning_effort="low", context_window=None),
        Cell(model="model-b", reasoning_effort="low", context_window=None),
    ]
    em = EfficiencyModel.from_db(db, cells, min_samples_per_cell=30)

    # Bucket 2 has data → model-a is cheaper (100 vs 500).
    assert em.has_complexity_data(2)
    assert em.best_cell(cells, complexity=2).model == "model-a"

    # Bucket 1 has no data → falls back to v1 → model-b is cheaper overall.
    assert not em.has_complexity_data(1)
    assert em.best_cell(cells, complexity=1).model == "model-b"


def test_v2_falls_back_per_cell_when_bucket_partially_covered(tmp_path: Path) -> None:
    """If complexity=1 has data for cell A but not cell B, B falls back to its v1 score."""
    db = tmp_path / "u.db"
    rows: list[tuple] = []
    # A: complexity 1 labeled cheap; B: no labeled rows but cheap overall.
    rows += [("model-a", "low", 50, 1) for _ in range(10)]
    rows += [("model-a", "low", 1500, None) for _ in range(30)]
    rows += [("model-b", "low", 200, None) for _ in range(30)]
    _make_db(db, rows)

    cells = [
        Cell(model="model-a", reasoning_effort="low", context_window=None),
        Cell(model="model-b", reasoning_effort="low", context_window=None),
    ]
    em = EfficiencyModel.from_db(db, cells, min_samples_per_cell=30)

    # complexity=1: A scored from v2 (50), B falls back to v1 (200). A wins.
    chosen = em.best_cell(cells, complexity=1)
    assert chosen.model == "model-a"
