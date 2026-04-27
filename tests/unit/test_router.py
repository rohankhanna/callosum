from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codex_proxy.cell_grid import Cell, build_cells
from codex_proxy.router import ExploiterRouter, ExplorerRouter


def _seed_log(db: Path, rows: list[tuple[str, str, str, int]]) -> None:
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
        rows,
    )
    conn.commit()
    conn.close()


def test_explorer_picks_first_cell_when_log_missing(tmp_path: Path) -> None:
    router = ExplorerRouter(usage_log_path=tmp_path / "no.sqlite")
    decision = router.choose()
    cells = build_cells()
    assert decision.cell == cells[0]


def test_explorer_picks_least_sampled_cell(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    cells = build_cells()
    # Saturate every cell except (model-a0c3, low) which has zero samples.
    saturated = [
        (c.model, c.reasoning_effort, "auto-learning", 200)
        for c in cells
        if c != Cell(model="model-a0c3", reasoning_effort="low")
    ] * 10
    _seed_log(db, saturated)
    router = ExplorerRouter(usage_log_path=db)
    decision = router.choose()
    assert decision.cell == Cell(model="model-a0c3", reasoning_effort="low")


def test_explorer_decision_carries_reason(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    _seed_log(db, [("model-a0e7", "xhigh", "auto-learning", 200)])
    router = ExplorerRouter(usage_log_path=db)
    decision = router.choose()
    # Whatever cell wins, the reason should mention the round-robin policy.
    assert "round-robin" in decision.reason
    # The chosen cell should NOT be the one we just inserted (it has 1 sample
    # while every other cell has 0).
    assert decision.cell != Cell(model="model-a0e7", reasoning_effort="xhigh")


def test_exploiter_raises_not_trained() -> None:
    router = ExploiterRouter(usage_log_path=None)
    with pytest.raises(ExploiterRouter.NotTrained):
        router.choose()
