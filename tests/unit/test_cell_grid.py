from __future__ import annotations

import sqlite3
from pathlib import Path

from callosum.cell_grid import (
    DEFAULT_MODELS,
    REASONING_LEVELS,
    CellCoverage,
    build_cells,
    cell_sample_counts,
    coverage_from_db,
    is_completion_model,
    live_completion_models,
    model_strength_key,
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


def test_cell_sample_counts_windows_and_ignores_routing_mode(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT, reasoning_effort TEXT, routing_mode TEXT,
            status INTEGER, ts_start REAL
        )
        """
    )
    now = 1_000_000.0
    conn.executemany(
        "INSERT INTO requests (model, reasoning_effort, routing_mode, status, ts_start)"
        " VALUES (?, ?, ?, ?, ?)",
        [
            # Counted regardless of routing_mode (unlike coverage_from_db):
            ("model-a0c3", "low", "auto", 200, now - 10),
            ("model-a0c3", "low", "pass-through", 200, now - 20),
            ("model-a0c3", "low", "quota_explore", 200, now - 30),
            # Excluded: failure status
            ("model-a0c3", "low", "auto", 500, now - 40),
            # Excluded: outside the window
            ("model-a0c3", "low", "auto", 200, now - 10_000),
            # Different cell, in window
            ("model-a0e7", "xhigh", "auto", 200, now - 5),
        ],
    )
    conn.commit()
    conn.close()
    cells = build_cells()
    cov = cell_sample_counts(db, cells, window_seconds=1_000, now=now)
    mini_low = next(c for c in cells if c.model == "model-a0c3" and c.reasoning_effort == "low")
    xhigh = next(c for c in cells if c.model == "model-a0e7" and c.reasoning_effort == "xhigh")
    assert cov.counts[mini_low] == 3  # 3 successes in window across routing modes
    assert cov.counts[xhigh] == 1


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
    # Use cells from build_cells to ensure context_window matches
    cell_5_4_mini_low = next(c for c in cells if c.model == "model-a0c3" and c.reasoning_effort == "low")
    cell_5_4_xhigh = next(c for c in cells if c.model == "model-a0e7" and c.reasoning_effort == "xhigh")
    cell_5_2_medium = next(c for c in cells if c.model == "model-a0e6" and c.reasoning_effort == "medium")
    assert cov.counts[cell_5_4_mini_low] == 3
    assert cov.counts[cell_5_4_xhigh] == 2
    # Cells not appearing in the data are zero, not missing.
    assert cov.counts[cell_5_2_medium] == 0
    assert cov.total_samples() == 5


# ---------- completion-model filter + strength ordering ---------------------


def test_is_completion_model_accepts_chat_targets() -> None:
    assert is_completion_model("model-a0e7") is True
    assert is_completion_model("model-a0c3") is True
    assert is_completion_model("model-a0b8") is True
    assert is_completion_model("model-a0e6") is True


def test_is_completion_model_rejects_special_purpose() -> None:
    # codex-auto-review is the canonical reason this filter exists.
    assert is_completion_model("codex-auto-review") is False
    assert is_completion_model("text-embedding-3-small") is False
    assert is_completion_model("model-a0a7") is False
    assert is_completion_model("not-a-model") is False
    assert is_completion_model("") is False


def test_model_strength_key_orders_strongest_first() -> None:
    """Lower key = stronger. Higher major.minor wins; among same version,
    full > codex > mini.
    """
    models = [
        "model-a0e6",
        "model-a0e7",
        "model-a0c3",
        "model-a0b8",
        "model-a0b9",
        "codex-auto-review",  # non-completion, should sort to end
    ]
    sorted_models = sorted(models, key=model_strength_key)
    assert sorted_models[0] == "model-a0e7"  # strongest
    assert sorted_models[1] == "model-a0b9"  # same version, full > codex
    assert sorted_models[2] == "model-a0c3"  # same version, mini last
    assert sorted_models[3] == "model-a0b8"
    assert sorted_models[4] == "model-a0e6"
    assert sorted_models[-1] == "codex-auto-review"  # non-completion at end


def test_live_completion_models_filters_and_sorts() -> None:
    """Given a backend's advertised_models pool, return only completion-style
    ids in strongest-first order.
    """
    pool = frozenset(
        {
            "model-a0e7",
            "model-a0e6",
            "model-a0c3",
            "model-a0b8",
            "codex-auto-review",  # excluded — not a completion target
            "model-a0b9",
        }
    )
    result = live_completion_models(pool)
    assert "codex-auto-review" not in result
    assert result == ("model-a0e7", "model-a0b9", "model-a0c3", "model-a0b8", "model-a0e6")


def test_live_completion_models_handles_empty_input() -> None:
    assert live_completion_models(frozenset()) == ()
    # Pool with only non-completion models also returns empty.
    assert live_completion_models(frozenset({"codex-auto-review", "embedding-x"})) == ()


def test_live_completion_models_handles_unknown_future_models() -> None:
    """Brand-new gpt-X.Y models from a future Codex release should slot in
    correctly without code edits — that's the whole point of the dynamic grid.
    """
    pool = frozenset({"model-a0e7", "model-a0f1", "model-a0c5"})
    result = live_completion_models(pool)
    # model-a0f1 is the strongest (highest major.minor); model-a0e7 sorts after.
    assert result[0] == "model-a0f1"
    assert result[1] == "model-a0c5"
    assert result[2] == "model-a0e7"
