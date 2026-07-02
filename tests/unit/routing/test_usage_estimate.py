from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from callosum.cell_grid import Cell
from callosum.routing.usage_estimate import (
    Estimate,
    EstimateInput,
    OutputTokenForecast,
    OutputTokenForecaster,
    _percentile,
)

_SCHEMA = """
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL,
    model TEXT,
    reasoning_effort TEXT,
    status INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER
)
"""


def _make_db(path: Path, rows: list[tuple]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    conn.executemany(
        "INSERT INTO requests"
        " (ts_start, model, reasoning_effort, status, prompt_tokens, completion_tokens)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


def _ratio_rows(model: str, effort: str, n: int, *, now: float, prompt: int, completion: int) -> list[tuple]:
    return [(now - 100, model, effort, 200, prompt, completion) for _ in range(n)]


def test_percentile_basics() -> None:
    assert _percentile([], 0.5) == 0.0
    assert _percentile([5.0], 0.95) == 5.0
    assert _percentile([0.0, 10.0], 0.5) == 5.0
    assert _percentile([0.0, 5.0, 10.0], 0.0) == 0.0
    assert _percentile([0.0, 5.0, 10.0], 1.0) == 10.0


def test_missing_db_uses_fallback_ratio(tmp_path: Path) -> None:
    f = OutputTokenForecaster(tmp_path / "nope.sqlite", fallback_ratio=1.5)
    fc = f.forecast(Cell("model-a0e8", "medium"), 1000)
    assert fc.source == "fallback"
    assert fc.n_obs == 0
    assert fc.p50 == 1500.0
    assert fc.p95 == 3000.0  # widened band at cold start


def test_cell_measured_forecast_scales_with_input(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    # 30 rows where completion = 2 * prompt -> ratio 2.
    _make_db(db, _ratio_rows("model-a0e8", "medium", 30, now=now, prompt=1000, completion=2000))
    f = OutputTokenForecaster(db, min_obs=20)
    fc = f.forecast(Cell("model-a0e8", "medium"), 500)
    assert fc.source == "cell-measured"
    assert fc.n_obs == 30
    assert fc.p50 == 1000.0  # ratio 2 * input 500
    assert fc.p95 == 1000.0  # all rows identical ratio


def test_cold_start_uses_global_prior(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    rows: list[tuple] = []
    # Plenty of GLOBAL signal from one cell (ratio 1)...
    rows += _ratio_rows("model-a0e7", "medium", 40, now=now, prompt=1000, completion=1000)
    # ...but the queried cell has only a couple of rows (< min_obs).
    rows += _ratio_rows("model-a0e8", "high", 2, now=now, prompt=1000, completion=9000)
    _make_db(db, rows)
    f = OutputTokenForecaster(db, min_obs=20)
    fc = f.forecast(Cell("model-a0e8", "high"), 1000)
    assert fc.source == "global-prior"
    # Global ratio is ~1 (dominated by the 40 model-a0e7 rows), not the cell's 9.
    assert 900.0 <= fc.p50 <= 1100.0


def test_value_objects_are_frozen() -> None:
    fc = OutputTokenForecast(p50=1.0, p95=2.0, source="x", n_obs=3)
    inp = EstimateInput(cell=Cell("model-a0e8", "medium"), input_tokens=10, output=fc)
    est = Estimate(point=1.0, low=1.0, high=2.0, unit="weekly_used_percent", source="s", verifiable=True)
    assert inp.output is fc
    assert est.high == 2.0
    assert inp.modalities == frozenset({"text"})
