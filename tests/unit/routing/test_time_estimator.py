from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from callosum.cell_grid import Cell
from callosum.routing.local_performance import build_local_performance_model
from callosum.routing.time_estimator import (
    TimeModelProvider,
    TimeUsageEstimator,
    aggregate_time_accuracy,
)
from callosum.routing.usage_estimate import EstimateInput, OutputTokenForecast

_SCHEMA = """
CREATE TABLE requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL,
    model TEXT,
    reasoning_effort TEXT,
    status INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms INTEGER
)
"""


def _make_db(path: Path, rows: list[dict[str, Any]]) -> None:
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    conn.executemany(
        "INSERT INTO requests"
        " (ts_start, model, reasoning_effort, status,"
        "  prompt_tokens, completion_tokens, total_tokens, latency_ms)"
        " VALUES (:ts_start, :model, :effort, :status,"
        "  :prompt, :completion, :total, :latency)",
        rows,
    )
    conn.commit()
    conn.close()


def _row(
    model: str,
    *,
    now: float,
    total: int,
    latency: float,
    effort: str = "medium",
    status: int = 200,
) -> dict[str, Any]:
    # Split total evenly across prompt/completion; since v1 fits a == b, only
    # the total matters for the slope (and predict(prompt, completion) reduces
    # to a·total + c).
    return {
        "ts_start": now - 100,
        "model": model,
        "effort": effort,
        "status": status,
        "prompt": total // 2,
        "completion": total - total // 2,
        "total": total,
        "latency": latency,
    }


def _linear_rows(
    model: str, *, now: float, m: float, c: float, totals: list[int], effort: str = "medium"
) -> list[dict[str, Any]]:
    """Rows whose latency lies exactly on m·total + c (residuals ≈ 0)."""
    return [_row(model, now=now, total=t, latency=m * t + c, effort=effort) for t in totals]


def _forecast(p50: float, p95: float) -> OutputTokenForecast:
    return OutputTokenForecast(p50=p50, p95=p95, source="test", n_obs=99)


# --------------------------------------------------------------------------- #
# TimeModelProvider                                                           #
# --------------------------------------------------------------------------- #


def test_cell_measured_recovers_slope_and_intercept(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    totals = [500, 1000, 1500, 2000, 2500, 3000] * 2  # 12 rows, varied tokens
    _make_db(db, _linear_rows("model-a0e8", now=now, m=2.0, c=100.0, totals=totals))
    p = TimeModelProvider(db, min_samples=10)
    model = p.model_for(Cell("model-a0e8", "medium"))
    assert model.source == "cell-measured"
    assert model.a == model.b  # v1 blends input/output into one slope
    assert abs(model.a - 2.0) < 1e-6
    assert abs(model.c - 100.0) < 1e-3
    # Exact linear data -> residuals ~0.
    assert abs(model.resid_p95) < 1e-3
    # predict(input, output) == 2*(input+output) + 100.
    assert abs(model.predict(500, 1000) - (2 * 1500 + 100)) < 1e-3


def test_residual_band_captures_dispersion(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    # All rows share the same token count: no slope is identifiable, so the fit
    # is flat (m=0, c=mean) and the spread lands entirely in the residuals.
    latencies = [1000.0 + 100.0 * i for i in range(20)]  # 1000..2900
    rows = [_row("model-a0e8", now=now, total=1000, latency=v) for v in latencies]
    _make_db(db, rows)
    p = TimeModelProvider(db, min_samples=10)
    model = p.model_for(Cell("model-a0e8", "medium"))
    assert model.a == 0.0  # no token variation -> flat fit
    assert model.resid_p95 > model.resid_p50  # honest upper-tail dispersion


def test_negative_slope_is_clamped_flat(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    # Latency DECREASES with tokens (noise); slope must clamp to 0, not go neg.
    _make_db(db, _linear_rows("model-a0e8", now=now, m=-1.0, c=5000.0, totals=[500, 1000, 1500, 2000] * 3))
    p = TimeModelProvider(db, min_samples=10)
    model = p.model_for(Cell("model-a0e8", "medium"))
    assert model.a == 0.0
    assert model.predict(1000, 1000) >= 0.0


def test_falls_back_to_model_pool_across_efforts(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    # The queried (model, effort) cell has too few rows, but the model pooled
    # across efforts has enough.
    rows = _linear_rows("model-a0e7", now=now, m=1.0, c=0.0, totals=[1000] * 12, effort="low")
    rows += _linear_rows("model-a0e7", now=now, m=1.0, c=0.0, totals=[2000, 3000], effort="high")
    _make_db(db, rows)
    p = TimeModelProvider(db, min_samples=10)
    model = p.model_for(Cell("model-a0e7", "high"))
    assert model.source == "model-measured"


def test_local_cell_gets_slowed_global_prior(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    # Only remote (fast) traffic exists. A local cell with no data of its own
    # inherits the global prior, slowed down by local_slowdown.
    _make_db(db, _linear_rows("model-a0e8", now=now, m=1.0, c=0.0, totals=[1000, 2000, 3000] * 4))
    p = TimeModelProvider(db, min_samples=10, local_slowdown=3.0)
    local = p.model_for(Cell("model-a0b6", "medium"))
    assert local.source == "global-prior"
    # Slope tilted up ~3x relative to the measured remote slope (~1.0).
    assert abs(local.a - 3.0) < 0.2
    # And a remote cell keeps the untilted prior.
    remote = p.model_for(Cell("model-a0f0", "medium"))  # unmeasured remote
    assert abs(remote.a - 1.0) < 0.2


def test_override_wins(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    _make_db(db, _linear_rows("model-a0e8", now=now, m=2.0, c=100.0, totals=[1000] * 12))
    p = TimeModelProvider(db, min_samples=10, overrides={"model-a0e8": [5.0, 200.0]})
    model = p.model_for(Cell("model-a0e8", "medium"))
    assert model.source == "override"
    assert model.a == 5.0 and model.c == 200.0
    assert model.predict(100, 0) == 5.0 * 100 + 200.0


def test_disabled_uses_fallback_but_overrides_apply(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    _make_db(db, _linear_rows("model-a0e8", now=now, m=2.0, c=100.0, totals=[1000] * 12))
    p = TimeModelProvider(
        db,
        enabled=False,
        fallback_ms_per_token=7.0,
        fallback_base_ms=300.0,
        overrides={"pinned": [9.0, 50.0]},
    )
    assert p.model_for(Cell("model-a0e8", "medium")).source == "fallback"
    assert p.model_for(Cell("pinned", "medium")).source == "override"


def test_no_data_uses_fallback(tmp_path: Path) -> None:
    p = TimeModelProvider(tmp_path / "missing.sqlite", fallback_ms_per_token=8.0, fallback_base_ms=400.0)
    model = p.model_for(Cell("model-a0e8", "medium"))
    assert model.source == "fallback"
    assert model.a == 8.0 and model.c == 400.0


# --------------------------------------------------------------------------- #
# TimeUsageEstimator                                                          #
# --------------------------------------------------------------------------- #


def test_estimate_propagates_forecast_and_residual_band(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    _make_db(db, _linear_rows("model-a0e8", now=now, m=2.0, c=100.0, totals=[500, 1000, 1500, 2000, 2500, 3000] * 2))
    est = TimeUsageEstimator(TimeModelProvider(db, min_samples=10))
    inp = EstimateInput(cell=Cell("model-a0e8", "medium"), input_tokens=500, output=_forecast(1000, 3000))
    out = est.estimate(inp)
    assert out.unit == "ms"
    # point at p50 output: 2*500 + 2*1000 + 100 = 3100 (residuals ~0).
    assert abs(out.point - 3100.0) < 1.0
    assert out.point == out.low
    # high at p95 output: 2*500 + 2*3000 + 100 = 7100.
    assert abs(out.high - 7100.0) < 1.0
    assert out.high > out.point
    assert out.verifiable is True
    assert "cell-measured" in out.source


def test_local_cell_is_not_zeroed(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    _make_db(db, _linear_rows("model-a0e8", now=now, m=1.0, c=0.0, totals=[1000, 2000, 3000] * 4))
    est = TimeUsageEstimator(TimeModelProvider(db, min_samples=10, local_slowdown=4.0))
    inp = EstimateInput(cell=Cell("model-a0b6", "medium"), input_tokens=1000, output=_forecast(2000, 5000))
    out = est.estimate(inp)
    # Unlike cost, local time is a real positive estimate, not 0.
    assert out.point > 0.0
    assert out.high >= out.point
    assert out.verifiable is True


def test_cold_start_estimate_is_still_verifiable(tmp_path: Path) -> None:
    # Contrast with cost: time has no integer-resolution problem, so even a
    # cold-start (fallback) estimate is verifiable — latency is always observed.
    est = TimeUsageEstimator(TimeModelProvider(tmp_path / "missing.sqlite"))
    inp = EstimateInput(cell=Cell("model-a0e8", "medium"), input_tokens=1000, output=_forecast(1000, 2000))
    out = est.estimate(inp)
    assert out.source.startswith("fallback")
    assert out.verifiable is True
    assert out.point > 0.0


def test_local_performance_model_overrides_generic_local_prior(tmp_path: Path) -> None:
    est = TimeUsageEstimator(
        TimeModelProvider(tmp_path / "missing.sqlite"),
        local_performance_model=lambda cell: (
            build_local_performance_model(
                model_id=cell.model,
                quantization="bf16",
                pool_bytes=121 * 1024**3,
                free_bytes=94 * 1024**3,
                weight_bytes=40 * 1024**3,
                kv_bytes_per_token=2 * 1024**2,
                activation_bytes=2 * 1024**3,
                estimated_tokens_per_second=20.0,
                prefill_ms_per_token=1.5,
            )
            if cell.model == "model-a0b6"
            else None
        ),
    )
    inp = EstimateInput(cell=Cell("model-a0b6", "medium"), input_tokens=1000, output=_forecast(1000, 2000))
    out = est.estimate(inp)
    assert out.source.startswith("local-underutilized")
    assert out.verifiable is True
    assert out.point > 0.0


# --------------------------------------------------------------------------- #
# finalize (no unverifiable / local-zero case)                                #
# --------------------------------------------------------------------------- #


def test_finalize_records_latency_always_verifiable() -> None:
    est = TimeUsageEstimator(TimeModelProvider(Path("/nonexistent")))
    obs = est.finalize(
        7,
        cell=Cell("model-a0e8", "medium"),
        observed_output_tokens=1234,
        observed_value=4200.0,
    )
    assert obs.request_id == 7
    assert obs.observed_value == 4200.0
    assert obs.unit == "ms"
    assert obs.verifiable is True
    assert obs.source == "latency"


def test_finalize_local_is_real_latency_not_zero() -> None:
    est = TimeUsageEstimator(TimeModelProvider(Path("/nonexistent")))
    obs = est.finalize(
        9,
        cell=Cell("model-a0b6", "medium"),
        observed_output_tokens=50,
        observed_value=41000.0,
    )
    # No local-zero branch: local latency is recorded as-is.
    assert obs.observed_value == 41000.0
    assert obs.verifiable is True
    assert obs.source == "latency"


# --------------------------------------------------------------------------- #
# aggregate calibration                                                       #
# --------------------------------------------------------------------------- #


def test_aggregate_accuracy_in_sample_is_well_calibrated(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    rows: list[dict[str, Any]] = []
    rows += _linear_rows("model-a0e8", now=now, m=2.0, c=100.0, totals=[1000, 2000, 3000] * 14)
    rows += _linear_rows("model-a0e7", now=now, m=1.0, c=500.0, totals=[1500, 2500] * 21)
    _make_db(db, rows)
    p = TimeModelProvider(db, min_samples=10)
    acc = aggregate_time_accuracy(db, p)
    assert "__overall__" in acc
    overall = acc["__overall__"]
    assert overall.ratio is not None
    # OLS mean fit on linear data -> in-sample sum of predictions == sum actual.
    assert 0.98 <= overall.ratio <= 1.02
    assert "model-a0e8" in acc and "model-a0e7" in acc


def test_aggregate_accuracy_includes_local_rows(tmp_path: Path) -> None:
    now = time.time()
    db = tmp_path / "u.sqlite"
    rows = _linear_rows("model-a0e8", now=now, m=1.0, c=0.0, totals=[1000] * 12)
    rows += _linear_rows("model-a0b6", now=now, m=20.0, c=1000.0, totals=[500, 1000, 1500] * 4)
    _make_db(db, rows)
    p = TimeModelProvider(db, min_samples=10)
    acc = aggregate_time_accuracy(db, p)
    # Time matters most for local cells -> they ARE included (unlike cost).
    assert "model-a0b6" in acc
    assert "model-a0e8" in acc


def test_aggregate_accuracy_missing_db_is_empty(tmp_path: Path) -> None:
    p = TimeModelProvider(tmp_path / "missing.sqlite")
    assert aggregate_time_accuracy(tmp_path / "missing.sqlite", p) == {}
