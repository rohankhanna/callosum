"""Hermetic unit tests for the pure helpers in routing/usage_rates.

The reporting layer in usage_rates.py mixes sqlite-backed loaders
(_load_rows / _meter_report / usage_rate_report /
meter_relationship_report) with pure helpers that operate on in-memory
_Sample / Cell inputs. Only the pure helpers are pinned here — no
sqlite, no network, no app, no GPU — just numpy (a dep) and the stdlib.

Pinned helpers:
- _fit_rate      — numpy lstsq per-cell quota-rate fit with collinearity
                       detection, cached>uncached clamp, negative-coef clamp,
                       and insufficient-data fallbacks.
- _component     — rate-dict builder (available / rate / unit / source).
- _local_rate    — local-cell zero-rate dict.
- _insufficient_rate — insufficient-data rate dict.
- _window_sums_by_cell — pure aggregation over a window list.
- _iso           — UTC timestamp formatting.

The sqlite-backed _load_rows / _meter_report / usage_rate_report /
meter_relationship_report paths are integration-shaped (they need a
UsageLog-built db) and are out of scope for this pure-logic pin.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from callosum.cell_grid import Cell
from callosum.routing.usage_rates import (
    _component,
    _fit_rate,
    _insufficient_rate,
    _iso,
    _local_rate,
    _Sample,
    _window_sums_by_cell,
)

UNIT = "weekly_used_percent"


def _sample(
    *,
    uncached: float,
    cached: float = 0.0,
    output: float = 0.0,
    reasoning: float = 0.0,
    delta: float,
    row_id: int = 0,
    ts_start: float = 0.0,
) -> _Sample:
    return _Sample(
        row_id=row_id,
        model="model-a0e7",
        effort="high",
        uncached_input=uncached,
        cached_input=cached,
        output=output,
        reasoning=reasoning,
        delta=delta,
        ts_start=ts_start,
    )


# ---------- _fit_rate: clean linear fit --------------------------------------


def test_fit_rate_clean_linear_recovers_slopes() -> None:
    # delta = 2 * uncached + 1 * cached + 3 * output exactly; reasoning held
    # at 0 so it is inactive (zero sum-of-abs). The three active features are
    # mutually non-collinear (linear / quadratic / primes) so the design
    # matrix is full rank and OLS recovers all three slopes. With cached (1)
    # below uncached (2), no clamp fires and confidence stays "measured".
    samples = [
        _sample(uncached=10.0, cached=1.0, output=2.0, delta=27.0),
        _sample(uncached=20.0, cached=4.0, output=3.0, delta=53.0),
        _sample(uncached=30.0, cached=9.0, output=5.0, delta=84.0),
        _sample(uncached=40.0, cached=16.0, output=7.0, delta=117.0),
        _sample(uncached=50.0, cached=25.0, output=11.0, delta=158.0),
    ]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=5, samples=samples, min_usable_samples=4, updated_at=None, unit=UNIT)
    assert out["input_uncached"]["rate"] == pytest.approx(2.0)
    assert out["input_cached"]["rate"] == pytest.approx(1.0)
    assert out["output"]["rate"] == pytest.approx(3.0)
    # reasoning was inactive -> not in coeffs -> rate is None.
    assert out["reasoning"]["rate"] is None
    assert out["confidence"] == "measured"
    assert out["source"] == "measured_from_weekly_quota_log"
    assert out["samples"] == {"total": 5, "usable": 5}
    assert out["cache_effect"]["available"] is True
    assert out["cache_effect"]["rate_difference"] == pytest.approx(1.0)


def test_fit_rate_negative_coef_clamped_to_zero() -> None:
    # delta = 2 * uncached - 1 * output. lstsq recovers output coef = -1,
    # but coeffs clamps each coefficient with max(value, 0.0) -> output = 0.0.
    # uncached/output are non-collinear (quadratic output) so the fit is
    # full rank.
    samples = [
        _sample(uncached=10.0, output=1.0, delta=19.0),
        _sample(uncached=20.0, output=4.0, delta=36.0),
        _sample(uncached=30.0, output=9.0, delta=51.0),
        _sample(uncached=40.0, output=16.0, delta=64.0),
    ]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=4, samples=samples, min_usable_samples=4, updated_at=None, unit=UNIT)
    assert out["input_uncached"]["rate"] == pytest.approx(2.0)
    # negative slope clamped to 0.0 (not None: 0.0 is not None -> available).
    assert out["output"]["rate"] == pytest.approx(0.0)
    assert out["output"]["available"] is True


# ---------- _fit_rate: cached > uncached clamp -------------------------------


def test_fit_rate_cached_exceeds_uncached_is_clamped_to_none() -> None:
    # delta = 1 * uncached + 5 * cached -> cached coef (5) > uncached coef (1).
    # The cached rate is dropped to None and a note is attached; confidence
    # degrades to insufficient_data and cache_effect becomes unavailable.
    # uncached/cached are non-collinear (quadratic cached) so the fit is
    # full rank.
    samples = [
        _sample(uncached=10.0, cached=1.0, delta=15.0),
        _sample(uncached=20.0, cached=4.0, delta=40.0),
        _sample(uncached=30.0, cached=9.0, delta=75.0),
        _sample(uncached=40.0, cached=16.0, delta=120.0),
    ]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=4, samples=samples, min_usable_samples=4, updated_at=None, unit=UNIT)
    assert out["input_uncached"]["rate"] == pytest.approx(1.0)
    assert out["input_cached"]["rate"] is None
    assert out["input_cached"]["available"] is False
    assert out["input_cached"]["note"] == "cached_input_rate_exceeds_uncached_input_rate"
    assert out["confidence"] == "insufficient_data"
    assert out["source"] == "insufficient_data"
    assert out["cache_effect"]["available"] is False
    assert out["cache_effect"]["note"] == "cached_input_rate_exceeds_uncached_input_rate"


# ---------- _fit_rate: insufficient-data fallbacks -----------------------------


def test_fit_rate_too_few_samples_returns_insufficient() -> None:
    # len(samples) < min_usable_samples -> insufficient rate with the
    # "too_few_verifiable_samples" reason.
    samples = [_sample(uncached=10.0, delta=5.0), _sample(uncached=20.0, delta=10.0)]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=2, samples=samples, min_usable_samples=4, updated_at=None, unit=UNIT)
    assert out["source"] == "insufficient_data"
    assert out["confidence"] == "insufficient_data"
    assert out["input_uncached"]["rate"] is None
    assert out["input_uncached"]["note"] == "too_few_verifiable_samples"
    assert out["samples"] == {"total": 2, "usable": 2}


def test_fit_rate_no_independent_token_variation_returns_insufficient() -> None:
    # Every feature column is constant/zero -> no active features -> the
    # "no_independent_token_variation" insufficient branch.
    samples = [
        _sample(uncached=0.0, delta=5.0),
        _sample(uncached=0.0, delta=10.0),
        _sample(uncached=0.0, delta=15.0),
        _sample(uncached=0.0, delta=20.0),
    ]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=4, samples=samples, min_usable_samples=4, updated_at=None, unit=UNIT)
    assert out["source"] == "insufficient_data"
    assert out["input_uncached"]["note"] == "no_independent_token_variation"


def test_fit_rate_collinear_features_returns_insufficient() -> None:
    # cached = 2 * uncached (perfectly collinear) -> matrix_rank < n_active ->
    # "collinear_token_features" insufficient branch.
    samples = [
        _sample(uncached=10.0, cached=20.0, delta=5.0),
        _sample(uncached=20.0, cached=40.0, delta=10.0),
        _sample(uncached=30.0, cached=60.0, delta=15.0),
        _sample(uncached=40.0, cached=80.0, delta=20.0),
    ]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=4, samples=samples, min_usable_samples=4, updated_at=None, unit=UNIT)
    assert out["source"] == "insufficient_data"
    assert out["input_uncached"]["note"] == "collinear_token_features"


def test_fit_rate_updated_at_propagated_as_iso() -> None:
    # updated_at flows through _iso into the returned dict.
    samples = [
        _sample(uncached=10.0, output=1.0, delta=23.0),
        _sample(uncached=20.0, output=4.0, delta=52.0),
        _sample(uncached=30.0, output=9.0, delta=87.0),
        _sample(uncached=40.0, output=16.0, delta=128.0),
    ]
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _fit_rate(cell, total_samples=4, samples=samples, min_usable_samples=4, updated_at=1735689600.0, unit=UNIT)
    assert out["updated_at"] == "2025-01-01T00:00:00Z"


# ---------- _component -------------------------------------------------------


def test_component_none_rate_is_unavailable() -> None:
    out = _component(None, source="measured_from_quota_log", unit=UNIT)
    assert out == {
        "available": False,
        "rate": None,
        "unit": f"{UNIT}_per_token",
        "source": "insufficient_data",
        "note": None,
    }


def test_component_positive_rate_rounded_and_source_preserved() -> None:
    out = _component(2.123456789012345, source="measured_from_quota_log", unit=UNIT, note="n")
    assert out["available"] is True
    assert out["rate"] == pytest.approx(2.123456789012)
    assert out["source"] == "measured_from_quota_log"
    assert out["unit"] == f"{UNIT}_per_token"
    assert out["note"] == "n"


def test_component_zero_rate_is_available() -> None:
    # 0.0 is not None -> available=True, source preserved (not overridden).
    out = _component(0.0, source="measured_from_quota_log", unit=UNIT)
    assert out["available"] is True
    assert out["rate"] == 0.0
    assert out["source"] == "measured_from_quota_log"


# ---------- _local_rate ------------------------------------------------------


def test_local_rate_shape_and_zero_rates() -> None:
    cell = Cell(model="model-a0e1", reasoning_effort="")
    out = _local_rate(cell, total_samples=7, updated_at=None, unit=UNIT)
    assert out["model"] == "model-a0e1"
    # empty effort -> None (not "").
    assert out["reasoning_effort"] is None
    assert out["samples"] == {"total": 7, "usable": 7}
    assert out["source"] == "local_zero"
    assert out["confidence"] == "not_applicable"
    for key in ("input_uncached", "input_cached", "output", "reasoning"):
        comp = out[key]
        assert comp["available"] is True
        assert comp["rate"] == 0.0
        assert comp["source"] == "local_zero"
        assert comp["unit"] == f"{UNIT}_per_token"
    assert out["cache_effect"]["available"] is False
    assert out["cache_effect"]["source"] == "not_applicable"
    assert out["updated_at"] is None


def test_local_rate_updated_at_iso_and_effort_passed_through() -> None:
    cell = Cell(model="model-a0e1", reasoning_effort="medium")
    out = _local_rate(cell, total_samples=3, updated_at=1735689600.0, unit=UNIT)
    assert out["reasoning_effort"] == "medium"
    assert out["updated_at"] == "2025-01-01T00:00:00Z"


# ---------- _insufficient_rate -----------------------------------------------


def test_insufficient_rate_shape_and_fields() -> None:
    cell = Cell(model="model-a0e7", reasoning_effort="high")
    out = _insufficient_rate(cell, 10, 3, 1735689600.0, "too_few_verifiable_samples", unit=UNIT)
    assert out["model"] == "model-a0e7"
    assert out["reasoning_effort"] == "high"
    assert out["samples"] == {"total": 10, "usable": 3}
    assert out["source"] == "insufficient_data"
    assert out["confidence"] == "insufficient_data"
    assert out["updated_at"] == "2025-01-01T00:00:00Z"
    for key in ("input_uncached", "input_cached", "output", "reasoning"):
        comp = out[key]
        assert comp["available"] is False
        assert comp["rate"] is None
        assert comp["source"] == "insufficient_data"
        assert comp["note"] == "too_few_verifiable_samples"
    assert out["cache_effect"]["available"] is False
    assert out["cache_effect"]["note"] == "too_few_verifiable_samples"


# ---------- _window_sums_by_cell ---------------------------------------------


def test_window_sums_empty_returns_empty_dict() -> None:
    assert _window_sums_by_cell([]) == {}


def test_window_sums_aggregates_delta_count_and_rows_per_cell() -> None:
    windows = [
        SimpleNamespace(model="model-a0e7", reasoning_effort="high", delta=3.0, row_count=2),
        SimpleNamespace(model="model-a0e7", reasoning_effort="high", delta=5.0, row_count=4),
        SimpleNamespace(model="model-a0e7", reasoning_effort="low", delta=1.0, row_count=1),
        SimpleNamespace(model="model-a0e9", reasoning_effort="high", delta=7.0, row_count=3),
    ]
    out = _window_sums_by_cell(windows)
    assert out[("model-a0e7", "high")] == (8.0, 2, 6)
    assert out[("model-a0e7", "low")] == (1.0, 1, 1)
    assert out[("model-a0e9", "high")] == (7.0, 1, 3)


# ---------- _iso -------------------------------------------------------------


def test_iso_none_returns_none() -> None:
    assert _iso(None) is None


def test_iso_formats_utc_and_is_lexicographically_ordered() -> None:
    # Fixed-width zero-padded UTC format -> lexicographic order matches
    # chronological order.
    earlier = _iso(1735689600.0)  # 2025-01-01T00:00:00Z
    later = _iso(1735776000.0)  # 2025-01-02T00:00:00Z
    assert earlier == "2025-01-01T00:00:00Z"
    assert later == "2025-01-02T00:00:00Z"
    assert earlier < later
    # round-trip: parsing the formatted string back through gmtime recovers ts.
    parsed = time.mktime(time.strptime(earlier, "%Y-%m-%dT%H:%M:%SZ"))
    # mktime interprets struct_time in local time; gmtime produced UTC, so
    # compensate by the local-tz offset to recover the original epoch.
    offset = time.mktime(time.gmtime(0)) - 0.0
    assert parsed - offset == pytest.approx(1735689600.0)
