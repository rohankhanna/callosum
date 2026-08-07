"""Hermetic unit tests for the pure _fit helper in time_estimator.

_fit fits latency ≈ m·total_tokens + c by ordinary least squares,
clamps a negative (unphysical) slope to a flat mean line, and reports the
residual p50/p95 around the *final* (possibly clamped) line. It takes a
list[_Sample] and returns a CellTimeModel | None — no sqlite, no
network, no app — so these tests build _Sample instances directly in
memory and pin that contract exactly.

The sqlite-backed _build_timing_samples and the TimeModelProvider /
TimeUsageEstimator / aggregate_time_accuracy paths are exercised in
test_time_estimator.py (integration-shaped) and are out of scope here.
"""

from __future__ import annotations

import pytest

from callosum.routing.time_estimator import CellTimeModel, _fit, _Sample

# ---------- clean linear fit -------------------------------------------------


def test_fit_clean_linear_recovers_slope_and_intercept() -> None:
    # latency = 2 * total_tokens + 100, exactly. OLS must recover it with
    # zero residuals.
    samples = [_Sample(total_tokens=float(t), latency_ms=2.0 * t + 100.0) for t in (10, 20, 30, 40, 50)]
    model = _fit(samples, min_samples=5, source="cell-measured")
    assert model is not None
    # v1 contract: a == b == m (single per-total-token slope).
    assert model.a == pytest.approx(2.0)
    assert model.b == pytest.approx(2.0)
    assert model.c == pytest.approx(100.0)
    # Perfect in-sample fit -> residuals vanish.
    assert model.resid_p50 == pytest.approx(0.0)
    assert model.resid_p95 == pytest.approx(0.0)
    assert model.source == "cell-measured"
    assert model.n_obs == 5


# ---------- negative-slope clamping ------------------------------------------


def test_fit_negative_slope_clamped_to_zero_slope() -> None:
    # latency decreases as tokens grow (unphysical noise): y = -2x + 220.
    # The slope must be clamped to 0 and the intercept to the mean latency.
    samples = [_Sample(total_tokens=float(t), latency_ms=-2.0 * t + 220.0) for t in (10, 20, 30, 40, 50)]
    model = _fit(samples, min_samples=5, source="global-prior")
    assert model is not None
    assert model.a == pytest.approx(0.0)
    assert model.b == pytest.approx(0.0)
    # c = ybar - m*xbar = ybar when m is clamped to 0; mean of the five
    # latencies (200,180,160,140,120) is 160.0.
    assert model.c == pytest.approx(160.0)
    assert model.source == "global-prior"
    assert model.n_obs == 5


def test_fit_clamped_fit_residuals_around_clamped_line() -> None:
    # Same anti-correlated data as above. The unclamped OLS line would fit
    # perfectly (residuals 0); the clamped flat line is y = 160 (the mean),
    # so residuals are the deviations from the mean. Pinning the non-zero
    # p95 proves residuals are taken around the CLAMPED line actually used,
    # as the docstring promises — not around the discarded raw fit.
    samples = [_Sample(total_tokens=float(t), latency_ms=-2.0 * t + 220.0) for t in (10, 20, 30, 40, 50)]
    model = _fit(samples, min_samples=5, source="cell-measured")
    assert model is not None
    # sorted residuals around y=160: [-40, -20, 0, 20, 40]
    assert model.resid_p50 == pytest.approx(0.0)
    # p95 at pos 0.95*(5-1)=3.8 -> 20*0.2 + 40*0.8 = 36.0
    assert model.resid_p95 == pytest.approx(36.0)


# ---------- insufficient / degenerate data ----------------------------------


def test_fit_insufficient_samples_returns_none() -> None:
    # Below min_samples -> None, whether empty or merely under-threshold.
    assert _fit([], min_samples=1, source="cell-measured") is None
    samples = [_Sample(total_tokens=10.0, latency_ms=100.0)]
    assert _fit(samples, min_samples=2, source="cell-measured") is None
    assert _fit(samples, min_samples=10, source="cell-measured") is None


def test_fit_single_sample_min_samples_one_is_flat() -> None:
    # One sample, threshold satisfied: sxx == 0 (no variation), so the slope
    # is taken as 0 and the intercept as the single observed latency. The
    # residual band is degenerate (0).
    samples = [_Sample(total_tokens=42.0, latency_ms=500.0)]
    model = _fit(samples, min_samples=1, source="cell-measured")
    assert model is not None
    assert model.a == pytest.approx(0.0)
    assert model.b == pytest.approx(0.0)
    assert model.c == pytest.approx(500.0)
    assert model.resid_p50 == pytest.approx(0.0)
    assert model.resid_p95 == pytest.approx(0.0)
    assert model.n_obs == 1


def test_fit_constant_tokens_uses_mean_with_deviation_band() -> None:
    # All rows share the same total_tokens -> sxx == 0 -> m = 0, c = mean
    # latency. The residual band carries the latency spread the slope
    # cannot explain.
    samples = [_Sample(total_tokens=100.0, latency_ms=float(v)) for v in (100, 200, 300, 400, 500)]
    model = _fit(samples, min_samples=5, source="model-measured")
    assert model is not None
    assert model.a == pytest.approx(0.0)
    assert model.b == pytest.approx(0.0)
    assert model.c == pytest.approx(300.0)
    # sorted residuals around mean 300: [-200, -100, 0, 100, 200]
    assert model.resid_p50 == pytest.approx(0.0)
    # p95 at pos 3.8 -> 100*0.2 + 200*0.8 = 180.0
    assert model.resid_p95 == pytest.approx(180.0)
    assert model.source == "model-measured"
    assert model.n_obs == 5


# ---------- provenance pass-through ------------------------------------------


def test_fit_source_and_n_obs_propagated() -> None:
    # The caller-supplied source string and the sample count flow through
    # to the returned model verbatim, across the different resolution tiers.
    samples = [_Sample(total_tokens=float(t), latency_ms=3.0 * t + 50.0) for t in (0, 5, 10, 15, 20, 25)]
    for src in ("cell-measured", "model-measured", "global-prior", "override"):
        model = _fit(samples, min_samples=6, source=src)
        assert model is not None
        assert model.source == src
        assert model.n_obs == 6
        assert isinstance(model, CellTimeModel)
