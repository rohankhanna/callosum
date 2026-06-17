"""Shared scaffolding for the usage estimators (cost + time).

Two forward-looking estimators answer "how much will this codex request
consume" before (or alongside) the request runs:

  * the COST estimator (this is its home contract; impl in
    routing/cost_estimator.py) predicts ChatGPT weekly-quota burn in
    weekly_used_percent points;
  * the TIME estimator (sibling, work tracker ) predicts
    wall-clock latency in milliseconds.

Both depend on the SAME unknown — how many output tokens the response will
contain — which is only known after completion. Rather than let each
estimator guess output length its own way (and diverge), this module owns
the single shared resolution recorded in work tracker :

  1. OutputTokenForecaster produces ONE OutputTokenForecast per
     request. Both estimators consume the *same object* — this is the
     anti-divergence mechanism. Cold-start uses a global prior output:input
     ratio; it yields to per-cell measured data as a cell accrues traffic.

  2. The forecast is a DISTRIBUTION (p50 / p95), not a point. Each
     estimator propagates it into a RANGE so pre-flight consumers (status
     "≈X%–Y% of weekly quota", a user-facing ETA) stay honest about
     uncertainty instead of claiming a fake-exact number.

  3. Realized truth is finalized POST-HOC from the request log. Pre-flight
     consumers read the range; post-hoc consumers (the routing reward /
     bandit cost term in , data-driven timeout tuning)
     wait for finalized truth.

The label-quality caveats from the substrate verification ()
are designed around here, not rediscovered:

  * weekly_used_percent is integer-resolution, so a small request logs a
    per-call delta of 0. A 0 delta is NOT a 0-cost label; finalize marks
    it verifiable=False so it feeds aggregate calibration only, never a
    per-request point fit. (Time has no such problem — latency_ms is
    fully observed — so its finalize is always verifiable.)
  * Per-request quota-delta attribution is clean only on serialized
    credential traffic; concurrency blurs it and is not flagged in the data.
    The cost model averages over many rows, robust for ordering even if any
    one row is misattributed; a strict per-credential serialization filter
    is the documented follow-up.

Nothing here is cost- or time-specific: OutputTokenForecast,
EstimateInput, Estimate, FinalizedObservation and the
UsageEstimator protocol are the reusable contract both estimators
import. The time estimator MUST import these types and the forecaster — it
must not copy them.
"""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from callosum.cell_grid import Cell

# Recent rows only: bounds the scan over a large request log and tracks the
# current upstream output-length regime rather than months-old behaviour.
_DEFAULT_WINDOW_SECONDS = 30 * 24 * 3600


# --------------------------------------------------------------------------- #
# Shared value objects (the contract both estimators speak)                   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class OutputTokenForecast:
    """How many output (completion) tokens the response is expected to hold.

    p50 / p95 are token counts (not ratios). source records how
    the forecast was produced ("cell-measured", "global-prior", ...) so
    downstream logging knows how much to trust it; n_obs is the number
    of measured observations that backed it (0 at cold start).
    """

    p50: float
    p95: float
    source: str
    n_obs: int


@dataclass(frozen=True, slots=True)
class EstimateInput:
    """Everything an estimator needs to score one request, identical for both.

    cell is the (model, reasoning_effort) pair that will serve the
    request. input_tokens is the pre-request input estimate
    (features.tokens / _approx_tokens). output is the SHARED
    forecast — pass the same object to every estimator. modalities and
    needs_tools carry the remaining request-side facts the estimators
    may key on later (kept on the shared input so neither estimator invents
    its own request struct).
    """

    cell: Cell
    input_tokens: int
    output: OutputTokenForecast
    session_id: str | None = None
    modalities: frozenset[str] = frozenset({"text"})
    needs_tools: bool = False


@dataclass(frozen=True, slots=True)
class Estimate:
    """A range-first estimate. point is the p50; low/high bracket
    the propagated p50→p95 forecast band.

    Per  the band is built by pushing the forecast's p50 and
    p95 output-token counts through the per-cell model: low = value at the
    p50 forecast (the point), high = value at the p95 forecast. Pre-flight
    consumers show low–high; post-hoc consumers use point only as a
    prior. unit is "weekly_used_percent" (cost) or "ms" (time). source
    mirrors the forecast / model provenance. verifiable is whether a
    realized label for this unit can be observed post-hoc (always True for
    time; for cost, False when the integer-% delta resolves to 0).
    """

    point: float
    low: float
    high: float
    unit: str
    source: str
    verifiable: bool


@dataclass(frozen=True, slots=True)
class FinalizedObservation:
    """Realized truth for one served request, produced by finalize.

    observed_value is in the estimator's unit. verifiable follows
    the same rule as Estimate.verifiable: a cost row whose integer-%
    quota delta is 0 is recorded verifiable=False (unobservable, NOT a
    0-cost label) and is only safe to use in aggregate calibration.
    """

    request_id: int
    cell: Cell
    observed_output_tokens: int | None
    observed_value: float | None
    unit: str
    verifiable: bool
    source: str = ""


class UsageEstimator(Protocol):
    """A forward usage estimator over the shared contract.

    estimate is the pre-flight call (range-first Estimate).
    finalize is the post-request call, hooked at the existing app.py
    logging path, that records realized truth with the verifiable flag.
    """

    @property
    def id(self) -> str: ...

    @property
    def unit(self) -> str: ...

    def estimate(self, inp: EstimateInput) -> Estimate: ...

    def finalize(
        self,
        request_id: int,
        *,
        cell: Cell,
        observed_output_tokens: int | None,
        observed_value: float | None,
        verifiable: bool,
    ) -> FinalizedObservation: ...


# --------------------------------------------------------------------------- #
# Shared output-token forecaster                                              #
# --------------------------------------------------------------------------- #


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list. q in
    [0, 1]. Empty list returns 0.0."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@dataclass(frozen=True, slots=True)
class _RatioStats:
    """Per-key output:input ratio percentiles, measured from the log."""

    ratio_p50: float
    ratio_p95: float
    n_obs: int


def _build_ratio_stats(
    usage_log_path: Path,
    *,
    window_seconds: int,
    now: float | None,
) -> tuple[dict[tuple[str, str], _RatioStats], _RatioStats]:
    """Per-cell and global output:input ratio percentiles from the request log.

    Uses the ratio (completion_tokens / prompt_tokens) rather than the raw
    output count so the forecast scales with THIS request's input size. The
    global pool is the cold-start prior; a cell yields to its own measured
    ratios once it has observations.
    """
    empty: dict[tuple[str, str], _RatioStats] = {}
    if not usage_log_path.exists():
        return empty, _RatioStats(0.0, 0.0, 0)
    cutoff = (now if now is not None else time.time()) - window_seconds
    conn = sqlite3.connect(usage_log_path)
    try:
        rows = conn.execute(
            "SELECT model, reasoning_effort, prompt_tokens, completion_tokens"
            " FROM requests"
            " WHERE status = 200"
            "   AND prompt_tokens IS NOT NULL AND prompt_tokens > 0"
            "   AND completion_tokens IS NOT NULL AND completion_tokens >= 0"
            "   AND ts_start > ?",
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()
    per_cell: dict[tuple[str, str], list[float]] = {}
    pooled: list[float] = []
    for model, effort, prompt, completion in rows:
        if model is None or not prompt:
            continue
        ratio = float(completion) / float(prompt)
        key = (model, effort or "")
        per_cell.setdefault(key, []).append(ratio)
        pooled.append(ratio)
    stats: dict[tuple[str, str], _RatioStats] = {}
    for key, ratios in per_cell.items():
        ratios.sort()
        stats[key] = _RatioStats(
            ratio_p50=_percentile(ratios, 0.50),
            ratio_p95=_percentile(ratios, 0.95),
            n_obs=len(ratios),
        )
    pooled.sort()
    global_stats = _RatioStats(
        ratio_p50=_percentile(pooled, 0.50),
        ratio_p95=_percentile(pooled, 0.95),
        n_obs=len(pooled),
    )
    return stats, global_stats


class OutputTokenForecaster:
    """The single shared output-length forecaster ().

    Call forecast ONCE per request and pass the resulting
    OutputTokenForecast to every estimator's EstimateInput.output.
    The same object flowing into both estimators is what keeps them from
    diverging on output length.

    Cold start (a cell with fewer than min_obs measured rows) uses the
    global pooled output:input ratio; the cell yields to its own measured
    ratios once it crosses min_obs. When the log has no rows at all, the
    forecast falls back to fallback_ratio (a flat output:input guess).
    Stats are recomputed at most once per refresh_seconds to keep the
    aggregate scan off the hot path.
    """

    def __init__(
        self,
        usage_log_path: Path,
        *,
        min_obs: int = 20,
        fallback_ratio: float = 1.0,
        window_seconds: int = _DEFAULT_WINDOW_SECONDS,
        refresh_seconds: int = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._path = usage_log_path
        self._min_obs = min_obs
        self._fallback_ratio = fallback_ratio
        self._window_seconds = window_seconds
        self._refresh_seconds = refresh_seconds
        self._clock = clock
        self._cell_stats: dict[tuple[str, str], _RatioStats] = {}
        self._global_stats = _RatioStats(0.0, 0.0, 0)
        self._last_built: float | None = None

    def _maybe_refresh(self) -> None:
        now = self._clock()
        if self._last_built is None or (now - self._last_built) >= self._refresh_seconds:
            self._cell_stats, self._global_stats = _build_ratio_stats(
                self._path, window_seconds=self._window_seconds, now=None
            )
            self._last_built = now

    def forecast(self, cell: Cell, input_tokens: int, features: object = None) -> OutputTokenForecast:
        """Forecast output-token count for one request.

        features is accepted for forward compatibility (richer
        conditioning later) but unused in v1; the ratio model keys on the
        cell and scales by input_tokens.
        """
        self._maybe_refresh()
        inp = max(0, int(input_tokens))
        key = cell.as_tuple()
        cell_stats = self._cell_stats.get(key)
        if cell_stats is not None and cell_stats.n_obs >= self._min_obs:
            return OutputTokenForecast(
                p50=cell_stats.ratio_p50 * inp,
                p95=cell_stats.ratio_p95 * inp,
                source="cell-measured",
                n_obs=cell_stats.n_obs,
            )
        if self._global_stats.n_obs >= self._min_obs:
            return OutputTokenForecast(
                p50=self._global_stats.ratio_p50 * inp,
                p95=self._global_stats.ratio_p95 * inp,
                source="global-prior",
                n_obs=self._global_stats.n_obs,
            )
        # No usable measured signal anywhere: flat fallback ratio, with a
        # widened p95 so the cold-start band is honestly uncertain.
        return OutputTokenForecast(
            p50=self._fallback_ratio * inp,
            p95=2.0 * self._fallback_ratio * inp,
            source="fallback",
            n_obs=0,
        )
