"""Regression gate for the exploration doom loop (work tracker ````).

This is the failing→green tiered-gate case the handoff asks for: it reproduces
the doom loop and pins the two-part fix.

PLAIN-LANGUAGE PROBLEM (from the node): routing=auto with
effective_routing_mode=quota_explore picked a local cell that FIT the
context window but could not FINISH within the stall-guard first-byte timeout.
The turn ran to the 300s timeout, returned status=0 (no completion), and
fell back. Because a timed-out turn records no completed sample, the cell's
sample count stayed at 0 — below the even-split exploration floor — so the
quota re-targeted the SAME cell indefinitely. GPU pegged, every turn failed.

FIX (two halves, both tested here):

  1. Feasibility-aware candidate filter (routing/feasibility.py): a cell is
     exploration-eligible only when its predicted p95 completion sits under the
     stall-guard first-byte budget. Cold cells with no measured fit stay
     eligible (grace) so exploration is not starved.
  2. Post-timeout cooldown (cell_grid.recent_quota_cooldown_cells +
     quota.select_quota_deficit_cell(cooldown=...)): a cell that just timed
     out on a forced turn is skipped for the next cycle; it re-arms only after
     recording a real completed sample.

Pre-fix this file would FAIL on the doom-loop cases (the slow cell gets forced
indefinitely); post-fix it is GREEN. That is the merge-gate seed for
````.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

from callosum.cell_grid import Cell, CellCoverage, recent_quota_cooldown_cells
from callosum.routing.capability import CapabilityFilter
from callosum.routing.embedding.noop import NoopEmbeddingProvider
from callosum.routing.feasibility import feasibility_eligible
from callosum.routing.protocols import CellCapabilities, PromptFeatures
from callosum.routing.quota import select_quota_deficit_cell
from callosum.routing.router import Router
from callosum.routing.selector.cost_weighted import CostWeightedSelector
from callosum.routing.usage_estimate import Estimate, EstimateInput, OutputTokenForecast

# Stall-guard first-byte budget used across these cases (mirrors the
# production default CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S=180s).
BUDGET_S = 180.0


def _cell(model: str, effort: str = "medium") -> Cell:
    return Cell(model=model, reasoning_effort=effort)


# --------------------------------------------------------------------------- #
# Stubs                                                                        #
# --------------------------------------------------------------------------- #


class _FixedForecaster:
    """Output forecaster stub returning a tiny fixed forecast (output length is
    irrelevant to the feasibility check — only the latency p95 matters)."""

    def forecast(self, cell: Cell, input_tokens: int, features: object = None) -> OutputTokenForecast:
        del cell, input_tokens, features
        return OutputTokenForecast(p50=100.0, p95=100.0, source="fixed", n_obs=1)


class _StubTimeEstimator:
    """Time estimator stub keyed by cell → (source, p95_ms).

    Returns an Estimate whose high is the p95 and whose source carries
    the trust stamp feasibility_eligible reads: a cell-measured / override
    source is trusted (hard constraint); anything else gets cold-cell grace.
    """

    id = "stub-time"
    unit = "ms"

    def __init__(self, table: dict[Cell, tuple[str, float]]) -> None:
        self._table = table

    def estimate(self, inp: EstimateInput) -> Estimate:
        source, high = self._table[inp.cell]
        return Estimate(point=high, low=high, high=high, unit="ms", source=source, verifiable=True)


class _FixedPredictor:
    id = "fixed"

    def __init__(self, scores: dict[Cell, float]) -> None:
        self._scores = scores

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        del features
        return {cell: self._scores[cell] for cell in candidates}


# --------------------------------------------------------------------------- #
# Part 1 — feasibility-aware candidate filter                                  #
# --------------------------------------------------------------------------- #


def test_trusted_measured_fit_under_budget_is_eligible() -> None:
    """A cell the estimator has measured and predicts to finish in time stays
    selectable for forced exploration."""
    fast = _cell("fast-local")
    est = _StubTimeEstimator({fast: ("cell-measured+fixed", 10_000.0)})  # 10s < 180s
    assert (
        feasibility_eligible(fast, 262_000, time_estimator=est, output_forecaster=_FixedForecaster(), budget_s=BUDGET_S)
        is True
    )


def test_trusted_measured_fit_over_budget_is_excluded() -> None:
    """The doom-loop cell: fits the window, but its MEASURED p95 exceeds the
    stall-guard budget. It must be EXCLUDED from forced exploration — this is
    the half the pre-fix candidate filter missed (it checked window fit only)."""
    slow = _cell("slow-local")
    est = _StubTimeEstimator({slow: ("cell-measured+fixed", 200_000.0)})  # 200s > 180s
    assert (
        feasibility_eligible(slow, 262_000, time_estimator=est, output_forecaster=_FixedForecaster(), budget_s=BUDGET_S)
        is False
    )


def test_cold_cell_graced_not_excluded_on_prior() -> None:
    """A cold cell (no measured fit, only a global-prior prediction) that the
    prior says is over budget is NOT hard-excluded — hard-excluding on an
    unmeasured prior would starve exploration of exactly the cells it needs to
    sample. The cooldown (part 2) bounds the cost to one failed forced turn."""
    cold = _cell("cold-local")
    est = _StubTimeEstimator({cold: ("global-prior+fixed", 200_000.0)})  # over budget, but untrusted
    assert (
        feasibility_eligible(cold, 262_000, time_estimator=est, output_forecaster=_FixedForecaster(), budget_s=BUDGET_S)
        is True
    )


def test_no_estimator_is_grace() -> None:
    """Without a time estimator (router cold-starts without one) feasibility
    must not gate — every cell is eligible."""
    cell = _cell("any")
    assert feasibility_eligible(cell, 262_000, time_estimator=None, output_forecaster=None, budget_s=BUDGET_S) is True


def test_cold_start_router_excludes_slow_cell_from_random_exploration() -> None:
    """End-to-end on the cold-start path: with feasibility on, the router's
    random exploration never hands a large turn to the slow measured cell —
    it picks the feasible one — while the slow cell stays a retry candidate
    (soft preference, not hard drop from the candidate list)."""
    slow = _cell("slow-local")
    fast = _cell("fast-local")
    caps = {
        slow: CellCapabilities(
            context_window=400_000, modalities=frozenset({"text"}), supports_tools=False, cost_rank=0
        ),
        fast: CellCapabilities(
            context_window=400_000, modalities=frozenset({"text"}), supports_tools=False, cost_rank=0
        ),
    }
    router = Router(
        embedding=NoopEmbeddingProvider(),
        predictor=_FixedPredictor({slow: 0.5, fast: 0.5}),
        selector=CostWeightedSelector(),
        capability_filter=CapabilityFilter(capabilities_of=caps.__getitem__),
        time_estimator=_StubTimeEstimator(  # type: ignore[arg-type]
            {slow: ("cell-measured+fixed", 200_000.0), fast: ("cell-measured+fixed", 10_000.0)}
        ),
        output_forecaster=_FixedForecaster(),  # type: ignore[arg-type]
        feasibility_enabled=True,
        feasibility_budget_s=BUDGET_S,
    )
    huge = "x" * 750_000  # ~250K tokens at chars/3 — both cells' windows fit it
    body = {"messages": [{"role": "user", "content": huge}]}
    # Over many cold-start draws, the slow cell is NEVER the primary pick.
    for _ in range(200):
        decision = asyncio.run(router.route(body, [slow, fast]))
        assert decision.cell == fast, "slow cell must be excluded from forced exploration"
        assert slow in decision.candidates, "slow cell stays a retry candidate (not hard-dropped)"


def test_cold_start_router_without_estimator_still_explores_both() -> None:
    """No estimator → feasibility is grace → cold-start exploration still
    spreads across both cells (no over-exclusion / no starvation)."""
    slow = _cell("slow-local")
    fast = _cell("fast-local")
    caps = {
        slow: CellCapabilities(
            context_window=400_000, modalities=frozenset({"text"}), supports_tools=False, cost_rank=0
        ),
        fast: CellCapabilities(
            context_window=400_000, modalities=frozenset({"text"}), supports_tools=False, cost_rank=0
        ),
    }
    router = Router(
        embedding=NoopEmbeddingProvider(),
        predictor=_FixedPredictor({slow: 0.5, fast: 0.5}),
        selector=CostWeightedSelector(),
        capability_filter=CapabilityFilter(capabilities_of=caps.__getitem__),
        time_estimator=None,
        output_forecaster=None,
        feasibility_enabled=True,
        feasibility_budget_s=BUDGET_S,
    )
    huge = "x" * 750_000
    body = {"messages": [{"role": "user", "content": huge}]}
    seen = {asyncio.run(router.route(body, [slow, fast])).cell for _ in range(200)}
    assert seen == {slow, fast}


# --------------------------------------------------------------------------- #
# Part 2 — post-timeout cooldown stops the re-arm                               #
# --------------------------------------------------------------------------- #


def _seed_requests(db_path, rows):
    """Write (model, effort, status, ts_start, effective_routing_mode) rows."""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "CREATE TABLE requests ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts_start REAL NOT NULL,"
            " ts_end REAL NOT NULL,"
            " latency_ms INTEGER NOT NULL,"
            " route TEXT NOT NULL,"
            " stream INTEGER NOT NULL,"
            " backend_id TEXT NOT NULL,"
            " status INTEGER NOT NULL,"
            " model TEXT,"
            " reasoning_effort TEXT,"
            " effective_routing_mode TEXT)"
        )
        conn.executemany(
            "INSERT INTO requests (ts_start, ts_end, latency_ms, route, stream, backend_id,"
            " status, model, reasoning_effort, effective_routing_mode)"
            " VALUES (?, ?, 0, 'r', 0, 'b', ?, ?, ?, ?)",
            [(ts, ts, status, model, effort, mode) for (model, effort, status, ts, mode) in rows],
        )
        conn.commit()
    finally:
        conn.close()


def test_cooldown_skips_cell_that_just_timed_out(tmp_path) -> None:
    """The doom loop's second half: a cell that just timed out on a forced
    (quota_explore) turn is skipped this cycle, so the quota stops re-targeting
    it and forces the next under-floor cell instead."""
    db = tmp_path / "requests.sqlite"
    slow = _cell("slow-local")
    other = _cell("other-local")
    now = time.time()
    _seed_requests(
        db,
        [
            # slow's most recent forced turn timed out (status 0), no success since.
            ("slow-local", "medium", 0, now - 60, "quota_explore"),
            # other has no traffic.
        ],
    )
    cooldown = recent_quota_cooldown_cells(db, [slow, other], window_seconds=600)
    assert cooldown == frozenset({slow})

    # Both under floor (0 samples); without cooldown `slow` wins by input order.
    coverage = CellCoverage(counts={slow: 0, other: 0})
    forced = select_quota_deficit_cell([slow, other], coverage, floor_pct=0.10, cooldown=cooldown)
    assert forced == other, "cooled cell must be skipped so the quota stops re-targeting it"


def test_cooldown_rearms_after_completed_sample(tmp_path) -> None:
    """The cell re-arms only after it records a real completed sample
    (status=200), even on an organic (non-forced) turn — any completion proves
    the cell can finish something."""
    db = tmp_path / "requests.sqlite"
    slow = _cell("slow-local")
    other = _cell("other-local")
    now = time.time()
    _seed_requests(
        db,
        [
            ("slow-local", "medium", 0, now - 600, "quota_explore"),  # old timeout
            ("slow-local", "medium", 200, now - 60, "auto"),  # recent organic success
        ],
    )
    cooldown = recent_quota_cooldown_cells(db, [slow, other], window_seconds=600)
    assert cooldown == frozenset(), "a completed sample after the failure re-arms the cell"


def test_cooldown_failure_outside_window_does_not_cool(tmp_path) -> None:
    """A forced timeout older than the cooldown window no longer cools the
    cell — the cooldown is short by design."""
    db = tmp_path / "requests.sqlite"
    slow = _cell("slow-local")
    now = time.time()
    _seed_requests(db, [("slow-local", "medium", 0, now - 4000, "quota_explore")])
    cooldown = recent_quota_cooldown_cells(db, [slow], window_seconds=600)
    assert cooldown == frozenset()


def test_cooldown_all_cooled_returns_none(tmp_path) -> None:
    """If every candidate is cooling, forcing nothing (None) is correct —
    forcing onto a cell we know will time out is worse than letting the
    router's optimal pick stand."""
    db = tmp_path / "requests.sqlite"
    a = _cell("a-local")
    b = _cell("b-local")
    now = time.time()
    _seed_requests(
        db,
        [
            ("a-local", "medium", 0, now - 60, "quota_explore"),
            ("b-local", "medium", 0, now - 60, "quota_explore"),
        ],
    )
    cooldown = recent_quota_cooldown_cells(db, [a, b], window_seconds=600)
    coverage = CellCoverage(counts={a: 0, b: 0})
    assert select_quota_deficit_cell([a, b], coverage, floor_pct=0.10, cooldown=cooldown) is None


def test_cooldown_off_behaves_like_pre_fix() -> None:
    """cooldown=None restores the pre-fix behavior: the least-sampled under-
    floor cell is forced regardless of recent failures (the doom loop). This
    pins backward compatibility for the escape hatch."""
    slow = _cell("slow-local")
    other = _cell("other-local")
    coverage = CellCoverage(counts={slow: 0, other: 0})
    assert select_quota_deficit_cell([slow, other], coverage, floor_pct=0.10) == slow


# --------------------------------------------------------------------------- #
# Combined: the full doom-loop break (the gate seed)                           #
# --------------------------------------------------------------------------- #


def test_doom_loop_broken_end_to_end(tmp_path) -> None:
    """The merge-gate seed. Two local cells both fit the window and both sit
    under the floor (0 completed samples). The slow cell's MEASURED p95 exceeds
    the stall-guard budget; the fast cell's is well under.

    PRE-FIX: select_quota_deficit_cell forces the slow cell (least-sampled,
    input-order tiebreak), it times out, records no sample, stays under floor,
    and gets re-targeted forever — the doom loop.

    POST-FIX: feasibility excludes the slow cell from the forced pool, so the
    quota forces the fast cell. When the slow cell LATER times out on a forced
    turn anyway (e.g. its measured fit was optimistic), the cooldown skips it
    on the next cycle — the quota does not re-target it. Both halves together
    break the loop; this is the case that fails on main and goes green here.
    """
    db = tmp_path / "requests.sqlite"
    slow = _cell("slow-local")
    fast = _cell("fast-local")
    forecaster = _FixedForecaster()
    estimator = _StubTimeEstimator({slow: ("cell-measured+fixed", 200_000.0), fast: ("cell-measured+fixed", 10_000.0)})
    coverage = CellCoverage(counts={slow: 0, fast: 0})
    candidates = [slow, fast]

    # --- Pre-fix baseline: no feasibility, no cooldown -> slow is forced. ---
    pre_fix = select_quota_deficit_cell(candidates, coverage, floor_pct=0.10)
    assert pre_fix == slow, "pre-fix: the slow cell is forced (the doom loop)"

    # --- Post-fix half 1: feasibility filters the slow cell out of forcing. ---
    feasible = [
        c
        for c in candidates
        if feasibility_eligible(c, 262_000, time_estimator=estimator, output_forecaster=forecaster, budget_s=BUDGET_S)
    ]
    assert feasible == [fast]
    post_fix_forced = select_quota_deficit_cell(feasible, coverage, floor_pct=0.10)
    assert post_fix_forced == fast, "post-fix: the fast, feasible cell is forced instead"

    # --- Post-fix half 2: if the slow cell DID get forced and timed out, the
    # cooldown skips it next cycle so the quota does not re-target it. ---
    now = time.time()
    _seed_requests(db, [("slow-local", "medium", 0, now - 60, "quota_explore")])
    cooldown = recent_quota_cooldown_cells(db, candidates, window_seconds=600)
    assert cooldown == frozenset({slow})
    # Even with feasibility disabled (escape hatch), the cooldown alone stops
    # the re-targeting — the belt-and-suspenders the handoff asks for.
    guarded = select_quota_deficit_cell(candidates, coverage, floor_pct=0.10, cooldown=cooldown)
    assert guarded == fast, "cooldown alone stops the slow cell being re-targeted"
