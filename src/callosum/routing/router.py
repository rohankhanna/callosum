"""Router orchestrator — wires the routing pipeline.

    request → features → capability filter → predict → select → RoutingDecision

Depends only on the Protocols defined in routing/protocols.py and the
existing Cell type. No knowledge of how predictions are made or how cost
is ranked — each is injected at construction.
"""

from __future__ import annotations

import random
from typing import Any

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.feasibility import feasibility_eligible
from callosum.routing.features import extract_features
from callosum.routing.protocols import (
    CellSelector,
    QualityPredictor,
    RoutingDecision,
)
from callosum.routing.time_estimator import TimeUsageEstimator
from callosum.routing.usage_estimate import EstimateInput, OutputTokenForecaster


class NoCompatibleCellError(RuntimeError):
    """Raised when the capability filter empties the cell list.

    Means no available cell can physically serve this request — modality
    mismatch (e.g. image-bearing prompt with no vision-capable cells) or
    no tool-capable cell when tools are requested. Context-window
    overflow is NOT a reason this fires; window fit is a soft preference
    at selection time, and upstream is the source of truth for actual
    token-count rejections.
    """


# Output headroom retained for the soft window-fit signal even though
# the hard filter no longer uses it — see _window_fit_factor below.
# Generous enough that a 252K-token prompt still scores 1.0 against a
# 256K-window cell.
_OUTPUT_HEADROOM_TOKENS = 4096

# Default coverage-feasibility budget: a forced-coverage cell whose
# predicted p95 completion exceeds this is excluded from the cold-start
# random pool (and, in app.py, from quota forcing) — it would trip the local
# stall guard, time out, and (under the even-split quota) get re-targeted
# forever. Mirrors CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S (the deadline a
# large-context prefill trips); the env-tunable value is plumbed in from
# app.py via build_router, so this is only the test/headless default.
# See routing/feasibility.py and work tracker .
_DEFAULT_FEASIBILITY_BUDGET_S = 180.0


def _window_fit_factor(window: int, prompt_tokens: int) -> float:
    """Multiplicative score factor in (0, 1] reflecting how comfortably
    a cell's advertised context window fits the estimated prompt size.

    Returns 1.0 when the prompt + headroom fits inside the window.
    Tapers linearly toward an asymptotic floor as the prompt overflows.
    Never zero — the proxy's prompt-size estimate is a chars/3 heuristic
    that's ~30% high on Codex CLI corpora (see features.py), so a cell
    that LOOKS too small often actually fits. Keeping a non-zero floor
    lets the selector still pick the least-bad cell when every option
    appears to overflow, instead of pretending none exist.
    """
    budget = prompt_tokens + _OUTPUT_HEADROOM_TOKENS
    if window <= 0:
        return 1.0  # unknown window — no penalty, trust upstream
    if window >= budget:
        return 1.0
    # Linear taper. ratio < 1 because window < budget. Floor at 0.1
    # so even a 10x-overflowing cell stays in the pool.
    return max(0.1, window / max(1, budget))


def _has_differentiated_predictions(predictions: dict[Cell, float]) -> bool:
    if len(predictions) < 2:
        return False
    vals = list(predictions.values())
    return max(vals) - min(vals) > 0.01


def _admitted_local_cells(cells: list[Cell], capabilities_map: dict[Cell, Any]) -> list[Cell]:
    return [c for c in cells if getattr(capabilities_map[c], "local_catalog_admitted", None) is True]


class Router:
    """Routes incoming requests through the configured pipeline.

    Stateless across requests except for the predictor's internal index,
    which is mutated by `reload()` calls outside the hot path (called
    by the predictor-training Dispatch job).
    """

    def __init__(
        self,
        *,
        predictor: QualityPredictor,
        selector: CellSelector,
        capability_filter: CapabilityFilter,
        time_estimator: TimeUsageEstimator | None = None,
        output_forecaster: OutputTokenForecaster | None = None,
        feasibility_enabled: bool = True,
        feasibility_budget_s: float = _DEFAULT_FEASIBILITY_BUDGET_S,
    ) -> None:
        self._predictor = predictor
        self._selector = selector
        self._filter = capability_filter
        self._time_estimator = time_estimator
        self._output_forecaster = output_forecaster
        self._feasibility_enabled = feasibility_enabled
        self._feasibility_budget_s = feasibility_budget_s

    async def route(self, body: dict[str, Any], cells: list[Cell]) -> RoutingDecision:
        """Run the pipeline once for an incoming request body.

        `cells` is the live, routability-filtered cell grid from the
        caller (already excludes backends in cooldown / weekly-exhausted
        / offline). The capability filter further drops cells whose
        physical capabilities don't match the request.
        """
        features = await extract_features(body)
        compatible = self._filter.filter(cells, features)
        if not compatible:
            raise NoCompatibleCellError(
                f"no cell satisfies request capabilities "
                f"(modalities={set(features.modalities)}, "
                f"needs_tools={features.needs_tools})"
            )
        predictions = self._predictor.predict(features, compatible)
        capabilities_map = {c: self._filter._capabilities_of(c) for c in compatible}

        if not _has_differentiated_predictions(predictions):
            # COLD START — no learned quality signal yet. Do NOT guess quality
            # from heuristics: the removed difficulty->effort prior biased every
            # turn to the highest effort (xhigh) and skewed the very dataset the
            # model will learn from. EXPLORE instead — pick a RANDOM capable
            # cell so coverage accrues unbiased. Restrict to cells that fully
            # fit the context window so we don't 4xx on overflow; fall back to
            # the whole compatible pool when nothing fully fits. Once enough
            # peer-quality data exists, the cell-majority-prior predictor returns
            # differentiated scores and the exploit branch below takes over.
            fitting = [
                c for c in compatible if _window_fit_factor(capabilities_map[c].context_window, features.tokens) >= 1.0
            ]
            # Feasibility: among window-fitting cells, keep only those predicted
            # to FINISH within the stall-guard budget, so cold-start coverage
            # doesn't hand a large real turn to a slow local cell that will time
            # out and record no sample (the doom loop, ). Cold
            # cells with no measured fit stay eligible (grace) — see
            # routing/feasibility.py. Fall back to the window-fitting pool, then
            # the whole compatible pool, so feasibility never empties selection
            # (better to try the least-bad cell than to 4xx).
            if self._feasibility_enabled:
                feasible = [
                    c
                    for c in fitting
                    if feasibility_eligible(
                        c,
                        features.tokens,
                        time_estimator=self._time_estimator,
                        output_forecaster=self._output_forecaster,
                        budget_s=self._feasibility_budget_s,
                    )
                ]
            else:
                feasible = fitting
            primary_pool = feasible or fitting or compatible
            local_primary_pool = _admitted_local_cells(primary_pool, capabilities_map)
            chosen = random.choice(local_primary_pool or primary_pool)
            # Retry order still prefers fitting, cheap cells (Dispatch retries
            # on 5xx); the random PRIMARY pick is what spreads coverage across
            # the grid during cold start.
            rest = sorted(
                (c for c in compatible if c != chosen),
                key=lambda c: (
                    0 if capabilities_map[c].local_catalog_admitted is True else 1,
                    -_window_fit_factor(capabilities_map[c].context_window, features.tokens),
                    capabilities_map[c].cost_rank,
                ),
            )
            return RoutingDecision(
                cell=chosen,
                features=features,
                predictions={f"{c.model} {c.reasoning_effort}": p for c, p in predictions.items()},
                candidates=(chosen, *rest),
                predictor_id=self._predictor.id,
            )

        # EXPLOIT — differentiated quality signal from the model. Soft
        # window-fit scaling (tighter-fitting cells penalized but never
        # excluded), then cost-weighted selection (cheapest capable cell above
        # the 0.5 quality bar).
        scaled_predictions = {
            c: predictions[c] * _window_fit_factor(capabilities_map[c].context_window, features.tokens)
            for c in compatible
        }
        time_estimates = self._estimate_time_ms(compatible, features.tokens)
        # When window-fit pushes EVERY cell below the 0.5 bar, the selector's
        # "cheapest best-effort" fallback would ignore fit — the wrong call when
        # fit is the reason nothing qualifies. Override with the best-fitting
        # cell so the primary attempt has the best chance upstream (Dispatch
        # retries 5xx only, not context-overflow 4xx). Cost tiebreaks equal fits.
        if all(p < 0.5 for p in scaled_predictions.values()):
            chosen = max(
                compatible,
                key=lambda c: (
                    scaled_predictions[c],
                    -capabilities_map[c].cost_rank,
                ),
            )
        else:
            chosen = self._selector.select(
                scaled_predictions,
                capabilities_map,
                time_estimates_ms=time_estimates,
            )

        # Candidate ordering for cell-retry: primary first, then the rest
        # ranked by scaled prediction (so retries also prefer fitting
        # cells), cost as tiebreaker. Dispatch only retries on 5xx, so
        # this mostly matters for backend-side failures; context-overflow
        # 4xx from upstream propagates without retry.
        def _rank_key(c: Cell) -> tuple[float, int]:
            return (-scaled_predictions[c], capabilities_map[c].cost_rank)

        rest = sorted((c for c in compatible if c != chosen), key=_rank_key)
        return RoutingDecision(
            cell=chosen,
            features=features,
            # Log the UNSCALED predictions — the window-fit factor is
            # a selection-time bias, not a quality claim. Logging the
            # raw probabilities keeps the predictor's calibration
            # readable downstream.
            predictions={f"{c.model} {c.reasoning_effort}": p for c, p in predictions.items()},
            candidates=(chosen, *rest),
            time_estimates_ms={f"{c.model} {c.reasoning_effort}": eta for c, eta in time_estimates.items()},
            predictor_id=self._predictor.id,
        )

    def _estimate_time_ms(self, cells: list[Cell], input_tokens: int) -> dict[Cell, float]:
        if self._time_estimator is None or self._output_forecaster is None:
            return {}
        estimates: dict[Cell, float] = {}
        for cell in cells:
            forecast = self._output_forecaster.forecast(cell, input_tokens)
            estimate = self._time_estimator.estimate(
                EstimateInput(
                    cell=cell,
                    input_tokens=input_tokens,
                    output=forecast,
                )
            )
            estimates[cell] = estimate.point
        return estimates
