"""Factory: build a wired Router from RoutingConfig.

The config names concrete implementations by string id; the factory
resolves each to a class and instantiates it. Adding a new predictor or
selector means adding one branch here and one case in the relevant
impls/ subdirectory — never touching the Router.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.predictor.cell_prior import CellMajorityPriorPredictor, CellMeanPriorPredictor
from callosum.routing.predictor.uniform import UniformPriorPredictor
from callosum.routing.protocols import (
    CellCapabilities,
    CellSelector,
    QualityPredictor,
)
from callosum.routing.router import Router
from callosum.routing.selector.cost_weighted import CostWeightedSelector
from callosum.routing.time_estimator import TimeUsageEstimator
from callosum.routing.usage_estimate import OutputTokenForecaster


class RoutingConfig(BaseModel):
    """Operator-facing routing config. Each field names ONE implementation.

    Defaults are cold-start safe — Router built with all defaults runs
    end-to-end with no ML deps and picks the cheapest compatible cell.
    """

    model_config = ConfigDict(extra="forbid")

    quality_predictor: str = "uniform"
    cell_selector: str = "cost-weighted"


_PREDICTOR_IMPLS: dict[str, Callable[[], QualityPredictor]] = {
    "uniform": UniformPriorPredictor,
    "cell_majority_prior": CellMajorityPriorPredictor,
    "cell_mean_prior": CellMeanPriorPredictor,
}

_SELECTOR_IMPLS: dict[str, Callable[[], CellSelector]] = {
    "cost-weighted": CostWeightedSelector,
}


def build_predictor(predictor_id: str) -> QualityPredictor:
    """Resolve a predictor id to a fresh, untrained instance.

    Lets offline tools (shadow-eval probe, training jobs) build a single
    predictor by id without constructing a whole Router. Raises ValueError
    on an unknown id, mirroring build_router.
    """
    try:
        predictor_cls = _PREDICTOR_IMPLS[predictor_id]
    except KeyError as e:
        raise ValueError(f"unknown quality_predictor {predictor_id!r}; available: {sorted(_PREDICTOR_IMPLS)}") from e
    return predictor_cls()


def build_router(
    config: RoutingConfig,
    capabilities_of: Callable[[Cell], CellCapabilities],
    *,
    time_estimator: TimeUsageEstimator | None = None,
    output_forecaster: OutputTokenForecaster | None = None,
    feasibility_enabled: bool = True,
    feasibility_budget_s: float | None = None,
) -> Router:
    """Resolve string ids → concrete impls → wired Router."""
    try:
        predictor_cls = _PREDICTOR_IMPLS[config.quality_predictor]
    except KeyError as e:
        raise ValueError(
            f"unknown quality_predictor {config.quality_predictor!r}; available: {sorted(_PREDICTOR_IMPLS)}"
        ) from e
    try:
        selector_cls = _SELECTOR_IMPLS[config.cell_selector]
    except KeyError as e:
        raise ValueError(f"unknown cell_selector {config.cell_selector!r}; available: {sorted(_SELECTOR_IMPLS)}") from e
    router_kwargs: dict[str, object] = {
        "predictor": predictor_cls(),
        "selector": selector_cls(),
        "capability_filter": CapabilityFilter(capabilities_of=capabilities_of),
        "time_estimator": time_estimator,
        "output_forecaster": output_forecaster,
        "feasibility_enabled": feasibility_enabled,
    }
    if feasibility_budget_s is not None:
        router_kwargs["feasibility_budget_s"] = feasibility_budget_s
    return Router(**router_kwargs)  # type: ignore[arg-type]
