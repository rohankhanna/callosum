"""Factory: build a wired Router from RoutingConfig.

The config names concrete implementations by string id; the factory
resolves each to a class and instantiates it. Adding a new embedding
provider / predictor / selector means adding one branch here and one
case in the relevant impls/ subdirectory — never touching the Router.

Phase 1 ships only no-op / cold-start impls. Phase 4 adds the BGE
provider and KNN predictor; the factory grows by two branches.
"""

from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.embedding.noop import NoopEmbeddingProvider
from callosum.routing.predictor.knn import KNNPredictor
from callosum.routing.predictor.uniform import UniformPriorPredictor
from callosum.routing.protocols import (
    CellCapabilities,
    CellSelector,
    EmbeddingProvider,
    QualityPredictor,
)
from callosum.routing.router import Router
from callosum.routing.selector.cost_weighted import CostWeightedSelector
from callosum.routing.time_estimator import TimeUsageEstimator
from callosum.routing.usage_estimate import OutputTokenForecaster


def _bge_provider_factory() -> EmbeddingProvider:
    """Lazy import + construct of the BGE provider so the module
    can be imported without sentence-transformers installed. Selected
    only when operator configures embedding_provider='bge-large-en-v1.5'."""
    from callosum.routing.embedding.bge import BGELargeEmbeddingProvider

    return BGELargeEmbeddingProvider()


class RoutingConfig(BaseModel):
    """Operator-facing routing config. Each field names ONE implementation.

    Defaults are cold-start safe — Router built with all defaults runs
    end-to-end with no ML deps and picks the cheapest compatible cell.
    """

    model_config = ConfigDict(extra="forbid")

    embedding_provider: str = "noop"
    quality_predictor: str = "uniform"
    cell_selector: str = "cost-weighted"


_EMBEDDING_IMPLS: dict[str, Callable[[], EmbeddingProvider]] = {
    "noop": NoopEmbeddingProvider,
    "bge-large-en-v1.5": _bge_provider_factory,
}

_PREDICTOR_IMPLS: dict[str, Callable[[], QualityPredictor]] = {
    "uniform": UniformPriorPredictor,
    "knn": KNNPredictor,
}

_SELECTOR_IMPLS: dict[str, Callable[[], CellSelector]] = {
    "cost-weighted": CostWeightedSelector,
}


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
        embedding_cls = _EMBEDDING_IMPLS[config.embedding_provider]
    except KeyError as e:
        raise ValueError(
            f"unknown embedding_provider {config.embedding_provider!r}; available: {sorted(_EMBEDDING_IMPLS)}"
        ) from e
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
        "embedding": embedding_cls(),
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
