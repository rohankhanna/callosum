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
from dataclasses import dataclass

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.embedding.noop import NoopEmbeddingProvider
from callosum.routing.predictor.uniform import UniformPriorPredictor
from callosum.routing.protocols import CellCapabilities
from callosum.routing.router import Router
from callosum.routing.selector.cost_weighted import CostWeightedSelector


@dataclass(frozen=True)
class RoutingConfig:
    """Operator-facing routing config. Each field names ONE implementation.

    Defaults are cold-start safe — Router built with all defaults runs
    end-to-end with no ML deps and picks the cheapest compatible cell.
    """

    enabled: bool = False
    embedding_provider: str = "noop"
    quality_predictor: str = "uniform"
    cell_selector: str = "cost-weighted"


_EMBEDDING_IMPLS: dict[str, Callable[[], object]] = {
    "noop": NoopEmbeddingProvider,
}

_PREDICTOR_IMPLS: dict[str, Callable[[], object]] = {
    "uniform": UniformPriorPredictor,
}

_SELECTOR_IMPLS: dict[str, Callable[[], object]] = {
    "cost-weighted": CostWeightedSelector,
}


def build_router(
    config: RoutingConfig,
    capabilities_of: Callable[[Cell], CellCapabilities],
) -> Router:
    """Resolve string ids → concrete impls → wired Router."""
    try:
        embedding_cls = _EMBEDDING_IMPLS[config.embedding_provider]
    except KeyError as e:
        raise ValueError(
            f"unknown embedding_provider {config.embedding_provider!r}; "
            f"available: {sorted(_EMBEDDING_IMPLS)}"
        ) from e
    try:
        predictor_cls = _PREDICTOR_IMPLS[config.quality_predictor]
    except KeyError as e:
        raise ValueError(
            f"unknown quality_predictor {config.quality_predictor!r}; "
            f"available: {sorted(_PREDICTOR_IMPLS)}"
        ) from e
    try:
        selector_cls = _SELECTOR_IMPLS[config.cell_selector]
    except KeyError as e:
        raise ValueError(
            f"unknown cell_selector {config.cell_selector!r}; "
            f"available: {sorted(_SELECTOR_IMPLS)}"
        ) from e
    return Router(
        embedding=embedding_cls(),
        predictor=predictor_cls(),
        selector=selector_cls(),
        capability_filter=CapabilityFilter(capabilities_of=capabilities_of),
    )
