"""Router orchestrator — wires the routing pipeline.

    request → features → capability filter → predict → select → RoutingDecision

Depends only on the Protocols defined in routing/protocols.py and the
existing Cell type. No knowledge of how embeddings are computed, how
predictions are made, or how cost is ranked — each is injected at
construction.
"""

from __future__ import annotations

from typing import Any

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.features import extract_features
from callosum.routing.protocols import (
    CellSelector,
    EmbeddingProvider,
    QualityPredictor,
    RoutingDecision,
)


class NoCompatibleCellError(RuntimeError):
    """Raised when the capability filter empties the cell list.

    Means no available cell can physically serve this request (e.g. an
    image-bearing prompt with no vision-capable cells advertised). The
    dispatch layer surfaces this to the client as a 4xx, since trying
    anyway would just produce upstream errors.
    """


class Router:
    """Routes incoming requests through the configured pipeline.

    Stateless across requests except for the predictor's internal index,
    which is mutated by `reload()` calls outside the hot path (called
    by the predictor-training Dispatch job).
    """

    def __init__(
        self,
        *,
        embedding: EmbeddingProvider,
        predictor: QualityPredictor,
        selector: CellSelector,
        capability_filter: CapabilityFilter,
    ) -> None:
        self._embedding = embedding
        self._predictor = predictor
        self._selector = selector
        self._filter = capability_filter

    async def route(
        self, body: dict[str, Any], cells: list[Cell]
    ) -> RoutingDecision:
        """Run the pipeline once for an incoming request body.

        `cells` is the live, routability-filtered cell grid from the
        caller (already excludes backends in cooldown / weekly-exhausted
        / offline). The capability filter further drops cells whose
        physical capabilities don't match the request.
        """
        features = await extract_features(body, self._embedding)
        compatible = self._filter.filter(cells, features)
        if not compatible:
            raise NoCompatibleCellError(
                f"no cell satisfies (tokens={features.tokens}, "
                f"modalities={set(features.modalities)}, "
                f"needs_tools={features.needs_tools})"
            )
        predictions = self._predictor.predict(features, compatible)
        capabilities_map = {c: self._filter._capabilities_of(c) for c in compatible}
        chosen = self._selector.select(predictions, capabilities_map)
        return RoutingDecision(
            cell=chosen,
            features=features,
            predictions={
                f"{c.model} {c.reasoning_effort}": p
                for c, p in predictions.items()
            },
            predictor_id=self._predictor.id,
        )
