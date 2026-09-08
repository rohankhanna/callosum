"""Cost-weighted cell selector.

Selection rule: pick the cheapest cell whose predicted satisfaction
probability is at the binary-classifier decision boundary or above
(P >= 0.5). If no cell qualifies, pick the cheapest cell overall —
the request is best-effort, not refused.

Within a cost tier, the selector keeps quality primary and uses the
forward time estimate only as a bounded tie-break: cells must first land
in the same 5-point quality bucket before lower estimated latency can
win. Admitted local cells then prefer lower GPU opportunity cost, faster
throughput, and the smaller model when performance is otherwise tied;
non-local fallback keeps the larger-capacity tiebreak.

The 0.5 cutoff isn't a tuning knob; it's the natural decision boundary
for a binary classifier predicting "did this cell satisfy this prompt?"
Cold-start (uniform predictor) returns 0.5 for everyone → all qualify
→ cheapest wins → local-first.
"""

from __future__ import annotations

from callosum.cell_grid import Cell
from callosum.routing.protocols import CellCapabilities

# Max-likelihood decision boundary for a binary classifier. Not a
# routing-quality threshold — the boundary at which P(satisfy) > P(fail)
# crosses 50/50.
_DECISION_BOUNDARY = 0.5
_QUALITY_BUCKET_WIDTH = 0.05


def _quality_bucket(prediction: float) -> int:
    return int(prediction / _QUALITY_BUCKET_WIDTH)


class CostWeightedSelector:
    """CellSelector impl picking cheapest candidate above the decision
    boundary; first prefer curated local-fleet cells, then within that
    hierarchy tier prefer lower cost, highest quality bucket, lower estimated
    latency, higher local throughput, and exact-fit size.
    Cheapest overall remains the best-effort fallback within the hierarchy
    tier.

    Local throughput is an informational same-tier signal for local cells:
    when the local LLM gateway exposes a measured or estimated tokens/s figure, the
    faster local cell should win before we fall back to model size. For
    admitted local cells, smaller wins that final tie because the hierarchy
    is an exact-fit ladder. For non-local fallback, parameter_count remains
    the final capacity tiebreaker.
    """

    id: str = "cost-weighted"

    def select(
        self,
        predictions: dict[Cell, float],
        capabilities: dict[Cell, CellCapabilities],
        *,
        time_estimates_ms: dict[Cell, float] | None = None,
    ) -> Cell:
        if not predictions:
            raise ValueError("CostWeightedSelector.select called with empty predictions")
        qualifying = [c for c, p in predictions.items() if p >= _DECISION_BOUNDARY]
        pool = qualifying if qualifying else list(predictions.keys())
        time_estimates_ms = time_estimates_ms or {}

        def _size_rank(caps: CellCapabilities) -> int:
            if caps.local_catalog_admitted is True:
                # Exact-fit local hierarchy: once quality and performance are
                # otherwise tied, spend the smallest admitted local model that
                # can serve the request. Unknown size sorts after known sizes.
                return caps.parameter_count if caps.parameter_count is not None else 10**30
            # Non-local / fallback hierarchy: preserve the existing
            # bigger-capacity deterministic tiebreak.
            return -(caps.parameter_count or 0)

        def _rank(c: Cell) -> tuple[int, int, int, float, float, float, float, int]:
            caps = capabilities[c]
            eta_ms = time_estimates_ms.get(c, float("inf"))
            local_hierarchy_rank = 0 if caps.local_catalog_admitted is True else 1
            local_gpu_cost = (
                caps.local_gpu_seconds_per_token
                if caps.local_catalog_admitted is True and caps.local_gpu_seconds_per_token is not None
                else float("inf")
            )
            # Negate throughput so larger values sort FIRST (Python tuple sort
            # is ascending). Missing values become 0 and therefore sort last
            # within the same earlier tiers.
            return (
                local_hierarchy_rank,
                caps.cost_rank,
                -_quality_bucket(predictions[c]),
                eta_ms,
                -predictions[c],
                local_gpu_cost,
                -(caps.local_throughput_tps or 0.0),
                _size_rank(caps),
            )

        return min(pool, key=_rank)
