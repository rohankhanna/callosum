"""Cost-weighted cell selector.

Selection rule: pick the cheapest cell whose predicted satisfaction
probability is at the binary-classifier decision boundary or above
(P >= 0.5). If no cell qualifies, pick the cheapest cell overall —
the request is best-effort, not refused.

Within a cost tier, the selector keeps quality primary and uses the
forward time estimate only as a bounded tie-break: cells must first land
in the same 5-point quality bucket before lower estimated latency can
win. Parameter count remains the final deterministic tie-break.

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
    boundary; within a cost tier, prefer the highest quality bucket,
    then the lower estimated latency, then parameter_count descending.
    Cheapest overall remains the best-effort fallback.

    The parameter_count tiebreaker is principled, not arbitrary: model
    capacity is a measurable property, and within a cost tier the more-
    parameterized cell is empirically more capable on agentic/tool-use
    workloads. Cells without parameter_count (typically remote cells
    where the backend doesn't expose it) tiebreak as 0, sorting last
    within their cost tier — fine, since cost mostly differentiates
    remote cells already.
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

        def _rank(c: Cell) -> tuple[int, int, float, float, int]:
            caps = capabilities[c]
            eta_ms = time_estimates_ms.get(c, float("inf"))
            # Negate parameter_count so larger values sort FIRST (Python
            # tuple sort is ascending). None becomes 0 → sorts last
            # within its cost tier.
            return (
                caps.cost_rank,
                -_quality_bucket(predictions[c]),
                eta_ms,
                -predictions[c],
                -(caps.parameter_count or 0),
            )

        return min(pool, key=_rank)
