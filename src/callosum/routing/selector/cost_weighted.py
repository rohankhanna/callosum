"""Cost-weighted cell selector.

Selection rule: pick the cheapest cell whose predicted satisfaction
probability is at the binary-classifier decision boundary or above
(P >= 0.5). If no cell qualifies, pick the cheapest cell overall —
the request is best-effort, not refused.

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


class CostWeightedSelector:
    """CellSelector impl picking cheapest candidate above the decision
    boundary; tiebreak by parameter_count descending (most capable
    within tier wins); cheapest overall as a best-effort fallback.

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
    ) -> Cell:
        if not predictions:
            raise ValueError(
                "CostWeightedSelector.select called with empty predictions"
            )
        qualifying = [
            c for c, p in predictions.items() if p >= _DECISION_BOUNDARY
        ]
        pool = qualifying if qualifying else list(predictions.keys())

        def _rank(c: Cell) -> tuple[int, int]:
            caps = capabilities[c]
            # Negate parameter_count so larger values sort FIRST (Python
            # tuple sort is ascending). None becomes 0 → sorts last
            # within its cost tier.
            return (caps.cost_rank, -(caps.parameter_count or 0))

        return min(pool, key=_rank)
