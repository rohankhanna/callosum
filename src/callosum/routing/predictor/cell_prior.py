"""Extension point: cell-prior quality predictor.

The trained predictor (per-cell majority/mean label priors from request-log
quality scores) is private.  This stub preserves the interface so the routing
factory can reference it by id; the default predictor is ``UniformPriorPredictor``
(see ``uniform.py``), which returns 0.5 for every candidate — mathematically
the max-likelihood cold-start prior.

To implement a custom predictor:

    from callosum.routing.protocols import LabeledRow, PromptFeatures
    from callosum.cell_grid import Cell

    class MyPredictor:
        id = "my_predictor"

        def predict(self, features, candidates):
            # Return P(satisfy) in [0, 1] for each candidate.
            return {c: 0.5 for c in candidates}

        def reload(self, labeled):
            # Learn from labeled request-log rows.
            ...

Then register it in ``routing/factory.py``'s ``_PREDICTOR_IMPLS``.
"""

from __future__ import annotations

from collections.abc import Iterable

from callosum.cell_grid import Cell
from callosum.routing.protocols import LabeledRow, PromptFeatures


class CellMajorityPriorPredictor:
    """Stub: returns the uniform 0.5 prior for every candidate.

    The trained majority-label logic is private.  Override ``reload`` and
    ``predict`` to implement a custom version.
    """

    id: str = "cell_majority_prior"

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        del features
        return {c: 0.5 for c in candidates}

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        return None


class CellMeanPriorPredictor:
    """Stub: returns the uniform 0.5 prior for every candidate.

    The trained mean-outcome logic is private.  Override ``reload`` and
    ``predict`` to implement a custom version.
    """

    id: str = "cell_mean_prior"

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        del features
        return {c: 0.5 for c in candidates}

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        return None
