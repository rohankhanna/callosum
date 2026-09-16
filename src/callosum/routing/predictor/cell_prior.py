"""Extension point: cell-prior quality predictor.

This module defines a neutral cell-prior implementation.  It preserves the
interface used by the routing factory and returns 0.5 for every candidate,
which is the maximum-likelihood cold-start prior.  The default predictor is
``UniformPriorPredictor`` (see ``uniform.py``).

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
    """Neutral cell-prior implementation.

    Override ``reload`` and ``predict`` to learn from labeled rows and produce
    a custom majority-label prior.
    """

    id: str = "cell_majority_prior"

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        del features
        return {c: 0.5 for c in candidates}

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        return None


class CellMeanPriorPredictor:
    """Neutral cell-mean implementation.

    Override ``reload`` and ``predict`` to learn from labeled rows and produce
    a custom mean-outcome prior.
    """

    id: str = "cell_mean_prior"

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        del features
        return {c: 0.5 for c in candidates}

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        return None
