"""Uniform-prior predictor — returns 0.5 for every candidate cell.

The cold-start predictor. With every cell predicted at the binary-
classifier decision boundary, the CostWeightedSelector falls through
to cost ordering — i.e. local-first by default until labeled data
arrives. Mathematically principled: 0.5 IS the max-likelihood prior
for a binary outcome when nothing is known.
"""

from __future__ import annotations

from collections.abc import Iterable

from callosum.cell_grid import Cell
from callosum.routing.protocols import LabeledRow, PromptFeatures


class UniformPriorPredictor:
    """QualityPredictor impl that returns 0.5 for every candidate.

    Always callable — never raises on missing embeddings, missing labels,
    or unknown cells. Phase 1's default predictor; later phases swap in
    KNNPredictor or a trained classifier without touching the Router
    that consumes this output.
    """

    id: str = "uniform"

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        return {c: 0.5 for c in candidates}

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        # No state to reload — the uniform predictor is data-independent
        # by construction. Method exists to satisfy the Protocol.
        return None
