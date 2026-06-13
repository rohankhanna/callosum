"""k-NN predictor over labeled prompt embeddings.

Storage: in-memory numpy matrix of normalized embeddings + per-row
metadata (which cell was used, what outcome was observed). Cosine
similarity is a dot product since vectors are unit-normalized.

Query: given a new prompt embedding, find the K nearest neighbors,
group them by which cell they ran on, average outcomes per cell,
map [-1, +1] outcomes to [0, 1] probabilities. Cells with zero
neighbors in the top-K fall back to 0.5 (uniform prior) so the
selector treats them as "no opinion yet" rather than "predicted bad."

Reload semantics: `reload(labeled_iterable)` rebuilds the index from
scratch. Cheap when row count is in the thousands; if/when the corpus
grows, switch to incremental updates without changing the call
interface.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from callosum.cell_grid import Cell
from callosum.routing.protocols import LabeledRow, PromptFeatures


class KNNPredictor:
    """k-NN over prompt embeddings, predicting P(cell satisfies prompt).

    Phase 4 builds the mechanics; Phase 5/6 deliver the labels that
    make this predictor actually distinguish cells. Until labels
    accumulate, behaves identically to UniformPriorPredictor (returns
    0.5 for every candidate) because the index is empty.
    """

    id: str = "knn"

    def __init__(self, k: int = 8) -> None:
        self._k = k
        # Lazy numpy import — keeps the module importable in places
        # that don't actually exercise the predictor (e.g. some unit
        # tests that only check protocol shape).
        import numpy as np

        self._np = np
        self._embeddings = np.zeros((0, 1), dtype=np.float32)  # placeholder
        self._cell_keys: list[str] = []
        self._outcomes = np.zeros((0,), dtype=np.float32)

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        # Cold start (no labels yet) → uniform prior for everyone.
        if features.embedding is None or len(self._cell_keys) == 0:
            return {c: 0.5 for c in candidates}
        np = self._np
        try:
            q = np.frombuffer(features.embedding, dtype=np.float32)
        except (ValueError, TypeError):
            return {c: 0.5 for c in candidates}
        if q.shape[0] != self._embeddings.shape[1]:
            # Dimensionality mismatch — embedding model changed since
            # the index was built. Fall back to prior so we don't
            # produce garbage predictions; the next training run will
            # rebuild the index against the new model.
            return {c: 0.5 for c in candidates}
        # Cosine similarity for normalized vectors == dot product.
        sims = self._embeddings @ q
        # Top-K nearest neighbors. argpartition is O(N) vs argsort's
        # O(N log N); only worth the extra branch when we have more
        # than K candidates to choose from.
        top_idx = (  # noqa: SIM108 — kept as if/else for the inline complexity note above
            np.argsort(-sims) if sims.shape[0] <= self._k else np.argpartition(-sims, self._k)[: self._k]
        )
        # Group neighbors by which cell they ran on; average outcomes.
        per_cell_sum: dict[str, float] = {}
        per_cell_count: dict[str, int] = {}
        for i in top_idx:
            ck = self._cell_keys[int(i)]
            per_cell_sum[ck] = per_cell_sum.get(ck, 0.0) + float(self._outcomes[int(i)])
            per_cell_count[ck] = per_cell_count.get(ck, 0) + 1
        out: dict[Cell, float] = {}
        for c in candidates:
            ck = f"{c.model} {c.reasoning_effort}"
            n = per_cell_count.get(ck, 0)
            if n == 0:
                # No neighbors used this cell — no opinion yet.
                out[c] = 0.5
                continue
            avg_outcome = per_cell_sum[ck] / n
            # Map outcome [-1, +1] → P(satisfy) [0, 1].
            #   -1 (definitely bad) → 0.0
            #    0 (neutral / unknown) → 0.5
            #   +1 (definitely good) → 1.0
            out[c] = max(0.0, min(1.0, (avg_outcome + 1.0) / 2.0))
        return out

    def reload(self, labeled: Iterable[LabeledRow]) -> None:
        """Rebuild the index from a labeled-row iterable.

        Caller is expected to filter to rows that have prompt_embedding
        populated (the embedding-backfill job fills these); rows with
        empty embeddings are skipped silently here as a defensive
        second check.
        """
        np = self._np
        embeddings: list[Any] = []
        cell_keys: list[str] = []
        outcomes: list[float] = []
        for row in labeled:
            if not row.prompt_embedding:
                continue
            try:
                vec = np.frombuffer(row.prompt_embedding, dtype=np.float32)
            except (ValueError, TypeError):
                continue
            embeddings.append(vec)
            cell_keys.append(row.cell_used)
            outcomes.append(float(row.outcome))
        if not embeddings:
            self._embeddings = np.zeros((0, 1), dtype=np.float32)
            self._cell_keys = []
            self._outcomes = np.zeros((0,), dtype=np.float32)
            return
        self._embeddings = np.stack(embeddings)
        self._cell_keys = cell_keys
        self._outcomes = np.asarray(outcomes, dtype=np.float32)
