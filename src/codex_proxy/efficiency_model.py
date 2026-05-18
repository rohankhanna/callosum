"""Efficiency model for cost-optimal routing.

v1: averages total_tokens per cell. Complexity-agnostic.
v2: averages total_tokens per (complexity_class, model, reasoning_effort).
    Falls back to v1 cell averages when a complexity bucket has no data.

Both models are computed in one DB pass and live in the same object; the
caller passes `complexity` into `best_cell` to engage v2 lookup, or omits
it to get v1 behavior.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codex_proxy.cell_grid import Cell


class EfficiencyModel:
    """Per-cell efficiency scores computed from logged requests.

    - v1 scores: `(model, reasoning_effort) -> avg_total_tokens` over all
      auto-learning rows regardless of prompt complexity. Lower is cheaper.
    - v2 scores: `(complexity_class, model, reasoning_effort) -> avg_total_tokens`
      computed only from rows where `prompt_complexity_class` is populated.
      Lets the router pick the cheapest cell *for prompts of this complexity*
      instead of an average smeared across all complexities.

    `best_cell(complexity=None)` uses v1. `best_cell(complexity=N)` uses v2
    when that bucket has data, falling back to v1 for any cell missing from
    the v2 map (so a partially-trained v2 still routes intelligently).
    """

    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        n_samples: dict[tuple[str, str], int],
        ready: bool,
        scores_by_complexity: dict[tuple[int, str, str], float] | None = None,
        n_samples_by_complexity: dict[tuple[int, str, str], int] | None = None,
    ) -> None:
        self._scores = scores
        self._n_samples = n_samples
        self._ready = ready
        self._scores_by_complexity = scores_by_complexity or {}
        self._n_samples_by_complexity = n_samples_by_complexity or {}

    @property
    def is_ready(self) -> bool:
        """True when the v1 model has sufficient data for every live cell."""
        return self._ready

    @property
    def scores(self) -> dict[tuple[str, str], float]:
        """v1 efficiency scores per cell. For logging only — do not modify."""
        return self._scores

    @property
    def scores_by_complexity(self) -> dict[tuple[int, str, str], float]:
        """v2 efficiency scores per (complexity, model, effort). Read-only."""
        return self._scores_by_complexity

    def has_complexity_data(self, complexity: int) -> bool:
        """True iff at least one cell has training data for this complexity bucket."""
        return any(c == complexity for c, _, _ in self._scores_by_complexity)

    @classmethod
    def from_db(
        cls,
        path: Path | None,
        cells: list[Cell],
        *,
        min_samples_per_cell: int = 30,
    ) -> "EfficiencyModel":
        """Load and compute efficiency scores from the usage log database.

        Computes both the v1 (per-cell) and v2 (per-complexity-per-cell) maps
        in a single DB connection. `is_ready` reflects v1 readiness; v2 is
        used opportunistically per-complexity-bucket via `has_complexity_data`.
        """
        scores: dict[tuple[str, str], float] = {}
        n_samples: dict[tuple[str, str], int] = {}
        scores_by_complexity: dict[tuple[int, str, str], float] = {}
        n_samples_by_complexity: dict[tuple[int, str, str], int] = {}

        if path is None or not path.exists():
            return cls(scores={}, n_samples={}, ready=False)

        try:
            conn = sqlite3.connect(path, check_same_thread=False)
            cursor = conn.cursor()
            # v1: per-cell averages across all complexities.
            cursor.execute(
                """
                SELECT model, reasoning_effort,
                       AVG(total_tokens) AS avg_tokens,
                       COUNT(*) AS n
                FROM requests
                WHERE routing_mode IN ('auto-learning', 'auto-learning-synthetic')
                  AND status = 200
                  AND model IS NOT NULL
                  AND reasoning_effort IS NOT NULL
                  AND total_tokens IS NOT NULL
                GROUP BY model, reasoning_effort
                """
            )
            for model, effort, avg_tokens, n in cursor.fetchall():
                scores[(model, effort)] = avg_tokens
                n_samples[(model, effort)] = n

            # v2: per-(complexity, cell) averages. Only rows where the
            # classifier marker was successfully extracted contribute here.
            cursor.execute(
                """
                SELECT prompt_complexity_class, model, reasoning_effort,
                       AVG(total_tokens) AS avg_tokens,
                       COUNT(*) AS n
                FROM requests
                WHERE routing_mode IN ('auto-learning', 'auto-learning-synthetic')
                  AND status = 200
                  AND model IS NOT NULL
                  AND reasoning_effort IS NOT NULL
                  AND total_tokens IS NOT NULL
                  AND prompt_complexity_class IS NOT NULL
                GROUP BY prompt_complexity_class, model, reasoning_effort
                """
            )
            for complexity, model, effort, avg_tokens, n in cursor.fetchall():
                key = (int(complexity), model, effort)
                scores_by_complexity[key] = avg_tokens
                n_samples_by_complexity[key] = n

            conn.close()
        except (sqlite3.Error, OSError):
            return cls(scores={}, n_samples={}, ready=False)

        ready = all(
            n_samples.get((c.model, c.reasoning_effort), 0) >= min_samples_per_cell
            for c in cells
        )

        return cls(
            scores=scores,
            n_samples=n_samples,
            ready=ready,
            scores_by_complexity=scores_by_complexity,
            n_samples_by_complexity=n_samples_by_complexity,
        )

    def best_cell(
        self,
        cells: list[Cell],
        *,
        complexity: int | None = None,
        session_prompt_tokens: int | None = None,
        router_context_safety_margin: int = 8192,
    ) -> Cell:
        """Return the cell with best (lowest) efficiency from the candidate list.

        When `complexity` is set and v2 has any data for that bucket, scores
        come from the (complexity, model, effort) map with a per-cell fallback
        to the v1 score (so a partially-trained v2 still picks intelligently).
        When `complexity` is None or v2 has no data for that bucket, falls
        back entirely to v1 scoring.

        Cells whose context window is too small for the live prompt size are
        filtered out (with a largest-context fallback if all are filtered).
        """
        candidates = cells
        if session_prompt_tokens is not None:
            min_required = session_prompt_tokens + router_context_safety_margin
            filtered = [
                c for c in candidates
                if c.context_window is None or c.context_window >= min_required
            ]
            if filtered:
                candidates = filtered
            else:
                candidates = sorted(
                    candidates,
                    key=lambda c: (c.context_window is None, -(c.context_window or 0))
                )

        use_v2 = complexity is not None and self.has_complexity_data(complexity)

        def score_of(c: Cell) -> float:
            if use_v2:
                key = (complexity, c.model, c.reasoning_effort)
                v2_score = self._scores_by_complexity.get(key)
                if v2_score is not None:
                    return v2_score
            # Fallback to v1 cell score; +inf if neither has data for this cell.
            return self._scores.get((c.model, c.reasoning_effort), float("inf"))

        return min(candidates, key=score_of)
