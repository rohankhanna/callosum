"""Efficiency model for cost-optimal routing.

v1: averages total_tokens per cell (cheaper = lower tokens)
v2 hook: quality-weighted once labels accumulate (quality_score / total_tokens)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from codex_proxy.cell_grid import Cell


class EfficiencyModel:
    """Per-cell efficiency scores computed from logged requests.

    v1 routes to the cell with lowest avg_total_tokens (as a proxy for cost).
    v2 will route to highest efficiency = avg(quality_score) / avg(total_tokens).
    """

    def __init__(
        self,
        scores: dict[tuple[str, str], float],
        n_samples: dict[tuple[str, str], int],
        ready: bool,
    ) -> None:
        """Initialize with pre-computed efficiency scores.

        `scores` maps (model, reasoning_effort) to a float. In v1, this is
        avg_total_tokens. Lower is better.

        `n_samples` maps the same key to the number of requests that went into
        the average (for logging/debugging).

        `ready` is True when all currently-visible cells have sufficient data.
        """
        self._scores = scores
        self._n_samples = n_samples
        self._ready = ready

    @property
    def is_ready(self) -> bool:
        """True when the model has sufficient training data for all live cells."""
        return self._ready

    @property
    def scores(self) -> dict[tuple[str, str], float]:
        """Efficiency scores per cell. For logging only — do not modify."""
        return self._scores

    @classmethod
    def from_db(
        cls,
        path: Path | None,
        cells: list[Cell],
        *,
        min_samples_per_cell: int = 30,
    ) -> "EfficiencyModel":
        """Load and compute efficiency scores from the usage log database.

        `cells` is the current live grid (may include dynamically-discovered models).
        For the model to be ready, every cell must have >= min_samples_per_cell
        successful requests with actual token counts.

        Returns NotReady model if no DB exists or insufficient data.
        """
        scores: dict[tuple[str, str], float] = {}
        n_samples: dict[tuple[str, str], int] = {}

        if path is None or not path.exists():
            return cls(scores={}, n_samples={}, ready=False)

        try:
            conn = sqlite3.connect(path, check_same_thread=False)
            cursor = conn.cursor()
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
            rows = cursor.fetchall()
            conn.close()

            for model, effort, avg_tokens, n in rows:
                key = (model, effort)
                scores[key] = avg_tokens
                n_samples[key] = n
        except (sqlite3.Error, OSError):
            # DB access failed, can't compute — return not ready
            return cls(scores={}, n_samples={}, ready=False)

        # Ready only if every cell in the live grid has >= min_samples
        ready = all(
            n_samples.get((c.model, c.reasoning_effort), 0) >= min_samples_per_cell
            for c in cells
        )

        return cls(scores=scores, n_samples=n_samples, ready=ready)

    def best_cell(
        self,
        cells: list[Cell],
        *,
        session_prompt_tokens: int | None = None,
        router_context_safety_margin: int = 8192,
    ) -> Cell:
        """Return the cell with best efficiency from the candidate list.

        If session_prompt_tokens is provided, filter out cells whose context
        window is too small. Falls back to largest-context if all are filtered.

        Returns the cell with lowest efficiency score (lowest avg_tokens in v1).
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
                # All filtered out — fall back to largest context window
                candidates = sorted(
                    candidates,
                    key=lambda c: (c.context_window is None, -(c.context_window or 0))
                )

        # Pick the candidate with the lowest score (cheapest in v1)
        best = min(
            candidates,
            key=lambda c: self._scores.get((c.model, c.reasoning_effort), float("inf"))
        )
        return best
