from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codex_proxy.cell_grid import (
    Cell,
    build_cells,
    coverage_from_db,
)
from codex_proxy.efficiency_model import EfficiencyModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RouterDecision:
    """A request body rewrite the router wants applied before dispatch.

    `cell` carries the (model, reasoning_effort) the router chose. The dispatch
    layer overwrites the request body's `model` and `reasoning.effort` fields
    with these values, then routes through the normal selector.
    """

    cell: Cell
    reason: str  # short tag for logging — e.g. "round-robin: cell at min(coverage)"


class ExplorerRouter:
    """Picks the next (model, reasoning_effort) cell to fill data for the cost
    model. v1 is round-robin: always pick the cell with the fewest samples,
    breaking ties by the cell-grid's natural order.

    Stateless — every choose() reads current coverage from the usage_log so
    decisions reflect the latest data, including from concurrent calls.

    `routing_mode` selects which tier of samples to count when computing
    coverage. Organic ('auto-learning') and synthetic ('auto-learning-synthetic')
    keep independent coverage so neither tier advances the other's least-
    sampled query.
    """

    def __init__(
        self,
        usage_log_path: Path | None,
        *,
        routing_mode: str = "auto-learning",
        cells_fn: Callable[[], list[Cell]] | None = None,
    ) -> None:
        self._usage_log_path = usage_log_path
        self._routing_mode = routing_mode
        # `cells_fn` is called on every choose() so the cell grid can adapt
        # to dynamic upstream catalog discovery without restarting the proxy.
        # Defaults to `build_cells()` (static DEFAULT_MODELS) for backward
        # compat in tests and cold-start scenarios.
        self._cells_fn: Callable[[], list[Cell]] = cells_fn or build_cells

    @property
    def cells(self) -> list[Cell]:
        """Return the current cell grid. Recomputed on each access via
        the configured cells_fn — never returns a stale snapshot.
        """
        return list(self._cells_fn())

    def choose(
        self,
        *,
        allowed_models: frozenset[str] | None = None,
        session_prompt_tokens: int | None = None,
        router_context_safety_margin: int = 8192,
    ) -> RouterDecision:
        """Return the cell to vary into for the next auto-learning request.

        When `allowed_models` is provided, only cells whose model is in that
        set are candidates. Used by the synthetic worker to constrain the
        explorer to models the forced backend actually advertises.

        When `session_prompt_tokens` is provided (current session's context size),
        filter out cells whose model has a known context window that would be
        exceeded. Add `router_context_safety_margin` as headroom.
        """
        cells = self._cells_fn()
        if allowed_models is not None:
            cells = [c for c in cells if c.model in allowed_models]

        # Filter by context window if we know the current session size
        if session_prompt_tokens is not None:
            min_required_window = session_prompt_tokens + router_context_safety_margin
            cells = [
                c
                for c in cells
                if c.context_window is None or c.context_window >= min_required_window
            ]
            # If all cells were filtered out, fall back to the largest known context window
            if not cells:
                cells = self._cells_fn()
                if allowed_models is not None:
                    cells = [c for c in cells if c.model in allowed_models]
                # Sort by context_window descending, None goes last
                cells.sort(
                    key=lambda c: (c.context_window is None, -(c.context_window or 0))
                )

        if not cells:
            # No cells available (e.g. backends haven't refreshed their
            # catalogs yet). Caller will see a clear failure rather than
            # routing to a phantom model.
            msg = "explorer router has no cells"
            if allowed_models is not None:
                msg += f" for models {sorted(allowed_models)}"
            if session_prompt_tokens is not None:
                msg += f"; session has {session_prompt_tokens} tokens, no model fits"
            msg += "; backends not yet ready"
            raise RuntimeError(msg)
        coverage = (
            coverage_from_db(self._usage_log_path, cells, routing_mode=self._routing_mode)
            if self._usage_log_path is not None
            else None
        )
        if coverage is None:
            # No usage log configured — can't observe coverage. Default to the
            # first cell so behavior is deterministic in tests.
            return RouterDecision(
                cell=cells[0],
                reason="no usage_log configured; defaulting to first cell",
            )
        chosen = coverage.least_sampled(cells)
        return RouterDecision(
            cell=chosen,
            reason=f"round-robin: {coverage.counts[chosen]} samples (min)",
        )


class ExploiterRouter:
    """Cost-optimal router. Routes to the cell with lowest average token cost.

    v1 metric: avg_total_tokens per cell (cheaper = lower tokens, across all sessions)
    v2 will use: quality_score / total_tokens once labels accumulate

    Fits the efficiency model from the usage log on demand (via fit() or fit_if_ready()).
    The model is ready only when every live cell has >= min_samples_per_cell successful
    requests.
    """

    class NotTrained(RuntimeError):
        pass

    def __init__(
        self,
        usage_log_path: Path | None,
        *,
        cells_fn: Callable[[], list[Cell]] | None = None,
    ) -> None:
        self._usage_log_path = usage_log_path
        self._cells_fn: Callable[[], list[Cell]] = cells_fn or build_cells
        self._model: EfficiencyModel | None = None
        self._last_fit: float = 0.0

    def fit(self, min_samples_per_cell: int = 30) -> None:
        """Refit the efficiency model from the usage log.

        Safe to call from any context (thread, task). Updates _model and _last_fit.
        """
        cells = self._cells_fn()
        self._model = EfficiencyModel.from_db(
            self._usage_log_path,
            cells,
            min_samples_per_cell=min_samples_per_cell,
        )
        self._last_fit = time.time()
        if self._model.is_ready:
            logger.warning(
                "✓ ExploiterRouter READY. auto routing is active. "
                "Routing to cheapest cells based on learned token costs."
            )
        else:
            # Count how many cells have data
            with_data = sum(1 for c in cells if (c.model, c.reasoning_effort) in self._model.scores)
            logger.info(
                "ExploiterRouter training: %d/%d cells have data. "
                "auto routing unavailable until all cells reach %d samples. "
                "Use model: 'auto-learning' for now.",
                with_data, len(cells),
                30,  # min_samples_per_cell hard-coded for clarity in log
            )

    def choose(
        self,
        *,
        model_hint: str | None = None,
        session_prompt_tokens: int | None = None,
        router_context_safety_margin: int = 8192,
    ) -> RouterDecision:
        del model_hint  # Hook for future: complexity-aware routing
        if self._model is None or not self._model.is_ready:
            raise ExploiterRouter.NotTrained(
                "auto-routing is not ready: the cost model has not been trained yet. "
                "Use `auto-learning` to keep collecting data, or pick a model explicitly."
            )
        cells = self._cells_fn()
        cell = self._model.best_cell(
            cells,
            session_prompt_tokens=session_prompt_tokens,
            router_context_safety_margin=router_context_safety_margin,
        )
        avg_tokens = self._model.scores.get((cell.model, cell.reasoning_effort), 0)
        return RouterDecision(
            cell=cell,
            reason=f"exploiter: cheapest cell avg_tokens={avg_tokens:.0f}",
        )
