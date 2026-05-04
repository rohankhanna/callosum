from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from codex_proxy.cell_grid import (
    Cell,
    build_cells,
    coverage_from_db,
)


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

    def choose(self, *, allowed_models: frozenset[str] | None = None) -> RouterDecision:
        """Return the cell to vary into for the next auto-learning request.

        When `allowed_models` is provided, only cells whose model is in that
        set are candidates. Used by the synthetic worker to constrain the
        explorer to models the forced backend actually advertises.
        """
        cells = self._cells_fn()
        if allowed_models is not None:
            cells = [c for c in cells if c.model in allowed_models]
        if not cells:
            # No cells available (e.g. backends haven't refreshed their
            # catalogs yet). Caller will see a clear failure rather than
            # routing to a phantom model.
            msg = "explorer router has no cells"
            if allowed_models is not None:
                msg += f" for models {sorted(allowed_models)}"
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
    """Cost-optimal router. Stub for now — returns a not-trained signal so the
    'auto' virtual model can fail loud rather than silently doing the wrong
    thing. Fills in once the cost model is fit on a complete cell grid.
    """

    class NotTrained(RuntimeError):
        pass

    def __init__(self, usage_log_path: Path | None) -> None:
        self._usage_log_path = usage_log_path

    def choose(self, *, model_hint: str | None = None) -> RouterDecision:
        del model_hint
        raise ExploiterRouter.NotTrained(
            "auto-routing is not ready: the cost model has not been trained yet."
            " Use `auto-learning` to keep collecting data, or pick a model"
            " explicitly (model-a0e7, model-a0c3, model-a0b8, model-a0e6)."
        )
