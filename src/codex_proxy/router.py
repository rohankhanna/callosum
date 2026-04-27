from __future__ import annotations

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
    ) -> None:
        self._usage_log_path = usage_log_path
        self._routing_mode = routing_mode
        self._cells = build_cells()

    @property
    def cells(self) -> list[Cell]:
        return list(self._cells)

    def choose(self) -> RouterDecision:
        """Return the cell to vary into for the next auto-learning request."""
        coverage = (
            coverage_from_db(self._usage_log_path, self._cells, routing_mode=self._routing_mode)
            if self._usage_log_path is not None
            else None
        )
        if coverage is None:
            # No usage log configured — can't observe coverage. Default to the
            # first cell so behavior is deterministic in tests.
            return RouterDecision(
                cell=self._cells[0],
                reason="no usage_log configured; defaulting to first cell",
            )
        chosen = coverage.least_sampled(self._cells)
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
