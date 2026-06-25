"""Extension point: per-cell minimum-coverage quota.

The quota enforcement (per-cell minimum-usage floor) is private.  This stub
preserves the interface used by the router but performs no enforcement:
``effective_floor_pct`` returns 0, ``select_quota_deficit_cell`` returns None,
and ``filter_cells_by_effort_cap`` returns all candidates unchanged.

To implement custom quota enforcement, override these functions in your own
module and wire it into the router.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from callosum.cell_grid import Cell, CellCoverage


def effective_floor_pct(
    n_candidates: int,
    *,
    budget_pct: float,
    min_floor_pct: float = 0.0,
) -> float:
    """Stub: no per-cell floor (returns 0)."""
    return 0.0


def select_quota_deficit_cell(
    candidates: Sequence[Cell],
    coverage: CellCoverage,
    *,
    floor_pct: float,
    cooldown: frozenset[Cell] | None = None,
) -> Cell | None:
    """Stub: no forced coverage (returns None)."""
    return None


def filter_cells_by_effort_cap(
    candidates: Sequence[Cell],
    coverage: CellCoverage,
    *,
    capped_efforts: frozenset[str],
    immune_models: frozenset[str] = frozenset(),
    cap_pct: float,
) -> tuple[Cell, ...]:
    """Stub: no effort capping (returns all candidates)."""
    return tuple(candidates)


def min_coverage_quota_report(
    cells: Sequence[Cell],
    coverage: CellCoverage,
    *,
    floor_pct: float,
) -> dict[str, Any]:
    """Stub: empty coverage report."""
    return {}
