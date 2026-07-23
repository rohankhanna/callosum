"""Extension point: coverage-greedy cell ordering.

The coverage enforcement (minimum-usage floor on organic traffic) is private.
This module preserves the pure ``coverage_order`` function used by the routing
pipeline; the quota enforcer that called it is stubbed in ``routing/quota.py``.
"""

from __future__ import annotations

from collections.abc import Sequence

from callosum.cell_grid import Cell, CellCoverage


def coverage_order(
    candidates: Sequence[Cell],
    coverage: CellCoverage,
) -> list[Cell]:
    """Return *candidates* reordered least-sampled-first."""
    decorated = sorted(
        enumerate(candidates),
        key=lambda t: (coverage.counts.get(t[1], 0), t[0]),
    )
    return [c for _, c in decorated]
