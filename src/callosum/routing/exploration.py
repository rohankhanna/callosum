"""Coverage-greedy cell ordering for the exploration quota.

The cost-weighted selector picks the *cheapest qualifying* cell. With a
cold-start uniform predictor that collapses all traffic to one cell, the rest
of the (model, reasoning_effort) grid is never sampled, so the quality
predictor, the measured cost model, and the peer-quality matrix have nothing to
learn from for the under-sampled cells.

This module is the reusable coverage engine: given the candidate cells and a
coverage snapshot, order them least-sampled-first so a caller can steer toward
under-covered cells. It is pure and side-effect free — the caller supplies the
coverage and decides the policy (the per-cell minimum-usage floor lives in the
quota enforcer; see docs/architecture/exploration_quota.md).

Salvaged from the removed synthetic exploration tier ().
Mechanism stays pure coverage-greedy; reward-aware exploration
(epsilon-greedy / Thompson / UCB) is a later refinement on the same substrate.
"""

from __future__ import annotations

from collections.abc import Sequence

from callosum.cell_grid import Cell, CellCoverage


def exploration_order(
    candidates: Sequence[Cell],
    coverage: CellCoverage,
) -> list[Cell]:
    """Return candidates reordered least-sampled-first.

    The first element is the cell with the fewest recorded samples (the
    exploration pick); the rest follow in ascending sample order. Ties break by
    the input order, so a router that already ranked candidates by quality/cost
    keeps that ordering among equally-covered cells. Pure and side-effect free —
    the caller supplies the coverage snapshot.
    """
    decorated = sorted(
        enumerate(candidates),
        key=lambda t: (coverage.counts.get(t[1], 0), t[0]),
    )
    return [c for _, c in decorated]
