"""Arm-level exploration for synthetic auto-learning traffic.

The cost-weighted selector picks the *cheapest qualifying* cell. With a
flat remote cost_rank and a cold-start uniform predictor, that collapses
ALL remote traffic to a single model — every synthetic request lands on the
same cell and the rest of the (model, reasoning_effort) grid never gets
sampled. No coverage means the downstream quality predictor, the measured
cost model, and the future contextual bandit have nothing to learn from for
the under-sampled cells.

Exploration fixes that for SYNTHETIC traffic only: instead of the cheapest
cell, synthetic requests deliberately target the *least-sampled* compatible
cell, so coverage accumulates evenly across the live cell grid. Organic
traffic is never touched — it keeps the cost/quality-optimal pick.

This is the coverage engine that feeds:
  * the cross-model peer-quality matrix (router quality signal),
  * the measured-quota cost model (:mod:`callosum.routing.cost_model`),
  * the eventual contextual-bandit reward.

Mechanism: pure coverage-greedy (round-robin over under-sampled arms). It is
deterministic and ties break by the candidate order the router produced, so
the choice is reproducible and testable. Reward-aware exploration
(epsilon-greedy / Thompson / UCB) is a later refinement layered on the same
coverage substrate — kept out of v1 on purpose, since the immediate need is
*coverage*, not reward maximization.
"""

from __future__ import annotations

from collections.abc import Sequence

from callosum.cell_grid import Cell, CellCoverage

# The virtual model name a client sends to opt a request into the synthetic
# auto-learning explorer tier. Kept in sync with
# callosum.cell_grid.VIRTUAL_MODELS; this is the one that should explore.
SYNTHETIC_ROUTING_MODE = "auto-learning-synthetic"


def is_exploration_request(requested_model: str | None) -> bool:
    """True when this request belongs to the synthetic-explorer tier.

    Only synthetic traffic explores; organic auto / auto-learning and
    concrete model requests keep the cost/quality-optimal routing decision.
    """
    return requested_model == SYNTHETIC_ROUTING_MODE


def exploration_order(
    candidates: Sequence[Cell],
    coverage: CellCoverage,
) -> list[Cell]:
    """Return candidates reordered least-sampled-first.

    The first element is the cell with the fewest recorded synthetic samples
    (the exploration pick); the rest follow in ascending sample order. Ties
    break by the input order, so a router that already ranked candidates by
    quality/cost keeps that ordering among equally-covered cells. Pure and
    side-effect free — the caller supplies the coverage snapshot.
    """
    decorated = sorted(
        enumerate(candidates),
        key=lambda t: (coverage.counts.get(t[1], 0), t[0]),
    )
    return [c for _, c in decorated]
