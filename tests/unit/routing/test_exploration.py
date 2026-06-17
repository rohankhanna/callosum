from __future__ import annotations

from callosum.cell_grid import Cell, CellCoverage
from callosum.routing.exploration import (
    SYNTHETIC_ROUTING_MODE,
    exploration_order,
    is_exploration_request,
)


def _cell(model: str, effort: str) -> Cell:
    return Cell(model=model, reasoning_effort=effort)


def test_is_exploration_request_only_for_synthetic() -> None:
    assert is_exploration_request(SYNTHETIC_ROUTING_MODE) is True
    assert is_exploration_request("auto") is False
    assert is_exploration_request("auto-learning") is False
    assert is_exploration_request("model-a0e8") is False
    assert is_exploration_request(None) is False


def test_exploration_order_puts_least_sampled_first() -> None:
    a = _cell("model-a0e8", "high")
    b = _cell("model-a0e7", "low")
    c = _cell("model-a0c3", "medium")
    # b has the fewest samples -> must be picked first.
    coverage = CellCoverage(counts={a: 10, b: 1, c: 5})
    ordered = exploration_order([a, b, c], coverage)
    assert ordered[0] == b
    assert ordered == [b, c, a]


def test_exploration_order_treats_missing_as_zero() -> None:
    a = _cell("model-a0e8", "high")
    b = _cell("model-a0e7", "low")
    # b absent from coverage -> counts as 0 -> least sampled.
    coverage = CellCoverage(counts={a: 3})
    ordered = exploration_order([a, b], coverage)
    assert ordered[0] == b


def test_exploration_order_breaks_ties_by_input_order() -> None:
    a = _cell("model-a0e8", "high")
    b = _cell("model-a0e7", "low")
    c = _cell("model-a0c3", "medium")
    coverage = CellCoverage(counts={a: 0, b: 0, c: 0})
    # All equal -> preserve the order the router produced.
    assert exploration_order([a, b, c], coverage) == [a, b, c]
    assert exploration_order([c, b, a], coverage) == [c, b, a]


def test_exploration_order_does_not_mutate_input() -> None:
    a = _cell("model-a0e8", "high")
    b = _cell("model-a0e7", "low")
    original = [a, b]
    coverage = CellCoverage(counts={a: 5, b: 1})
    _ = exploration_order(original, coverage)
    assert original == [a, b]
