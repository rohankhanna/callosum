"""Tests for the no-op predictor + cost-weighted selector."""

from __future__ import annotations

from callosum.cell_grid import Cell
from callosum.routing.predictor.uniform import UniformPriorPredictor
from callosum.routing.protocols import CellCapabilities, PromptFeatures
from callosum.routing.selector.cost_weighted import CostWeightedSelector


def _features():
    return PromptFeatures(
        text="x", tokens=100, modalities=frozenset({"text"}), needs_tools=False
    )


def _caps(cost: int) -> CellCapabilities:
    return CellCapabilities(
        context_window=128_000,
        modalities=frozenset({"text"}),
        supports_tools=False,
        cost_rank=cost,
    )


# ---------- UniformPriorPredictor -----------------------------------------


def test_uniform_predictor_returns_half_for_every_cell() -> None:
    p = UniformPriorPredictor()
    cells = [
        Cell(model="a", reasoning_effort="default"),
        Cell(model="b", reasoning_effort="low"),
    ]
    out = p.predict(_features(), cells)
    assert out == {cells[0]: 0.5, cells[1]: 0.5}


def test_uniform_predictor_reload_is_a_noop() -> None:
    p = UniformPriorPredictor()
    # Doesn't crash on any input. Doesn't change subsequent output.
    p.reload([])
    out = p.predict(_features(), [Cell(model="a", reasoning_effort="default")])
    assert out == {Cell(model="a", reasoning_effort="default"): 0.5}


def test_uniform_predictor_has_stable_id() -> None:
    assert UniformPriorPredictor().id == "uniform"


# ---------- CostWeightedSelector ------------------------------------------


def test_selector_picks_cheapest_when_all_qualify() -> None:
    cheap = Cell(model="local", reasoning_effort="default")
    expensive = Cell(model="remote", reasoning_effort="high")
    predictions = {cheap: 0.7, expensive: 0.9}
    caps = {cheap: _caps(0), expensive: _caps(10)}
    chosen = CostWeightedSelector().select(predictions, caps)
    assert chosen == cheap


def test_selector_skips_below_boundary_picks_cheapest_above() -> None:
    """The cheapest cell has P below 0.5; selector picks the next-cheapest
    that's above the boundary. This is how learned predictions promote
    capable cells past unsuitable cheap ones once labels exist."""
    cheap_but_unsuitable = Cell(model="local", reasoning_effort="default")
    capable = Cell(model="remote-mid", reasoning_effort="medium")
    expensive_capable = Cell(model="remote-high", reasoning_effort="high")
    predictions = {
        cheap_but_unsuitable: 0.3,  # below boundary
        capable: 0.6,
        expensive_capable: 0.9,
    }
    caps = {
        cheap_but_unsuitable: _caps(0),
        capable: _caps(5),
        expensive_capable: _caps(10),
    }
    chosen = CostWeightedSelector().select(predictions, caps)
    assert chosen == capable  # cheapest above boundary


def test_selector_falls_back_to_cheapest_when_nobody_qualifies() -> None:
    """Every cell predicted below the boundary → best-effort: pick
    cheapest anyway. Refusing to route is worse than serving with low
    expected quality."""
    a = Cell(model="a", reasoning_effort="default")
    b = Cell(model="b", reasoning_effort="default")
    predictions = {a: 0.2, b: 0.4}
    caps = {a: _caps(0), b: _caps(10)}
    chosen = CostWeightedSelector().select(predictions, caps)
    assert chosen == a  # cheapest even though below boundary


def test_selector_cold_start_uniform_predictor_picks_cheapest() -> None:
    """End-to-end cold-start scenario: uniform predictor returns 0.5 for
    everyone → all qualify → cheapest wins. This is the local-first
    default the user wants."""
    p = UniformPriorPredictor()
    local = Cell(model="local", reasoning_effort="default")
    remote = Cell(model="remote", reasoning_effort="medium")
    predictions = p.predict(_features(), [local, remote])
    caps = {local: _caps(0), remote: _caps(10)}
    chosen = CostWeightedSelector().select(predictions, caps)
    assert chosen == local


def test_selector_raises_on_empty_predictions() -> None:
    """Empty predictions is a programmer error — should never happen
    if the capability filter ran first. Surface immediately rather
    than picking some arbitrary 'nothing'."""
    import pytest
    with pytest.raises(ValueError):
        CostWeightedSelector().select({}, {})
