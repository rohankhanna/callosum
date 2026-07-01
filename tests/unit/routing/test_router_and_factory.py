"""Tests for the Router orchestrator + factory wiring."""

from __future__ import annotations

import asyncio

import pytest

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.embedding.noop import NoopEmbeddingProvider
from callosum.routing.factory import RoutingConfig, build_router
from callosum.routing.protocols import CellCapabilities, LabeledRow, PromptFeatures
from callosum.routing.router import NoCompatibleCellError, Router
from callosum.routing.selector.cost_weighted import CostWeightedSelector
from callosum.routing.usage_estimate import Estimate, EstimateInput, OutputTokenForecast

# Hand-built capabilities map for testing.
LOCAL = Cell(model="local-llm", reasoning_effort="default")
REMOTE_MID = Cell(model="remote-mid", reasoning_effort="medium")
REMOTE_HIGH = Cell(model="remote-high", reasoning_effort="high")

CAPS = {
    LOCAL: CellCapabilities(
        context_window=128_000,
        modalities=frozenset({"text"}),
        supports_tools=False,
        cost_rank=0,
    ),
    REMOTE_MID: CellCapabilities(
        context_window=256_000,
        modalities=frozenset({"text", "image"}),
        supports_tools=True,
        cost_rank=10,
    ),
    REMOTE_HIGH: CellCapabilities(
        context_window=400_000,
        modalities=frozenset({"text", "image", "audio"}),
        supports_tools=True,
        cost_rank=20,
    ),
}


def _build():
    return build_router(RoutingConfig(), capabilities_of=CAPS.__getitem__)


class _FixedPredictor:
    id = "fixed"

    def __init__(self, scores: dict[Cell, float]) -> None:
        self._scores = scores

    def predict(self, features: PromptFeatures, candidates: list[Cell]) -> dict[Cell, float]:
        del features
        return {cell: self._scores[cell] for cell in candidates}

    def reload(self, labeled: list[LabeledRow]) -> None:
        del labeled


class _FixedOutputForecaster:
    def forecast(self, cell: Cell, input_tokens: int) -> OutputTokenForecast:
        del cell, input_tokens
        return OutputTokenForecast(p50=100.0, p95=100.0, source="fixed", n_obs=1)


class _FixedTimeEstimator:
    id = "fixed-time"
    unit = "ms"

    def __init__(self, estimates: dict[Cell, float]) -> None:
        self._estimates = estimates

    def estimate(self, inp: EstimateInput) -> Estimate:
        point = self._estimates[inp.cell]
        return Estimate(
            point=point,
            low=point,
            high=point,
            unit="ms",
            source="fixed",
            verifiable=True,
        )


def test_factory_defaults_yield_a_working_cold_start_router() -> None:
    """RoutingConfig() with no overrides → no-op embedding + uniform
    predictor + cost-weighted selector → a working pipeline. With no quality
    signal, cold start EXPLORES: a random capable cell is chosen."""
    router = _build()
    body = {"messages": [{"role": "user", "content": "hello"}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    assert decision.cell in (LOCAL, REMOTE_MID, REMOTE_HIGH)
    assert decision.predictor_id == "uniform"


def test_cold_start_explores_randomly_across_capable_cells() -> None:
    """With a uniform (undifferentiated) predictor there is no quality signal,
    so cold start must NOT collapse to one cell (cheapest, or a heuristic
    'hardest') — it picks randomly so coverage accrues unbiased. Over many
    runs every capable cell should get chosen at least once."""
    router = _build()
    body = {"messages": [{"role": "user", "content": "hello"}]}
    seen = {
        asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH])).cell
        for _ in range(200)
    }
    assert seen == {LOCAL, REMOTE_MID, REMOTE_HIGH}


def test_cold_start_never_picks_an_incompatible_cell() -> None:
    """Random exploration is among CAPABILITY-FILTERED cells only — a
    text-only cell is never chosen for an image prompt, across many runs."""
    router = _build()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
    }
    for _ in range(100):
        decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
        assert decision.cell in (REMOTE_MID, REMOTE_HIGH)  # LOCAL is text-only
        assert LOCAL not in decision.candidates


def test_router_picks_only_tool_capable_cell() -> None:
    """tools field present → LOCAL drops out (supports_tools=False); only
    tool-capable cells remain in the pool, across many runs."""
    router = _build()
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    for _ in range(100):
        decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
        assert decision.cell in (REMOTE_MID, REMOTE_HIGH)
        assert LOCAL not in decision.candidates


def test_cold_start_explores_only_among_cells_that_fit_the_window() -> None:
    """A 250K-token prompt still fits MID (256K) and HIGH (400K) but overflows
    LOCAL (128K). Cold-start exploration restricts the random pick to fully-
    fitting cells, so LOCAL is never the primary — but it stays a candidate
    (soft preference, not exclusion)."""
    router = _build()
    huge = "x" * 750_000  # ~250K tokens at chars/3
    body = {"messages": [{"role": "user", "content": huge}]}
    for _ in range(100):
        decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
        assert decision.cell in (REMOTE_MID, REMOTE_HIGH)
        assert LOCAL in decision.candidates


def test_router_falls_back_to_some_cell_when_nothing_fits() -> None:
    """When every cell overflows the (over-counted) estimate, exploration
    falls back to the whole compatible pool and still returns a decision;
    the retry order still prefers the biggest-window cells."""
    router = _build()
    huge = "x" * 1_500_000  # ~500K tokens — exceeds even HIGH's 400K window
    body = {"messages": [{"role": "user", "content": huge}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    assert decision.cell in (LOCAL, REMOTE_MID, REMOTE_HIGH)
    # Retry order (the non-primary candidates) prefers larger windows.
    rest = [c for c in decision.candidates if c != decision.cell]
    if REMOTE_HIGH in rest and LOCAL in rest:
        assert rest.index(REMOTE_HIGH) < rest.index(LOCAL)


def test_router_raises_when_nothing_compatible() -> None:
    """A prompt whose requirements no cell can meet → NoCompatibleCellError.
    Caller (dispatch) converts to a client-facing 4xx."""
    router = _build()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "input_video", "video": "..."}],
            }
        ]
    }
    # None of LOCAL/MID/HIGH supports video.
    with pytest.raises(NoCompatibleCellError):
        asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))


def test_factory_rejects_unknown_impl_names() -> None:
    """Mistyped config key surfaces immediately at startup, not later
    when a request fires."""
    with pytest.raises(ValueError, match="embedding_provider"):
        build_router(
            RoutingConfig(embedding_provider="does-not-exist"),
            capabilities_of=CAPS.__getitem__,
        )
    with pytest.raises(ValueError, match="quality_predictor"):
        build_router(
            RoutingConfig(quality_predictor="does-not-exist"),
            capabilities_of=CAPS.__getitem__,
        )
    with pytest.raises(ValueError, match="cell_selector"):
        build_router(
            RoutingConfig(cell_selector="does-not-exist"),
            capabilities_of=CAPS.__getitem__,
        )


def test_decision_carries_predictions_keyed_by_cell_string() -> None:
    """RoutingDecision.predictions is keyed by 'model effort' strings so
    the request-log writer can serialize without a custom encoder. In cold
    start these are the raw uniform priors (0.5), not heuristic-adjusted —
    the difficulty heuristic was removed in favor of random exploration."""
    router = _build()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID]))
    assert decision.predictions == {
        "local-llm default": 0.5,
        "remote-mid medium": 0.5,
    }


def test_router_wires_time_estimates_into_bounded_selector_preference() -> None:
    slow = Cell(model="same-cost-slow", reasoning_effort="medium")
    fast = Cell(model="same-cost-fast", reasoning_effort="medium")
    caps = {
        slow: CellCapabilities(
            context_window=128_000,
            modalities=frozenset({"text"}),
            supports_tools=False,
            cost_rank=0,
        ),
        fast: CellCapabilities(
            context_window=128_000,
            modalities=frozenset({"text"}),
            supports_tools=False,
            cost_rank=0,
        ),
    }
    router = Router(
        embedding=NoopEmbeddingProvider(),
        predictor=_FixedPredictor({slow: 0.82, fast: 0.80}),
        selector=CostWeightedSelector(),
        capability_filter=CapabilityFilter(capabilities_of=caps.__getitem__),
        time_estimator=_FixedTimeEstimator({slow: 900.0, fast: 200.0}),  # type: ignore[arg-type]
        output_forecaster=_FixedOutputForecaster(),  # type: ignore[arg-type]
    )
    decision = asyncio.run(
        router.route({"messages": [{"role": "user", "content": "hello"}]}, [slow, fast])
    )
    assert decision.cell == fast
    assert decision.predictions == {
        "same-cost-slow medium": 0.82,
        "same-cost-fast medium": 0.8,
    }
    assert decision.time_estimates_ms == {
        "same-cost-slow medium": 900.0,
        "same-cost-fast medium": 200.0,
    }
