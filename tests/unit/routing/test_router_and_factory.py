"""Tests for the Router orchestrator + factory wiring."""

from __future__ import annotations

import asyncio

import pytest

from callosum.cell_grid import Cell
from callosum.routing.factory import RoutingConfig, build_router
from callosum.routing.protocols import CellCapabilities
from callosum.routing.router import NoCompatibleCellError


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


def test_factory_defaults_yield_a_working_cold_start_router() -> None:
    """RoutingConfig() with no overrides → no-op embedding + uniform
    predictor + cost-weighted selector → a working pipeline that picks
    the cheapest compatible cell."""
    router = _build()
    body = {"messages": [{"role": "user", "content": "hello"}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    assert decision.cell == LOCAL  # cheapest compatible
    assert decision.predictor_id == "uniform"


def test_router_picks_only_modality_capable_cell() -> None:
    """Image-bearing prompt → LOCAL drops out (text-only), MID and HIGH
    survive, cheapest (MID) wins."""
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
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    assert decision.cell == REMOTE_MID


def test_router_picks_only_tool_capable_cell() -> None:
    """tools field present → LOCAL drops out (supports_tools=False),
    cheapest tool-capable (MID) wins."""
    router = _build()
    body = {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    assert decision.cell == REMOTE_MID


def test_router_picks_large_context_cell_when_prompt_is_huge() -> None:
    """A 250K-token prompt won't fit in 128K LOCAL; the 256K MID has
    just enough; HIGH has plenty. Cheapest compatible wins."""
    router = _build()
    huge = "x" * 750_000  # ~250K tokens at chars/3
    body = {"messages": [{"role": "user", "content": huge}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    # 250K tokens + 4K headroom > 128K → LOCAL out. 256K > 254K → MID in.
    assert decision.cell == REMOTE_MID


def test_router_raises_when_nothing_compatible() -> None:
    """A prompt whose requirements no cell can meet → NoCompatibleCellError.
    Caller (dispatch) converts to a client-facing 4xx."""
    router = _build()
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_video", "video": "..."}
                ],
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
    the request-log writer can serialize without a custom encoder."""
    router = _build()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID]))
    assert decision.predictions == {
        "local-llm default": 0.5,
        "remote-mid medium": 0.5,
    }
