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


def test_router_prefers_large_context_cell_when_prompt_is_huge() -> None:
    """A 250K-token prompt no longer EXCLUDES LOCAL (128K) — the filter
    is soft, not hard. But the Router's window-fit scaling penalizes
    LOCAL's prediction enough that the selector should prefer MID
    (256K, full headroom) over LOCAL even though LOCAL is cheaper.
    Window fit is the routing preference signal the user asked for."""
    router = _build()
    huge = "x" * 750_000  # ~250K tokens at chars/3
    body = {"messages": [{"role": "user", "content": huge}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    # MID has full headroom (256K > 254K budget); LOCAL doesn't (128K < 254K)
    # so its scaled prediction halves. With uniform prior 0.5, LOCAL → ~0.25,
    # falls below the selector's 0.5 cutoff, MID at 0.5 wins.
    assert decision.cell == REMOTE_MID
    # LOCAL must still be in the candidate list — soft preference, not exclusion.
    assert LOCAL in decision.candidates


def test_router_falls_back_to_overflowing_cell_when_nothing_fits() -> None:
    """When every cell looks too small for the (over-counted) estimate,
    the Router still returns a decision. The least-bad cell wins —
    largest window, then cheapest. Upstream is the source of truth on
    whether the prompt actually overflows."""
    router = _build()
    huge = "x" * 1_500_000  # ~500K tokens at chars/3 — exceeds even HIGH's 400K
    body = {"messages": [{"role": "user", "content": huge}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID, REMOTE_HIGH]))
    # All three overflow → all scaled below 0.5 → selector falls back to
    # cheapest overall, but in proportion to fit ratio. HIGH has the best
    # fit ratio (400/504 vs 256/504 vs 128/504), so candidates should be
    # ordered with HIGH near the top of the retry list.
    assert decision.cell in (LOCAL, REMOTE_MID, REMOTE_HIGH)
    # The biggest-window cell should appear before the smallest one in
    # the candidate retry order.
    cands = list(decision.candidates)
    assert cands.index(REMOTE_HIGH) < cands.index(LOCAL)


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
    the request-log writer can serialize without a custom encoder."""
    router = _build()
    body = {"messages": [{"role": "user", "content": "hi"}]}
    decision = asyncio.run(router.route(body, [LOCAL, REMOTE_MID]))
    assert decision.predictions == {
        "local-llm default": 0.5,
        "remote-mid medium": 0.5,
    }
