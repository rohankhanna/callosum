"""Tests for routing/capability.py — deterministic filter."""

from __future__ import annotations

from callosum.cell_grid import Cell
from callosum.routing.capability import CapabilityFilter
from callosum.routing.protocols import CellCapabilities, PromptFeatures


def _features(*, tokens: int = 1000, modalities=("text",), needs_tools: bool = False):
    return PromptFeatures(
        text="x",
        tokens=tokens,
        modalities=frozenset(modalities),
        needs_tools=needs_tools,
    )


def _caps(*, ctx: int = 128_000, modalities=("text",), tools: bool = False, cost: int = 0):
    return CellCapabilities(
        context_window=ctx,
        modalities=frozenset(modalities),
        supports_tools=tools,
        cost_rank=cost,
    )


def test_filter_drops_cell_whose_context_is_too_small() -> None:
    small = Cell(model="tiny", reasoning_effort="default")
    big = Cell(model="big", reasoning_effort="default")
    caps = {small: _caps(ctx=4_000), big: _caps(ctx=256_000)}
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter([small, big], _features(tokens=10_000))
    assert big in out
    assert small not in out


def test_filter_drops_cell_without_required_modality() -> None:
    text_only = Cell(model="text-only", reasoning_effort="default")
    vision = Cell(model="vision-capable", reasoning_effort="default")
    caps = {
        text_only: _caps(modalities=("text",)),
        vision: _caps(modalities=("text", "image")),
    }
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter(
        [text_only, vision], _features(modalities=("text", "image"))
    )
    assert vision in out
    assert text_only not in out


def test_filter_drops_cell_without_tool_support_when_needed() -> None:
    no_tools = Cell(model="no-tools", reasoning_effort="default")
    with_tools = Cell(model="with-tools", reasoning_effort="default")
    caps = {no_tools: _caps(tools=False), with_tools: _caps(tools=True)}
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter([no_tools, with_tools], _features(needs_tools=True))
    assert with_tools in out
    assert no_tools not in out


def test_filter_keeps_cell_when_tools_not_needed() -> None:
    no_tools = Cell(model="no-tools", reasoning_effort="default")
    caps = {no_tools: _caps(tools=False)}
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter([no_tools], _features(needs_tools=False))
    assert no_tools in out


def test_filter_returns_empty_when_no_cell_qualifies() -> None:
    """No physical cell can serve a 1M-token image-bearing prompt with
    only text-only 128K cells available. Empty list is correct — the
    Router converts this to a NoCompatibleCellError."""
    text_only = Cell(model="text-only", reasoning_effort="default")
    caps = {text_only: _caps(ctx=128_000, modalities=("text",))}
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter(
        [text_only],
        _features(tokens=1_000_000, modalities=("text", "image")),
    )
    assert out == []
