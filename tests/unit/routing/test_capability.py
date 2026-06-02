"""Tests for routing/capability.py — deterministic filter.

The filter only enforces HARD requirements (modality, tools). Context
window is intentionally a soft preference applied later by the Router,
not a hard filter — see capability.py module docstring for why.
"""

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


def test_filter_no_longer_drops_on_context_window() -> None:
    """A 250K-token prompt against a 128K cell used to drop. Now the
    filter keeps the cell — window fit is a soft preference applied by
    the Router's selection step, not a hard filter. Upstream is the
    source of truth for actual context overflow."""
    small = Cell(model="tiny", reasoning_effort="default")
    big = Cell(model="big", reasoning_effort="default")
    caps = {small: _caps(ctx=4_000), big: _caps(ctx=256_000)}
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter([small, big], _features(tokens=10_000))
    assert small in out
    assert big in out


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


def test_filter_returns_empty_only_on_modality_or_tool_mismatch() -> None:
    """An image-bearing prompt against text-only cells yields empty.
    Router converts empty → NoCompatibleCellError → client-facing 4xx.
    Note: token count is intentionally absurd here to prove it does NOT
    affect the filter outcome — only modality does."""
    text_only = Cell(model="text-only", reasoning_effort="default")
    caps = {text_only: _caps(ctx=128_000, modalities=("text",))}
    f = CapabilityFilter(capabilities_of=caps.__getitem__)
    out = f.filter(
        [text_only],
        _features(tokens=1_000_000, modalities=("text", "image")),
    )
    assert out == []


# ---------- at-scale gate (consumes capability harness findings) -----------


def _at_scale_features(*, chars: int, needs_tools: bool = True) -> PromptFeatures:
    """Build a PromptFeatures with `chars` characters of body text — the
    at-scale gate keys on raw `len(features.text)` because the chars/3
    token estimate is too noisy at this granularity."""
    return PromptFeatures(
        text="x" * chars,
        tokens=chars // 3,
        modalities=frozenset({"text"}),
        needs_tools=needs_tools,
    )


def test_at_scale_gate_excludes_failing_cell_above_threshold() -> None:
    """Cell with a failing tool_call_at_scale finding is dropped when
    the request exceeds the at-scale character threshold AND tools are
    needed. This is the routing-side consumer of the harness — the
    finding produced on disk causes a real change in cell selection."""
    failing = Cell(model="fails-at-scale", reasoning_effort="default")
    healthy = Cell(model="works-at-scale", reasoning_effort="default")
    caps = {failing: _caps(tools=True), healthy: _caps(tools=True)}
    f = CapabilityFilter(
        capabilities_of=caps.__getitem__,
        at_scale_fails_for=lambda m: m == "fails-at-scale",
        at_scale_chars_threshold=10_000,
    )
    out = f.filter(
        [failing, healthy], _at_scale_features(chars=20_000)
    )
    assert healthy in out
    assert failing not in out


def test_at_scale_gate_keeps_failing_cell_below_threshold() -> None:
    """Small-context requests stay routable on the failing cell — the
    harness finding describes context-size-dependent breakage, not a
    blanket loss of tool support. Preserves the cell's small-context
    utility instead of nuking it from tool routing entirely."""
    failing = Cell(model="fails-at-scale", reasoning_effort="default")
    caps = {failing: _caps(tools=True)}
    f = CapabilityFilter(
        capabilities_of=caps.__getitem__,
        at_scale_fails_for=lambda m: m == "fails-at-scale",
        at_scale_chars_threshold=10_000,
    )
    out = f.filter([failing], _at_scale_features(chars=2_000))
    assert failing in out


def test_at_scale_gate_inert_when_tools_not_needed() -> None:
    """A request that doesn't need tools can land on the failing cell
    regardless of size — the gate is specifically about tool-call
    integrity at scale, not about general output quality."""
    failing = Cell(model="fails-at-scale", reasoning_effort="default")
    caps = {failing: _caps(tools=False)}
    f = CapabilityFilter(
        capabilities_of=caps.__getitem__,
        at_scale_fails_for=lambda m: m == "fails-at-scale",
        at_scale_chars_threshold=10_000,
    )
    out = f.filter(
        [failing], _at_scale_features(chars=100_000, needs_tools=False)
    )
    assert failing in out


def test_at_scale_gate_disabled_when_callable_is_none() -> None:
    """Passing None for at_scale_fails_for restores pre-harness
    behavior — the filter ignores at-scale findings entirely. Lets
    callers opt out at construction without changing the threshold."""
    failing = Cell(model="fails-at-scale", reasoning_effort="default")
    caps = {failing: _caps(tools=True)}
    f = CapabilityFilter(
        capabilities_of=caps.__getitem__,
        at_scale_fails_for=None,
        at_scale_chars_threshold=10_000,
    )
    out = f.filter([failing], _at_scale_features(chars=100_000))
    assert failing in out
