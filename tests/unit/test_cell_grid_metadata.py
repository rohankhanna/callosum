"""Tests for the API-metadata-aware cell grid builders.

These exist to remove two of the remaining hardcoded assumptions in the
proxy: which models are "completion-shaped" (was a regex) and what
reasoning effort levels each supports (was a global constant). Now both
come from /backend-api/codex/models when the upstream response includes
them; the regex + constant remain as fallbacks for older responses.
"""

from __future__ import annotations

from callosum.cell_grid import (
    ModelMetadata,
    build_cells_from_metadata,
    live_completion_models_from_metadata,
    reasoning_levels_for,
)


def _md(
    slug: str,
    *,
    priority: int | None = None,
    visibility: str | None = None,
    supported_in_api: bool | None = None,
    levels: tuple[str, ...] = (),
    context_window: int | None = None,
) -> ModelMetadata:
    return ModelMetadata(
        slug=slug,
        priority=priority,
        visibility=visibility,
        supported_in_api=supported_in_api,
        supported_reasoning_levels=levels,
        context_window=context_window,
    )


def test_live_completion_models_orders_by_priority_ascending() -> None:
    """Lower priority is stronger per upstream's convention; ranking should
    reflect that — strongest model first."""
    metadata = {
        "model-a0c3": _md("model-a0c3", priority=23, supported_in_api=True, visibility="list"),
        "model-a0e7": _md("model-a0e7", priority=16, supported_in_api=True, visibility="list"),
        "model-a0e6": _md("model-a0e6", priority=29, supported_in_api=True, visibility="list"),
    }
    assert live_completion_models_from_metadata(metadata) == (
        "model-a0e7",
        "model-a0c3",
        "model-a0e6",
    )


def test_live_completion_models_excludes_hidden() -> None:
    """`visibility=hide` (e.g. codex-auto-review) must not appear in the grid."""
    metadata = {
        "model-a0e7": _md("model-a0e7", priority=16, supported_in_api=True, visibility="list"),
        "codex-auto-review": _md("codex-auto-review", priority=43, supported_in_api=True, visibility="hide"),
    }
    out = live_completion_models_from_metadata(metadata)
    assert out == ("model-a0e7",)


def test_live_completion_models_excludes_not_supported_in_api() -> None:
    """`supported_in_api=False` excludes the model even if visibility says 'list'."""
    metadata = {
        "model-a0e7": _md("model-a0e7", priority=16, supported_in_api=True, visibility="list"),
        "internal-only-thing": _md("internal-only-thing", priority=99, supported_in_api=False, visibility="list"),
    }
    assert live_completion_models_from_metadata(metadata) == ("model-a0e7",)


def test_live_completion_models_falls_back_to_regex_when_metadata_absent() -> None:
    """If neither supported_in_api NOR visibility is present, fall back to
    the name-shape regex so we don't accidentally route to embeddings /
    audio / review-shaped models when the API response is stripped."""
    metadata = {
        # No supported_in_api, no visibility — must use regex filter
        "model-a0e7": _md("model-a0e7", priority=16),
        "text-embedding-3-large": _md("text-embedding-3-large", priority=0),
        "tts-1-hd": _md("tts-1-hd", priority=0),
    }
    out = live_completion_models_from_metadata(metadata)
    assert out == ("model-a0e7",)


def test_reasoning_levels_for_uses_api_when_present() -> None:
    metadata = {
        "gpt-NEW": _md("gpt-NEW", levels=("low", "medium", "high", "ultra")),
    }
    assert reasoning_levels_for("gpt-NEW", metadata) == (
        "low",
        "medium",
        "high",
        "ultra",
    )


def test_reasoning_levels_for_falls_back_when_absent() -> None:
    """If the API didn't include supported_reasoning_levels, fall back."""
    metadata = {"gpt-NO-LEVELS": _md("gpt-NO-LEVELS")}  # empty levels
    out = reasoning_levels_for("gpt-NO-LEVELS", metadata)
    # default fallback is REASONING_LEVELS
    assert out == ("low", "medium", "high", "xhigh")


def test_build_cells_from_metadata_uses_per_model_levels() -> None:
    """Different models can advertise different effort sets. The cell grid
    should respect that, not project a global REASONING_LEVELS onto all."""
    metadata = {
        "gpt-A": _md(
            "gpt-A",
            priority=1,
            supported_in_api=True,
            visibility="list",
            levels=("low", "high"),
            context_window=128_000,
        ),
        "gpt-B": _md(
            "gpt-B", priority=2, supported_in_api=True, visibility="list", levels=("medium",), context_window=64_000
        ),
    }
    cells = build_cells_from_metadata(metadata)
    # gpt-A has 2 efforts → 2 cells; gpt-B has 1 effort → 1 cell.
    assert len(cells) == 3
    a_cells = [c for c in cells if c.model == "gpt-A"]
    b_cells = [c for c in cells if c.model == "gpt-B"]
    assert {c.reasoning_effort for c in a_cells} == {"low", "high"}
    assert {c.reasoning_effort for c in b_cells} == {"medium"}
    # Context window propagates from metadata.
    assert a_cells[0].context_window == 128_000
    assert b_cells[0].context_window == 64_000


def test_build_cells_from_metadata_orders_by_priority() -> None:
    """Model order in the cell list reflects priority (stronger first)."""
    metadata = {
        "gpt-mid": _md("gpt-mid", priority=20, supported_in_api=True, visibility="list", levels=("low",)),
        "gpt-strongest": _md("gpt-strongest", priority=5, supported_in_api=True, visibility="list", levels=("low",)),
        "gpt-weakest": _md("gpt-weakest", priority=99, supported_in_api=True, visibility="list", levels=("low",)),
    }
    cells = build_cells_from_metadata(metadata)
    assert [c.model for c in cells] == ["gpt-strongest", "gpt-mid", "gpt-weakest"]


def test_build_cells_from_metadata_handles_empty() -> None:
    assert build_cells_from_metadata({}) == []
