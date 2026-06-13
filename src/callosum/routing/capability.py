"""Deterministic capability filter.

Drops cells whose CellCapabilities can't satisfy the request's HARD
requirements: modality support, tool-use support, and (when the cell
has a failing capability-harness `tool_call_at_scale` finding) at-scale
tool-call reliability.

Context window is intentionally NOT a filter criterion. The proxy's
inbound token count is a chars/3 heuristic (see routing/features.py)
and is wrong by ~30% on Codex CLI corpora — it over-counts mixed
code/English text, and the encrypted_content blobs in Responses-API
reasoning items aren't counted at all. Hard-rejecting on that estimate
caused requests upstream would have accepted to be 4xx'd at the proxy.
The Router applies window-fit as a soft preference in selection
instead, and the upstream's own tokenizer is the source of truth for
actual context overflow.
"""

from __future__ import annotations

from collections.abc import Callable

from callosum.capability.gating import (
    DEFAULT_AT_SCALE_CHARS_THRESHOLD,
    at_scale_tool_call_fails,
)
from callosum.cell_grid import Cell
from callosum.routing.protocols import CellCapabilities, PromptFeatures


class CapabilityFilter:
    """Filters a cell list down to cells whose physical capabilities CAN
    serve a given request.

    `capabilities_of` is a callable provided at construction time; the
    filter doesn't know how capabilities are sourced (Codex backend
    metadata vs. LiteLLM gateway vs. operator config). Keeps the filter
    decoupled from backend-shape detail.

    `at_scale_fails_for` is an optional callable that returns True when
    the cell's persisted capability profile shows a failing
    `tool_call_at_scale` finding. Injectable so tests can stub it
    without filesystem dependencies; production uses the default reader
    from `callosum.capability.gating`. None disables the at-scale gate
    entirely (matches the pre-harness behavior).
    """

    def __init__(
        self,
        capabilities_of: Callable[[Cell], CellCapabilities],
        *,
        at_scale_fails_for: Callable[[str], bool] | None = at_scale_tool_call_fails,
        at_scale_chars_threshold: int = DEFAULT_AT_SCALE_CHARS_THRESHOLD,
    ) -> None:
        self._capabilities_of = capabilities_of
        self._at_scale_fails_for = at_scale_fails_for
        self._at_scale_chars_threshold = at_scale_chars_threshold

    def filter(self, cells: list[Cell], features: PromptFeatures) -> list[Cell]:
        """Return cells whose capabilities cover the features' HARD
        requirements:

          * features.modalities is a subset of cell.modalities
          * if features.needs_tools, cell.supports_tools must be True
          * if features.needs_tools AND the request is at-scale AND the
            cell has a failing `tool_call_at_scale` harness finding,
            the cell is excluded for this request only. Small-context
            tool requests on the same cell remain eligible — the
            finding describes a context-size-dependent failure, not a
            blanket loss of tool support.

        Context window is *not* a filter criterion — see module docstring.
        Cells whose advertised window may be too small for the estimated
        prompt are kept here and deprioritized by the Router's soft
        window-fit scoring at selection time.
        """
        is_at_scale_request = (
            features.needs_tools
            and self._at_scale_fails_for is not None
            and len(features.text) >= self._at_scale_chars_threshold
        )
        out: list[Cell] = []
        for c in cells:
            caps = self._capabilities_of(c)
            if not features.modalities <= caps.modalities:
                continue
            if features.needs_tools and not caps.supports_tools:
                continue
            if is_at_scale_request and self._at_scale_fails_for is not None and self._at_scale_fails_for(c.model):
                continue
            out.append(c)
        return out
