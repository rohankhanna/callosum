"""Deterministic capability filter.

Drops cells whose CellCapabilities don't satisfy the request's
requirements. No thresholds, no scores — pure set membership and
numeric comparison.
"""

from __future__ import annotations

from callosum.cell_grid import Cell
from callosum.routing.protocols import CellCapabilities, PromptFeatures


# Token budget kept free for the completion in addition to the input
# estimate. Not a routing heuristic — every backend needs SOME output
# headroom in its context window. Conservative enough that a 256K-window
# cell still serves 252K-token prompts cleanly.
_OUTPUT_HEADROOM_TOKENS = 4096


class CapabilityFilter:
    """Filters a cell list down to cells that CAN serve a given request.

    `capabilities_of` is a callable provided at construction time; the
    filter doesn't know how capabilities are sourced (Codex backend
    metadata vs. LiteLLM gateway vs. operator config). Keeps the filter
    decoupled from backend-shape detail.
    """

    def __init__(self, capabilities_of):
        # capabilities_of: Callable[[Cell], CellCapabilities]
        self._capabilities_of = capabilities_of

    def filter(
        self, cells: list[Cell], features: PromptFeatures
    ) -> list[Cell]:
        """Return cells whose capabilities cover the features' requirements.

        Filtering rules (all must hold):
          * cell.context_window >= features.tokens + OUTPUT_HEADROOM
          * features.modalities is a subset of cell.modalities
          * if features.needs_tools, cell.supports_tools must be True
        """
        out: list[Cell] = []
        budget = features.tokens + _OUTPUT_HEADROOM_TOKENS
        for c in cells:
            caps = self._capabilities_of(c)
            if caps.context_window < budget:
                continue
            if not features.modalities <= caps.modalities:
                continue
            if features.needs_tools and not caps.supports_tools:
                continue
            out.append(c)
        return out
