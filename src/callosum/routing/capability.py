"""Deterministic capability filter.

Drops cells whose CellCapabilities can't satisfy the request's HARD
requirements: modality support, and tool-use support. Pure set
membership / boolean checks — no thresholds, no estimates, no
heuristics.

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

from callosum.cell_grid import Cell
from callosum.routing.protocols import PromptFeatures


class CapabilityFilter:
    """Filters a cell list down to cells whose physical capabilities CAN
    serve a given request.

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
        """Return cells whose capabilities cover the features' HARD
        requirements:

          * features.modalities is a subset of cell.modalities
          * if features.needs_tools, cell.supports_tools must be True

        Context window is *not* a filter criterion — see module docstring.
        Cells whose advertised window may be too small for the estimated
        prompt are kept here and deprioritized by the Router's soft
        window-fit scoring at selection time.
        """
        out: list[Cell] = []
        for c in cells:
            caps = self._capabilities_of(c)
            if not features.modalities <= caps.modalities:
                continue
            if features.needs_tools and not caps.supports_tools:
                continue
            out.append(c)
        return out
