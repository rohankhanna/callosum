"""Per-cell payload transforms.

The harness produces structured findings about what each cell can and
can't do. Some of those failure modes are fixable with a transform on
the request or response payload — parsing tool-call-shaped JSON out of
message text, capping prompt size for cells that break at scale,
prefixing a model-specific system prompt, etc. This package is where
those transforms live.

Design constraints, from the agentic-loop discussion:

  * **Tightly bounded blast radius.** Transforms live in their own
    package and the dev loop (when we wire it up) is permitted to
    modify ONLY this package and its tests. Everything else — the
    router, the backends, the harness, the gating layer — stays
    out of scope.

  * **No transform registered by default.** An empty registry is a
    valid state. The request-path hook walks the registry; an empty
    walk is a no-op. Shipping this package introduces zero behavior
    change until someone registers a concrete transform.

  * **Self-describing applicability.** Each transform decides whether
    it applies to a given cell via `applies_to(ctx)`. The context
    carries the cell, its weight identity (so transforms can target
    a group of cells sharing weights), and the cell's capability
    profile (so transforms can target specific finding patterns).
    This is what lets the dev loop write a transform that targets
    "any cell that failed `tool_call_at_scale`" rather than
    enumerating cell names by hand.

  * **Request-side AND response-side hooks.** Some transforms
    pre-process the request (e.g. prompt cap); others post-process
    the response (e.g. parsing tool calls out of message text).
    Request transforms run in registration order; response
    transforms run in reverse so the substrate behaves like
    middleware.

  * **Defensive.** A transform that raises is logged and skipped;
    the rest of the pipeline continues with the unmodified payload.
    A buggy transform must not 5xx an otherwise-good request.

Concrete transforms (when authored) will live under this package as
`tool_call_from_text.py`, `prompt_size_cap.py`, etc. Each module
exports its transform instance via a `TRANSFORM` module-level binding;
`registry.build_default_registry()` discovers and registers them.
"""

from callosum.transforms.protocol import Transform, TransformContext
from callosum.transforms.registry import (
    TransformRegistry,
    build_default_registry,
)

__all__ = [
    "Transform",
    "TransformContext",
    "TransformRegistry",
    "build_default_registry",
]
