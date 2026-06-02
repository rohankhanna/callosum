"""TransformRegistry: holds transforms and applies them per request.

The registry is the substrate's runtime entry point. The request path
calls `apply_request` after cell selection (before forwarding to the
backend) and `apply_response` once a response is available. Each call
walks the registered transforms, filters by `applies_to(ctx)`, and
runs the applicable ones as middleware.

Two safety properties this class enforces:

  * **Per-transform error isolation.** A transform that raises is
    caught, logged, and skipped. The unmodified payload flows on.
    One buggy transform cannot 5xx an otherwise-healthy request.

  * **No transform required.** An empty registry — the default
    state — is a valid configuration. apply_request and apply_response
    on an empty registry return the body unchanged. This is what
    makes shipping the substrate a zero-behavior-change deploy.

Registration: transforms register via `register()`. The order of
registration determines the order in which transforms apply for
requests. Response transforms run in reverse order so an outer
wrap-on-request can be unwrapped on response.
"""

from __future__ import annotations

import logging
from typing import Any

from callosum.transforms.protocol import Transform, TransformContext

logger = logging.getLogger(__name__)


class TransformRegistry:
    """A mutable, ordered list of transforms.

    Thread-safety: registration is not thread-safe; do it during
    application startup before serving traffic. Reads (apply_request /
    apply_response) ARE safe to call concurrently from multiple
    request handlers — the underlying list is only iterated, never
    mutated, in the read path.
    """

    def __init__(self) -> None:
        self._transforms: list[Transform] = []

    def register(self, transform: Transform) -> None:
        """Add a transform to the end of the list. Duplicate names
        are a configuration error and raise — the registry's behavior
        with two same-named transforms would be ambiguous."""
        for existing in self._transforms:
            if existing.name == transform.name:
                raise ValueError(
                    f"transform name {transform.name!r} is already "
                    f"registered; pick a unique name"
                )
        self._transforms.append(transform)

    def names(self) -> list[str]:
        """Return the registered transform names in registration order.
        Useful for /status and the operator's mental model of what's
        active."""
        return [t.name for t in self._transforms]

    def __len__(self) -> int:
        return len(self._transforms)

    def applicable_for(
        self, ctx: TransformContext
    ) -> list[Transform]:
        """Filter the registered transforms down to those whose
        applies_to(ctx) returns True. Order matches registration order.

        Defensive against a transform's applies_to raising — that
        transform is skipped (treated as not applicable) and the
        exception is logged."""
        out: list[Transform] = []
        for t in self._transforms:
            try:
                if t.applies_to(ctx):
                    out.append(t)
            except Exception:
                logger.exception(
                    "transform %s: applies_to raised; skipping",
                    t.name,
                )
        return out

    def apply_request(
        self,
        body: dict[str, Any],
        ctx: TransformContext,
    ) -> dict[str, Any]:
        """Apply all applicable request transforms in registration
        order. Returns the (possibly modified) body. A transform
        whose transform_request raises is logged and skipped — the
        body the next transform sees is the body the previous one
        produced (or, if every transform raises, the original)."""
        current = body
        for t in self.applicable_for(ctx):
            try:
                current = t.transform_request(current, ctx)
            except Exception:
                logger.exception(
                    "transform %s: transform_request raised; "
                    "skipping this transform for cell %s",
                    t.name, ctx.cell.model,
                )
        return current

    def apply_response(
        self,
        body: dict[str, Any],
        ctx: TransformContext,
    ) -> dict[str, Any]:
        """Apply all applicable response transforms in REVERSE
        registration order — the substrate behaves like middleware,
        so a transform that wrapped the request gets to unwrap the
        response. Same per-transform error isolation as
        apply_request."""
        current = body
        for t in reversed(self.applicable_for(ctx)):
            try:
                current = t.transform_response(current, ctx)
            except Exception:
                logger.exception(
                    "transform %s: transform_response raised; "
                    "skipping this transform for cell %s",
                    t.name, ctx.cell.model,
                )
        return current


def build_default_registry() -> TransformRegistry:
    """Build the standard registry callosum uses at startup.

    Currently empty: no transforms are shipped by default. Adding a
    new transform is one new module under callosum/transforms/ plus
    one line in this function. Centralizing the wiring here keeps
    discovery explicit — there's no auto-import magic that could
    activate a transform someone didn't know was shipped.
    """
    return TransformRegistry()
