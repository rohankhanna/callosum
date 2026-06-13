"""Transform protocol + context.

A transform is a thing that, given a request or response payload and
a context describing the target cell, returns a (possibly modified)
payload. Transforms compose like middleware — request transforms run
in order, response transforms run in reverse order, so a transform
that wraps the request can match its wrap on the response.

The protocol is intentionally narrow. A transform implementation
needs four things and no more:

  * `name` — a stable identifier used in logs and the request log's
    `transforms_applied` column (future). Must be unique within a
    registry.

  * `applies_to(ctx) -> bool` — does this transform fire for this
    cell? Called once per request. Should be cheap.

  * `transform_request(body, ctx) -> body` — pre-process. Return the
    body unchanged when no work is needed (don't return None;
    None is reserved for "drop the request entirely," not yet
    supported but reserved).

  * `transform_response(body, ctx) -> body` — post-process. Same
    contract.

A transform that only modifies requests can implement
`transform_response` as a passthrough (and vice versa). The default
implementations on a base class (provided in the same module for
convenience) do exactly this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from callosum.capability.profile import CapabilityProfile
from callosum.capability.weight_identity import WeightIdentity
from callosum.cell_grid import Cell


@dataclass(frozen=True, slots=True)
class TransformContext:
    """What a transform sees at apply time.

    `cell` is the routing-selected target. `weight_identity` is the
    cell's underlying-weights identity (None when no provider knew
    about it). `capability_profile` is the cell's on-disk profile
    with all known findings (None when no harness sweep has produced
    one yet — a brand-new cell hasn't been probed). `endpoint`
    identifies which entry-point the request arrived on (e.g.
    "codex" for POST /codex; None or "generic" for the legacy
    /v1/* endpoints) so a transform can scope itself to a specific
    CLI's contract. Transforms can inspect any of these to decide
    whether to fire and what to do.

    Construction is the caller's responsibility — the request handler
    builds one per request, passes it to every applicable transform.
    The fields are frozen so a transform can't accidentally mutate
    one transform's view and affect another's.
    """

    cell: Cell
    weight_identity: WeightIdentity | None
    capability_profile: CapabilityProfile | None
    endpoint: str | None = None


class Transform(Protocol):
    """Per-cell payload transform.

    Implementations are typically module-level singletons. A transform
    instance is shared across all requests; do NOT keep per-request
    state inside the instance. If you need per-request scratch space,
    derive it from the context inside the transform method.
    """

    @property
    def name(self) -> str:
        """Stable identifier. Used in logs and (future) per-row
        provenance. Must be unique within a registry."""
        ...

    def applies_to(self, ctx: TransformContext) -> bool:
        """Decide whether this transform should fire for the given
        cell/context. Called once per request. Should be cheap — no
        I/O, no network. Inspect ctx.cell.model, ctx.weight_identity,
        and ctx.capability_profile.findings as needed."""
        ...

    def transform_request(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        """Pre-process the request body. Return the body unchanged
        when no work is needed (DO NOT return None — that's reserved
        for "drop the request" which isn't yet supported). The body
        is mutable in principle, but treating it as immutable and
        returning a new dict is the safer pattern."""
        ...

    def transform_response(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        """Post-process the response body. Same contract as
        transform_request."""
        ...


class TransformBase:
    """Convenience base class implementing pass-through for both
    request and response. Subclasses override only the side they
    actually transform.

    Using this base is optional — anything matching the Transform
    protocol is registerable. The base just removes boilerplate for
    the common case of "I only transform responses."
    """

    @property
    def name(self) -> str:
        # Subclasses override; we surface a recognizable default so
        # an accidental missing-override is easy to spot in logs.
        return type(self).__name__

    def applies_to(self, ctx: TransformContext) -> bool:
        return False

    def transform_request(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        return body

    def transform_response(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        return body
