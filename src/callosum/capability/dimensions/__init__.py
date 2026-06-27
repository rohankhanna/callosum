"""Capability dimensions.

Each dimension is a module exporting a `probe(cell, call_responses)`
coroutine that returns a DimensionFinding. Dimensions are independent
of how they get invoked — pytest harness, callosum's background
scheduler, and any future operator command all call the same function.

The `DIMENSIONS` registry lists every dimension in execution order.
Order matters: cheaper dimensions run first so their results can
short-circuit later expensive ones (the at-scale tool-call probe
skips when the small-prompt probe already failed).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from callosum.capability.dimensions import tool_call_at_scale, tool_call_shape
from callosum.capability.profile import CapabilityProfile, DimensionFinding

# Type alias for the probe callable each dimension implements.
ProbeFn = Callable[
    [str, Callable[[dict[str, Any]], Awaitable[dict[str, Any]]], CapabilityProfile],
    Awaitable[DimensionFinding],
]


# Execution order matters — see module docstring.
DIMENSIONS: list[tuple[str, ProbeFn]] = [
    ("tool_call_shape", tool_call_shape.probe),
    ("tool_call_at_scale", tool_call_at_scale.probe),
]


__all__ = ["DIMENSIONS", "ProbeFn"]
