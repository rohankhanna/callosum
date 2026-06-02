"""Compat shim — request shapes now live in callosum.capability.

Older test code imports the body-builders from this module. Re-export
from the canonical location. New code should import directly from
`callosum.capability.request_shapes`.
"""

from __future__ import annotations

from callosum.capability.request_shapes import (  # noqa: F401
    tool_call_simple_body,
    tool_call_with_context_body,
)
