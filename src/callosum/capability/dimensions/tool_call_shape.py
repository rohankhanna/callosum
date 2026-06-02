"""Dimension: tool_call_shape.

Small-prompt baseline: does this cell emit OpenAI-shaped tool_calls
when given a realistic-but-minimal Codex-shape request? A cell that
fails this can't be used for ANY tool-using traffic.

Pass: response.output[] contains at least one valid function_call AND
no message text content parses as a tool-call-shaped JSON object.

Fail-mode classification (each gets a different adapter_hint):
  * text-as-JSON only — adapter parses message text and lifts to
    structured tool_calls (model-a0e5 family quirk).
  * dual-emit — adapter strips the duplicate JSON-text when a
    structured tool_call is already present.
  * refused / empty — no adapter feasible; cell denied for tools.
  * transport error — not a model quirk; backend issue.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from callosum.capability.dimensions._shape_utils import classify_response
from callosum.capability.profile import CapabilityProfile, DimensionFinding
from callosum.capability.request_shapes import tool_call_simple_body

DIMENSION_NAME = "tool_call_shape"


async def probe(
    cell: str,
    call_responses: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    profile: CapabilityProfile,  # noqa: ARG001 — kept for symmetry with at-scale
) -> DimensionFinding:
    """Run the small-prompt tool-call probe. `call_responses` takes a
    Responses-API body and returns the upstream response dict; it's
    the abstraction over how the request actually reaches the cell
    (admin/cell-call endpoint, direct backend call, or stub in tests).
    """
    body = tool_call_simple_body()
    body["model"] = cell
    try:
        response = await call_responses(body)
    except Exception as exc:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="error",
            summary=f"call_responses raised: {type(exc).__name__}: {exc}",
            evidence={"exception": f"{type(exc).__name__}: {exc}"},
            adapter_hint=(
                "transport failure — not a model quirk. Investigate "
                "callosum's backend health for this cell."
            ),
        )

    cls = classify_response(response)

    if cls.has_structured_call and not cls.text_json_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="pass",
            summary=(
                "structured function_call emitted; no tool-call-JSON "
                "leak in message text"
            ),
            evidence={
                "function_calls_count": cls.function_calls_count,
                "text_excerpts": cls.text_excerpts[:3],
            },
        )
    if cls.has_structured_call and cls.text_json_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="fail",
            summary=(
                "dual-emit: structured function_call AND duplicate "
                "tool-call-shaped JSON in message text. Codex CLI's "
                "parser will render the text JSON as visible junk."
            ),
            evidence={
                "text_json_leak_examples": cls.text_json_leak_examples[:2],
                "text_excerpts": cls.text_excerpts[:3],
            },
            adapter_hint=(
                "adapter must strip message-text content matching "
                'shape {"name": str, "arguments": str|object} before '
                "forwarding to the client. The structured tool_calls "
                "are usable as-is."
            ),
        )
    if cls.text_json_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="fail",
            summary=(
                "text-as-JSON only: model emits tool calls as JSON in "
                "message text, no structured function_call items. The "
                "model-a0e5 quirk class."
            ),
            evidence={
                "text_json_leak_examples": cls.text_json_leak_examples[:2],
            },
            adapter_hint=(
                "adapter must parse message-text content matching shape "
                '{"name": str, "arguments": str|object} and lift each '
                "match into a structured function_call item in output[]. "
                "Strip the original text. Then deliver the rewritten "
                "response to the client."
            ),
        )
    return DimensionFinding(
        dimension=DIMENSION_NAME,
        status="fail",
        summary=(
            "no structured function_call AND no tool-call-shaped JSON "
            "in text. Model refused to call a tool or produced an "
            "empty/text-only response."
        ),
        evidence={
            "text_excerpts": cls.text_excerpts[:3],
            "output_item_types": cls.output_item_types,
        },
        adapter_hint=(
            "no adapter feasible — this model can't be coerced into "
            "tool use through the current prompt. Either the system "
            "prompt needs major rework (low confidence this would "
            "work) or the cell should be denied for tool-using traffic."
        ),
    )
