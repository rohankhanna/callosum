"""Dimension: tool_call_shape.

Small-prompt baseline: does this cell emit OpenAI-shaped tool_calls
when given a realistic-but-minimal Codex-shape request? A cell that
fails this can't be used for ANY tool-using traffic.

Pass: response.output[] contains at least one valid function_call AND
no message text content looks like a tool call emitted as text (JSON
object OR Hermes tag format -- see _shape_utils).

Fail-mode classification (each gets a different adapter_hint):
  * text-as-tool-call only -- adapter parses message text (JSON or
    Hermes tag format) and lifts to structured tool_calls
    (model-a0e5 text-as-JSON quirk; model-a0d4 Hermes-tag quirk).
  * dual-emit -- adapter strips the duplicate text tool-call when a
    structured tool_call is already present.
  * refused / empty -- no adapter feasible; cell denied for tools.
  * transport error -- not a model quirk; backend issue.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from callosum.capability.dimensions._shape_utils import classify_response
from callosum.capability.profile import CapabilityProfile, DimensionFinding
from callosum.capability.request_shapes import tool_call_simple_body
from callosum.substrate_contract import ContractAction

DIMENSION_NAME = "tool_call_shape"

# Class-B tool-call-shape gaps are model quirks in how THIS cell emits tool
# calls; the substrate owns generic tool-call shape translation for cells it
# fronts. The temporary adapter (parse/lift/strip) retires once the substrate
# fronting this cell translates tool-call shape both directions.
_TOOL_CALL_UPSTREAM_OWNER = "the local LLM gateway / LiteLLM tool-call shape translation"
_TOOL_CALL_CLOSE_CONDITION = "substrate fronting this cell translates tool-call shape both directions"

# Prose names for the Hermes-family tag formats recognized by
# _shape_utils._looks_like_tool_call_hermes (kept here without literal
# angle-bracket tag glyphs so the hint text stays readable and the formats are
# named once, authoritatively, in _shape_utils).
_HERMES_TAG_FORMATS_PROSE = (
    "the Hermes tag formats: a function=NAME block, a tool_call special-token block, or a tool_call pipe marker"
)


async def probe(
    cell: str,
    call_responses: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    profile: CapabilityProfile,  # noqa: ARG001 -- kept for symmetry with at-scale
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
                "transport failure -- not a model quirk. Investigate callosum's backend health for this cell."
            ),
        )

    cls = classify_response(response)

    if cls.has_structured_call and not cls.text_tool_call_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="pass",
            summary=("structured function_call emitted; no tool-call-shaped text leak in message text"),
            evidence={
                "function_calls_count": cls.function_calls_count,
                "text_excerpts": cls.text_excerpts[:3],
            },
        )
    if cls.has_structured_call and cls.text_tool_call_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="fail",
            summary=(
                "dual-emit: structured function_call AND duplicate "
                "tool-call-shaped text in message text. Codex CLI's "
                "parser will render the text tool-call as visible junk."
            ),
            evidence={
                "text_tool_call_leak_examples": cls.text_tool_call_leak_examples[:2],
                "text_excerpts": cls.text_excerpts[:3],
            },
            adapter_hint=(
                "adapter must strip message-text content that looks like "
                'a tool call (a JSON object of shape {"name": str, '
                '"arguments": str|object}, or any of '
                + _HERMES_TAG_FORMATS_PROSE
                + ") before forwarding to the client. The structured "
                "tool_calls are usable as-is."
            ),
            gap_class="B",
            suggested_action=ContractAction.AUTHOR_TEMPORARY_ADAPTER,
            upstream_owner=_TOOL_CALL_UPSTREAM_OWNER,
            close_condition=_TOOL_CALL_CLOSE_CONDITION,
        )
    if cls.text_tool_call_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="fail",
            summary=(
                "text-as-tool-call only: model emits tool calls as text "
                "(a JSON object or a Hermes tag format) in message text, "
                "no structured function_call items. The model-a0e5 "
                "text-as-JSON and model-a0d4 Hermes-tag quirk classes."
            ),
            evidence={
                "text_tool_call_leak_examples": cls.text_tool_call_leak_examples[:2],
            },
            adapter_hint=(
                "adapter must parse message-text content that looks like "
                'a tool call (a JSON object of shape {"name": str, '
                '"arguments": str|object}, or any of '
                + _HERMES_TAG_FORMATS_PROSE
                + ") and lift each match into a structured function_call "
                "item in output[]. Strip the original text. Then deliver "
                "the rewritten response to the client."
            ),
            gap_class="B",
            suggested_action=ContractAction.AUTHOR_TEMPORARY_ADAPTER,
            upstream_owner=_TOOL_CALL_UPSTREAM_OWNER,
            close_condition=_TOOL_CALL_CLOSE_CONDITION,
        )
    return DimensionFinding(
        dimension=DIMENSION_NAME,
        status="fail",
        summary=(
            "no structured function_call AND no tool-call-shaped text "
            "in text. Model refused to call a tool or produced an "
            "empty/text-only response."
        ),
        evidence={
            "text_excerpts": cls.text_excerpts[:3],
            "output_item_types": cls.output_item_types,
        },
        adapter_hint=(
            "no adapter feasible -- this model can't be coerced into "
            "tool use through the current prompt. Either the system "
            "prompt needs major rework (low confidence this would "
            "work) or the cell should be denied for tool-using traffic."
        ),
        gap_class=None,
        suggested_action=ContractAction.QUARANTINE_CELL,
    )
