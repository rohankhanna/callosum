"""Dimension: tool_call_at_scale.

Same tool-call shape probe as tool_call_shape but at realistic Codex
CLI context size (~80K chars ≈ 22K tokens). This is what catches
cells like model-a0c8 that pass the small-prompt probe but emit
tool calls as text (JSON object or Hermes tag format) under
real-traffic-shaped input.

Skips itself when tool_call_shape didn't pass -- no point burning
60-300s of GPU time on a cell that can't even pass the small probe.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from callosum.capability.dimensions._shape_utils import classify_response
from callosum.capability.profile import CapabilityProfile, DimensionFinding
from callosum.capability.request_shapes import tool_call_with_context_body
from callosum.substrate_contract import ContractAction

DIMENSION_NAME = "tool_call_at_scale"

# Class-B at-scale tool-call gaps share the tool-call-shape upstream owner;
# the substrate owns generic tool-call shape translation for cells it fronts.
_TOOL_CALL_UPSTREAM_OWNER = "local LLM gateway / LiteLLM tool-call shape translation"
_TOOL_CALL_CLOSE_CONDITION = (
    "substrate fronting this cell translates tool-call shape both directions"
)

# 80K chars ≈ 22K tokens. Substantial enough to expose context-
# sensitive failures but well under the 200K-token real-Codex extremes
# (each char costs probe time linearly in the model's tokenizer).
_PROMPT_CHARS = 80_000


async def probe(
    cell: str,
    call_responses: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    profile: CapabilityProfile,
) -> DimensionFinding:
    """Short-circuit when the small probe didn't pass. The dependency
    is encoded as a profile-state check rather than as a runner
    sequencing concern so re-running just this dimension still
    respects the gate."""
    small_finding = profile.findings.get("tool_call_shape")
    if small_finding is None or small_finding.status != "pass":
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="skipped",
            summary=("skipped — small-prompt tool_call_shape did not pass; no point burning minutes probing at scale"),
            evidence={
                "small_test_status": (small_finding.status if small_finding else "not_run"),
            },
        )

    body = tool_call_with_context_body(target_user_chars=_PROMPT_CHARS)
    body["model"] = cell
    try:
        response = await call_responses(body)
    except Exception as exc:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="error",
            summary=f"call_responses raised: {type(exc).__name__}: {exc}",
            evidence={
                "prompt_chars": _PROMPT_CHARS,
                "exception": f"{type(exc).__name__}: {exc}",
            },
            adapter_hint=(
                "context-size error or transport failure. If the error "
                "mentions context length, the model can't physically "
                "serve realistic Codex traffic."
            ),
        )

    cls = classify_response(response)

    if cls.has_structured_call and not cls.text_tool_call_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="pass",
            summary=(
                f"structured function_call emitted at ~{_PROMPT_CHARS} char "
                "prompt; no tool-call-shaped text leak"
            ),
            evidence={
                "prompt_chars": _PROMPT_CHARS,
                "text_excerpts": cls.text_excerpts[:3],
            },
        )
    if cls.text_tool_call_leak_examples:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="fail",
            summary=(
                "model emits tool calls as text (JSON object or Hermes "
                "tag format) in message text at "
                f"~{_PROMPT_CHARS}-char prompt size, even though the "
                "small-prompt probe passed. Real Codex CLI traffic "
                "will hit this failure mode."
            ),
            evidence={
                "prompt_chars": _PROMPT_CHARS,
                "text_tool_call_leak_examples": cls.text_tool_call_leak_examples[:2],
                "structured_call_also_present": cls.has_structured_call,
            },
            adapter_hint=(
                "model breaks down at realistic context size. Two "
                "adapter options: (a) parse tool-call-shaped text "
                "(JSON object or Hermes tag format) from message text "
                "and lift to structured function_call items, OR (b) "
                "restrict this cell to small-context routing only "
                "(denied for sessions whose accumulated context "
                "exceeds N tokens). (b) is simpler; (a) is more "
                "general but requires careful escaping logic."
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
            "no structured tool_call AND no tool-call-shaped text at "
            "scale. Model produced text-only or empty output despite "
            "passing the small-prompt probe."
        ),
        evidence={
            "prompt_chars": _PROMPT_CHARS,
            "text_excerpts": cls.text_excerpts[:3],
        },
        adapter_hint=(
            "context-degradation: model loses tool-calling competence "
            "as prompt grows. Restrict this cell to small-context "
            "routing only."
        ),
        gap_class=None,
        suggested_action=ContractAction.QUARANTINE_CELL,
    )
