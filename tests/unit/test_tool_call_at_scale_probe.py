"""Tests for the tool_call_at_scale capability dimension PROBE.

Pins the profile-state skip short-circuit (the small-prompt
tool_call_shape finding must be pass before the at-scale probe
runs) plus the at-scale pass / text-leak-fail / context-degradation-fail
/ transport-error branches. The at-scale probe reuses
_shape_utils.classify_response (covered by test_shape_utils.py),
so these tests pin the probe's branch logic and the prompt_chars
evidence, not the classifier.

All hermetic: a fake call_responses callable + a CapabilityProfile
carrying the small-prompt finding. No GPU, no network, no app.
"""

from __future__ import annotations

from typing import Any

from callosum.capability.dimensions import tool_call_at_scale
from callosum.capability.dimensions.tool_call_at_scale import probe
from callosum.capability.profile import CapabilityProfile, DimensionFinding
from callosum.substrate_contract import ContractAction

LT, GT = chr(60), chr(62)


def _msg(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _function_call(name: str = "exec_command", arguments: str = '{"cmd":"ls"}') -> dict[str, Any]:
    return {"type": "function_call", "name": name, "arguments": arguments, "call_id": "fc_1"}


def _resp(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"output": items}


def _json_leak() -> str:
    return '{"name": "exec_command", "arguments": {"cmd": "ls"}}'


def _profile(small_status: str | None = "pass") -> CapabilityProfile:
    """A profile whose small-prompt tool_call_shape finding is small_status.

    small_status=None -> no tool_call_shape finding at all (not run).
    """
    if small_status is None:
        return CapabilityProfile(model_id="cell")
    return CapabilityProfile(
        model_id="cell",
        findings={
            "tool_call_shape": DimensionFinding(
                dimension="tool_call_shape",
                status=small_status,
                summary="small-prompt result",
            ),
        },
    )


async def _probe_with(
    response: dict[str, Any],
    *,
    small_status: str | None = "pass",
) -> Any:
    async def call_responses(body: dict[str, Any]) -> dict[str, Any]:
        return response

    return await probe("cell", call_responses, _profile(small_status))


# ---------- skip short-circuit --------------------------------------------


async def test_probe_skipped_when_small_not_run() -> None:
    async def call_responses(body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("at-scale probe must not call upstream when small probe did not run")

    finding = await probe("cell", call_responses, _profile(None))
    assert finding.status == "skipped"
    assert finding.evidence["small_test_status"] == "not_run"


async def test_probe_skipped_when_small_failed() -> None:
    async def call_responses(body: dict[str, Any]) -> dict[str, Any]:
        raise AssertionError("at-scale probe must not call upstream when small probe failed")

    finding = await probe("cell", call_responses, _profile("fail"))
    assert finding.status == "skipped"
    assert finding.evidence["small_test_status"] == "fail"


# ---------- pass -----------------------------------------------------------


async def test_probe_pass_at_scale() -> None:
    finding = await _probe_with(_resp([_function_call(), _msg("done")]))
    assert finding.dimension == "tool_call_at_scale"
    assert finding.status == "pass"
    assert finding.evidence["prompt_chars"] == tool_call_at_scale._PROMPT_CHARS


# ---------- fail: text leak at scale ---------------------------------------


async def test_probe_fail_text_leak_at_scale() -> None:
    leak = _json_leak()
    finding = await _probe_with(_resp([_msg(leak)]))
    assert finding.status == "fail"
    assert finding.gap_class == "B"
    assert finding.suggested_action == ContractAction.AUTHOR_TEMPORARY_ADAPTER
    assert finding.evidence["prompt_chars"] == tool_call_at_scale._PROMPT_CHARS
    assert finding.evidence["text_tool_call_leak_examples"] == [leak]
    assert finding.evidence["structured_call_also_present"] is False


async def test_probe_fail_dual_emit_at_scale_records_structured_also_present() -> None:
    leak = _json_leak()
    finding = await _probe_with(_resp([_function_call(), _msg(leak)]))
    # dual-emit (structured + leak) is NOT pass (leak present); it falls
    # into the text-leak branch with structured_call_also_present=True.
    assert finding.status == "fail"
    assert finding.gap_class == "B"
    assert finding.evidence["structured_call_also_present"] is True


# ---------- fail: context degradation -------------------------------------


async def test_probe_fail_context_degradation_at_scale() -> None:
    finding = await _probe_with(_resp([_msg("just text, no tool")]))
    assert finding.status == "fail"
    assert finding.gap_class is None
    assert finding.suggested_action == ContractAction.QUARANTINE_CELL
    assert finding.evidence["prompt_chars"] == tool_call_at_scale._PROMPT_CHARS


# ---------- error: transport ----------------------------------------------


async def test_probe_error_at_scale() -> None:
    async def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("context too long")

    finding = await probe("cell", boom, _profile("pass"))
    assert finding.status == "error"
    assert finding.evidence["prompt_chars"] == tool_call_at_scale._PROMPT_CHARS
    assert "RuntimeError" in finding.evidence["exception"]


# ---------- dimension registration -----------------------------------------


def test_dimension_name_constant() -> None:
    assert tool_call_at_scale.DIMENSION_NAME == "tool_call_at_scale"
