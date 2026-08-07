"""Tests for the tool_call_shape capability dimension PROBE.

_shape_utils.classify_response is already covered by
test_shape_utils.py; these tests pin the probe that wraps it and
builds the DimensionFinding — the 4-branch fail-mode classification
(transport error, pass, dual-emit, text-as-tool-call-only, refused/
empty) and the structured gap-report fields (gap_class / suggested_action
/ upstream_owner / close_condition) each branch must populate.

All hermetic: a fake call_responses callable returns a canned
Responses-API dict; no GPU, no network, no app. Mirrors the sibling
test_capability_reasoning_channel.py harness.
"""

from __future__ import annotations

from typing import Any

from callosum.capability.dimensions import tool_call_shape
from callosum.capability.dimensions.tool_call_shape import probe
from callosum.capability.profile import CapabilityProfile
from callosum.substrate_contract import ContractAction

# Angle-bracket glyphs built via chr() so this source contains no literal
# Hermes tag sequences (same convention as test_shape_utils.py).
LT, GT = chr(60), chr(62)


def _msg(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _function_call(name: str = "exec_command", arguments: str = '{"cmd":"ls"}') -> dict[str, Any]:
    return {"type": "function_call", "name": name, "arguments": arguments, "call_id": "fc_1"}


def _resp(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"output": items}


def _json_leak() -> str:
    return '{"name": "exec_command", "arguments": {"cmd": "ls"}}'


def _func_tag(body: str, name: str = "exec_command") -> str:
    return LT + "function=" + name + GT + body + LT + "/function" + GT


def _profile() -> CapabilityProfile:
    return CapabilityProfile(model_id="cell")


async def _probe_with(response: dict[str, Any]) -> Any:
    async def call_responses(body: dict[str, Any]) -> dict[str, Any]:
        return response

    return await probe("cell", call_responses, _profile())


# ---------- pass -----------------------------------------------------------


async def test_probe_pass_structured_call_no_leak() -> None:
    finding = await _probe_with(_resp([_function_call(), _msg("done")]))
    assert finding.dimension == "tool_call_shape"
    assert finding.status == "pass"
    assert finding.evidence["function_calls_count"] == 1
    # pass / error / skipped leave the structured gap report unset
    assert finding.gap_class is None
    assert finding.suggested_action is None


# ---------- fail: dual-emit ------------------------------------------------


async def test_probe_fail_dual_emit_structured_and_text_leak() -> None:
    leak = _json_leak()
    finding = await _probe_with(_resp([_function_call(), _msg(leak)]))
    assert finding.status == "fail"
    assert finding.gap_class == "B"
    assert finding.suggested_action == ContractAction.AUTHOR_TEMPORARY_ADAPTER
    assert finding.upstream_owner is not None
    assert finding.close_condition is not None
    assert finding.adapter_hint is not None
    assert finding.evidence["text_tool_call_leak_examples"] == [leak]


# ---------- fail: text-as-tool-call only -----------------------------------


async def test_probe_fail_text_as_tool_call_only_json() -> None:
    leak = _json_leak()
    finding = await _probe_with(_resp([_msg(leak)]))
    assert finding.status == "fail"
    assert finding.gap_class == "B"
    assert finding.suggested_action == ContractAction.AUTHOR_TEMPORARY_ADAPTER
    assert finding.evidence["text_tool_call_leak_examples"] == [leak]


async def test_probe_fail_text_as_tool_call_only_hermes() -> None:
    tag = _func_tag('{"name":"exec_command","arguments":"ls"}')
    finding = await _probe_with(_resp([_msg(tag)]))
    assert finding.status == "fail"
    assert finding.gap_class == "B"
    assert finding.suggested_action == ContractAction.AUTHOR_TEMPORARY_ADAPTER
    assert finding.evidence["text_tool_call_leak_examples"] == [tag]


# ---------- fail: refused / empty ------------------------------------------


async def test_probe_fail_refused_plain_text() -> None:
    finding = await _probe_with(_resp([_msg("I cannot use tools")]))
    assert finding.status == "fail"
    # refusal is not a substrate-owned quirk -> no gap_class, quarantine
    assert finding.gap_class is None
    assert finding.suggested_action == ContractAction.QUARANTINE_CELL
    assert finding.evidence["output_item_types"] == ["message"]
    assert finding.evidence["text_excerpts"] == ["I cannot use tools"]


async def test_probe_fail_empty_output() -> None:
    finding = await _probe_with(_resp([]))
    assert finding.status == "fail"
    assert finding.gap_class is None
    assert finding.suggested_action == ContractAction.QUARANTINE_CELL


# ---------- error: transport ----------------------------------------------


async def test_probe_error_transport() -> None:
    async def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("network down")

    finding = await probe("cell", boom, _profile())
    assert finding.status == "error"
    assert "RuntimeError" in finding.evidence["exception"]
    assert finding.adapter_hint is not None
    assert "transport" in finding.adapter_hint.lower()


# ---------- dimension registration -----------------------------------------


def test_dimension_name_constant() -> None:
    assert tool_call_shape.DIMENSION_NAME == "tool_call_shape"
