"""Unit tests for callosum.capability.conformance.

P5-prep materializes the dormant substrate-contract conformance pipeline.
These pin three layers: the seven pure invariant checks (fake Responses
payloads, no GPU), the build_contract_profile builder (a stub
call_responses transport), and the persistence round-trip.

The most important assertion is test_builder_transport_error_yields_false_not_none:
classify_contract quarantines only on an explicit False (it uses
SurfaceConformance.violations()), so the builder must never leave an
always-applicable invariant as None — otherwise an all-None surface
would *falsely* route_native.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from callosum.capability.conformance import (
    build_contract_profile,
    check_finish_reason_mapping,
    check_ordering_reasoning_then_function_call_then_message,
    check_output_never_empty,
    check_parallel_tool_call_collapse,
    check_reasoning_alias_unification,
    check_tool_call_at_scale_probe_through_substrate,
    check_usage_input_tokens_present,
    contract_profile_path,
    load_contract_profile,
    save_contract_profile,
)
from callosum.substrate_contract import (
    REQUIRED_CAPABILITY_FIELDS,
    CellContractProfile,
    ContractAction,
    Surface,
    SurfaceConformance,
    classify_contract,
)
from callosum.substrate_contract import (
    ConformanceInvariant as Inv,
)

# --------------------------------------------------------------------
# Response fixtures (Responses-API shaped, no GPU).
# --------------------------------------------------------------------


def _usage(input_tokens: int = 10) -> dict[str, Any]:
    return {"input_tokens": input_tokens, "output_tokens": 5}


def _function_call(name: str = "exec_command", arguments: str = '{"cmd":"ls"}') -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": "fc",
        "call_id": "call_fc",
        "name": name,
        "arguments": arguments,
    }


def _message(text: str = "done") -> dict[str, Any]:
    return {
        "type": "message",
        "id": "m",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _reasoning_item(text: str = "careful thought") -> dict[str, Any]:
    return {
        "type": "reasoning",
        "id": "rs",
        "summary": [{"type": "summary_text", "text": text}],
    }


def _resp(
    output: list[dict[str, Any]],
    *,
    status: str = "completed",
    usage: dict[str, Any] | None = None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    r: dict[str, Any] = {
        "id": "r",
        "object": "response",
        "status": status,
        "output": output,
    }
    if usage is not None:
        r["usage"] = usage
    if incomplete_reason is not None:
        r["incomplete_details"] = {"reason": incomplete_reason}
    return r


# --------------------------------------------------------------------
# Pure invariant checks.
# --------------------------------------------------------------------


# --- invariant 1: usage.input_tokens present -----------------------------


def test_usage_input_tokens_present_pass() -> None:
    assert check_usage_input_tokens_present(_resp([], usage=_usage(10))) is True


def test_usage_input_tokens_present_zero_is_false() -> None:
    assert check_usage_input_tokens_present(_resp([], usage=_usage(0))) is False


def test_usage_input_tokens_present_missing_usage_is_false() -> None:
    # a real response without usage is a hard violation (Codex hard-fails)
    assert check_usage_input_tokens_present(_resp([_message()], usage=None)) is False


def test_usage_input_tokens_present_non_dict_is_none() -> None:
    assert check_usage_input_tokens_present(None) is None
    assert check_usage_input_tokens_present("not a dict") is None  # type: ignore[arg-type]


# --- invariant 2: output never empty -------------------------------------


def test_output_never_empty_pass() -> None:
    assert check_output_never_empty(_resp([_message()])) is True


def test_output_never_empty_empty_is_false() -> None:
    assert check_output_never_empty(_resp([])) is False


def test_output_never_empty_missing_output_is_false() -> None:
    assert check_output_never_empty({"status": "completed"}) is False


def test_output_never_empty_non_dict_is_none() -> None:
    assert check_output_never_empty(None) is None


# --- invariant 3: ordering reasoning -> function_call -> message --------


def test_ordering_pass_in_order() -> None:
    assert (
        check_ordering_reasoning_then_function_call_then_message(
            _resp([_reasoning_item(), _function_call(), _message()])
        )
        is True
    )


def test_ordering_pass_message_only_vacuously() -> None:
    assert check_ordering_reasoning_then_function_call_then_message(_resp([_message()])) is True


def test_ordering_violation_message_before_function_call() -> None:
    assert check_ordering_reasoning_then_function_call_then_message(_resp([_message(), _function_call()])) is False


def test_ordering_non_dict_is_none() -> None:
    assert check_ordering_reasoning_then_function_call_then_message(None) is None


# --- invariant 4: parallel tool-call collapse ----------------------------


def test_parallel_tool_call_collapse_pass_two_calls() -> None:
    assert (
        check_parallel_tool_call_collapse(
            _resp([_function_call(name="read_file"), _function_call(name="list_directory")])
        )
        is True
    )


def test_parallel_tool_call_collapse_none_when_one_call() -> None:
    # the model chose not to parallelize — a model choice, not a substrate violation
    assert check_parallel_tool_call_collapse(_resp([_function_call()])) is None


def test_parallel_tool_call_collapse_none_when_no_calls() -> None:
    assert check_parallel_tool_call_collapse(_resp([_message()])) is None


def test_parallel_tool_call_collapse_non_dict_is_none() -> None:
    assert check_parallel_tool_call_collapse(None) is None


# --- invariant 5: finish-reason mapping ----------------------------------


def test_finish_reason_mapping_completed_pass() -> None:
    assert check_finish_reason_mapping(_resp([_message()], status="completed")) is True


def test_finish_reason_mapping_incomplete_with_reason_pass() -> None:
    assert (
        check_finish_reason_mapping(_resp([_message()], status="incomplete", incomplete_reason="max_output_tokens"))
        is True
    )


def test_finish_reason_mapping_incomplete_without_reason_is_false() -> None:
    assert check_finish_reason_mapping(_resp([_message()], status="incomplete")) is False


def test_finish_reason_mapping_missing_status_is_false() -> None:
    assert check_finish_reason_mapping({"output": [_message()]}) is False


def test_finish_reason_mapping_non_dict_is_none() -> None:
    assert check_finish_reason_mapping(None) is None


# --- invariant 6: reasoning alias unification ----------------------------


def test_reasoning_alias_unification_native_pass() -> None:
    assert check_reasoning_alias_unification(_resp([_reasoning_item(), _message("answer")])) is True


def test_reasoning_alias_unification_inband_tags_is_false() -> None:
    assert check_reasoning_alias_unification(_resp([_message("<thought>thinking</thought>answer")])) is False


def test_reasoning_alias_unification_none_channel_is_none() -> None:
    # non-thinking turn — not applicable
    assert check_reasoning_alias_unification(_resp([_message("just the answer")])) is None


def test_reasoning_alias_unification_non_dict_is_none() -> None:
    assert check_reasoning_alias_unification(None) is None


# --- invariant 7: tool-call at scale through substrate -------------------


def test_tool_call_at_scale_pass_structured_no_leak() -> None:
    assert check_tool_call_at_scale_probe_through_substrate(_resp([_function_call(), _message()])) is True


def test_tool_call_at_scale_no_structured_call_is_false() -> None:
    assert check_tool_call_at_scale_probe_through_substrate(_resp([_message("no tool call")])) is False


def test_tool_call_at_scale_text_tool_call_leak_is_false() -> None:
    leak = '{"name":"exec_command","arguments":"{\\"cmd\\":\\"ls\\"}"}'
    assert check_tool_call_at_scale_probe_through_substrate(_resp([_function_call(), _message(leak)])) is False


def test_tool_call_at_scale_hermes_tag_leak_is_false() -> None:
    # Hermes tag format: a function=NAME block wrapping a tool-call JSON
    # object (built via chr() so no literal angle-bracket tag glyphs appear
    # in this source). The classifier must recognize it as a text tool-call
    # leak even though it is not a bare JSON object.
    lt, gt = chr(60), chr(62)
    leak = f"{lt}function=exec_command{gt}" + '{"name":"exec_command","arguments":"ls"}' + f"{lt}/function{gt}"
    assert check_tool_call_at_scale_probe_through_substrate(_resp([_function_call(), _message(leak)])) is False


def test_tool_call_at_scale_non_dict_is_none() -> None:
    assert check_tool_call_at_scale_probe_through_substrate(None) is None


# --------------------------------------------------------------------
# Builder (stub transport).
# --------------------------------------------------------------------


class _StubTransport:
    """A stub call_responses returning fixed responses by call index,
    optionally raising on a set of indices. Records call count + bodies."""

    def __init__(
        self,
        responses: list[dict[str, Any]],
        raise_on: frozenset[int] = frozenset(),
    ) -> None:
        self._responses = responses
        self._raise_on = raise_on
        self.call_count = 0
        self.bodies: list[dict[str, Any]] = []

    async def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        i = self.call_count
        self.call_count += 1
        self.bodies.append(body)
        if i in self._raise_on:
            raise RuntimeError("transport error")
        return self._responses[i]


def _all_pass_responses() -> list[dict[str, Any]]:
    """Four conformant responses for probes A (small tool-call), B
    (reasoning), C (parallel), D (at-scale)."""
    return [
        # A: usage + structured call + message, status completed
        _resp([_function_call(), _message()], usage=_usage(10)),
        # B: reasoning item + message (native channel, ordering ok)
        _resp([_reasoning_item(), _message("answer")]),
        # C: two function_call items (parallel collapse)
        _resp([_function_call(name="read_file"), _function_call(name="list_directory")]),
        # D: structured call + message, no leak (at-scale)
        _resp([_function_call(), _message()], usage=_usage(10)),
    ]


def _full_capability_fields() -> frozenset[str]:
    return frozenset(REQUIRED_CAPABILITY_FIELDS)


async def test_builder_responses_surface_all_pass_routes_native() -> None:
    transport = _StubTransport(_all_pass_responses())
    profile = await build_contract_profile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    assert transport.call_count == 4  # A, B, C, D (small probe passed)
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.ROUTE_NATIVE
    conf = profile.conformance[Surface.RESPONSES]
    assert conf.violations() == ()


async def test_builder_usage_violation_quarantines() -> None:
    responses = list(_all_pass_responses())
    # probe A omits usage -> invariant 1 is False
    responses[0] = _resp([_function_call(), _message()])  # no usage key
    transport = _StubTransport(responses)
    profile = await build_contract_profile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert Inv.USAGE_INPUT_TOKENS_PRESENT in verdict.violated_invariants


async def test_builder_inband_reasoning_quarantines() -> None:
    responses = list(_all_pass_responses())
    # probe B leaks in-band reasoning tags -> invariant 6 is False
    responses[1] = _resp([_message("<thought>thinking</thought>answer")])
    transport = _StubTransport(responses)
    profile = await build_contract_profile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert Inv.REASONING_ALIAS_UNIFICATION in verdict.violated_invariants


async def test_builder_at_scale_skipped_when_small_probe_failed() -> None:
    responses = list(_all_pass_responses())
    # probe A is text-only -> no structured call -> at-scale gate fails -> D skipped
    responses[0] = _resp([_message("no tool, just text")], usage=_usage(10))
    # only A, B, C are issued; drop D so the stub wouldn't satisfy a 4th call
    responses = responses[:3]
    transport = _StubTransport(responses)
    profile = await build_contract_profile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    assert transport.call_count == 3  # A, B, C — D skipped
    inv7 = profile.conformance[Surface.RESPONSES].invariant_results[Inv.TOOL_CALL_AT_SCALE_PROBE_THROUGH_SUBSTRATE]
    assert inv7 is None


async def test_builder_chat_advertised_not_probed_quarantines() -> None:
    transport = _StubTransport(_all_pass_responses())
    profile = await build_contract_profile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.CHAT, Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    # only RESPONSES was probed; CHAT has no conformance entry
    assert Surface.RESPONSES in profile.conformance
    assert Surface.CHAT not in profile.conformance
    verdict = classify_contract(profile, Surface.CHAT)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert "not probed" in verdict.reason


async def test_builder_transport_error_yields_false_not_none() -> None:
    # The None-trap guard: probe A (small tool-call) transport-errors, so the
    # always-applicable invariants 1/2/5 become False (not None). Without this
    # guard an all-None surface would falsely route_native. The other probes
    # (B, C) succeed so the surface has a mix of True/None/False — the False
    # from inv 1/2/5 is what quarantines it.
    transport = _StubTransport(
        responses=[
            _resp([_reasoning_item(), _message("answer")]),  # index 1 -> probe B
            _resp([_function_call(name="a"), _function_call(name="b")]),  # index 2 -> probe C
        ],
        raise_on=frozenset({0}),  # index 0 -> probe A raises
    )
    profile = await build_contract_profile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    conf = profile.conformance[Surface.RESPONSES]
    # always-applicable invariants are False (not None) after the transport error
    assert conf.invariant_results[Inv.USAGE_INPUT_TOKENS_PRESENT] is False
    assert conf.invariant_results[Inv.OUTPUT_NEVER_EMPTY] is False
    assert conf.invariant_results[Inv.FINISH_REASON_MAPPING] is False
    verdict = classify_contract(profile, Surface.RESPONSES)
    assert verdict.action is ContractAction.QUARANTINE_CELL
    assert Inv.USAGE_INPUT_TOKENS_PRESENT in verdict.violated_invariants
    assert verdict.action is not ContractAction.ROUTE_NATIVE


async def test_builder_sets_model_on_every_probe_body() -> None:
    transport = _StubTransport(_all_pass_responses())
    await build_contract_profile(
        cell_id="my-cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        capability_fields=_full_capability_fields(),
        call_responses=transport,
    )
    assert transport.call_count == 4
    assert all(body["model"] == "my-cell" for body in transport.bodies)


# --------------------------------------------------------------------
# Persistence round-trip.
# --------------------------------------------------------------------


def test_surface_conformance_roundtrip() -> None:
    sc = SurfaceConformance(
        surface=Surface.RESPONSES,
        invariant_results={
            Inv.USAGE_INPUT_TOKENS_PRESENT: True,
            Inv.OUTPUT_NEVER_EMPTY: False,
            Inv.ORDERING_REASONING_THEN_FUNCTION_CALL_THEN_MESSAGE: None,
        },
    )
    revived = SurfaceConformance.from_dict(sc.to_dict())
    assert revived.surface is Surface.RESPONSES
    assert revived.invariant_results[Inv.USAGE_INPUT_TOKENS_PRESENT] is True
    assert revived.invariant_results[Inv.OUTPUT_NEVER_EMPTY] is False
    assert revived.invariant_results[Inv.ORDERING_REASONING_THEN_FUNCTION_CALL_THEN_MESSAGE] is None
    assert revived.violations() == (Inv.OUTPUT_NEVER_EMPTY,)


def test_contract_profile_roundtrip() -> None:
    sc = SurfaceConformance(
        surface=Surface.RESPONSES,
        invariant_results={inv: True for inv in Inv},
    )
    profile = CellContractProfile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES, Surface.CHAT}),
        conformance={Surface.RESPONSES: sc},
        capability_fields=frozenset({"context_window", "quantization"}),
        residual_translation=True,
    )
    revived = CellContractProfile.from_dict(profile.to_dict())
    assert revived.cell_id == "cell"
    assert revived.advertised_surfaces == frozenset({Surface.RESPONSES, Surface.CHAT})
    assert revived.capability_fields == frozenset({"context_window", "quantization"})
    assert revived.residual_translation is True
    assert Surface.RESPONSES in revived.conformance
    assert revived.conformance[Surface.RESPONSES].is_conformant()


def test_contract_profile_roundtrip_json_safe() -> None:
    # to_dict must produce a structure json can serialize (frozenset -> list, etc.)
    profile = CellContractProfile(
        cell_id="cell",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        conformance={},
        capability_fields=frozenset({"a", "b"}),
    )
    serialized = json.dumps(profile.to_dict(), sort_keys=True)
    revived = CellContractProfile.from_dict(json.loads(serialized))
    assert revived == profile


def test_contract_profile_from_dict_drops_unknown_surfaces() -> None:
    # forward-compatible load: a future schema surface is dropped, not fatal
    data = {
        "cell_id": "cell",
        "advertised_surfaces": ["responses", "future_surface"],
        "conformance": {},
        "capability_fields": ["context_window"],
        "residual_translation": False,
    }
    revived = CellContractProfile.from_dict(data)
    assert revived.advertised_surfaces == frozenset({Surface.RESPONSES})


@pytest.fixture(autouse=True)
def _redirect_contract_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate contract-profile persistence to a tmp dir per test."""
    monkeypatch.setattr("callosum.capability.conformance.DEFAULT_CONTRACT_DIR", tmp_path)
    yield


def test_save_load_contract_profile(tmp_path: Path) -> None:
    sc = SurfaceConformance(
        surface=Surface.RESPONSES,
        invariant_results={inv: True for inv in Inv},
    )
    profile = CellContractProfile(
        cell_id="models/local/model-a0d5",
        advertised_surfaces=frozenset({Surface.RESPONSES}),
        conformance={Surface.RESPONSES: sc},
        capability_fields=frozenset(REQUIRED_CAPABILITY_FIELDS),
        residual_translation=False,
    )
    path = save_contract_profile(profile)
    assert path == contract_profile_path("models/local/model-a0d5")
    assert path.parent == tmp_path
    revived = load_contract_profile("models/local/model-a0d5")
    assert revived is not None
    assert revived == profile


def test_load_missing_returns_none() -> None:
    assert load_contract_profile("never-probed") is None


def test_load_corrupt_returns_none(tmp_path: Path) -> None:
    path = contract_profile_path("corrupt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ not valid json", encoding="utf-8")
    assert load_contract_profile("corrupt") is None


def test_load_non_dict_returns_none(tmp_path: Path) -> None:
    path = contract_profile_path("weird")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")  # valid JSON, not a dict
    assert load_contract_profile("weird") is None
