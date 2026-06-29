"""Tests for the reasoning_channel capability dimension.

Fake upstream Responses payloads (no GPU): a model that inlines
`<think>` tags classifies as `inband_tags` with the observed tag set; a
model with a clean reasoning item classifies as `native`; a plain answer
classifies as `none`; empty/malformed classifies as `unknown` (status
error) so the transform's gate stays inert.
"""

from __future__ import annotations

from typing import Any

from callosum.capability.dimensions import reasoning_channel
from callosum.capability.dimensions.reasoning_channel import (
    classify_reasoning_channel,
    probe,
)
from callosum.capability.profile import CapabilityProfile


def _resp(output: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": "r", "object": "response", "status": "completed", "output": output}


def _message(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}


def _reasoning_item(text: str) -> dict[str, Any]:
    return {"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": text}]}


# ---------- classifier -----------------------------------------------------


def test_classify_inband_tags_records_observed_tag_set() -> None:
    cls = classify_reasoning_channel(_resp([_message("<think>hmm</think>Answer")]))
    assert cls.channel == "inband_tags"
    assert cls.observed_tags == [["<think>", "</think>"]]


def test_classify_inband_thought_variant() -> None:
    cls = classify_reasoning_channel(_resp([_message("<thought>x</thought>y")]))
    assert cls.channel == "inband_tags"
    assert cls.observed_tags == [["<thought>", "</thought>"]]


def test_classify_native_reasoning_item() -> None:
    cls = classify_reasoning_channel(_resp([_reasoning_item("careful thought"), _message("Answer")]))
    assert cls.channel == "native"


def test_classify_plain_answer_is_none() -> None:
    cls = classify_reasoning_channel(_resp([_message("just the answer")]))
    assert cls.channel == "none"


def test_classify_inband_wins_over_native_when_both_present() -> None:
    cls = classify_reasoning_channel(_resp([_reasoning_item("native"), _message("<think>leak</think>a")]))
    assert cls.channel == "inband_tags"


def test_classify_lone_open_tag_is_unknown() -> None:
    cls = classify_reasoning_channel(_resp([_message("<think>never closed")]))
    assert cls.channel == "unknown"
    assert cls.saw_lone_open_tag is True


def test_classify_empty_output_is_unknown() -> None:
    assert classify_reasoning_channel(_resp([])).channel == "unknown"
    assert classify_reasoning_channel(None).channel == "unknown"
    assert classify_reasoning_channel({"output": "nonsense"}).channel == "unknown"


# ---------- probe ----------------------------------------------------------


def _profile() -> CapabilityProfile:
    return CapabilityProfile(model_id="cell")


async def _probe_with(response: Any):
    async def call_responses(body: dict[str, Any]) -> dict[str, Any]:
        return response

    return await probe("cell", call_responses, _profile())


async def test_probe_inband_fails_with_tag_set_and_hint() -> None:
    finding = await _probe_with(_resp([_message("<think>cot</think>Final")]))
    assert finding.dimension == "reasoning_channel"
    assert finding.status == "fail"
    assert finding.evidence["reasoning_channel"] == "inband_tags"
    assert finding.evidence["observed_tags"] == [["<think>", "</think>"]]
    assert finding.adapter_hint is not None


async def test_probe_native_passes() -> None:
    finding = await _probe_with(_resp([_reasoning_item("t"), _message("a")]))
    assert finding.status == "pass"
    assert finding.evidence["reasoning_channel"] == "native"


async def test_probe_none_passes() -> None:
    finding = await _probe_with(_resp([_message("plain")]))
    assert finding.status == "pass"
    assert finding.evidence["reasoning_channel"] == "none"


async def test_probe_unknown_is_error_so_gate_stays_inert() -> None:
    finding = await _probe_with(_resp([]))
    assert finding.status == "error"
    assert finding.evidence["reasoning_channel"] == "unknown"


async def test_probe_transport_error_is_error() -> None:
    async def boom(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("network down")

    finding = await probe("cell", boom, _profile())
    assert finding.status == "error"
    assert finding.evidence["reasoning_channel"] == "unknown"


def test_dimension_registered_in_execution_order() -> None:
    from callosum.capability.dimensions import DIMENSIONS

    names = [name for name, _ in DIMENSIONS]
    assert "reasoning_channel" in names


def test_dimension_name_constant() -> None:
    assert reasoning_channel.DIMENSION_NAME == "reasoning_channel"
