"""Tests for callosum.transforms.inband_reasoning.

Three layers:

  * The stateful splitter — single-chunk, two-delta straddle of both the
    opening AND closing tag, tag-free passthrough (incl. a literal `<`),
    a lone trailing `<` held until flush, and an unclosed reasoning span
    flushed as reasoning.
  * The non-streaming transform — lifts in-band spans into a reasoning
    item, strips tags from the visible answer, and stays inert for cells
    that aren't classified `inband_tags`.
  * The streaming gate `inband_splitter_for_model` — returns a configured
    splitter only for an `inband_tags` profile on disk.
"""

from __future__ import annotations

from callosum.capability.profile import (
    CapabilityProfile,
    DimensionFinding,
    save_profile,
)
from callosum.cell_grid import Cell
from callosum.transforms import TransformContext
from callosum.transforms.inband_reasoning import (
    InbandReasoningSplitter,
    InbandReasoningTransform,
    inband_splitter_for_model,
)

# ---------- splitter -------------------------------------------------------


def _drain(splitter: InbandReasoningSplitter, *chunks: str) -> list[tuple[str, str]]:
    """Feed chunks through the splitter and coalesce consecutive same-
    channel runs so assertions speak to the semantic invariant."""
    raw: list[tuple[str, str]] = []
    for chunk in chunks:
        raw += splitter.push(chunk)
    raw += splitter.flush()
    coalesced: list[tuple[str, str]] = []
    for channel, text in raw:
        if coalesced and coalesced[-1][0] == channel:
            coalesced[-1] = (channel, coalesced[-1][1] + text)
        else:
            coalesced.append((channel, text))
    return coalesced


def test_splitter_single_chunk_splits_thought_from_answer() -> None:
    segments = _drain(InbandReasoningSplitter(), "<thought>ponder</thought>Paris")
    assert segments == [("reasoning", "ponder"), ("content", "Paris")]


def test_splitter_tag_straddles_two_deltas() -> None:
    """The opening AND closing tags are each split across a delta boundary."""
    segments = _drain(
        InbandReasoningSplitter(),
        "<thi",  # opening tag begins...
        "nk>reason",  # ...completes, then reasoning starts
        "ing</thi",  # closing tag begins...
        "nk>Paris",  # ...completes, then the answer
    )
    assert segments == [("reasoning", "reasoning"), ("content", "Paris")]


def test_splitter_passes_through_content_without_tags() -> None:
    """A literal `<` that never becomes a tag must survive as content."""
    segments = _drain(InbandReasoningSplitter(), "x ", "< y ", "still text")
    assert segments == [("content", "x < y still text")]


def test_splitter_holds_lone_trailing_angle_until_flush() -> None:
    splitter = InbandReasoningSplitter()
    # A trailing '<' is a possible tag start, so it is held back, not emitted.
    assert splitter.push("answer<") == [("content", "answer")]
    # It turns out not to be a tag; flush releases it as content.
    assert splitter.flush() == [("content", "<")]


def test_splitter_unclosed_reasoning_flushes_as_reasoning() -> None:
    splitter = InbandReasoningSplitter()
    segments = splitter.push("<think>still thinking")
    segments += splitter.flush()
    assert segments == [("reasoning", "still thinking")]


def test_splitter_split_once_partitions_full_string() -> None:
    reasoning, content = InbandReasoningSplitter().split_once("<think> a </think>final")
    assert reasoning == " a "
    assert content == "final"


def test_splitter_uses_configured_tag_set() -> None:
    # A custom tag the default set does not know about.
    segments = _drain(InbandReasoningSplitter((("<reason>", "</reason>"),)), "<reason>r</reason>ans")
    assert segments == [("reasoning", "r"), ("content", "ans")]


# ---------- non-streaming transform ---------------------------------------


def _inband_ctx(model: str = "model-a0d5", tags: list[list[str]] | None = None) -> TransformContext:
    profile = CapabilityProfile(model_id=model)
    profile.upsert(
        DimensionFinding(
            dimension="reasoning_channel",
            status="fail",
            summary="in-band",
            evidence={
                "reasoning_channel": "inband_tags",
                "observed_tags": tags if tags is not None else [["<think>", "</think>"]],
            },
        )
    )
    return TransformContext(
        cell=Cell(model=model, reasoning_effort="default"),
        weight_identity=None,
        capability_profile=profile,
    )


def _responses_body(text: str, *, resp_id: str = "resp1") -> dict:
    return {
        "id": resp_id,
        "object": "response",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        ],
    }


def test_transform_applies_only_for_inband_tags_finding() -> None:
    t = InbandReasoningTransform()
    assert t.applies_to(_inband_ctx()) is True
    # No profile → inert.
    assert (
        t.applies_to(
            TransformContext(
                cell=Cell(model="x", reasoning_effort="default"),
                weight_identity=None,
                capability_profile=None,
            )
        )
        is False
    )


def test_transform_inert_for_native_channel() -> None:
    profile = CapabilityProfile(model_id="native-cell")
    profile.upsert(
        DimensionFinding(
            dimension="reasoning_channel",
            status="pass",
            summary="native",
            evidence={"reasoning_channel": "native"},
        )
    )
    ctx = TransformContext(
        cell=Cell(model="native-cell", reasoning_effort="default"),
        weight_identity=None,
        capability_profile=profile,
    )
    t = InbandReasoningTransform()
    assert t.applies_to(ctx) is False
    body = _responses_body("hello")
    # Even if called directly, a non-inband ctx must not rewrite.
    assert t.transform_response(body, ctx) == body


def test_transform_lifts_reasoning_and_strips_tags() -> None:
    t = InbandReasoningTransform()
    ctx = _inband_ctx()
    body = _responses_body("<think>the ball is $0.05</think>The ball costs $0.05.")
    out = t.transform_response(body, ctx)
    # First item is the lifted reasoning.
    assert out["output"][0]["type"] == "reasoning"
    assert out["output"][0]["summary"][0]["text"] == "the ball is $0.05"
    # The message keeps only the cleaned answer — no tags.
    msg = out["output"][1]
    assert msg["type"] == "message"
    assert msg["content"][0]["text"] == "The ball costs $0.05."
    assert "<think>" not in msg["content"][0]["text"]


def test_transform_pure_reasoning_collapses_message() -> None:
    """When the message was nothing but reasoning, the empty message item
    collapses and only the reasoning item remains."""
    t = InbandReasoningTransform()
    out = t.transform_response(_responses_body("<think>only thinking</think>"), _inband_ctx())
    assert [item["type"] for item in out["output"]] == ["reasoning"]
    assert out["output"][0]["summary"][0]["text"] == "only thinking"


def test_transform_no_tags_leaves_body_unchanged() -> None:
    """An inband_tags cell that didn't emit tags this turn is untouched —
    we never rewrite content we didn't change."""
    t = InbandReasoningTransform()
    body = _responses_body("plain answer, no markup")
    assert t.transform_response(body, _inband_ctx()) == body


# ---------- streaming gate -------------------------------------------------


def _write_profile(tmp_path, model: str, channel: str, tags=None) -> None:
    profile = CapabilityProfile(model_id=model)
    evidence: dict = {"reasoning_channel": channel}
    if tags is not None:
        evidence["observed_tags"] = tags
    profile.upsert(
        DimensionFinding(
            dimension="reasoning_channel",
            status="fail" if channel == "inband_tags" else "pass",
            summary=channel,
            evidence=evidence,
        )
    )
    save_profile(profile, profile_dir=tmp_path)


def test_gate_returns_splitter_for_inband_cell(tmp_path) -> None:
    _write_profile(tmp_path, "model-a0d5-inband", "inband_tags", tags=[["<think>", "</think>"]])
    splitter = inband_splitter_for_model("model-a0d5-inband", profile_dir=tmp_path)
    assert splitter is not None
    assert splitter.split_once("<think>r</think>a") == ("r", "a")


def test_gate_returns_none_for_native_cell(tmp_path) -> None:
    _write_profile(tmp_path, "clean", "native")
    assert inband_splitter_for_model("clean", profile_dir=tmp_path) is None


def test_gate_returns_none_for_unprobed_cell(tmp_path) -> None:
    assert inband_splitter_for_model("never-probed", profile_dir=tmp_path) is None
