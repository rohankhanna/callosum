"""Dimension: reasoning_channel.

Thinking-class models deliver chain-of-thought in one of several ways.
This dimension probes the OBSERVED behavior — a declared/static "is a
thinking model" flag is not the same as how the CoT actually reaches a
Responses-API client (the same lesson the tool_call_shape dimension
encodes for tool calls). It sends a thinking-trigger prompt and
classifies the cell's reasoning channel into one of:

  * native        — CoT arrives in a separate field (the backend already
                    emits clean Responses `reasoning` items, e.g. a cell
                    fronted by local LLM gateway's responses-proxy, or a
                    runtime that surfaces `thinking`/`reasoning_content`
                    which callosum's translator lifts into a reasoning
                    item). No transform needed.
  * inband_tags   — CoT arrives INSIDE message content wrapped in the
                    model's own markup (`<think>…</think>` /
                    `<thought>…</thought>`). The leak case. The observed
                    tag set is recorded in the finding so the transform
                    is data-driven, not hardcoded.
  * none          — no reasoning emitted (non-thinking model).
  * unknown       — probe could not classify (empty/error, or markup seen
                    but not a complete pair). Maps to `status=error` so
                    the transform's gate stays inert (fail safe) and the
                    sweep re-runs it next pass.

The finding's status maps the four channels onto the harness contract:
`native` and `none` are clean (`pass`), `inband_tags` needs an adapter
(`fail`, carrying the tag set + adapter_hint), `unknown` is `error`.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from callosum.capability.profile import CapabilityProfile, DimensionFinding
from callosum.capability.request_shapes import reasoning_channel_probe_body

DIMENSION_NAME = "reasoning_channel"

# Candidate in-band tag pairs the probe looks for in message content.
# Kept in sync with transforms.inband_reasoning.DEFAULT_REASONING_TAGS.
_CANDIDATE_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thought>", "</thought>"),
)


@dataclass
class ReasoningChannelClassification:
    """Result of inspecting a probe response."""

    channel: str  # "native" | "inband_tags" | "none" | "unknown"
    observed_tags: list[list[str]]  # [[open, close], ...] for inband_tags
    has_reasoning_item: bool
    saw_lone_open_tag: bool
    text_excerpts: list[str]


def _message_texts(output: list[Any]) -> list[str]:
    texts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
    return texts


def _reasoning_item_has_text(output: list[Any]) -> bool:
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        summary = item.get("summary")
        if isinstance(summary, list):
            for part in summary:
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                    return True
    return False


def classify_reasoning_channel(response: dict[str, Any] | None) -> ReasoningChannelClassification:
    """Inspect an upstream Responses payload and classify how reasoning
    was delivered. Pass-through-safe on None / malformed input
    (classifies as `unknown` so the transform never fires)."""
    if not isinstance(response, dict):
        return ReasoningChannelClassification("unknown", [], False, False, [])
    output = response.get("output")
    if not isinstance(output, list):
        return ReasoningChannelClassification("unknown", [], False, False, [])

    texts = _message_texts(output)
    has_reasoning_item = _reasoning_item_has_text(output)

    observed: list[list[str]] = []
    saw_lone_open = False
    for text in texts:
        for open_tag, close_tag in _CANDIDATE_TAGS:
            open_idx = text.find(open_tag)
            if open_idx == -1:
                continue
            close_idx = text.find(close_tag, open_idx + len(open_tag))
            if close_idx != -1:
                if [open_tag, close_tag] not in observed:
                    observed.append([open_tag, close_tag])
            else:
                saw_lone_open = True

    excerpts = [t[:300] for t in texts[:3]]

    # In-band tags win: a complete pair is unambiguous markup leakage and
    # needs the transform regardless of whether a native item also exists.
    if observed:
        return ReasoningChannelClassification("inband_tags", observed, has_reasoning_item, saw_lone_open, excerpts)
    # A clean separate reasoning item → native.
    if has_reasoning_item:
        return ReasoningChannelClassification("native", [], True, saw_lone_open, excerpts)
    # Markup seen but never closed → can't safely classify; fail safe.
    if saw_lone_open:
        return ReasoningChannelClassification("unknown", [], False, True, excerpts)
    # Visible answer with no reasoning of any kind → non-thinking turn.
    if any(t.strip() for t in texts):
        return ReasoningChannelClassification("none", [], False, False, excerpts)
    # Nothing usable.
    return ReasoningChannelClassification("unknown", [], False, False, excerpts)


async def probe(
    cell: str,
    call_responses: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
    profile: CapabilityProfile,  # noqa: ARG001 — kept for dimension-signature symmetry
) -> DimensionFinding:
    """Send the thinking-trigger prompt and classify the reasoning
    channel. `call_responses` takes a Responses-API body and returns the
    upstream response dict (the abstraction over how the probe reaches
    the cell — admin cell-call, direct backend call, or a test stub)."""
    body = reasoning_channel_probe_body()
    body["model"] = cell
    try:
        response = await call_responses(body)
    except Exception as exc:
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="error",
            summary=f"call_responses raised: {type(exc).__name__}: {exc}",
            evidence={"reasoning_channel": "unknown", "exception": f"{type(exc).__name__}: {exc}"},
            adapter_hint="transport failure — not a model quirk. Investigate callosum's backend health for this cell.",
        )

    cls = classify_reasoning_channel(response)

    if cls.channel == "inband_tags":
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="fail",
            summary=(
                "in-band reasoning: chain-of-thought is inlined into "
                f"message content wrapped in {[o for o, _ in cls.observed_tags]}. "
                "Without an adapter the markup leaks into the visible answer."
            ),
            evidence={
                "reasoning_channel": "inband_tags",
                "observed_tags": cls.observed_tags,
                "also_has_native_item": cls.has_reasoning_item,
                "text_excerpts": cls.text_excerpts,
            },
            adapter_hint=(
                "register the inband_reasoning transform: split message "
                "output_text on the observed tag set, lift the spans into "
                "a Responses reasoning item, and leave the visible answer "
                "tag-free (streaming + non-streaming)."
            ),
        )
    if cls.channel == "native":
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="pass",
            summary="native reasoning channel: clean Responses reasoning item, no in-band markup. No transform needed.",
            evidence={"reasoning_channel": "native", "text_excerpts": cls.text_excerpts},
        )
    if cls.channel == "none":
        return DimensionFinding(
            dimension=DIMENSION_NAME,
            status="pass",
            summary="no reasoning emitted (non-thinking turn). Nothing to re-route.",
            evidence={"reasoning_channel": "none", "text_excerpts": cls.text_excerpts},
        )
    return DimensionFinding(
        dimension=DIMENSION_NAME,
        status="error",
        summary=(
            "inconclusive: could not classify the reasoning channel "
            "(empty response, or markup seen without a complete tag pair). "
            "Transform stays inert; sweep will re-probe."
        ),
        evidence={
            "reasoning_channel": "unknown",
            "saw_lone_open_tag": cls.saw_lone_open_tag,
            "text_excerpts": cls.text_excerpts,
        },
    )
