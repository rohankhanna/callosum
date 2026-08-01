"""In-band reasoning re-routing.

Some thinking-class local models do not expose chain-of-thought in a
separate field. Over an OpenAI-compatible /v1/chat/completions surface
(e.g. model-a0d5 via ollama) the CoT is inlined into `message.content`
wrapped in the model's own markup -- `<think>…</think>` or
`<thought>…</thought>`. When callosum translates that chat upstream into
the Responses API a Codex CLI client expects, the markup leaks into the
visible answer instead of rendering as a collapsible reasoning block.

This module re-routes those in-band spans out of the visible content
channel and into Responses-API reasoning items. It exposes:

  * `InbandReasoningSplitter` -- a stateful splitter (ported from
    local LLM gateway's responses-proxy, proven there) that partitions a
    content stream into ("reasoning", text) / ("content", text)
    segments. It is robust to a tag straddling an SSE delta boundary
    (`…<thou` then `ght>…`) because it buffers the longest trailing
    proper-prefix of any candidate tag across calls.

  * `InbandReasoningTransform` -- a `Transform` (the dev-loop-authorable
    contract) that fixes the NON-streaming path: it reads the cell's
    `reasoning_channel` capability finding, and only when that finding
    is `inband_tags` does it split each message item's `output_text`,
    lifting reasoning spans into a reasoning item and leaving the
    visible answer tag-free.

  * `inband_splitter_for_model` -- the gate the STREAMING chat→Responses
    translator calls. callosum's transform framework deliberately does
    not process streaming responses (byte pass-through + deferred), and
    the only place callosum owns the chat→Responses *stream* translation
    is the gateway translator -- never a byte-passed Responses stream.
    The translator asks this helper for a configured splitter (or None
    when the cell is not an `inband_tags` cell), keeping the gating
    logic and the tag set in one reusable place.

ARCHITECTURE NOTE (the pass-through invariant): this transform only ever
acts where callosum itself translates a chat-completions upstream into
Responses output. Cells served natively over /v1/responses (remote
Codex, and local cells fronted by local LLM gateway's responses-proxy, which
already strips in-band tags upstream) probe as `native`/`none`, so both
the non-stream transform's `applies_to` and the streaming gate return
inert for them. callosum never parses a byte-passed Responses stream to
run this.

TEMPORARY-DEBT (ADR `docs/adr/2026-07-28-substrate-compatibility-contract.md`
section 5, the canonical Class-B `author_temporary_adapter`):

  * Upstream owner: local LLM gateway responses-proxy `_InbandReasoningSplitter`
    (the original this was ported from). Per the ADR's 2026-07-31 accuracy
    amendment, that splitter is NOT in local LLM gateway committed history —
    it lives on the in-flight `the compatibility branch`
    branch (live via editable install, unmerged). If that branch is
    abandoned, this transform becomes the sole owner (not a port) and the
    Class A/B classification of this surface must be rechecked.
  * Close condition: a substrate fronts this cell's responses surface and
    strips in-band reasoning tags on both the stream and non-stream
    paths. When that fires, this transform (and its streaming gate) is
    deleted via the `remove_shim` / `verify_fix` action and `build_default_registry`
    shrinks back to empty — the P5 shim-retirement end-state.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from callosum.transforms.protocol import TransformBase, TransformContext

# The capability dimension that classifies a cell's reasoning channel.
# Kept as a string constant (rather than importing the dimension module)
# so the transform package has no dependency on the capability dimension
# package beyond the on-disk finding contract.
REASONING_CHANNEL_DIMENSION = "reasoning_channel"

# Default candidate tag pairs. The transform reads the OBSERVED tag set
# from the capability finding so it is data-driven; this default is the
# fallback used when a caller constructs a splitter without a finding
# (e.g. the non-streaming one-shot helper) and matches the dimension's
# probe candidates.
DEFAULT_REASONING_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<thought>", "</thought>"),
)

_REASONING_CHANNEL = "reasoning"
_CONTENT_CHANNEL = "content"


def _earliest_tag(buffer: str, tags: Sequence[str]) -> tuple[int, str | None]:
    """Return the (index, tag) of the earliest complete tag occurrence.

    On ties (same index) the longer tag wins -- irrelevant for the
    default set (`<think>` and `<thought>` cannot start at the same
    index) but keeps the scan well-defined for arbitrary configured
    tags. Returns (-1, None) when no complete tag is present.
    """
    best_index = -1
    best_tag: str | None = None
    for tag in tags:
        if not tag:
            continue
        index = buffer.find(tag)
        if index == -1:
            continue
        if best_index == -1 or index < best_index or (index == best_index and len(tag) > len(best_tag or "")):
            best_index = index
            best_tag = tag
    return best_index, best_tag


def _split_partial_tag_suffix(buffer: str, tags: Sequence[str]) -> tuple[str, str]:
    """Split buffer into (emit_now, hold) at the longest trailing partial tag.

    hold is the longest suffix of buffer that is a *proper*
    prefix of some candidate tag (length 1 .. len(tag)-1); a complete tag
    would already have been consumed by :func:`_earliest_tag`. Holding
    that suffix back across calls is what makes the splitter robust to a
    tag straddling an SSE delta boundary.
    """
    hold = 0
    for tag in tags:
        if not tag:
            continue
        max_prefix = min(len(tag) - 1, len(buffer))
        for prefix_len in range(max_prefix, hold, -1):
            if buffer.endswith(tag[:prefix_len]):
                hold = prefix_len
                break
    if hold == 0:
        return buffer, ""
    return buffer[:-hold], buffer[-hold:]


def _append_segment(segments: list[tuple[str, str]], channel: str, text: str) -> None:
    """Append text to segments, coalescing with a same-channel trailing run."""
    if not text:
        return
    if segments and segments[-1][0] == channel:
        segments[-1] = (channel, segments[-1][1] + text)
    else:
        segments.append((channel, text))


class InbandReasoningSplitter:
    """Stateful splitter re-routing in-band reasoning tags out of content.

    Feed it content text incrementally with :meth:`push`; it yields
    (channel, text) segments where channel is "reasoning" or
    "content". A partial opening/closing tag at the end of a chunk is
    buffered and reconsidered on the next :meth:`push`, so a tag split
    across SSE delta boundaries (…<thou then ght>…) is still
    recognized. Call :meth:`flush` at end-of-stream to release any still-
    buffered text on the active channel.

    The same instance handles a whole turn (streaming). For the non-
    streaming path, a fresh instance is fed the full content in one
    push + flush.

    Ported from local LLM gateway's responses-proxy `_InbandReasoningSplitter`
    where the algorithm is proven against real model-a0d5/model-a0g3 traffic.
    """

    def __init__(self, tags: Sequence[tuple[str, str]] | None = None) -> None:
        self._tags = tuple(tags) if tags is not None else DEFAULT_REASONING_TAGS
        self._open_tags = tuple(open_tag for open_tag, _ in self._tags)
        self._close_by_open = {open_tag: close_tag for open_tag, close_tag in self._tags}
        self._in_reasoning = False
        self._current_close: str | None = None
        self._pending = ""

    @property
    def in_reasoning(self) -> bool:
        """True when the splitter is currently inside an open reasoning
        span (no matching close tag seen yet). Lets a streaming caller
        decide which channel a mid-stream item belongs to."""
        return self._in_reasoning

    def push(self, text: str) -> list[tuple[str, str]]:
        if not text:
            return []
        buffer = self._pending + text
        self._pending = ""
        segments: list[tuple[str, str]] = []
        while buffer:
            if not self._in_reasoning:
                index, tag = _earliest_tag(buffer, self._open_tags)
                if tag is not None:
                    _append_segment(segments, _CONTENT_CHANNEL, buffer[:index])
                    buffer = buffer[index + len(tag) :]
                    self._in_reasoning = True
                    self._current_close = self._close_by_open[tag]
                    continue
                emit_now, self._pending = _split_partial_tag_suffix(buffer, self._open_tags)
                _append_segment(segments, _CONTENT_CHANNEL, emit_now)
                break

            close = self._current_close or ""
            index = buffer.find(close)
            if index != -1:
                _append_segment(segments, _REASONING_CHANNEL, buffer[:index])
                buffer = buffer[index + len(close) :]
                self._in_reasoning = False
                self._current_close = None
                continue
            emit_now, self._pending = _split_partial_tag_suffix(buffer, (close,))
            _append_segment(segments, _REASONING_CHANNEL, emit_now)
            break
        return segments

    def flush(self) -> list[tuple[str, str]]:
        if not self._pending:
            return []
        channel = _REASONING_CHANNEL if self._in_reasoning else _CONTENT_CHANNEL
        leftover = self._pending
        self._pending = ""
        return [(channel, leftover)]

    def split_once(self, text: str) -> tuple[str, str]:
        """One-shot convenience for the non-streaming path: split a whole
        content string into (reasoning_text, content_text). Equivalent to
        push(text) + flush() with same-channel coalescing."""
        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        for channel, segment in [*self.push(text), *self.flush()]:
            if channel == _REASONING_CHANNEL:
                reasoning_parts.append(segment)
            else:
                content_parts.append(segment)
        return "".join(reasoning_parts), "".join(content_parts)


# --------------------------------------------------------------------------
# Capability-finding plumbing: read the observed tag set from a profile so
# both the non-stream transform and the streaming gate are data-driven.
# --------------------------------------------------------------------------


def _tags_from_finding_evidence(evidence: Any) -> tuple[tuple[str, str], ...] | None:
    """Pull the observed tag set out of a `reasoning_channel` finding's
    evidence. Returns the configured tags, or None when the finding is
    not an in-band-tags finding (so the caller stays inert).

    The evidence shape (written by the dimension probe) is::

        {"reasoning_channel": "inband_tags",
         "observed_tags": [["<think>", "</think>"], ...]}
    """
    if not isinstance(evidence, dict):
        return None
    if evidence.get("reasoning_channel") != "inband_tags":
        return None
    raw = evidence.get("observed_tags")
    if not isinstance(raw, list):
        return None
    tags: list[tuple[str, str]] = []
    for pair in raw:
        if isinstance(pair, (list, tuple)) and len(pair) == 2:
            open_tag, close_tag = pair[0], pair[1]
            if isinstance(open_tag, str) and isinstance(close_tag, str) and open_tag and close_tag:
                tags.append((open_tag, close_tag))
    if not tags:
        return None
    return tuple(tags)


def _inband_tags_for_ctx(ctx: TransformContext) -> tuple[tuple[str, str], ...] | None:
    """Return the configured in-band tag set for this cell, or None when
    the cell's profile does not classify it as `inband_tags`. This is the
    single gate both the transform's `applies_to` and the response split
    consult, so they can never disagree."""
    profile = ctx.capability_profile
    if profile is None:
        return None
    finding = profile.findings.get(REASONING_CHANNEL_DIMENSION)
    if finding is None or finding.status != "fail":
        return None
    return _tags_from_finding_evidence(finding.evidence)


def inband_splitter_for_model(
    model: str,
    *,
    profile_dir: Any = None,
) -> InbandReasoningSplitter | None:
    """Gate for the streaming chat→Responses translator.

    Loads `model`'s capability profile from disk and returns a freshly-
    configured `InbandReasoningSplitter` (with the observed tag set) when
    the cell classifies as `inband_tags`. Returns None for every other
    case -- `native`, `none`, `unknown`, no profile -- so the translator
    keeps its existing behavior unchanged (the splitter never engages).

    Loading the small profile JSON is the same cheap read the request
    handler already does per request; the translator calls this once at
    stream start.
    """
    # Imported lazily to avoid a transforms→capability import at module
    # load (mirrors how the request handler defers the profile import).
    from callosum.capability.profile import load_profile, profile_path

    try:
        path = profile_path(model, profile_dir)
        if not path.exists():
            return None
        profile = load_profile(model, profile_dir=path.parent)
    except Exception:
        return None
    finding = profile.findings.get(REASONING_CHANNEL_DIMENSION)
    if finding is None or finding.status != "fail":
        return None
    tags = _tags_from_finding_evidence(finding.evidence)
    if tags is None:
        return None
    return InbandReasoningSplitter(tags)


# --------------------------------------------------------------------------
# Non-streaming transform.
# --------------------------------------------------------------------------


def _message_output_text(item: dict[str, Any]) -> str | None:
    """Return the concatenated output_text of a Responses message item,
    or None when the item is not a text-bearing message."""
    if item.get("type") != "message":
        return None
    content = item.get("content")
    if not isinstance(content, list):
        return None
    parts: list[str] = []
    found_text_part = False
    for part in content:
        if isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str):
            parts.append(part["text"])
            found_text_part = True
    if not found_text_part:
        return None
    return "".join(parts)


class InbandReasoningTransform(TransformBase):
    """Non-streaming in-band reasoning re-routing.

    Fires only for cells whose `reasoning_channel` capability finding is
    `inband_tags`. For those cells it walks the Responses output items,
    splits each message item's `output_text` with the observed tag set,
    lifts the reasoning spans into a reasoning item, and rewrites the
    message so the visible answer is tag-free.

    The streaming counterpart lives in the gateway's chat→Responses
    stream translator (see `inband_splitter_for_model`) because
    callosum's transform framework does not process streaming responses.
    """

    @property
    def name(self) -> str:
        return "inband_reasoning"

    def applies_to(self, ctx: TransformContext) -> bool:
        return _inband_tags_for_ctx(ctx) is not None

    def transform_response(self, body: dict[str, Any], ctx: TransformContext) -> dict[str, Any]:
        tags = _inband_tags_for_ctx(ctx)
        if tags is None:
            return body
        output = body.get("output")
        if not isinstance(output, list):
            return body

        new_output: list[dict[str, Any]] = []
        # Reasoning lifted out of content this turn, concatenated in
        # output order so the summary reads coherently.
        lifted_reasoning: list[str] = []
        rebuilt = False
        for item in output:
            if not isinstance(item, dict):
                new_output.append(item)
                continue
            text = _message_output_text(item)
            if text is None:
                new_output.append(item)
                continue
            # A fresh splitter per message item -- a span never straddles
            # two message items in a single response.
            reasoning_text, content_text = InbandReasoningSplitter(tags).split_once(text)
            if not reasoning_text:
                # No in-band span in this message -- leave it untouched so
                # we never rewrite content we didn't change.
                new_output.append(item)
                continue
            rebuilt = True
            lifted_reasoning.append(reasoning_text)
            # Keep the cleaned message only when it still carries visible
            # text; an item that was pure reasoning collapses away (the
            # reasoning item below carries its content).
            if content_text.strip():
                cleaned = dict(item)
                cleaned["content"] = [{"type": "output_text", "text": content_text}]
                new_output.append(cleaned)
        if not rebuilt:
            return body

        reasoning_item = {
            "type": "reasoning",
            "id": f"rs_{body.get('id', 'inband')}",
            "summary": [{"type": "summary_text", "text": "".join(lifted_reasoning)}],
        }
        # Reasoning leads the output array, matching the o1/o3 convention
        # the existing chat→Responses translator already follows.
        result = dict(body)
        result["output"] = [reasoning_item, *new_output]
        return result


# Module-level transform instance discovered by the registry builder.
TRANSFORM = InbandReasoningTransform()
