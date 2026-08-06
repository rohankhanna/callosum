"""Shared response-shape helpers for the dimension probes.

Both tool_call_shape and tool_call_at_scale do the same response
inspection: walk output items, classify whether structured
function_calls are present, scan message text for tool-call-shaped
text, build evidence. The classification is identical between the
two dimensions — only the prompt size differs — so the logic lives
here once.

A "text tool-call leak" is any message-text content that IS a tool
call the model emitted as text instead of as a structured
function_call output item — i.e. the substrate (local LLM gateway /
LiteLLM / the responses-proxy) failed to translate the model's
text-format tool call into a structured call. Two canonical text
formats are recognized (both are well-documented Hermes-family
renderings; vLLM ships a hermes_tool_parser for them, and LiteLLM
has a long tail of parsing bugs against them — the substrate owns
translation, callosum only classifies):

  * JSON object — {"name": "...", "arguments": ...} (the model-a0e5
    text-as-JSON quirk).
  * Hermes tag format — <function=NAME>...</function> or a
    <|tool_call|> /  ...  block wrapping a tool-call
    JSON object (model-a0d4 and other Hermes-template models).

Recognizing the tag formats matters because a model emitting them is
clearly *attempting* a tool call, not refusing — bucketing it as
Class-B (substrate-owned, AUTHOR_TEMPORARY_ADAPTER, retires when
the substrate translates) instead of QUARANTINE_CELL (sticky
false-negative, no path back into tool routing) is the correct
attribution and avoids the exact sticky-false-negative the
cell_capabilities docstring warns against.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ResponseClassification:
    """The classification result the dimension probes turn into a
    DimensionFinding. Keeps the shape inspection separate from the
    finding construction so dimensions can decorate it with
    dimension-specific summary / hint text.

    text_tool_call_leak_examples is format-agnostic: it collects
    message-text snippets that look like a tool call the model emitted
    as text (JSON object OR Hermes tag format), regardless of which
    text format was used. The snippet itself shows the format.
    """

    has_structured_call: bool
    text_tool_call_leak_examples: list[str]
    text_excerpts: list[str]
    output_item_types: list[str]
    function_calls_count: int


def classify_response(response: dict[str, Any] | None) -> ResponseClassification:
    """Inspect an upstream response and produce a structured
    classification ready to turn into a DimensionFinding.

    Pass-through-safe: handles None / non-dict / missing-output
    cases gracefully (every field empty / False).
    """
    if not isinstance(response, dict):
        return ResponseClassification(
            has_structured_call=False,
            text_tool_call_leak_examples=[],
            text_excerpts=[],
            output_item_types=[],
            function_calls_count=0,
        )
    output = response.get("output") or []
    if not isinstance(output, list):
        return ResponseClassification(
            has_structured_call=False,
            text_tool_call_leak_examples=[],
            text_excerpts=[],
            output_item_types=[],
            function_calls_count=0,
        )

    has_structured = False
    text_tool_call_leaks: list[str] = []
    text_excerpts: list[str] = []
    types: list[str] = []
    function_calls_count = 0
    for item in output:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if isinstance(t, str):
            types.append(t)
        if t == "function_call":
            function_calls_count += 1
            if isinstance(item.get("name"), str) and item.get("name") and isinstance(item.get("arguments"), str):
                has_structured = True
        elif t == "message":
            for text in _text_parts(item):
                text_excerpts.append(text[:300])
                if _looks_like_tool_call_text(text):
                    text_tool_call_leaks.append(text[:500])
    return ResponseClassification(
        has_structured_call=has_structured,
        text_tool_call_leak_examples=text_tool_call_leaks,
        text_excerpts=text_excerpts,
        output_item_types=types,
        function_calls_count=function_calls_count,
    )


def _text_parts(message_item: dict[str, Any]) -> list[str]:
    content = message_item.get("content")
    if not isinstance(content, list):
        return []
    out: list[str] = []
    for part in content:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                out.append(text)
    return out


def _looks_like_tool_call_text(text: str) -> bool:
    """Return True when text looks like a tool call the model
    emitted as message text instead of as a structured function_call
    output item. Recognizes two canonical Hermes-family text tool-call
    formats in addition to the JSON-object shape:

      * JSON object — {"name": "...", "arguments": ...}.
      * Hermes tag format — <function=NAME>...</function> or a
        <|tool_call|> /  ...  block wrapping a tool-call
        JSON object.

    A model emitting either is *attempting* a tool call as text — the
    substrate failed to translate it — so the dimension buckets it as a
    Class-B text-leak, not a refusal.
    """
    return _looks_like_tool_call_json(text) or _looks_like_tool_call_hermes(text)


def _looks_like_tool_call_json(text: str) -> bool:
    """Strict-parse a string and return True when it parses as a
    JSON object matching the OpenAI tool-call shape (name + arguments).
    Cheap pre-check on `startswith("{")` keeps the common non-JSON
    case fast.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(parsed, dict):
        return False
    return (
        isinstance(parsed.get("name"), str)
        and bool(parsed["name"])
        and isinstance(parsed.get("arguments"), (str, dict))
    )


# <function=NAME>...</function> (observed from model-a0d4 via an
# un-translating responses-proxy). NAME is constrained to a tool-name-ish
# token (letters/digits/_/-/.) so ordinary prose with the literal
# substring "function=" does not trip it.
_HERMES_FUNCTION_TAG = re.compile(
    r"<function=[A-Za-z0-9_.\-]+\s*>.*?</function\s*>",
    re.DOTALL,
)
# <|tool_call|> marker (some Hermes-template models) optionally
# followed by a tool-call-shaped JSON object.
_HERMES_TOOL_CALL_MARKER = re.compile(r"<\|\s*tool_call\s*\|>")
#  ...  wrapping a tool-call-shaped JSON object (vLLM's
# canonical Hermes format — the special tokens are sometimes emitted
# verbatim as text when the runtime doesn't strip them).
_HERMES_TOOL_TOKEN_BLOCK = re.compile(
    r"<\s*tool_call\s*>(?P<body>.*?)<\s*/\s*tool_call\s*>",
    re.DOTALL,
)


def _looks_like_tool_call_hermes(text: str) -> bool:
    """Return True when text looks like a Hermes-family tag-format
    tool call: <function=NAME>...</function>, a <|tool_call|>
    marker, or a  ...  block wrapping a tool-call-shaped JSON
    object. These are well-documented model tool-call renderings (vLLM
    ships a hermes_tool_parser for them); when they reach callosum
    as message text it means the substrate did not translate them into
    structured function_call items.
    """
    if _HERMES_FUNCTION_TAG.search(text):
        return True
    if _HERMES_TOOL_CALL_MARKER.search(text):
        return True
    block = _HERMES_TOOL_TOKEN_BLOCK.search(text)
    return block is not None and _looks_like_tool_call_json(block.group("body"))
