"""Shared response-shape helpers for the dimension probes.

Both tool_call_shape and tool_call_at_scale do the same response
inspection: walk output items, classify whether structured
function_calls are present, scan message text for tool-call-shaped
JSON, build evidence. The classification is identical between the
two dimensions — only the prompt size differs — so the logic lives
here once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class ResponseClassification:
    """The classification result the dimension probes turn into a
    DimensionFinding. Keeps the shape inspection separate from the
    finding construction so dimensions can decorate it with
    dimension-specific summary / hint text."""

    has_structured_call: bool
    text_json_leak_examples: list[str]
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
            text_json_leak_examples=[],
            text_excerpts=[],
            output_item_types=[],
            function_calls_count=0,
        )
    output = response.get("output") or []
    if not isinstance(output, list):
        return ResponseClassification(
            has_structured_call=False,
            text_json_leak_examples=[],
            text_excerpts=[],
            output_item_types=[],
            function_calls_count=0,
        )

    has_structured = False
    text_json_leaks: list[str] = []
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
            if (
                isinstance(item.get("name"), str)
                and item.get("name")
                and isinstance(item.get("arguments"), str)
            ):
                has_structured = True
        elif t == "message":
            for text in _text_parts(item):
                text_excerpts.append(text[:300])
                if _looks_like_tool_call_json(text):
                    text_json_leaks.append(text[:500])
    return ResponseClassification(
        has_structured_call=has_structured,
        text_json_leak_examples=text_json_leaks,
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
