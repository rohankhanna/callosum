"""Dimension: tool_call_shape.

Sends a small Codex-shape tool-using request to each candidate cell
and records whether the cell emits a structured `function_call` item
in `output[]`. This is the baseline tool-call competence test — a
cell that fails this can't be used for ANY tool-using traffic.

What "pass" means: response.output[] contains at least one item with
type=="function_call", a non-empty `name`, and a string `arguments`,
AND no message text in the response parses as a tool-call-shaped
JSON object.

What "fail" means: any deviation from the above. The `adapter_hint`
field tells future readers what an adapter would need to do:

  * text-as-JSON only → adapter parses message content for
    tool-call-shaped JSON and lifts it to structured tool_calls.
  * empty output → no adapter feasible; cell is unusable for tools.
  * dual-emit (structured + JSON-text leak) → adapter strips the
    text-JSON duplicate before forwarding to Codex CLI.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.model_capability.codex_request_shape import tool_call_simple_body
from tests.model_capability.profile import (
    DimensionFinding,
    load_profile,
    save_profile,
)


def _text_parts(message_item: dict) -> list[str]:
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


@pytest.mark.model_probe
def test_tool_call_shape_per_cell(
    callosum_client: httpx.Client, cells_to_probe: list[str]
) -> None:
    """One test invocation iterates every advertised cell. Pytest
    sees a single test result but the capability profile JSON per
    model captures the per-cell finding. This shape is deliberate —
    if we parametrized over cells, a single bad cell would mark the
    whole test "failed" and obscure the per-cell findings."""
    for cell in cells_to_probe:
        body = tool_call_simple_body()
        r = callosum_client.post(
            "/admin/cell-call",
            json={"model": cell, "body": body},
        )
        if r.status_code != 200:
            finding = DimensionFinding(
                dimension="tool_call_shape",
                status="error",
                summary=f"admin/cell-call returned HTTP {r.status_code}",
                evidence={"http_status": r.status_code, "body": r.text[:500]},
                adapter_hint=(
                    "transport failure — not a model quirk. Investigate "
                    "callosum's backend health for this cell."
                ),
            )
            _persist(cell, finding)
            continue

        outcome = r.json()
        served_by = outcome.get("served_by")
        latency_ms = outcome.get("latency_ms")
        if outcome.get("status") != "ok":
            finding = DimensionFinding(
                dimension="tool_call_shape",
                status="error",
                summary=f"cell-call failed: {outcome.get('error')}",
                evidence={"upstream_error": outcome.get("error")},
                adapter_hint=(
                    "transport/backend failure — not a model quirk. "
                    "Investigate upstream availability."
                ),
                latency_ms=latency_ms,
            )
            _persist(cell, finding, served_by=served_by)
            continue

        response_body = outcome.get("response") or {}
        output = response_body.get("output") or []
        has_structured = False
        text_json_leak_examples: list[str] = []
        text_excerpts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "function_call":
                name = item.get("name")
                args = item.get("arguments")
                if (
                    isinstance(name, str) and name
                    and isinstance(args, str)
                ):
                    has_structured = True
            elif t == "message":
                for text in _text_parts(item):
                    text_excerpts.append(text[:300])
                    if _looks_like_tool_call_json(text):
                        text_json_leak_examples.append(text[:500])

        if has_structured and not text_json_leak_examples:
            finding = DimensionFinding(
                dimension="tool_call_shape",
                status="pass",
                summary=(
                    "structured function_call emitted; no tool-call-JSON "
                    "leak in message text"
                ),
                evidence={
                    "function_calls_count": sum(
                        1 for it in output
                        if isinstance(it, dict) and it.get("type") == "function_call"
                    ),
                    "text_excerpts": text_excerpts[:3],
                },
                latency_ms=latency_ms,
            )
        elif has_structured and text_json_leak_examples:
            finding = DimensionFinding(
                dimension="tool_call_shape",
                status="fail",
                summary=(
                    "dual-emit: structured function_call AND duplicate "
                    "tool-call-shaped JSON in message text. Codex CLI's "
                    "parser will render the text JSON as visible junk."
                ),
                evidence={
                    "text_json_leak_examples": text_json_leak_examples[:2],
                    "text_excerpts": text_excerpts[:3],
                },
                adapter_hint=(
                    "adapter must strip message-text content matching "
                    'shape {"name": str, "arguments": str|object} before '
                    "forwarding to the client. The structured tool_calls "
                    "are usable as-is."
                ),
                latency_ms=latency_ms,
            )
        elif not has_structured and text_json_leak_examples:
            finding = DimensionFinding(
                dimension="tool_call_shape",
                status="fail",
                summary=(
                    "text-as-JSON only: model emits tool calls as JSON "
                    "in message text content, no structured function_call "
                    "items. The model-a0e5 quirk class."
                ),
                evidence={
                    "text_json_leak_examples": text_json_leak_examples[:2],
                },
                adapter_hint=(
                    "adapter must parse message-text content matching "
                    'shape {"name": str, "arguments": str|object} and '
                    "lift each match into a structured function_call "
                    "item in output[]. Strip the original text. Then "
                    "deliver the rewritten response to the client."
                ),
                latency_ms=latency_ms,
            )
        else:
            finding = DimensionFinding(
                dimension="tool_call_shape",
                status="fail",
                summary=(
                    "no structured function_call AND no tool-call-shaped "
                    "JSON in text. Model refused to call a tool or "
                    "produced an empty/text-only response."
                ),
                evidence={
                    "text_excerpts": text_excerpts[:3],
                    "output_item_types": [
                        it.get("type")
                        for it in output
                        if isinstance(it, dict)
                    ],
                },
                adapter_hint=(
                    "no adapter feasible — this model can't be coerced "
                    "into tool use through the current prompt. Either "
                    "the system prompt needs major rework (low confidence "
                    "this would work) or the cell should be denied for "
                    "tool-using traffic."
                ),
                latency_ms=latency_ms,
            )

        _persist(cell, finding, served_by=served_by)


def _persist(
    cell: str, finding: DimensionFinding, *, served_by: str | None = None
) -> None:
    profile = load_profile(cell)
    if served_by is not None:
        profile.backend_id = served_by
    profile.upsert(finding)
    save_profile(profile)
