"""Dimension: tool_call_at_scale.

Same tool-call probe as `test_tool_call_shape`, but at realistic
Codex CLI context size (~80K tokens). This is the test that catches
cells like model-a0c8 that pass the small-prompt probe but emit
text-as-JSON when given a real-traffic-shaped input.

Why a separate test rather than just making `tool_call_shape` larger:

  * The small-prompt test is fast (<10s typically). Useful as a
    quick sanity check during model admission.
  * The realistic-size test is slow (60-300s for 31B-class cells on
    GPU). Useful only when the smaller test already passed — no point
    burning minutes probing a cell that fails at small scale.

Together they let the operator (or ) make a layered
admission decision: cells must pass BOTH dimensions to be considered
tools-capable for real Codex traffic.

Adapter hints from this dimension specifically describe the
context-size at which behavior breaks, which is the load-bearing
fact for any future "use small-context-only" adapter.
"""

from __future__ import annotations

import json

import httpx
import pytest

from tests.model_capability.codex_request_shape import (
    tool_call_with_context_body,
)
from tests.model_capability.profile import (
    DimensionFinding,
    load_profile,
    save_profile,
)

# Target prompt size for the at-scale probe. 80,000 characters is
# ~22K tokens at typical chars/token ratios — substantial enough to
# expose context-sensitive failures but well under the 200K-token
# real-Codex extremes (each character of padding costs probe time
# linearly in the model's tokenizer). Bump if 22K turns out to be
# below the cell's failure threshold.
_PROMPT_CHARS = 80_000


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
def test_tool_call_at_scale_per_cell(
    callosum_client: httpx.Client, cells_to_probe: list[str]
) -> None:
    """Skip cells that haven't already passed the small-prompt
    tool_call_shape test. Running the at-scale probe against a cell
    that fails at small scale is a waste — we'd just confirm what we
    already know, after burning 60-300s of GPU time.

    This dependency between dimensions is encoded as a check on the
    persisted profile rather than as a pytest dependency because:
      * pytest's dependency plugins don't compose well with the
        parametrized-over-cells shape used here.
      * Profiles are persistent, so a previous-day pass of the small
        test still gates today's scale test correctly.
    """
    for cell in cells_to_probe:
        profile = load_profile(cell)
        small_finding = profile.findings.get("tool_call_shape")
        if small_finding is None or small_finding.status != "pass":
            finding = DimensionFinding(
                dimension="tool_call_at_scale",
                status="skipped",
                summary=(
                    "skipped — small-prompt tool_call_shape did not pass; "
                    "no point burning minutes probing at scale"
                ),
                evidence={
                    "small_test_status": (
                        small_finding.status if small_finding else "not_run"
                    ),
                },
            )
            _persist(cell, finding)
            continue

        body = tool_call_with_context_body(target_user_chars=_PROMPT_CHARS)
        r = callosum_client.post(
            "/admin/cell-call",
            json={"model": cell, "body": body},
        )
        if r.status_code != 200:
            finding = DimensionFinding(
                dimension="tool_call_at_scale",
                status="error",
                summary=f"admin/cell-call HTTP {r.status_code}",
                evidence={"http_status": r.status_code, "body": r.text[:500]},
                adapter_hint="transport failure — investigate upstream.",
            )
            _persist(cell, finding)
            continue

        outcome = r.json()
        served_by = outcome.get("served_by")
        latency_ms = outcome.get("latency_ms")
        if outcome.get("status") != "ok":
            finding = DimensionFinding(
                dimension="tool_call_at_scale",
                status="error",
                summary=f"cell-call failed: {outcome.get('error')}",
                evidence={"upstream_error": outcome.get("error")},
                adapter_hint=(
                    "context-size error or transport failure. If the "
                    "error mentions context length, the model can't "
                    "physically serve realistic Codex traffic."
                ),
                latency_ms=latency_ms,
            )
            _persist(cell, finding, served_by=served_by)
            continue

        response_body = outcome.get("response") or {}
        output = response_body.get("output") or []
        has_structured = False
        text_json_leaks: list[str] = []
        text_excerpts: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            t = item.get("type")
            if t == "function_call":
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

        if has_structured and not text_json_leaks:
            finding = DimensionFinding(
                dimension="tool_call_at_scale",
                status="pass",
                summary=(
                    "structured function_call emitted at "
                    f"~{_PROMPT_CHARS} char prompt; no JSON-text leak"
                ),
                evidence={
                    "prompt_chars": _PROMPT_CHARS,
                    "text_excerpts": text_excerpts[:3],
                },
                latency_ms=latency_ms,
            )
        elif text_json_leaks:
            finding = DimensionFinding(
                dimension="tool_call_at_scale",
                status="fail",
                summary=(
                    "model emits tool-call-shaped JSON in message text "
                    f"at ~{_PROMPT_CHARS}-char prompt size, even though "
                    "the small-prompt probe passed. Real Codex CLI "
                    "traffic will hit this failure mode."
                ),
                evidence={
                    "prompt_chars": _PROMPT_CHARS,
                    "text_json_leak_examples": text_json_leaks[:2],
                    "structured_call_also_present": has_structured,
                },
                adapter_hint=(
                    "model breaks down at realistic context size. "
                    "Two adapter options: "
                    "(a) parse tool-call-shaped JSON from message text "
                    "and lift to structured function_call items, OR "
                    "(b) restrict this cell to small-context routing "
                    "only (denied for sessions whose accumulated context "
                    "exceeds N tokens). (b) is simpler; (a) is more "
                    "general but requires careful escaping logic."
                ),
                latency_ms=latency_ms,
            )
        else:
            finding = DimensionFinding(
                dimension="tool_call_at_scale",
                status="fail",
                summary=(
                    "no structured tool_call AND no JSON-text at scale. "
                    "Model produced text-only or empty output despite "
                    "passing the small-prompt probe."
                ),
                evidence={
                    "prompt_chars": _PROMPT_CHARS,
                    "text_excerpts": text_excerpts[:3],
                },
                adapter_hint=(
                    "context-degradation: model loses tool-calling "
                    "competence as prompt grows. Restrict this cell to "
                    "small-context routing only."
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
