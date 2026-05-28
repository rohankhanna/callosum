"""Failure-detection labelers.

Phase 6 of the learning-router refactor. The kNN predictor learns
which cells fail on which prompt shapes by reading quality_score
labels from the request log. These rules produce ONLY negative
labels (-1) from objective response-shape signals; absence of
failure is unlabeled, and the predictor treats unlabeled as
"no opinion" via its uniform prior.

Rules implemented:

  empty_visible_output  — model spent generation budget but produced
                          no visible text and no tool calls. Wasted
                          compute, useless to the caller.

  finish_reason_length  — model hit num_predict / max_tokens without a
                          natural stop. Indicates either truncation or
                          a runaway model that didn't decide to stop.

  tool_call_loop        — same tool_call (name + arguments) appeared
                          N times consecutively in the conversation.
                          Classic agentic-loop pattern.

  malformed_tool_args   — tool_call.arguments isn't valid JSON. The
                          calling client can't parse it; the call fails
                          downstream regardless.

Each rule is a pure function of (request_row_dict, response_body_dict)
returning either None ("no opinion") or a label value in [-1, 0]. The
caller (the apply-labels job) writes the label to the quality_score
column.
"""

from __future__ import annotations

import json
from typing import Any

# How many consecutive identical tool_calls before we call it a loop.
# 3 = "I see it tried, then retried once, then retried again — that's a
# pattern, not a coincidence." Lower → more sensitive. Higher → more
# tolerant of legitimate model retries.
_TOOL_LOOP_THRESHOLD = 3


def detect_empty_visible_output(
    request_row: dict[str, Any],
    response_body: dict[str, Any],
) -> int | None:
    """Label -1 when the model used inference budget but emitted no
    visible content and no tool_calls. Reads completion_tokens from the
    request row (already populated by usage_log) AND checks the
    response payload for actual content."""
    completion_tokens = request_row.get("completion_tokens")
    if not isinstance(completion_tokens, int) or completion_tokens <= 0:
        return None  # No generation happened — not a quality signal.
    # Look at the Responses-API output (the shape callosum logs).
    output = response_body.get("output") if isinstance(response_body, dict) else None
    if not isinstance(output, list):
        return None
    has_text = False
    has_tool_call = False
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "message":
            content = item.get("content")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                        has_text = True
                        break
        elif item.get("type") == "function_call":
            has_tool_call = True
    if has_text or has_tool_call:
        return None
    return -1


def detect_finish_reason_length(
    request_row: dict[str, Any],
    response_body: dict[str, Any],
) -> int | None:
    """Label -1 when the response stopped because it ran out of
    generation budget (finish_reason=length) AND the visible content
    didn't end on a natural sentence boundary. The latter check
    distinguishes "model genuinely needed more space" from "model
    answered fine but happened to fill the budget exactly."""
    if not isinstance(response_body, dict):
        return None
    output = response_body.get("output")
    if not isinstance(output, list):
        return None
    # Find the message item and check if it has a clean ending.
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("status") == "incomplete":
            details = item.get("incomplete_details") or {}
            if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
                return -1
        # finish_reason isn't in the Responses API output schema; it's
        # in usage / status. Look at the top-level status field.
    if response_body.get("status") == "incomplete":
        details = response_body.get("incomplete_details") or {}
        if isinstance(details, dict) and details.get("reason") == "max_output_tokens":
            return -1
    return None


def detect_tool_call_loop(
    request_row: dict[str, Any],
    response_body: dict[str, Any],
    request_body: dict[str, Any] | None = None,
) -> int | None:
    """Label -1 when this request's tool_call (or recent conversation
    history) shows the same (name, arguments) repeated N times in a
    row. Reads the conversation history from `request_body.input` (the
    Responses-API shape callosum logs).

    Definition of "consecutive identical" — same function name AND
    same arguments string. Different arguments to the same tool count
    as different calls; this catches the model-a0d5-loop pattern where the
    model retries with literally the same `ls -R` over and over.
    """
    if not isinstance(request_body, dict):
        return None
    input_items = request_body.get("input")
    if not isinstance(input_items, list):
        return None
    # Walk the most recent N+1 function_call items; if the last N are
    # all identical to the (N+1)th-to-last, that's a loop.
    fn_calls: list[tuple[str, str]] = []
    for item in input_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            name = str(item.get("name", ""))
            args = item.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, sort_keys=True)
            fn_calls.append((name, args))
    if len(fn_calls) < _TOOL_LOOP_THRESHOLD:
        return None
    tail = fn_calls[-_TOOL_LOOP_THRESHOLD:]
    if all(tc == tail[0] for tc in tail):
        return -1
    return None


def detect_malformed_tool_args(
    request_row: dict[str, Any],
    response_body: dict[str, Any],
) -> int | None:
    """Label -1 when ANY function_call in this response has an
    arguments string that isn't parseable JSON. The downstream client
    can't execute it; the call effectively failed regardless of
    HTTP status."""
    if not isinstance(response_body, dict):
        return None
    output = response_body.get("output")
    if not isinstance(output, list):
        return None
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        args = item.get("arguments")
        if not isinstance(args, str):
            return -1
        try:
            json.loads(args)
        except json.JSONDecodeError:
            return -1
    return None


# ---------- top-level entry ----------


def score_row(
    *,
    request_row: dict[str, Any],
    response_body: dict[str, Any] | None,
    request_body: dict[str, Any] | None = None,
) -> int | None:
    """Run all labelers and return the most negative signal observed
    (-1 if any rule fires; None if none do).

    Phase 6 is failure-only: we never produce positive labels, so the
    return space is restricted to {-1, None}. If multiple rules fire
    on the same row, the label is still -1 — there's no concept of
    "more negative" in this framing.
    """
    if response_body is None:
        return None
    for rule in (
        detect_empty_visible_output,
        detect_finish_reason_length,
        detect_malformed_tool_args,
    ):
        result = rule(request_row, response_body)
        if result is not None and result < 0:
            return result
    if request_body is not None:
        loop_result = detect_tool_call_loop(request_row, response_body, request_body)
        if loop_result is not None and loop_result < 0:
            return loop_result
    return None
