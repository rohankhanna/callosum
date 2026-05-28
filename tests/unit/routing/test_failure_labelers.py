"""Tests for the failure-detection labelers in Phase 6.

Each rule is a pure function; tests construct request/response/body
dicts directly and assert the rule's verdict.
"""

from __future__ import annotations

from callosum.routing.labeler.failures import (
    detect_empty_visible_output,
    detect_finish_reason_length,
    detect_malformed_tool_args,
    detect_tool_call_loop,
    score_row,
)


def test_empty_output_with_nonzero_tokens_labeled_negative() -> None:
    """Model generated tokens but produced neither text nor tool_calls.
    Wasted compute → -1."""
    req = {"completion_tokens": 17}
    resp = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": ""}]}
        ]
    }
    assert detect_empty_visible_output(req, resp) == -1


def test_empty_output_with_zero_tokens_returns_none() -> None:
    """No generation happened — not a quality signal."""
    req = {"completion_tokens": 0}
    resp = {"output": []}
    assert detect_empty_visible_output(req, resp) is None


def test_substantive_text_returns_none() -> None:
    """Non-empty visible text → not a failure."""
    req = {"completion_tokens": 5}
    resp = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
        ]
    }
    assert detect_empty_visible_output(req, resp) is None


def test_tool_call_only_returns_none() -> None:
    """Tool calls count as useful output too — even with no text."""
    req = {"completion_tokens": 12}
    resp = {
        "output": [
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'}
        ]
    }
    assert detect_empty_visible_output(req, resp) is None


def test_max_output_tokens_finish_labeled_negative() -> None:
    """status=incomplete with reason=max_output_tokens → -1."""
    req = {}
    resp = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    }
    assert detect_finish_reason_length(req, resp) == -1


def test_completed_status_returns_none() -> None:
    req = {}
    resp = {"status": "completed", "output": []}
    assert detect_finish_reason_length(req, resp) is None


def test_malformed_tool_args_labeled_negative() -> None:
    """arguments isn't valid JSON → -1."""
    req = {}
    resp = {
        "output": [
            {"type": "function_call", "name": "shell", "arguments": "{not json"}
        ]
    }
    assert detect_malformed_tool_args(req, resp) == -1


def test_well_formed_tool_args_returns_none() -> None:
    req = {}
    resp = {
        "output": [
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'}
        ]
    }
    assert detect_malformed_tool_args(req, resp) is None


def test_tool_call_loop_detected() -> None:
    """Three identical consecutive tool_calls in the conversation
    history → loop → -1."""
    req = {}
    resp = {}
    request_body = {
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "review"}]},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "output": "files"},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "output": "files"},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'},
        ]
    }
    assert detect_tool_call_loop(req, resp, request_body) == -1


def test_tool_call_loop_not_detected_when_args_differ() -> None:
    """Same tool, different arguments → not a loop (legitimate
    exploration)."""
    req = {}
    resp = {}
    request_body = {
        "input": [
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"ls"}'},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"pwd"}'},
            {"type": "function_call", "name": "shell", "arguments": '{"cmd":"cat README"}'},
        ]
    }
    assert detect_tool_call_loop(req, resp, request_body) is None


def test_score_row_returns_first_negative_signal() -> None:
    """If any rule fires, score_row returns -1. Failure-only label space."""
    req = {"completion_tokens": 17}
    resp = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": ""}]}
        ]
    }
    assert score_row(request_row=req, response_body=resp) == -1


def test_score_row_returns_none_when_no_rule_fires() -> None:
    """All rules pass → None (no opinion). Predictor treats as
    cold-start, falls back to uniform prior."""
    req = {"completion_tokens": 5}
    resp = {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}
        ]
    }
    assert score_row(request_row=req, response_body=resp) is None


def test_score_row_handles_missing_response_body() -> None:
    """Streaming rows may have no captured resp body — no signal."""
    req = {"completion_tokens": 5}
    assert score_row(request_row=req, response_body=None) is None
