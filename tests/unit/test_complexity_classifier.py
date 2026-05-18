"""Tests for the v2 cost router's prompt complexity classifier.

The classifier returns 1 / 2 / 3 buckets matching the same scale the model
itself emits when answering the auto-learning instruction. The numbers
need to be cheap to compute (no network call) and structural — not
semantic — so they can run on the dispatch hot path.
"""

from __future__ import annotations

from codex_proxy.complexity_classifier import classify_prompt_complexity


def test_short_chat_completion_prompt_classifies_as_simple() -> None:
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert classify_prompt_complexity(body) == 1


def test_medium_chat_completion_prompt_classifies_as_moderate() -> None:
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": "abc" * 1000}],
    }
    assert classify_prompt_complexity(body) == 2


def test_huge_chat_completion_prompt_classifies_as_complex() -> None:
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": "abc" * 10_000}],
    }
    assert classify_prompt_complexity(body) == 3


def test_responses_api_input_array_with_text_parts() -> None:
    body = {
        "model": "x",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Tiny question"}],
            }
        ],
    }
    assert classify_prompt_complexity(body) == 1


def test_session_token_count_overrides_body_scan() -> None:
    """When the proxy already knows session_prompt_tokens, prefer it."""
    body = {"model": "x", "messages": [{"role": "user", "content": "hi"}]}
    assert classify_prompt_complexity(body, session_prompt_tokens=10_000) == 3
    assert classify_prompt_complexity(body, session_prompt_tokens=2_000) == 2
    assert classify_prompt_complexity(body, session_prompt_tokens=50) == 1


def test_unknown_shape_defaults_to_moderate() -> None:
    """Empty / surprising body shouldn't bias routing toward the cheapest cell."""
    assert classify_prompt_complexity(None) == 2
    assert classify_prompt_complexity({}) == 2
    assert classify_prompt_complexity({"weird": "shape"}) == 2


def test_instructions_field_counts_toward_size() -> None:
    body = {
        "model": "x",
        "instructions": "a" * 30_000,
        "input": [],
    }
    assert classify_prompt_complexity(body) == 3
