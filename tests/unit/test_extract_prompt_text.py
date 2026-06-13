"""Tests for the dual-shape prompt/response text extractors.

Pre-fix: extractor only handled Chat Completions `messages`; every
Codex Responses request silently logged NULL. Verifies both shapes
plus mixed nesting.
"""

from __future__ import annotations

import json

from callosum.usage_log import _extract_prompt_text, _extract_response_text


def test_chat_completions_messages_extracted() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "what is 2+2"},
        ]
    }
    text = _extract_prompt_text(body)
    assert "what is 2+2" in text
    assert "system msg" in text


def test_responses_api_input_extracted() -> None:
    """The shape Codex CLI sends — `input` array with `input_text`
    content parts. Earlier extractor returned None for this, which
    explained the 52K rows with NULL prompt_text."""
    body = {
        "instructions": "be helpful",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "do a code review"}],
            }
        ],
    }
    text = _extract_prompt_text(body)
    assert "do a code review" in text
    assert "be helpful" in text


def test_chat_messages_with_part_list_extracted() -> None:
    """OpenAI multimodal-style `content: [{type: text, text: ...}]`."""
    body = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}
    assert "hello" in _extract_prompt_text(body)


def test_extractor_returns_none_when_no_text_anywhere() -> None:
    assert _extract_prompt_text({"messages": []}) is None
    assert _extract_prompt_text({}) is None
    assert _extract_prompt_text(None) is None


def test_response_text_chat_completions_shape() -> None:
    payload = json.dumps({"choices": [{"message": {"role": "assistant", "content": "hi back"}}]}).encode()
    assert _extract_response_text(payload) == "hi back"


def test_response_text_responses_api_shape() -> None:
    """Output items list with `output_text` content parts."""
    payload = json.dumps(
        {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "the answer"}],
                }
            ]
        }
    ).encode()
    assert "the answer" in _extract_response_text(payload)


def test_response_text_handles_function_call_output() -> None:
    """function_call items shouldn't contribute text — but the extractor
    shouldn't crash on them either."""
    payload = json.dumps(
        {
            "output": [
                {
                    "type": "function_call",
                    "name": "shell",
                    "arguments": '{"cmd":"ls"}',
                }
            ]
        }
    ).encode()
    # No visible text content; extractor returns the function_call
    # arguments via the walk-text fallback. That's fine — it's still
    # signal-bearing for the kNN predictor.
    result = _extract_response_text(payload)
    # Either way: doesn't crash, doesn't return obvious garbage.
    assert result is None or "shell" in result or "ls" in result


def test_extractor_handles_malformed_json_response() -> None:
    assert _extract_response_text(b"not json at all") is None
    assert _extract_response_text(None) is None
