"""Tests for the tool-call verification probe.

Covers the contract: a cell passes only when it returns a Responses-API
response carrying at least one structured `function_call` output item
AND no message text that parses as a tool-call-shaped JSON object.

Quirks the probe MUST catch:
  * tool calls emitted as JSON text inside message content (model-a0e5 quirk)
  * the dual-emit failure mode where the model produces BOTH a structured
    function_call AND duplicates it as JSON in message text
  * empty output (model refused or produced nothing)
  * function_call items with malformed/missing required fields
  * backend errors → probe fails closed
"""

from __future__ import annotations

from typing import Any

import pytest

from callosum.routing.probe import (
    build_probe_body,
    probe_supports_tools,
    response_has_structured_tool_call,
)


def test_probe_body_targets_named_model() -> None:
    body = build_probe_body("model-a0a9")
    assert body["model"] == "model-a0a9"
    # Tools list must be present and contain the hardened multi-tool
    # set so the assertion paths in response_has_structured_tool_call
    # match real-traffic shape rather than a single-tool probe.
    assert isinstance(body.get("tools"), list)
    tool_names = {t["name"] for t in body["tools"]}
    # At least 6 tools — the hardening that catches the model-a0d5-31b
    # false-positive case (single-tool probes degenerate into "always
    # call that tool" which is easier than real traffic).
    assert len(tool_names) >= 6
    assert "exec_command" in tool_names


def test_probe_body_is_fresh_copy_per_call() -> None:
    """If callers mutate the body (backends often do — apply_params etc),
    the mutation must not bleed into subsequent probes for a different
    cell. Verifies we don't accidentally hand out a shared mutable dict."""
    body_a = build_probe_body("model-a")
    body_a["model"] = "mutated"
    body_b = build_probe_body("model-b")
    assert body_b["model"] == "model-b"
    # The lists in body_b must be independent — mutating body_a's input
    # list shouldn't affect body_b's.
    body_a["input"].append({"type": "junk"})
    assert {"type": "junk"} not in body_b["input"]


def test_probe_body_is_realistic_size() -> None:
    """The hardened probe targets ~5KB of context (instructions + user
    message + tool definitions) so it stresses tool-disambiguation the
    way real Codex CLI traffic does. The earlier ~200-byte probe
    produced false positives — model-a0c8 passed the simple form and
    failed real traffic. Asserting a lower bound on the realistic
    size catches regressions where someone simplifies the probe back."""
    body = build_probe_body("test")
    import json as _json
    total_chars = len(_json.dumps(body))
    assert total_chars > 3000, (
        f"probe body is only {total_chars} chars — hardening regressed."
    )


# ---------- response_has_structured_tool_call --------------------------


def test_structured_tool_call_recognized_in_output() -> None:
    """The happy path: a properly-shaped function_call in output[] is
    recognized as the verifying signal."""
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "probe_echo",
                "arguments": '{"payload":"capability check"}',
                "call_id": "fc_001",
            }
        ]
    }
    assert response_has_structured_tool_call(response) is True


def test_text_as_json_in_content_is_rejected() -> None:
    """The model-a0e5 quirk: the model emits the tool call as JSON text
    inside a message item's content. Looks like a tool call to a
    naive reader, but doesn't follow the OpenAI shape — must fail
    the probe so callosum doesn't route real tool-using traffic to
    this cell."""
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"type":"function_call","name":"probe_echo","arguments":"{\\"payload\\":\\"capability check\\"}"}',
                    }
                ],
            }
        ]
    }
    assert response_has_structured_tool_call(response) is False


def test_dual_emit_failure_mode_rejected() -> None:
    """The harder case the previous probe missed: the model emits BOTH
    a structured function_call AND duplicates it as JSON in a message
    text part. Codex CLI's parser will handle the structured call but
    render the JSON-shaped text as visible terminal junk. The probe
    must fail the cell on this pattern even though one half of the
    output is correct.

    This is the case where model-a0c8 passed the earlier simpler probe
    (the structured call WAS present) but real Codex CLI traffic still
    showed JSON-as-text in the user's terminal.
    """
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "exec_command",
                "arguments": '{"cmd":"git status"}',
                "call_id": "fc_001",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": '{"name":"exec_command","arguments":{"cmd":"git status"}}',
                    }
                ],
            },
        ]
    }
    assert response_has_structured_tool_call(response) is False


def test_function_call_plus_real_explanatory_text_passes() -> None:
    """The healthy pattern: model emits a structured function_call AND
    a brief explanatory text message. Text content that isn't JSON-shaped
    must NOT trigger the dual-emit rejection — otherwise we'd reject
    every well-behaved model that explains what it's about to do."""
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "git_status",
                "arguments": "{}",
                "call_id": "fc_001",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Checking the current branch and uncommitted changes.",
                    }
                ],
            },
        ]
    }
    assert response_has_structured_tool_call(response) is True


def test_function_call_plus_json_text_unrelated_to_tool_call_passes() -> None:
    """Edge case: the model's explanatory text happens to contain a
    JSON OBJECT, but it doesn't have the `name`+`arguments` shape of a
    tool call. e.g. the model is explaining a config payload. Must NOT
    trigger the dual-emit rejection — the rejection condition is
    specifically tool-call-shaped JSON, not all JSON."""
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "read_file",
                "arguments": '{"path":"config.json"}',
                "call_id": "fc_001",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": 'The expected config shape is {"port": 8765, "host": "127.0.0.1"}.',
                    }
                ],
            },
        ]
    }
    assert response_has_structured_tool_call(response) is True


def test_empty_output_rejected() -> None:
    """The model produced no output items at all (refused or empty).
    The cell can't be trusted with tool-using requests."""
    assert response_has_structured_tool_call({"output": []}) is False


def test_text_only_output_rejected() -> None:
    """The model produced only a text message with no tool call. The
    most common refusal/misunderstanding mode."""
    response = {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "I'd be happy to help!"}
                ],
            }
        ]
    }
    assert response_has_structured_tool_call(response) is False


def test_function_call_missing_name_rejected() -> None:
    """A function_call item without a name field is malformed and
    can't be dispatched. Fail closed."""
    response = {
        "output": [
            {
                "type": "function_call",
                "arguments": '{"payload":"x"}',
            }
        ]
    }
    assert response_has_structured_tool_call(response) is False


def test_function_call_missing_arguments_rejected() -> None:
    """arguments is required by the OpenAI shape (even if empty
    string). Missing → malformed → probe fails."""
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "probe_echo",
            }
        ]
    }
    assert response_has_structured_tool_call(response) is False


def test_function_call_arguments_as_dict_rejected() -> None:
    """Some backends emit arguments as a dict instead of a JSON string.
    Strictly per OpenAI's shape this is malformed — Codex CLI's parser
    will reject it. Fail the probe so we don't route to those cells."""
    response = {
        "output": [
            {
                "type": "function_call",
                "name": "probe_echo",
                "arguments": {"payload": "capability check"},
            }
        ]
    }
    assert response_has_structured_tool_call(response) is False


def test_multiple_items_one_valid_function_call_passes() -> None:
    """A response can include a reasoning item, a message item, AND a
    function_call. As long as ONE valid function_call exists, the cell
    is tools-capable."""
    response = {
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Calling tool now."}],
            },
            {
                "type": "function_call",
                "name": "probe_echo",
                "arguments": '{"payload":"x"}',
            },
        ]
    }
    assert response_has_structured_tool_call(response) is True


def test_non_dict_response_rejected() -> None:
    """Defensive: non-dict responses (None, str, error shapes) shouldn't
    crash the probe — they're treated as failures."""
    assert response_has_structured_tool_call(None) is False  # type: ignore[arg-type]
    assert response_has_structured_tool_call("oops") is False  # type: ignore[arg-type]
    assert response_has_structured_tool_call({"error": "rate_limited"}) is False


def test_output_not_a_list_rejected() -> None:
    """A response with output that's a dict or string instead of a list
    is malformed."""
    assert response_has_structured_tool_call({"output": "junk"}) is False
    assert response_has_structured_tool_call({"output": {"type": "function_call"}}) is False


# ---------- probe_supports_tools (integration) -------------------------


@pytest.mark.asyncio
async def test_probe_returns_true_when_backend_emits_structured_call() -> None:
    """End-to-end: pass a fake call_responses that returns a response
    with a function_call; probe returns True."""
    captured_body: dict[str, Any] = {}

    async def fake_call(body: dict[str, Any]) -> dict[str, Any]:
        captured_body.update(body)
        return {
            "output": [
                {
                    "type": "function_call",
                    "name": "probe_echo",
                    "arguments": '{"payload":"capability check"}',
                }
            ]
        }

    result = await probe_supports_tools(
        model="some-local-model", call_responses=fake_call
    )
    assert result is True
    # Sanity: the probe DID hand the right model to the backend.
    assert captured_body["model"] == "some-local-model"


@pytest.mark.asyncio
async def test_probe_returns_false_when_backend_emits_text_only() -> None:
    """End-to-end: a cell that refuses to use tools fails the probe."""

    async def fake_call(body: dict[str, Any]) -> dict[str, Any]:
        return {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Sorry, I cannot use tools.",
                        }
                    ],
                }
            ]
        }

    assert (
        await probe_supports_tools(model="x", call_responses=fake_call)
        is False
    )


@pytest.mark.asyncio
async def test_probe_returns_false_when_backend_raises() -> None:
    """Network errors, timeouts, backend rejections — any exception
    means we can't verify the cell, which we treat as failure. Refusing
    to route is the safe default; we can re-probe later."""

    async def fake_call(body: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("upstream went away mid-probe")

    assert (
        await probe_supports_tools(model="x", call_responses=fake_call)
        is False
    )
