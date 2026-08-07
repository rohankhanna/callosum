"""Unit tests for capability/dimensions/_shape_utils.classify_response.

Covers:
  * structured function_call detection (valid + malformed).
  * JSON-object text tool-call leaks (and that non-tool JSON does not
    leak).
  * Hermes tag-format text tool-call leaks: the function=NAME block,
    the tool_call pipe-marker form, and the tool_call special-token
    block wrapping tool-call JSON (and that a block wrapping non-JSON
    prose does not leak).
  * ordinary prose containing "function=" not flagging as a leak.
  * dual-emit (structured call + text leak), JSON and Hermes.
  * pass-through safety for None / non-dict / missing / non-list output.
  * evidence fields (text_excerpts, output_item_types).

All Hermes tag glyphs are built via chr() so this source file contains
no literal angle-bracket tag sequences.
"""

from __future__ import annotations

import json
from typing import Any

from callosum.capability.dimensions._shape_utils import classify_response

# Angle-bracket / pipe glyphs built via chr() so no literal tag sequences
# appear in this source.
LT, GT, PIPE = chr(60), chr(62), chr(124)


def _msg(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _function_call(
    name: str = "exec_command",
    arguments: str | None = None,
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "name": name,
        "arguments": arguments if arguments is not None else '{"cmd":"ls"}',
        "call_id": "fc_1",
    }


def _resp(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"output": items}


def _func_tag(body: str, name: str = "exec_command") -> str:
    """The function=NAME block form wrapping `body`."""
    return LT + "function=" + name + GT + body + LT + "/function" + GT


def _tool_call_block(body: str) -> str:
    """The tool_call special-token block form wrapping `body`."""
    return LT + "tool_call" + GT + body + LT + "/tool_call" + GT


def _tool_call_pipe_marker() -> str:
    """The tool_call pipe-marker form."""
    return LT + PIPE + "tool_call" + PIPE + GT


# ---------- structured-call detection --------------------------------------


def test_structured_call_no_leak() -> None:
    cls = classify_response(_resp([_function_call(), _msg("done")]))
    assert cls.has_structured_call is True
    assert cls.function_calls_count == 1
    assert cls.text_tool_call_leak_examples == []


def test_malformed_function_call_not_structured() -> None:
    # missing arguments -> not a valid structured call
    cls = classify_response(_resp([{"type": "function_call", "name": "x"}]))
    assert cls.has_structured_call is False
    # still counted as a function_call item, just not a valid one
    assert cls.function_calls_count == 1


# ---------- JSON-object text leaks -----------------------------------------


def test_json_text_leak_detected() -> None:
    leak = json.dumps({"name": "exec_command", "arguments": '{"cmd":"ls"}'})
    cls = classify_response(_resp([_msg(leak)]))
    assert cls.has_structured_call is False
    assert cls.text_tool_call_leak_examples == [leak[:500]]


def test_non_tool_json_not_a_leak() -> None:
    # config-shape JSON: no name/arguments keys -> not a tool call
    text = '{"port": 8765, "host": "127.0.0.1"}'
    cls = classify_response(_resp([_msg(text)]))
    assert cls.text_tool_call_leak_examples == []


def test_arguments_as_object_is_leak() -> None:
    leak = '{"name": "exec_command", "arguments": {"cmd": "ls"}}'
    cls = classify_response(_resp([_msg(leak)]))
    assert cls.text_tool_call_leak_examples == [leak]


# ---------- Hermes tag-format text leaks -----------------------------------


def test_hermes_function_tag_leak_detected() -> None:
    tag = _func_tag('{"name":"exec_command","arguments":"ls"}')
    cls = classify_response(_resp([_msg(tag)]))
    assert cls.has_structured_call is False
    assert cls.text_tool_call_leak_examples == [tag[:500]]


def test_hermes_tool_call_pipe_marker_leak_detected() -> None:
    marker = _tool_call_pipe_marker()
    cls = classify_response(_resp([_msg(marker + ' {"name":"x","arguments":"y"}')]))
    assert cls.text_tool_call_leak_examples == [(marker + ' {"name":"x","arguments":"y"}')[:500]]


def test_hermes_tool_call_block_with_json_leak_detected() -> None:
    body = '{"name":"exec_command","arguments":"ls"}'
    block = _tool_call_block(body)
    cls = classify_response(_resp([_msg(block)]))
    assert cls.text_tool_call_leak_examples == [block[:500]]


def test_hermes_tool_call_block_without_json_not_leak() -> None:
    # tool_call block whose body is plain prose, not tool-call JSON
    block = _tool_call_block("thinking about which tool to use")
    cls = classify_response(_resp([_msg(block)]))
    assert cls.text_tool_call_leak_examples == []


def test_prose_with_function_substring_not_leak() -> None:
    # ordinary prose mentioning "function=" has no leading angle bracket,
    # so the constrained NAME regex must not trip.
    text = "The model's function= dispatch path is configurable."
    cls = classify_response(_resp([_msg(text)]))
    assert cls.text_tool_call_leak_examples == []


# ---------- dual-emit ------------------------------------------------------


def test_dual_emit_structured_and_json_leak() -> None:
    leak = json.dumps({"name": "exec_command", "arguments": '{"cmd":"ls"}'})
    cls = classify_response(_resp([_function_call(), _msg(leak)]))
    assert cls.has_structured_call is True
    assert cls.text_tool_call_leak_examples == [leak[:500]]


def test_dual_emit_structured_and_hermes_leak() -> None:
    tag = _func_tag('{"name":"exec_command","arguments":"ls"}')
    cls = classify_response(_resp([_function_call(), _msg(tag)]))
    assert cls.has_structured_call is True
    assert cls.text_tool_call_leak_examples == [tag[:500]]


# ---------- pass-through safety --------------------------------------------


def test_none_response() -> None:
    cls = classify_response(None)
    assert cls.has_structured_call is False
    assert cls.text_tool_call_leak_examples == []
    assert cls.function_calls_count == 0


def test_non_dict_response() -> None:
    cls = classify_response("oops")  # type: ignore[arg-type]
    assert cls.has_structured_call is False
    assert cls.text_tool_call_leak_examples == []


def test_missing_output() -> None:
    cls = classify_response({"status": "completed"})
    assert cls.has_structured_call is False
    assert cls.text_tool_call_leak_examples == []


def test_output_not_list() -> None:
    cls = classify_response({"output": "junk"})
    assert cls.has_structured_call is False
    assert cls.text_tool_call_leak_examples == []


# ---------- evidence fields ------------------------------------------------


def test_text_excerpts_captured() -> None:
    cls = classify_response(_resp([_msg("hello world this is a response")]))
    assert cls.text_excerpts == ["hello world this is a response"]


def test_output_item_types_recorded() -> None:
    cls = classify_response(_resp([_function_call(), _msg("x")]))
    assert cls.output_item_types == ["function_call", "message"]


def test_long_text_truncated_in_excerpts_and_leaks() -> None:
    long_text = "x" * 1000
    cls = classify_response(_resp([_msg(long_text)]))
    assert cls.text_excerpts == [long_text[:300]]
    # not a leak, so no leak examples
    assert cls.text_tool_call_leak_examples == []
