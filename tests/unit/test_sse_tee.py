from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import pytest

from callosum.sse_tee import (
    ResponsesStreamCollector,
    parse_response_completed,
)


async def _iter_chunks(chunks: Sequence[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def _build_stream(events: list[tuple[str, dict]]) -> bytes:
    parts: list[str] = []
    for event_name, payload in events:
        parts.append(f"event: {event_name}\ndata: {json.dumps(payload)}\n\n")
    return "".join(parts).encode()


def test_parse_response_completed_extracts_terminal_event_payload() -> None:
    blob = _build_stream(
        [
            ("response.created", {"type": "response.created", "id": "r1"}),
            (
                "response.output_text.delta",
                {"type": "response.output_text.delta", "delta": "hi"},
            ),
            (
                "response.completed",
                {
                    "type": "response.completed",
                    "response": {
                        "id": "r1",
                        "model": "model-a0e7",
                        "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
                    },
                },
            ),
        ]
    )
    parsed = parse_response_completed(blob)
    assert parsed is not None
    assert parsed["id"] == "r1"
    assert parsed["usage"]["total_tokens"] == 15


def test_parse_response_completed_returns_none_when_event_missing() -> None:
    blob = _build_stream([("response.output_text.delta", {"type": "response.output_text.delta", "delta": "x"})])
    assert parse_response_completed(blob) is None


def test_parse_response_completed_skips_malformed_event() -> None:
    # A `response.completed` event with invalid JSON must not crash; the
    # parser returns None and the caller falls back to null token counts.
    broken = b"event: response.completed\ndata: not-json\n\n"
    assert parse_response_completed(broken) is None


def test_parse_response_completed_handles_multi_data_line_payload() -> None:
    payload = {"type": "response.completed", "response": {"id": "r42", "usage": {}}}
    raw = json.dumps(payload)
    # Split the JSON across two data: lines (legal in SSE, rarely seen in
    # practice but the parser must cope).
    mid = len(raw) // 2
    blob = b"event: response.completed\ndata: " + raw[:mid].encode() + b"\ndata: " + raw[mid:].encode() + b"\n\n"
    parsed = parse_response_completed(blob)
    assert parsed is not None
    assert parsed["id"] == "r42"


async def test_collector_forwards_bytes_verbatim_and_exposes_summary() -> None:
    created = b'event: response.created\ndata: {"type":"response.created","id":"r1"}\n\n'
    delta = b'event: response.output_text.delta\ndata: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    completed = (
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"id":"r1",'
        b'"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}\n\n'
    )
    chunks = [created, delta, completed]
    collector = ResponsesStreamCollector(_iter_chunks(chunks))
    forwarded: list[bytes] = []
    async for chunk in collector.iter_through():
        forwarded.append(chunk)
    # Verbatim forwarding: the downstream consumer sees every original chunk
    # unchanged and in order.
    assert forwarded == chunks

    summary = collector.summary
    assert summary.total_bytes == sum(len(c) for c in chunks)
    assert summary.raw_blob == b"".join(chunks)
    assert summary.completed_response is not None
    assert summary.completed_response["usage"]["total_tokens"] == 3


async def test_collector_summary_is_safe_on_empty_stream() -> None:
    collector = ResponsesStreamCollector(_iter_chunks([]))
    async for _ in collector.iter_through():
        pytest.fail("expected empty stream to yield nothing")
    summary = collector.summary
    assert summary.total_bytes == 0
    assert summary.completed_response is None
    assert summary.raw_blob == b""


def test_extract_output_text_concatenates_done_events() -> None:
    """`response.output_text.done` carries the full assembled text per
    output_index. extract_output_text_from_blob concatenates them in
    stream order so non-stream callers see the visible response."""
    from callosum.sse_tee import extract_output_text_from_blob

    blob = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"Hello, "}\n\n'
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"world!"}\n\n'
        b"event: response.output_text.done\n"
        b'data: {"type":"response.output_text.done","text":"Hello, world!"}\n\n'
    )
    assert extract_output_text_from_blob(blob) == "Hello, world!"


def test_extract_output_text_falls_back_to_deltas_when_no_done() -> None:
    """If the stream was truncated and we only have deltas, reconstruct
    from those instead. Avoids returning empty text on aborted streams
    when partial visible content is still useful."""
    from callosum.sse_tee import extract_output_text_from_blob

    blob = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"Partial "}\n\n'
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"answer"}\n\n'
    )
    assert extract_output_text_from_blob(blob) == "Partial answer"


def test_assemble_completed_injects_text_when_output_is_empty() -> None:
    """The whole point of the helper: the Codex response.completed event
    ships with output:[], but the visible text exists in delta/done
    events. assemble_completed_with_text reconstructs a dict that
    actually contains the text in output[].content[].text."""
    from callosum.sse_tee import assemble_completed_with_text

    blob = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"model-a0e7 high"}\n\n'
        b"event: response.output_text.done\n"
        b'data: {"type":"response.output_text.done","text":"model-a0e7 high"}\n\n'
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"id":"r1","output":[],"usage":{"total_tokens":4}}}\n\n'
    )
    result = assemble_completed_with_text(blob)
    assert result is not None
    assert result["id"] == "r1"
    assert result["usage"]["total_tokens"] == 4
    # Injected message item with the assembled visible text.
    assert len(result["output"]) == 1
    msg = result["output"][0]
    assert msg["type"] == "message"
    assert msg["role"] == "assistant"
    assert msg["content"][0]["text"] == "model-a0e7 high"


def test_assemble_completed_preserves_existing_output_items() -> None:
    """If the completed event already had reasoning items in output[],
    those must survive the assembly; the text item is appended, not
    substituted."""
    from callosum.sse_tee import assemble_completed_with_text

    blob = (
        b"event: response.output_text.done\n"
        b'data: {"type":"response.output_text.done","text":"4"}\n\n'
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"id":"r1","output":[{"type":"reasoning","summary":[]}],"usage":{}}}\n\n'
    )
    result = assemble_completed_with_text(blob)
    assert result is not None
    assert len(result["output"]) == 2
    assert result["output"][0]["type"] == "reasoning"
    assert result["output"][1]["type"] == "message"
    assert result["output"][1]["content"][0]["text"] == "4"


def test_assemble_completed_skips_injection_when_text_already_present() -> None:
    """Defensive: if a future Codex version starts populating output[]
    with the assembled text, don't double-inject. Detect by looking for
    any message item that already has non-empty text."""
    from callosum.sse_tee import assemble_completed_with_text

    blob = (
        b"event: response.output_text.done\n"
        b'data: {"type":"response.output_text.done","text":"hello"}\n\n'
        b"event: response.completed\n"
        b'data: {"type":"response.completed","response":{"id":"r1","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}]}}\n\n'
    )
    result = assemble_completed_with_text(blob)
    assert result is not None
    # Only the existing message item; we did NOT append a duplicate.
    assert len(result["output"]) == 1
    assert result["output"][0]["content"][0]["text"] == "hello"


# ---------- namespace stripping on the response stream -------------------


async def _drain(it: AsyncIterator[bytes]) -> bytes:
    out = bytearray()
    async for chunk in it:
        out += chunk
    return bytes(out)


def test_namespaced_tool_names_from_request_collects_namespaced_tools() -> None:
    """`exec` (flat custom tool, no namespace) is NOT namespaced; tools inside
    a `namespace` group, and flat tools carrying their own `namespace`, are.
    Declarations are read from both top-level `tools` and the `additional_tools`
    input item the codex CLI ships."""
    from callosum.sse_tee import namespaced_tool_names_from_request

    body = {
        "tools": [{"type": "function", "name": "top_ns", "namespace": "grp"}],
        "input": [
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {"type": "custom", "name": "exec"},  # flat, no namespace
                    {
                        "type": "namespace",
                        "name": "collaboration",
                        "tools": [
                            {"type": "function", "name": "followup_task"},
                        ],
                    },
                ],
            },
        ],
    }
    assert namespaced_tool_names_from_request(body) == {"top_ns", "followup_task"}
    assert "exec" not in namespaced_tool_names_from_request(body)


@pytest.mark.asyncio
async def test_strip_namespace_stream_drops_bogus_namespace_for_unnamespaced_tool() -> None:
    """The upstream returns `custom_tool_call` with `namespace == name` for a
    tool the client declared WITHOUT a namespace; the stream strip removes it
    so the codex CLI dispatches by `name` alone. Non-tool events and fields
    like `obfuscation`/`sequence_number` pass through byte-for-byte."""
    from callosum.sse_tee import namespaced_tool_names_from_request, strip_namespace_stream

    body = {"input": [{"type": "additional_tools", "tools": [{"type": "custom", "name": "exec"}]}]}
    namespaced = namespaced_tool_names_from_request(body)
    added_event = (
        "event: response.output_item.added\n"
        'data: {"type":"response.output_item.added","item":{"id":"ctc_1",'
        '"type":"custom_tool_call","call_id":"call_x","name":"exec",'
        '"namespace":"exec","input":"await tools.exec_command({})"}}\n\n'
    )
    delta_event = (
        "event: response.custom_tool_call_input.delta\n"
        'data: {"type":"response.custom_tool_call_input.delta","delta":"await",'
        '"item_id":"ctc_1","obfuscation":"Z9","sequence_number":21}\n\n'
    )
    # Feed the two events split across chunk boundaries to exercise per-event
    # buffering (the added event straddles two chunks; its terminator lands in
    # the second chunk with the next event).
    stream = added_event.encode() + delta_event.encode()
    chunks = [stream[:40], stream[40:120], stream[120:]]
    out = await _drain(strip_namespace_stream(_iter_chunks(chunks), namespaced))
    # Re-serialize the rewritten added event's payload and check the item.
    added_blob = out.split(b"\n\n", 1)[0].decode("utf-8")
    added_obj = json.loads(added_blob.split("data: ", 1)[1])
    assert added_obj["item"]["type"] == "custom_tool_call"
    assert added_obj["item"]["name"] == "exec"
    assert "namespace" not in added_obj["item"]
    # The untouched delta event survives with its obfuscation/sequence_number
    # byte-for-byte (it was not a rewrite candidate).
    assert b'"obfuscation":"Z9"' in out
    assert b'"sequence_number":21' in out
    # Output is still valid SSE framing.
    assert out.count(b"\n\n") == 2


@pytest.mark.asyncio
async def test_strip_namespace_stream_preserves_legitimately_namespaced_tools() -> None:
    """A tool declared under a `namespace` group keeps its `namespace` in the
    response — the strip targets only tools the client declared flat."""
    from callosum.sse_tee import namespaced_tool_names_from_request, strip_namespace_stream

    body = {
        "input": [
            {
                "type": "additional_tools",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "collaboration",
                        "tools": [
                            {"type": "function", "name": "followup_task"},
                        ],
                    },
                ],
            }
        ]
    }
    namespaced = namespaced_tool_names_from_request(body)
    added_event = (
        "event: response.output_item.added\n"
        'data: {"type":"response.output_item.added","item":{"id":"fc_1",'
        '"type":"custom_tool_call","call_id":"call_y","name":"followup_task",'
        '"namespace":"collaboration","input":"{}"}}\n\n'
    )
    out = await _drain(strip_namespace_stream(_iter_chunks([added_event.encode()]), namespaced))
    assert b'"namespace":"collaboration"' in out


@pytest.mark.asyncio
async def test_strip_namespace_stream_passes_through_non_candidate_events() -> None:
    """Events with no `data:` line (e.g. a stray comment) and malformed JSON
    are passed through unchanged, never crash."""
    from callosum.sse_tee import strip_namespace_stream

    raw = b": keepalive ping\n\n" + b"event: bad\ndata: not-json\n\n"
    out = await _drain(strip_namespace_stream(_iter_chunks([raw]), set()))
    assert out == raw


def test_strip_namespace_from_completed_strips_unnamespaced_only() -> None:
    """Non-streaming path strips the bogus `namespace` from `custom_tool_call`
    items in a completed payload, leaving namespaced tools and the response's
    echoed tool *declarations* untouched."""
    from callosum.sse_tee import strip_namespace_from_completed

    payload = {
        "type": "response.completed",
        "response": {
            "id": "r1",
            "tools": [{"type": "custom", "name": "exec", "namespace": "exec"}],
            "output": [
                {"type": "custom_tool_call", "name": "exec", "namespace": "exec", "call_id": "c1"},
                {"type": "custom_tool_call", "name": "followup_task", "namespace": "collaboration", "call_id": "c2"},
            ],
        },
    }
    namespaced = {"followup_task"}
    strip_namespace_from_completed(payload, namespaced)
    out = payload["response"]["output"]
    assert "namespace" not in out[0]
    assert out[0]["name"] == "exec"
    assert out[1]["namespace"] == "collaboration"
    # The echoed tool declaration is a declaration, not a call — untouched.
    assert payload["response"]["tools"][0]["namespace"] == "exec"
