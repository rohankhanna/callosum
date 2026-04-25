from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

import pytest

from codex_proxy.sse_tee import (
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
    blob = _build_stream(
        [("response.output_text.delta", {"type": "response.output_text.delta", "delta": "x"})]
    )
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
    blob = (
        b"event: response.completed\n"
        b"data: " + raw[:mid].encode() + b"\n"
        b"data: " + raw[mid:].encode() + b"\n\n"
    )
    parsed = parse_response_completed(blob)
    assert parsed is not None
    assert parsed["id"] == "r42"


async def test_collector_forwards_bytes_verbatim_and_exposes_summary() -> None:
    created = b'event: response.created\ndata: {"type":"response.created","id":"r1"}\n\n'
    delta = (
        b"event: response.output_text.delta\n"
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
    )
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
