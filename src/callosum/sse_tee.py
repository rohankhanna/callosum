from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResponsesStreamSummary:
    """Outcome of observing a streamed `/responses` call.

    `completed_response` is the `response` object from the terminal
    `response.completed` SSE event (includes `usage`, `id`, `model`, etc.),
    or None if the stream ended without one (error, disconnect, etc.).
    `raw_blob` is the exact bytes that flowed through, kept so the logging
    layer can persist them when body-capture is on.
    """

    completed_response: dict[str, Any] | None
    total_bytes: int
    raw_blob: bytes


class ResponsesStreamCollector:
    """Wraps an upstream SSE iterator. Forwards chunks to the client verbatim
    while accumulating a copy so the terminal event can be parsed after.

    Typical use inside a StreamingResponse body:

        collector = ResponsesStreamCollector(upstream_iter)
        async for chunk in collector.iter_through():
            yield chunk
        summary = collector.summary  # safe after iteration ends
    """

    def __init__(self, source: AsyncIterator[bytes]) -> None:
        self._source = source
        self._chunks: list[bytes] = []

    async def iter_through(self) -> AsyncIterator[bytes]:
        async for chunk in self._source:
            self._chunks.append(chunk)
            yield chunk

    @property
    def summary(self) -> ResponsesStreamSummary:
        blob = b"".join(self._chunks)
        return ResponsesStreamSummary(
            completed_response=parse_response_completed(blob),
            total_bytes=len(blob),
            raw_blob=blob,
        )


def parse_response_completed(blob: bytes) -> dict[str, Any] | None:
    """Scan an SSE blob for the terminal `response.completed` event and
    return its `response` object, or None if the event is absent/malformed.

    Note: the upstream `response.completed` event ships with `output: []`
    even when the model emitted text — visible content lives only in the
    streamed `response.output_text.{delta,done}` events. Callers that
    need a non-stream-shaped dict with assembled text should use
    `assemble_completed_with_text` instead.

    SSE framing: events are separated by blank lines. Each event may have
    zero or more `data:` lines whose concatenation is the payload (JSON).
    We iterate events in order and return the first valid `response.completed`
    payload, which is always terminal in a Responses-API stream.
    """
    for raw_event in blob.split(b"\n\n"):
        if b"response.completed" not in raw_event:
            continue
        data_lines: list[bytes] = []
        for line in raw_event.split(b"\n"):
            if line.startswith(b"data:"):
                data_lines.append(line[len(b"data:") :].lstrip())
        if not data_lines:
            continue
        try:
            payload = json.loads(b"".join(data_lines))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("type") == "response.completed":
            response = payload.get("response")
            if isinstance(response, dict):
                return response
    return None


def extract_output_text_from_blob(blob: bytes) -> str:
    """Concatenate the `text` field across all `response.output_text.done`
    events in stream order.

    Codex's Responses API only emits visible text via streamed events; the
    terminal `response.completed` event ships `output: []` regardless. The
    `.done` events come once per output_index with the fully assembled text
    for that index — concatenating them in stream order reconstructs the
    full visible response. Falls back to deltas only if no `.done` events
    arrived (truncated/aborted stream); deltas carry the same text in
    incremental form.
    """
    done_texts: list[str] = []
    delta_parts: list[str] = []
    for raw_event in blob.split(b"\n\n"):
        is_done = b"response.output_text.done" in raw_event
        is_delta = b"response.output_text.delta" in raw_event
        if not (is_done or is_delta):
            continue
        data_lines: list[bytes] = []
        for line in raw_event.split(b"\n"):
            if line.startswith(b"data:"):
                data_lines.append(line[len(b"data:") :].lstrip())
        if not data_lines:
            continue
        try:
            payload = json.loads(b"".join(data_lines))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("type") == "response.output_text.done":
            text = payload.get("text")
            if isinstance(text, str):
                done_texts.append(text)
        elif payload.get("type") == "response.output_text.delta":
            delta = payload.get("delta")
            if isinstance(delta, str):
                delta_parts.append(delta)
    if done_texts:
        return "".join(done_texts)
    return "".join(delta_parts)


def assemble_completed_with_text(blob: bytes) -> dict[str, Any] | None:
    """Return the `response.completed` payload's response object with the
    visible text materialized into `output[]`.

    Codex's non-stream path is internally streaming (see
    CodexAuthVaultBackend.responses) — we need this assembly to give
    non-streaming callers a dict shape that actually contains the
    response text. Without it, `output[]` is always empty even when the
    model emitted hundreds of tokens.

    The materialized message item mirrors the shape of an output_text
    content part: `{"type":"message","role":"assistant","content":
    [{"type":"output_text","text":<assembled>}]}`. Other items present
    in the original `output[]` (reasoning, tool calls) are preserved
    verbatim and the message item is appended.
    """
    response = parse_response_completed(blob)
    if response is None:
        return None
    text = extract_output_text_from_blob(blob)
    if not text:
        return response
    existing_output = response.get("output")
    if not isinstance(existing_output, list):
        existing_output = []
    # If there's already a message item with text (shouldn't happen with
    # current Codex behavior, but be defensive against future changes),
    # don't double-inject — return the response as-is.
    for item in existing_output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for c in item.get("content") or []:
            if isinstance(c, dict) and isinstance(c.get("text"), str) and c["text"].strip():
                return response
    message_item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    return {**response, "output": [*existing_output, message_item]}
