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
