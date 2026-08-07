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


def namespaced_tool_names_from_request(body: dict[str, Any]) -> set[str]:
    """Names of tools the client declared UNDER a namespace.

    The chatgpt `/codex/responses` upstream echoes a `namespace` onto the
    `custom_tool_call` items it returns even for tools the client declared
    WITHOUT one (the codex CLI's `exec` JS orchestrator is a flat
    `{"type":"custom","name":"exec"}` declaration, yet the call comes back with
    `namespace == name == "exec"`). The codex CLI then dispatches a custom tool
    call by `namespace + name`, so `exec`+`exec` = `execexec` is rejected as an
    unsupported custom tool — tool calls never run.

    callosum reconciles this on the response side by stripping that bogus
    `namespace` for tools the client did NOT declare as namespaced. THIS
    function identifies the opposite set — tools the client DID declare under
    a namespace — so the strip leaves their (legitimate) `namespace` intact
    (e.g. a `collaboration` namespace group's `followup_task`).

    Declarations are collected from the standard top-level `tools` array and
    from the `additional_tools` input item the codex CLI ships its tools in.
    A `{"type":"namespace","tools":[...]}` group namespaces every tool inside
    it; a flat tool is namespaced only when it carries its own `namespace`.
    """
    namespaced: set[str] = set()

    def visit(tool: Any) -> None:
        if not isinstance(tool, dict):
            return
        if tool.get("type") == "namespace":
            for inner in tool.get("tools") or []:
                if isinstance(inner, dict) and isinstance(inner.get("name"), str):
                    namespaced.add(inner["name"])
            return
        name = tool.get("name")
        ns = tool.get("namespace")
        if isinstance(name, str) and isinstance(ns, str) and ns:
            namespaced.add(name)

    for tool in body.get("tools") or []:
        visit(tool)
    for item in body.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "additional_tools":
            for tool in item.get("tools") or []:
                visit(tool)
    return namespaced


def _strip_namespace_from_custom_tool_calls(obj: Any, namespaced_names: set[str]) -> bool:
    """Recursively drop `namespace` from every `custom_tool_call` dict whose
    tool was NOT declared as namespaced. Returns True if anything changed.

    Only `custom_tool_call` items are touched — tool *declarations* the
    response echoes back (type `custom`/`namespace`/`function`) are left alone,
    and `function_call` items (which carry no `namespace`) are untouched.
    Mutates in place; safe to call on the parsed `response.completed` payload
    or on a single parsed event."""
    changed = False
    if isinstance(obj, dict):
        if obj.get("type") == "custom_tool_call" and "namespace" in obj and obj.get("name") not in namespaced_names:
            del obj["namespace"]
            changed = True
        for v in obj.values():
            if _strip_namespace_from_custom_tool_calls(v, namespaced_names):
                changed = True
    elif isinstance(obj, list):
        for v in obj:
            if _strip_namespace_from_custom_tool_calls(v, namespaced_names):
                changed = True
    return changed


def strip_namespace_from_completed(payload: dict[str, Any], namespaced_names: set[str]) -> dict[str, Any]:
    """Non-streaming path: strip the bogus `namespace` from `custom_tool_call`
    items in a `response.completed` payload. Returns the (in-place mutated)
    payload so callers can chain."""
    _strip_namespace_from_custom_tool_calls(payload, namespaced_names)
    return payload


def _rewrite_responses_event(event: bytes, namespaced_names: set[str]) -> bytes:
    """Rewrite one SSE event (WITHOUT its trailing blank-line terminator):
    if its `data:` payload carries a `custom_tool_call` whose `namespace` the
    upstream added wrongly, drop that `namespace`. Returns the original bytes
    unchanged when the event isn't a rewrite candidate, so the events we don't
    touch pass through byte-for-byte (preserving `obfuscation`,
    `sequence_number`, and every other field the client depends on)."""
    if b"\ndata: " not in event:
        return event
    text = event.decode("utf-8")
    lines = text.split("\n")
    data_idx: int | None = None
    for i, ln in enumerate(lines):
        if ln.startswith("data: "):
            data_idx = i
            break
    if data_idx is None:
        return event
    try:
        obj = json.loads(lines[data_idx][len("data: ") :])
    except (ValueError, json.JSONDecodeError):
        return event
    if not _strip_namespace_from_custom_tool_calls(obj, namespaced_names):
        return event
    lines[data_idx] = "data: " + json.dumps(obj)
    return "\n".join(lines).encode("utf-8")


async def strip_namespace_stream(
    source: AsyncIterator[bytes],
    namespaced_names: set[str],
) -> AsyncIterator[bytes]:
    """Streaming path: drop the bogus `namespace` from `custom_tool_call`
    items as they flow through a `/responses` SSE stream.

    Buffers per-event — each Responses-API event is one complete `data:` JSON
    terminated by a blank line — so rewrites never split an event across a
    chunk boundary. Every event we don't rewrite is yielded with its original
    bytes (the `+ b"\\n\\n"` re-attaches the terminator we split on)."""
    buffer = b""
    async for chunk in source:
        buffer += chunk
        while b"\n\n" in buffer:
            event, buffer = buffer.split(b"\n\n", 1)
            yield _rewrite_responses_event(event, namespaced_names) + b"\n\n"
    if buffer:
        # Trailing bytes without a terminating blank line: a well-formed
        # Responses stream doesn't end this way, but pass anything leftover
        # through the same rewriter rather than dropping it.
        yield _rewrite_responses_event(buffer, namespaced_names)


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
