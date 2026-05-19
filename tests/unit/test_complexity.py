"""Tests for complexity classification extraction and handling."""

import asyncio
import json

from codex_proxy.app import (
    _extract_and_strip_complexity,
    _extract_complexity_class,
    _extract_complexity_from_stream,
    _strip_trailing_complexity_marker,
    _strip_trailing_complexity_marker_text,
)


def test_extract_complexity_class_finds_marker() -> None:
    """_extract_complexity_class identifies {{{N}}} markers and strips them."""
    text = "{{{1}}} This is a simple answer"
    complexity_class, cleaned = _extract_complexity_class(text)
    assert complexity_class == 1
    assert cleaned == "This is a simple answer"


def test_extract_complexity_class_ignores_whitespace_before_marker() -> None:
    """Whitespace before marker is stripped along with the marker."""
    text = "  \n{{{2}}} This is moderate"
    complexity_class, cleaned = _extract_complexity_class(text)
    assert complexity_class == 2
    assert cleaned == "This is moderate"


def test_extract_complexity_class_returns_none_when_marker_missing() -> None:
    """When no marker is found, return None and the original text."""
    text = "This is just a response"
    complexity_class, cleaned = _extract_complexity_class(text)
    assert complexity_class is None
    assert cleaned == text


def test_extract_complexity_class_handles_all_levels() -> None:
    """All three complexity levels (1, 2, 3) are correctly identified."""
    for level in [1, 2, 3]:
        text = "{{{" + str(level) + "}}} Response"
        complexity_class, cleaned = _extract_complexity_class(text)
        assert complexity_class == level
        assert cleaned == "Response"


def test_extract_complexity_class_strips_non_numeric_brace_markers() -> None:
    """Non-numeric {{{...}}} markers are stripped but produce no class.

    The proxy injects an instruction for the model to emit {{{1}}}, {{{2}}},
    or {{{3}}}; if the model emits any other leading {{{...}}} (e.g.
    {{{0}}}, {{{complexity: Low}}}), we still strip it so the user never
    sees a stray marker — we just can't classify it.
    """
    text = "{{{0}}} This has non-numeric marker"
    complexity_class, cleaned = _extract_complexity_class(text)
    assert complexity_class is None
    assert cleaned == "This has non-numeric marker"


def test_extract_and_strip_complexity_modifies_response_dict() -> None:
    """Extract and strip properly modifies the response dict structure."""
    response = {
        "choices": [
            {
                "message": {
                    "content": "{{{2}}} This is moderate complexity"
                }
            }
        ]
    }
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class == 2
    assert modified["choices"][0]["message"]["content"] == "This is moderate complexity"


def test_extract_and_strip_complexity_handles_missing_marker() -> None:
    """When no marker is found, return None and unchanged response."""
    response = {
        "choices": [
            {
                "message": {
                    "content": "Just a normal response"
                }
            }
        ]
    }
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None
    assert modified["choices"][0]["message"]["content"] == "Just a normal response"


def test_extract_and_strip_complexity_handles_malformed_response() -> None:
    """Gracefully handle responses with missing structure."""
    response = {}
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None
    assert modified == {}


def test_extract_and_strip_complexity_handles_missing_choices() -> None:
    """Gracefully handle responses with no choices array."""
    response = {"data": "something"}
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None
    assert modified == response


def test_extract_and_strip_complexity_handles_empty_choices() -> None:
    """Gracefully handle responses with empty choices array."""
    response = {"choices": []}
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None
    assert modified == response


def test_extract_and_strip_complexity_handles_missing_message() -> None:
    """Gracefully handle choices with no message field."""
    response = {"choices": [{"delta": {"content": "stream data"}}]}
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None


def test_extract_and_strip_complexity_handles_missing_content() -> None:
    """Gracefully handle messages with no content field."""
    response = {"choices": [{"message": {"role": "assistant"}}]}
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None


def test_extract_and_strip_complexity_handles_non_string_content() -> None:
    """Gracefully handle content that is not a string."""
    response = {"choices": [{"message": {"content": None}}]}
    complexity_class, modified = _extract_and_strip_complexity(response)
    assert complexity_class is None


class TestStreamingComplexityExtraction:
    """Tests for streaming SSE complexity extraction (both chat and Responses API formats)."""

    async def _extract_from_source(self, sse_bytes: bytes) -> bytes:
        """Helper to extract from SSE bytes and return the reconstructed result."""

        async def source():
            yield sse_bytes

        result_chunks = []
        async for chunk in _extract_complexity_from_stream(source()):
            result_chunks.append(chunk)
        return b"".join(result_chunks)

    async def test_chat_completions_format_with_marker(self) -> None:
        """Chat completions SSE format: marker is extracted and stripped."""
        sse_data = (
            b'data: {"choices":[{"delta":{"role":"assistant","content":"{{{1}}} hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # Marker should be stripped
        assert "{{{1}}}" not in result_str
        # Content should be preserved
        assert "hello" in result_str
        assert "world" in result_str

    async def test_chat_completions_format_without_marker(self) -> None:
        """Chat completions format without marker passes through unchanged."""
        sse_data = (
            b'data: {"choices":[{"delta":{"role":"assistant","content":"hello"}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            b'data: [DONE]\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # Content should be exactly as-is
        assert "hello" in result_str
        assert "world" in result_str

    async def test_responses_api_format_with_marker(self) -> None:
        """Responses API SSE format: marker is extracted from delta field."""
        sse_data = (
            b'data: {"type":"response.created","id":"resp-s"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"{{{2}}} hello"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":" world"}\n\n'
            b'data: {"type":"response.completed"}\n\n'
            b'data: [DONE]\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # Marker should be stripped
        assert "{{{2}}}" not in result_str
        # Content should be preserved
        assert "hello" in result_str
        assert "world" in result_str
        # Event types should be preserved
        assert "response.created" in result_str
        assert "response.completed" in result_str

    async def test_responses_api_format_without_marker(self) -> None:
        """Responses API format without marker passes through unchanged."""
        sse_data = (
            b'data: {"type":"response.created","id":"resp-s"}\n\n'
            b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n'
            b'data: {"type":"response.completed"}\n\n'
            b'data: [DONE]\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # Content should be exactly as-is
        assert "hello" in result_str
        assert "response.created" in result_str

    async def test_multiline_chunk_preservation(self) -> None:
        """Multiline chunks: all lines are preserved, only marker line is modified."""
        sse_data = (
            b'data: {"type":"response.created"}\n'
            b'data: {"type":"response.output_text.delta","delta":"{{{3}}} first"}\n'
            b'data: {"type":"response.output_text.delta","delta":" second"}\n'
            b'data: [DONE]\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # All event types should be present
        assert "response.created" in result_str
        assert "response.output_text.delta" in result_str

        # Marker stripped
        assert "{{{3}}}" not in result_str
        # Content preserved
        assert "first" in result_str
        assert "second" in result_str

    async def test_malformed_json_stops_extraction(self) -> None:
        """Malformed JSON: extraction stops but stream continues."""
        sse_data = (
            b'data: {"type":"response.output_text.delta","delta":"{{{1}}} hello"}\n\n'
            b'data: {broken json\n\n'
            b'data: {"type":"response.output_text.delta","delta":" world"}\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # Marker from first line should be stripped
        assert "{{{1}}}" not in result_str
        # Malformed line should pass through as-is
        assert "broken json" in result_str
        # Subsequent lines should pass through as-is
        assert "world" in result_str

    async def test_empty_stream(self) -> None:
        """Empty stream yields nothing."""
        sse_data = b""
        result = await self._extract_from_source(sse_data)
        assert result == b""

    async def test_done_marker_stops_extraction(self) -> None:
        """[DONE] marker causes extraction to stop checking."""
        sse_data = (
            b'data: {"type":"response.output_text.delta","delta":"{{{2}}} first"}\n\n'
            b'data: [DONE]\n\n'
            b'data: {"type":"response.output_text.delta","delta":"{{{3}}} should_not_extract"}\n\n'
        )
        result = await self._extract_from_source(sse_data)
        result_str = result.decode("utf-8")

        # First marker should be extracted
        assert "{{{2}}}" not in result_str
        # After [DONE], markers should not be extracted
        assert "{{{3}}}" in result_str

    async def _extract_from_event_sequence(self, events: list[bytes]) -> bytes:
        """Feed events one chunk at a time (each event = its own chunk)."""

        async def source():
            for ev in events:
                yield ev

        result_chunks = []
        async for chunk in _extract_complexity_from_stream(source()):
            result_chunks.append(chunk)
        return b"".join(result_chunks)

    async def test_marker_split_across_chunks_chat_completions(self) -> None:
        """Regression: marker tokenized as multiple SSE events must still be stripped.

        Tokenizers commonly split '{{{2}}}' into pieces like '{{{', '2', '}}}'.
        Each piece may arrive as its own data: event. The old extractor saw
        '{{{' in the first event, failed the strict regex, set complexity_found=True
        anyway, and let '2}}}' leak through to the client.
        """
        events = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"{{{"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"2"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"}}}"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" hello"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        result_str = (await self._extract_from_event_sequence(events)).decode("utf-8")
        assert "{{{" not in result_str
        assert "}}}" not in result_str
        assert "hello" in result_str

    async def test_marker_split_across_chunks_responses_api(self) -> None:
        """Same split-marker case for the Responses API delta format."""
        events = [
            b'data: {"type":"response.created","id":"r1"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"{{{"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"3"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"}}}"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":" world"}\n\n',
            b'data: [DONE]\n\n',
        ]
        result_str = (await self._extract_from_event_sequence(events)).decode("utf-8")
        assert "{{{" not in result_str
        assert "}}}" not in result_str
        assert "world" in result_str
        assert "response.created" in result_str

    async def test_event_split_across_byte_chunks(self) -> None:
        """One SSE event split across multiple TCP byte chunks (no \\n\\n yet)."""

        async def source():
            yield b'data: {"choices":[{"delta":{"content":"{{{1}'
            yield b'}} hello"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        result_chunks = []
        async for chunk in _extract_complexity_from_stream(source()):
            result_chunks.append(chunk)
        result_str = b"".join(result_chunks).decode("utf-8")
        assert "{{{1}}}" not in result_str
        assert "hello" in result_str

    async def test_bare_digit_marker_chat_completions(self) -> None:
        """Model dropped the braces and emitted just '2\\n\\n' before the answer."""
        events = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"2"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"\\n\\n"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"I checked the things."}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        result_str = (await self._extract_from_event_sequence(events)).decode("utf-8")
        # The bare leading "2" + blank line should be gone, but the real answer remains.
        assert "I checked the things." in result_str
        # Reconstruct visible delta content and confirm no leading bare digit.
        visible = ""
        for line in result_str.split("\n"):
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try:
                    data = json.loads(line[6:])
                    choices = data.get("choices")
                    if choices:
                        visible += choices[0].get("delta", {}).get("content") or ""
                except (json.JSONDecodeError, AttributeError, KeyError, IndexError, TypeError):
                    pass
        assert visible.lstrip().startswith("I checked"), f"visible content was {visible!r}"

    async def test_bare_digit_marker_responses_api(self) -> None:
        """Same bare-digit case on Responses API delta events."""
        events = [
            b'data: {"type":"response.created","id":"r1"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"3"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"\\n\\n"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"The answer is here."}\n\n',
            b'data: [DONE]\n\n',
        ]
        result_str = (await self._extract_from_event_sequence(events)).decode("utf-8")
        assert "The answer is here." in result_str
        visible = ""
        for line in result_str.split("\n"):
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try:
                    data = json.loads(line[6:])
                    if data.get("type") == "response.output_text.delta":
                        visible += data.get("delta") or ""
                except (json.JSONDecodeError, AttributeError, KeyError, IndexError, TypeError):
                    pass
        assert visible.lstrip().startswith("The answer"), f"visible content was {visible!r}"

    async def test_leading_digit_in_legit_content_not_stripped(self) -> None:
        """Response like '2 minutes is the limit' must NOT have its leading 2 stripped."""
        events = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"2 minutes is the limit"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        result_str = (await self._extract_from_event_sequence(events)).decode("utf-8")
        visible = ""
        for line in result_str.split("\n"):
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                try:
                    data = json.loads(line[6:])
                    choices = data.get("choices")
                    if choices:
                        visible += choices[0].get("delta", {}).get("content") or ""
                except (json.JSONDecodeError, AttributeError, KeyError, IndexError, TypeError):
                    pass
        assert visible == "2 minutes is the limit", f"visible content was {visible!r}"


# Async test wrapper for pytest
def test_chat_completions_format_with_marker() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_chat_completions_format_with_marker())


def test_chat_completions_format_without_marker() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_chat_completions_format_without_marker())


def test_responses_api_format_with_marker() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_responses_api_format_with_marker())


def test_responses_api_format_without_marker() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_responses_api_format_without_marker())


def test_multiline_chunk_preservation() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_multiline_chunk_preservation())


def test_malformed_json_stops_extraction() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_malformed_json_stops_extraction())


def test_empty_stream() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_empty_stream())


def test_done_marker_stops_extraction() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_done_marker_stops_extraction())


def test_marker_split_across_chunks_chat_completions() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_marker_split_across_chunks_chat_completions())


def test_marker_split_across_chunks_responses_api() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_marker_split_across_chunks_responses_api())


def test_event_split_across_byte_chunks() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_event_split_across_byte_chunks())


def test_bare_digit_marker_chat_completions() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_bare_digit_marker_chat_completions())


def test_bare_digit_marker_responses_api() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_bare_digit_marker_responses_api())


def test_leading_digit_in_legit_content_not_stripped() -> None:
    asyncio.run(TestStreamingComplexityExtraction().test_leading_digit_in_legit_content_not_stripped())


# ---------- Trailing-marker stripper tests ----------


def test_strip_trailing_marker_text_closing_tag() -> None:
    """{{{/2}}} at the very end is stripped."""
    text = "Here is the answer.\n\n{{{/2}}}"
    assert _strip_trailing_complexity_marker_text(text) == "Here is the answer.\n\n"


def test_strip_trailing_marker_text_no_marker() -> None:
    """Plain trailing content is preserved verbatim."""
    text = "Here is the answer."
    assert _strip_trailing_complexity_marker_text(text) == text


def test_strip_trailing_marker_text_marker_not_at_end_kept() -> None:
    """A {{{...}}} in the middle of the response is NOT stripped."""
    text = "Refer to {{{node_id}}} in the graph."
    assert _strip_trailing_complexity_marker_text(text) == text


class TestTrailingMarkerStream:
    async def _run(self, events: list[bytes]) -> str:
        async def source():
            for ev in events:
                yield ev
        chunks = []
        async for chunk in _strip_trailing_complexity_marker(source()):
            chunks.append(chunk)
        return b"".join(chunks).decode("utf-8")

    @staticmethod
    def _visible_chat(sse: str) -> str:
        out = ""
        for line in sse.split("\n"):
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload.strip() == "[DONE]":
                continue
            try:
                d = json.loads(payload)
            except json.JSONDecodeError:
                continue
            ch = d.get("choices")
            if ch:
                out += ch[0].get("delta", {}).get("content") or ""
            elif d.get("type") == "response.output_text.delta":
                out += d.get("delta") or ""
        return out

    async def test_trailing_closing_tag_chat_completions(self) -> None:
        events = [
            b'data: {"choices":[{"delta":{"content":"Step A: check\\n"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"Step B: confirm."}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"\\n\\n{{{/2}}}"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        visible = self._visible_chat(await self._run(events))
        assert visible == "Step A: check\nStep B: confirm.\n\n"

    async def test_trailing_closing_tag_responses_api(self) -> None:
        events = [
            b'data: {"type":"response.created","id":"r1"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"Here is the brief.\\n\\n"}\n\n',
            b'data: {"type":"response.output_text.delta","delta":"{{{/2}}}"}\n\n',
            b'data: {"type":"response.completed"}\n\n',
            b'data: [DONE]\n\n',
        ]
        visible = self._visible_chat(await self._run(events))
        assert visible == "Here is the brief.\n\n"

    async def test_trailing_marker_split_across_events(self) -> None:
        """Closing tag is split into pieces: '{{{', '/2', '}}}'."""
        events = [
            b'data: {"choices":[{"delta":{"content":"Done."}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"\\n\\n"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"{{{"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"/2"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":"}}}"}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        visible = self._visible_chat(await self._run(events))
        assert "{{{" not in visible and "}}}" not in visible
        assert visible == "Done.\n\n"

    async def test_no_trailing_marker_passthrough(self) -> None:
        events = [
            b'data: {"choices":[{"delta":{"content":"Just a plain answer."}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        visible = self._visible_chat(await self._run(events))
        assert visible == "Just a plain answer."

    async def test_marker_in_middle_not_stripped(self) -> None:
        """A {{{node_id}}} in the middle is NOT a trailing marker; preserve it."""
        events = [
            b'data: {"choices":[{"delta":{"content":"See {{{node_id}}}"}}]}\n\n',
            b'data: {"choices":[{"delta":{"content":" for details."}}]}\n\n',
            b'data: [DONE]\n\n',
        ]
        visible = self._visible_chat(await self._run(events))
        assert visible == "See {{{node_id}}} for details."


def test_trailing_closing_tag_chat_completions() -> None:
    asyncio.run(TestTrailingMarkerStream().test_trailing_closing_tag_chat_completions())


def test_trailing_closing_tag_responses_api() -> None:
    asyncio.run(TestTrailingMarkerStream().test_trailing_closing_tag_responses_api())


def test_trailing_marker_split_across_events() -> None:
    asyncio.run(TestTrailingMarkerStream().test_trailing_marker_split_across_events())


def test_no_trailing_marker_passthrough() -> None:
    asyncio.run(TestTrailingMarkerStream().test_no_trailing_marker_passthrough())


def test_marker_in_middle_not_stripped() -> None:
    asyncio.run(TestTrailingMarkerStream().test_marker_in_middle_not_stripped())


# ---------- done-event scrubbing (full-text accumulated events) ----------


class TestDoneEventScrubbing:
    """The delta filters strip incremental markers, but Codex also emits
    accumulated-text 'done' events at the end of each output. Hermes /
    codex-cli often read those for the final UI render, so they need
    their own scrub pass — otherwise a marker the model voluntarily
    echoed from poisoned conversation history leaks through the delta
    filters into the user's display via response.output_text.done.
    """

    async def _run(self, events: list[bytes]) -> str:
        from codex_proxy.app import _scrub_full_text_events

        async def source():
            for ev in events:
                yield ev

        chunks = []
        async for c in _scrub_full_text_events(source()):
            chunks.append(c)
        return b"".join(chunks).decode("utf-8")

    async def test_output_text_done_text_field_scrubbed(self) -> None:
        events = [
            b'data: {"type":"response.output_text.done","text":"{{{2}}}\\nHello world."}\n\n',
            b'data: [DONE]\n\n',
        ]
        result = await self._run(events)
        assert "{{{2}}}" not in result
        assert "Hello world." in result

    async def test_content_part_done_nested_text_field_scrubbed(self) -> None:
        events = [
            b'data: {"type":"response.content_part.done","part":{"type":"output_text","text":"{{{3}}}\\nThe answer."}}\n\n',
            b'data: [DONE]\n\n',
        ]
        result = await self._run(events)
        assert "{{{3}}}" not in result
        assert "The answer." in result

    async def test_output_item_done_deep_nested_text_field_scrubbed(self) -> None:
        events = [
            b'data: {"type":"response.output_item.done","item":{"id":"msg_x","content":[{"type":"output_text","text":"{{{1}}}\\nResponse here"}]}}\n\n',
            b'data: [DONE]\n\n',
        ]
        result = await self._run(events)
        assert "{{{1}}}" not in result
        assert "Response here" in result

    async def test_trailing_closing_tag_in_done_event_scrubbed(self) -> None:
        events = [
            b'data: {"type":"response.output_text.done","text":"Done.\\n\\n{{{/2}}}"}\n\n',
            b'data: [DONE]\n\n',
        ]
        result = await self._run(events)
        assert "{{{" not in result
        assert "Done." in result

    async def test_non_done_events_passed_through_unchanged(self) -> None:
        events = [
            b'data: {"type":"response.output_text.delta","delta":"hello"}\n\n',
            b'data: {"type":"response.function_call_arguments.delta","delta":"{\\"key\\":1}"}\n\n',
            b'data: [DONE]\n\n',
        ]
        result = await self._run(events)
        # delta events left untouched by this pass (separate filters scrub them)
        assert '"delta":"hello"' in result
        # function-call JSON braces preserved (the escaped form survives untouched)
        assert "\\\"key\\\"" in result


def test_output_text_done_text_field_scrubbed() -> None:
    asyncio.run(TestDoneEventScrubbing().test_output_text_done_text_field_scrubbed())


def test_content_part_done_nested_text_field_scrubbed() -> None:
    asyncio.run(TestDoneEventScrubbing().test_content_part_done_nested_text_field_scrubbed())


def test_output_item_done_deep_nested_text_field_scrubbed() -> None:
    asyncio.run(TestDoneEventScrubbing().test_output_item_done_deep_nested_text_field_scrubbed())


def test_trailing_closing_tag_in_done_event_scrubbed() -> None:
    asyncio.run(TestDoneEventScrubbing().test_trailing_closing_tag_in_done_event_scrubbed())


def test_non_done_events_passed_through_unchanged() -> None:
    asyncio.run(TestDoneEventScrubbing().test_non_done_events_passed_through_unchanged())
