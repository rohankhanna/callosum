"""Tests for complexity classification extraction and handling."""

import asyncio

from codex_proxy.app import (
    _extract_and_strip_complexity,
    _extract_complexity_class,
    _extract_complexity_from_stream,
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


def test_extract_complexity_class_ignores_invalid_markers() -> None:
    """Invalid markers like {{{0}}} or {{{4}}} are not matched."""
    text = "{{{0}}} This has invalid marker"
    complexity_class, cleaned = _extract_complexity_class(text)
    assert complexity_class is None
    # Text unchanged since regex doesn't match invalid levels
    assert cleaned == text


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
