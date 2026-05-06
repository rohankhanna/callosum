"""Tests for complexity classification extraction and handling."""

from codex_proxy.app import _extract_and_strip_complexity, _extract_complexity_class


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
