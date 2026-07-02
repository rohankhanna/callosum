import pytest

from callosum.selectors import (
    SelectorDecision,
    SelectorError,
    is_selector,
    parse_selector,
)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("callosum:auto", SelectorDecision(strategy="auto")),
        ("callosum:local-only", SelectorDecision(strategy="local-only")),
        ("callosum:remote-only", SelectorDecision(strategy="remote-only")),
        (
            "callosum:remote/model-a0e8",
            SelectorDecision(source="remote", pinned_model="model-a0e8"),
        ),
        (
            "callosum:remote/model-a0e8:high",
            SelectorDecision(source="remote", pinned_model="model-a0e8", pinned_effort="high"),
        ),
        (
            "callosum:local/model-a0b6".replace(":30b", ""),  # plain local pin
            SelectorDecision(source="local", pinned_model="model-a0d4"),
        ),
        (
            # Local pins accept an effort symmetric to remote ().
            # The parser gates only on REASONING_LEVELS; per-model support is
            # enforced downstream (catalog advertise + dispatch 503).
            "callosum:local/model-a0d2:high",
            SelectorDecision(source="local", pinned_model="model-a0d2", pinned_effort="high"),
        ),
    ],
)
def test_parse_valid(model, expected):
    assert parse_selector(model) == expected


def test_legacy_passthrough_returns_none():
    assert parse_selector("model-a0e8") is None
    assert parse_selector("auto") is None  # bare virtual model, not callosum:
    assert parse_selector("") is None
    assert parse_selector(None) is None


def test_is_selector():
    assert is_selector("callosum:auto") is True
    assert is_selector("model-a0e8") is False
    assert is_selector(None) is False


def test_offline_rejected():
    with pytest.raises(SelectorError):
        parse_selector("callosum:offline")


def test_unknown_strategy_rejected():
    with pytest.raises(SelectorError):
        parse_selector("callosum:turbo")


def test_unknown_pin_source_rejected():
    with pytest.raises(SelectorError):
        parse_selector("callosum:cloud/model-a0e8")


def test_remote_pin_invalid_effort_rejected():
    with pytest.raises(SelectorError):
        parse_selector("callosum:remote/model-a0e8:ultra")


def test_local_pin_invalid_effort_rejected():
    # Resolved (): local pins accept an effort, but it must be
    # a syntactically valid reasoning level — same gate as remote.
    with pytest.raises(SelectorError):
        parse_selector("callosum:local/model-a0d4:ultra")


def test_empty_and_missing_model_rejected():
    for bad in ("callosum:", "callosum:remote/", "callosum:local/"):
        with pytest.raises(SelectorError):
            parse_selector(bad)
