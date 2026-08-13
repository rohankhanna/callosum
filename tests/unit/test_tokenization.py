"""Tests for callosum.tokenization.count_tokens.

The peer-quality audit subtracts the audit prompt's token cost from the
Codex-facing usage meter, so the count must be biased to **round up**
(never under-count the audit) — a 1-char string must count as 1 token,
not 0. GPT models use tiktoken (exact); everything else uses a
rounded-up chars/3 fallback. These tests pin both paths and the
round-up invariant without depending on tiktoken being installed.
"""

from __future__ import annotations

import math

import pytest

from callosum import tokenization
from callosum.tokenization import _fallback_tokens, count_tokens


class _FakeEncoder:
    """A stand-in tiktoken encoder returning a fixed token list."""

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = tokens

    def encode(self, text: str) -> list[int]:
        return self._tokens


class _RaisingEncoder:
    def encode(self, text: str) -> list[int]:
        raise RuntimeError("encode failed")


# ---------- empty ---------------------------------------------------------


def test_empty_text_returns_zero() -> None:
    assert count_tokens("", model="model-a0f5") == 0
    assert count_tokens("", model="model-a0d4") == 0


# ---------- fallback (non-GPT) --------------------------------------------


def test_non_gpt_uses_fallback_rounded_up() -> None:
    assert count_tokens("hello", model="model-a0d4") == math.ceil(5 / 3)
    assert count_tokens("ab", model="model-a0g1-3") == math.ceil(2 / 3)


def test_fallback_tokens_round_up() -> None:
    # ceil(1/3) == 1, NOT 0 — the round-up-bias invariant.
    assert _fallback_tokens("a") == 1
    assert _fallback_tokens("abc") == 1
    assert _fallback_tokens("abcd") == 2
    assert _fallback_tokens("") == 0


def test_fallback_never_under_counts() -> None:
    for n in range(0, 20):
        assert _fallback_tokens("x" * n) >= n / 3


# ---------- GPT path with a controlled encoder ----------------------------


def test_gpt_with_encoder_uses_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokenization, "_encoder", lambda model: _FakeEncoder([1, 2, 3]))
    assert count_tokens("hello", model="model-a0f5") == 3


def test_gpt_encoder_none_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokenization, "_encoder", lambda model: None)
    assert count_tokens("hello", model="model-a0f5") == math.ceil(5 / 3)


def test_gpt_encode_raises_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokenization, "_encoder", lambda model: _RaisingEncoder())
    assert count_tokens("hello", model="model-a0f5") == math.ceil(5 / 3)


# ---------- encoder selection --------------------------------------------


def test_encoder_returns_none_for_non_gpt() -> None:
    # The real lru-cached _encoder returns None for non-GPT slugs so they
    # take the chars/3 fallback (tiktoken does not apply to local models).
    assert tokenization._encoder("model-a0d4") is None


def test_gpt_real_tiktoken_path_positive() -> None:
    """If tiktoken is installed, the GPT path returns a positive int.

    Skipped when tiktoken is unavailable so the suite is hermetic."""
    try:
        import tiktoken  # noqa: F401
    except Exception:
        pytest.skip("tiktoken not installed")
    n = count_tokens("hello world this is a prompt", model="model-a0f5")
    assert isinstance(n, int)
    assert n > 0
