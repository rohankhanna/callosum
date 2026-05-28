"""Tests for routing/features.py — request-body feature extraction."""

from __future__ import annotations

import asyncio

from callosum.routing.embedding.noop import NoopEmbeddingProvider
from callosum.routing.features import extract_features


def _features(body: dict):
    return asyncio.run(extract_features(body, NoopEmbeddingProvider()))


def test_extract_text_from_chat_messages() -> None:
    body = {
        "messages": [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hello"},
        ]
    }
    f = _features(body)
    assert "you are helpful" in f.text
    assert "hello" in f.text


def test_extract_text_from_codex_responses_input() -> None:
    body = {
        "instructions": "you are helpful",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "what's 2+2"}],
            }
        ],
    }
    f = _features(body)
    assert "you are helpful" in f.text
    assert "what's 2+2" in f.text


def test_modalities_text_only_by_default() -> None:
    body = {"messages": [{"role": "user", "content": "plain text"}]}
    f = _features(body)
    assert f.modalities == frozenset({"text"})


def test_modalities_detects_image_content_part() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "..."}},
                ],
            }
        ]
    }
    f = _features(body)
    assert "image" in f.modalities


def test_modalities_detects_codex_input_image() -> None:
    body = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "..."}],
            }
        ]
    }
    f = _features(body)
    assert "image" in f.modalities


def test_modalities_detects_audio_and_video() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "audio": "..."},
                    {"type": "input_video", "video": "..."},
                ],
            }
        ]
    }
    f = _features(body)
    assert "audio" in f.modalities
    assert "video" in f.modalities


def test_needs_tools_when_tools_present() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "tools": [{"type": "function", "function": {"name": "calc"}}],
    }
    f = _features(body)
    assert f.needs_tools is True


def test_needs_tools_accepts_legacy_functions_field() -> None:
    body = {
        "messages": [{"role": "user", "content": "x"}],
        "functions": [{"name": "calc"}],
    }
    f = _features(body)
    assert f.needs_tools is True


def test_needs_tools_false_when_field_empty_or_missing() -> None:
    assert _features({"messages": [{"role": "user", "content": "x"}]}).needs_tools is False
    assert _features({"messages": [{"role": "user", "content": "x"}], "tools": []}).needs_tools is False


def test_embedding_none_with_noop_provider() -> None:
    body = {"messages": [{"role": "user", "content": "x"}]}
    f = _features(body)
    assert f.embedding is None


def test_tokens_estimate_scales_with_text_length() -> None:
    short = _features({"messages": [{"role": "user", "content": "x"}]})
    long_text = "x" * 30_000
    long = _features({"messages": [{"role": "user", "content": long_text}]})
    assert long.tokens > short.tokens
    # Sanity: 30K chars at chars/3 ≈ 10K tokens.
    assert 8_000 < long.tokens < 12_000
