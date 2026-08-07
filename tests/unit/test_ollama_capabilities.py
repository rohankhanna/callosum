"""Tests for the shared ollama /api/show capability parse + fetch module.

Covers parse_ollama_show (well-formed, minimal, audio, malformed) and
fetch_ollama_capabilities (success via MockTransport, transport failure,
non-200, malformed JSON). The parser is pure-sync; the fetcher is async and
never raises — all failure paths return None.
"""

from __future__ import annotations

import contextlib
from typing import Any

import httpx

from callosum.backends._ollama_capabilities import (
    DEFAULT_HEALTH_TIMEOUT_S,
    fetch_ollama_capabilities,
    parse_ollama_show,
)

# ---------- parse_ollama_show -------------------------------------------------


def test_parse_well_formed_vision_tools() -> None:
    """capabilities=["vision","tools"] + model_info with context_length +
    parameter_count -> full OllamaShowCapabilities."""
    info: dict[str, Any] = {
        "capabilities": ["completion", "tools", "vision", "thinking"],
        "model_info": {
            "general.architecture": "model-a0f4",
            "general.context_length": 8192,
            "general.parameter_count": 3_100_000_000,
            "model-a0f4.context_length": 8192,
        },
    }
    caps = parse_ollama_show(info)
    assert caps is not None
    assert caps.modalities == frozenset({"text", "image"})
    assert caps.supports_tools is True
    assert caps.context_window == 8192
    assert caps.parameter_count == 3_100_000_000


def test_parse_minimal_tools_only() -> None:
    """capabilities=["tools"] only, no model_info -> fallbacks."""
    info: dict[str, Any] = {"capabilities": ["tools"]}
    caps = parse_ollama_show(info)
    assert caps is not None
    assert caps.modalities == frozenset({"text"})
    assert caps.supports_tools is True
    assert caps.context_window == 128_000
    assert caps.parameter_count is None


def test_parse_audio_modality() -> None:
    """capabilities=["audio"] -> text+audio, no tools."""
    info: dict[str, Any] = {"capabilities": ["audio"]}
    caps = parse_ollama_show(info)
    assert caps is not None
    assert caps.modalities == frozenset({"text", "audio"})
    assert caps.supports_tools is False
    assert caps.context_window == 128_000
    assert caps.parameter_count is None


def test_parse_malformed_returns_none() -> None:
    """Missing capabilities key, non-list capabilities, and empty dict -> None."""
    assert parse_ollama_show({}) is None
    assert parse_ollama_show({"capabilities": "not-a-list"}) is None
    assert parse_ollama_show({"model_info": {"general.context_length": 4096}}) is None


def test_parse_context_length_walk_picks_arch_key() -> None:
    """The walk finds a `<arch>.context_length` even when
    `general.context_length` is absent."""
    info: dict[str, Any] = {
        "capabilities": ["completion"],
        "model_info": {
            "general.architecture": "model-a0g2",
            "model-a0g2.context_length": 32768,
        },
    }
    caps = parse_ollama_show(info)
    assert caps is not None
    assert caps.context_window == 32768
    assert caps.parameter_count is None


def test_parse_ignores_non_int_parameter_count() -> None:
    """A non-int general.parameter_count leaves parameter_count None."""
    info: dict[str, Any] = {
        "capabilities": ["completion"],
        "model_info": {"general.parameter_count": "big"},
    }
    caps = parse_ollama_show(info)
    assert caps is not None
    assert caps.parameter_count is None


# ---------- fetch_ollama_capabilities -----------------------------------------


def _show_body() -> dict[str, Any]:
    return {
        "capabilities": ["completion", "vision"],
        "model_info": {
            "general.architecture": "model-a0f4",
            "model-a0f4.context_length": 8192,
        },
    }


async def test_fetch_success_via_mock_transport() -> None:
    """A 200 with a well-formed body returns parsed capabilities."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=_show_body()))
    async with contextlib.aclosing(httpx.AsyncClient(transport=transport)) as client:
        caps = await fetch_ollama_capabilities(client, endpoint="http://ollama.local", runtime_model="model-a0d7")
    assert caps is not None
    assert caps.modalities == frozenset({"text", "image"})
    assert caps.supports_tools is False
    assert caps.context_window == 8192
    assert caps.parameter_count is None


async def test_fetch_connect_error_returns_none() -> None:
    """A transport-level ConnectError never raises — returns None."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    transport = httpx.MockTransport(handler)
    async with contextlib.aclosing(httpx.AsyncClient(transport=transport)) as client:
        caps = await fetch_ollama_capabilities(client, endpoint="http://ollama.local", runtime_model="model-a0d7")
    assert caps is None


async def test_fetch_non_200_returns_none() -> None:
    """A 500 response is swallowed -> None (never raises)."""
    transport = httpx.MockTransport(lambda request: httpx.Response(500, text="boom"))
    async with contextlib.aclosing(httpx.AsyncClient(transport=transport)) as client:
        caps = await fetch_ollama_capabilities(client, endpoint="http://ollama.local", runtime_model="model-a0d7")
    assert caps is None


async def test_fetch_malformed_json_returns_none() -> None:
    """A 200 with non-JSON body -> None."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json", headers={"content-type": "application/json"})

    transport = httpx.MockTransport(handler)
    async with contextlib.aclosing(httpx.AsyncClient(transport=transport)) as client:
        caps = await fetch_ollama_capabilities(client, endpoint="http://ollama.local", runtime_model="model-a0d7")
    assert caps is None


async def test_fetch_strips_trailing_slash_and_posts_show() -> None:
    """endpoint with a trailing slash still hits /api/show, and the request
    body carries {"name": runtime_model}."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = request.content.decode()
        return httpx.Response(200, json=_show_body())

    transport = httpx.MockTransport(handler)
    async with contextlib.aclosing(httpx.AsyncClient(transport=transport)) as client:
        caps = await fetch_ollama_capabilities(
            client,
            endpoint="http://ollama.local/",
            runtime_model="model-a0d7",
            timeout_s=DEFAULT_HEALTH_TIMEOUT_S,
        )
    assert caps is not None
    assert seen["url"] == "http://ollama.local/api/show"
    assert '"model-a0d7"' in seen["body"]
