from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from codex_proxy.backend import UsageSnapshot
from codex_proxy.backends.openai_api_key import OpenAIApiKeyBackend
from codex_proxy.errors import BackendError
from codex_proxy.state import StateStore


async def test_chat_completions_forwards_body_and_returns_response() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        captured["body"] = request.content
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "model": "model-a0f5-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await backend.chat_completions(
            {"model": "model-a0f5-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response["id"] == "chatcmpl-123"
        assert response["choices"][0]["message"]["content"] == "hi"
        assert captured["url"] == "https://api.openai.com/v1/chat/completions"
        assert captured["auth"] == "Bearer sk-test"
    finally:
        await backend.aclose()


async def test_chat_completions_raises_classified_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"retry-after": "5"},
            json={"error": {"message": "rate limit"}},
        )

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "model-a0f5-mini"})
        assert excinfo.value.classification == "rate_limited"
        assert excinfo.value.status_code == 429
        assert excinfo.value.retry_after_s == 5.0
    finally:
        await backend.aclose()


async def test_chat_completions_stream_yields_upstream_bytes() -> None:
    stream_body = b'data: {"id":"1"}\n\ndata: {"id":"2"}\n\ndata: [DONE]\n\n'
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks = [c async for c in backend.chat_completions_stream({"model": "model-a0f5-mini"})]
        assert b"".join(chunks) == stream_body
        assert captured["auth"] == "Bearer sk-test"
    finally:
        await backend.aclose()


async def test_chat_completions_stream_raises_classified_backend_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=401, json={"error": {"message": "auth"}})

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            async for _ in backend.chat_completions_stream({"model": "model-a0f5-mini"}):
                pass
        assert excinfo.value.classification == "auth_invalid"
        assert excinfo.value.status_code == 401
    finally:
        await backend.aclose()


async def test_rate_limited_response_records_cooldown(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"retry-after": "7"},
            json={"error": {"message": "rate"}},
        )

    store = StateStore(tmp_path)
    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )
    try:
        with pytest.raises(BackendError):
            await backend.chat_completions({"model": "model-a0f5-mini"})
        usage = await backend.usage_snapshot()
        assert usage.cooldown_until_ts is not None
        persisted = store.load_usage("test")
        assert persisted is not None
        assert persisted.cooldown_until_ts == usage.cooldown_until_ts
    finally:
        await backend.aclose()


async def test_state_store_preloads_usage_on_construction(tmp_path: Path) -> None:
    store = StateStore(tmp_path)
    store.save_usage(
        "test",
        UsageSnapshot(
            remaining_fraction=0.25,
            cooldown_until_ts=9999.0,
            weekly_exhausted=False,
            probed_at_ts=1.0,
        ),
    )
    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        state_store=store,
    )
    try:
        usage = await backend.usage_snapshot()
        assert usage.cooldown_until_ts == 9999.0
        assert usage.remaining_fraction == 0.25
    finally:
        await backend.aclose()


async def test_health_and_usage_return_defaults() -> None:
    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    try:
        health = await backend.health()
        assert health.available
        assert health.reason == "ok"
        usage = await backend.usage_snapshot()
        assert usage.remaining_fraction is None
        assert not usage.weekly_exhausted
    finally:
        await backend.aclose()


async def test_responses_forwards_body_to_responses_endpoint() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"id": "resp-1", "object": "response"})

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        base_url="https://api.openai.example.com/v1",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await backend.responses({"model": "model-a0f5-mini", "input": []})
        assert captured["url"] == "https://api.openai.example.com/v1/responses"
        assert captured["auth"] == "Bearer sk-test"
        assert result["id"] == "resp-1"
    finally:
        await backend.aclose()


async def test_responses_stream_passes_sse_chunks_through() -> None:
    sse_body = b'data: {"type":"response.created"}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse_body)

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks = [c async for c in backend.responses_stream({"model": "model-a0f5-mini", "input": []})]
        assert b"".join(chunks) == sse_body
    finally:
        await backend.aclose()
