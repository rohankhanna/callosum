from __future__ import annotations

import httpx
import pytest

from codex_proxy.backends.openai_api_key import OpenAIApiKeyBackend


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


async def test_chat_completions_raises_on_upstream_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code=429, json={"error": {"message": "rate limit"}})

    backend = OpenAIApiKeyBackend(
        id="test",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await backend.chat_completions({"model": "model-a0f5-mini"})
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
