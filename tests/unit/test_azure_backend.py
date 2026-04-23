from __future__ import annotations

import httpx
import pytest

from codex_proxy.backends.azure_openai import AzureOpenAIBackend
from codex_proxy.errors import BackendError


async def test_chat_completions_constructs_deployment_url_and_api_key_header() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["api_key"] = request.headers.get("api-key")
        captured["authorization"] = request.headers.get("Authorization")
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-az",
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

    backend = AzureOpenAIBackend(
        id="azure",
        endpoint="https://example.openai.azure.com",
        api_key="az-secret",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
        transport=httpx.MockTransport(handler),
    )
    try:
        response = await backend.chat_completions(
            {"model": "model-a0f5-mini", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response["id"] == "chatcmpl-az"
        assert captured["url"] == (
            "https://example.openai.azure.com/openai/deployments/model-a0f5-mini-prod"
            "/chat/completions?api-version=2024-10-01-preview"
        )
        assert captured["api_key"] == "az-secret"
        assert captured["authorization"] is None
    finally:
        await backend.aclose()


async def test_advertised_models_are_deployment_keys() -> None:
    backend = AzureOpenAIBackend(
        id="azure",
        endpoint="https://example.openai.azure.com",
        api_key="az-secret",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5": "model-a0f5-prod", "model-a0f5-mini": "model-a0f5-mini-prod"},
    )
    try:
        assert backend.advertised_models == frozenset({"model-a0f5", "model-a0f5-mini"})
    finally:
        await backend.aclose()


async def test_empty_deployments_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="deployments"):
        AzureOpenAIBackend(
            id="azure",
            endpoint="https://example.openai.azure.com",
            api_key="az-secret",
            api_version="2024-10-01-preview",
            deployments={},
        )


async def test_missing_deployment_raises_unknown_model() -> None:
    backend = AzureOpenAIBackend(
        id="azure",
        endpoint="https://example.openai.azure.com",
        api_key="az-secret",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "unmapped"})
        assert excinfo.value.classification == "unknown_model"
    finally:
        await backend.aclose()


async def test_rate_limited_response_classifies_and_records_cooldown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"retry-after": "3"},
            json={"error": {"message": "rate"}},
        )

    backend = AzureOpenAIBackend(
        id="azure",
        endpoint="https://example.openai.azure.com",
        api_key="az-secret",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "model-a0f5-mini"})
        assert excinfo.value.classification == "rate_limited"
        usage = await backend.usage_snapshot()
        assert usage.cooldown_until_ts is not None
    finally:
        await backend.aclose()


async def test_stream_yields_upstream_bytes() -> None:
    stream_body = b'data: {"id":"1"}\n\ndata: [DONE]\n\n'
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    backend = AzureOpenAIBackend(
        id="azure",
        endpoint="https://example.openai.azure.com",
        api_key="az-secret",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks = [c async for c in backend.chat_completions_stream({"model": "model-a0f5-mini"})]
        assert b"".join(chunks) == stream_body
        assert "api-version=2024-10-01-preview" in str(captured["url"])
    finally:
        await backend.aclose()
