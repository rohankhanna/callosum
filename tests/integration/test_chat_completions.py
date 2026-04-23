from __future__ import annotations

import httpx
from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.backends.openai_api_key import OpenAIApiKeyBackend
from codex_proxy.fakes import InMemoryFakeBackend


def test_chat_completions_routes_to_fake_backend() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "model-a0f5-mini"
    assert body["id"] == "fake-fake"


def test_chat_completions_503_when_no_viable_backend() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"other-model"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": []},
        )
    assert response.status_code == 503


def test_chat_completions_rejects_missing_model() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0f5-mini"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400


def test_chat_completions_streaming_routes_through_fake_backend() -> None:
    chunks = (b'data: {"id":"1"}\n\n', b"data: [DONE]\n\n")
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0f5-mini"}),
        canned_stream_chunks=chunks,
    )
    with (
        TestClient(create_app(backends=[fake])) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-a0f5-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())
    assert body == b"".join(chunks)


def test_chat_completions_streaming_end_to_end_with_openai_backend_and_mock_upstream() -> None:
    stream_body = b'data: {"id":"c1"}\n\ndata: [DONE]\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    backend = OpenAIApiKeyBackend(
        id="primary",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    with (
        TestClient(create_app(backends=[backend])) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-a0f5-mini",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())
    assert body == stream_body


def test_chat_completions_end_to_end_with_openai_backend_and_mock_upstream() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-mocked",
                "object": "chat.completion",
                "model": "model-a0f5-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = OpenAIApiKeyBackend(
        id="primary",
        api_key="sk-test",
        advertised_models=frozenset({"model-a0f5-mini"}),
        transport=httpx.MockTransport(handler),
    )
    with TestClient(create_app(backends=[backend])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0f5-mini", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    assert response.json()["id"] == "chatcmpl-mocked"
