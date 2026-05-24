from __future__ import annotations

from fastapi.testclient import TestClient

from callosum.app import create_app
from callosum.fakes import InMemoryFakeBackend


def test_chat_completions_routes_to_fake_backend() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "model-a0d0"
    assert body["id"] == "fake-fake"


def test_chat_completions_503_when_no_viable_backend() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"other-model"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": []},
        )
    assert response.status_code == 503


def test_chat_completions_rejects_missing_model() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post("/v1/chat/completions", json={"messages": []})
    assert response.status_code == 400


def test_chat_completions_streaming_routes_through_fake_backend() -> None:
    chunks = (b'data: {"id":"1"}\n\n', b"data: [DONE]\n\n")
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0d0"}),
        canned_stream_chunks=chunks,
    )
    with (
        TestClient(create_app(backends=[fake])) as client,
        client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "model-a0d0",
                "stream": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())
    assert body == b"".join(chunks)
