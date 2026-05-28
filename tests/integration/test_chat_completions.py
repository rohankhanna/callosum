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


def test_chat_completions_rewrites_to_router_pick_when_requested_model_unavailable() -> None:
    """With route-all-models routing, requesting a model no backend serves
    no longer 503s — the router picks from cells that ARE served. The
    only failure mode tied to "no viable backend" is when the capability
    filter empties the candidate set entirely (NoCompatibleCellError →
    400), which is exercised by routing/router tests."""
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"other-model"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]},
        )
    assert response.status_code == 200
    # Router rewrote to a real served cell — fake serves only "other-model".
    assert response.json()["model"] == "other-model"


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
