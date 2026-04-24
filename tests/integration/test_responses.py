from __future__ import annotations

from fastapi.testclient import TestClient

from codex_proxy.app import create_app
from codex_proxy.backend import UsageSnapshot
from codex_proxy.fakes import InMemoryFakeBackend


def _usage(remaining: float) -> UsageSnapshot:
    return UsageSnapshot(
        remaining_fraction=remaining,
        cooldown_until_ts=None,
        weekly_exhausted=False,
        probed_at_ts=0.0,
    )


def test_responses_routes_to_fake_backend() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post(
            "/v1/responses",
            json={
                "model": "model-a0d0",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            },
        )
    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "resp-fake"
    assert body["object"] == "response"
    assert body["model"] == "model-a0d0"


def test_responses_rejects_missing_model() -> None:
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0d0"}),
    )
    with TestClient(create_app(backends=[fake])) as client:
        response = client.post("/v1/responses", json={"input": []})
    assert response.status_code == 400


def test_responses_streaming_passes_chunks_through() -> None:
    chunks = (
        b'data: {"type":"response.created"}\n\n',
        b'data: {"type":"response.output_text.delta","delta":"hel"}\n\n',
        b"data: [DONE]\n\n",
    )
    fake = InMemoryFakeBackend(
        id="fake",
        advertised_models=frozenset({"model-a0d0"}),
        canned_responses_stream_chunks=chunks,
    )
    with (
        TestClient(create_app(backends=[fake])) as client,
        client.stream(
            "POST",
            "/v1/responses",
            json={"model": "model-a0d0", "stream": True, "input": []},
        ) as response,
    ):
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = b"".join(response.iter_bytes())
    assert body == b"".join(chunks)


def test_session_binding_is_shared_between_routes() -> None:
    alpha = InMemoryFakeBackend(
        id="alpha",
        advertised_models=frozenset({"model-a0d0"}),
        usage=_usage(0.5),
    )
    beta = InMemoryFakeBackend(
        id="beta",
        advertised_models=frozenset({"model-a0d0"}),
        usage=_usage(0.5),
    )
    with TestClient(create_app(backends=[alpha, beta])) as client:
        # First touch via /v1/responses binds s1 -> alpha (id tiebreak).
        first = client.post(
            "/v1/responses",
            json={"model": "model-a0d0", "input": []},
            headers={"X-Codex-Session-Id": "s1"},
        )
        assert first.status_code == 200
        assert first.json()["id"] == "resp-alpha"

        # Rebalance to make beta the obvious default pick.
        beta.set_usage(_usage(0.99))
        alpha.set_usage(_usage(0.1))

        # Second request via /v1/chat/completions with the same session id
        # must honor the binding established on the other route.
        second = client.post(
            "/v1/chat/completions",
            json={"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Codex-Session-Id": "s1"},
        )
        assert second.status_code == 200
        assert second.json()["id"] == "fake-alpha"
