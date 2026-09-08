from __future__ import annotations

import json

import httpx
import pytest

from callosum.backends.codex_gateway import CodexGatewayBackend
from callosum.errors import BackendError


def _sse_response(response_payload: dict, *, status: int = 200, headers: dict | None = None) -> httpx.Response:
    events = [
        ("response.created", {"type": "response.created", "id": response_payload.get("id", "r1")}),
        ("response.completed", {"type": "response.completed", "response": response_payload}),
    ]
    body = "".join(f"event: {n}\ndata: {json.dumps(p)}\n\n" for n, p in events).encode()
    return httpx.Response(
        status,
        content=body,
        headers={"content-type": "text/event-stream", **(headers or {})},
    )


def _gateway(handler: httpx.MockTransport | None = None, **kwargs: object) -> CodexGatewayBackend:
    if handler is None:

        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("unexpected request")

    return CodexGatewayBackend(
        id="gw-a",
        api_key="gw-key-123",
        advertised_models=frozenset({"model-a"}),
        base_url="https://codex-gw.example.com",
        transport=httpx.MockTransport(handler),
        **kwargs,  # type: ignore[arg-type]
    )


async def test_chat_completions_translates_and_forwards_to_responses() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["body"] = json.loads(request.content)
        return _sse_response(
            {
                "id": "resp-1",
                "model": "model-a",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12},
            }
        )

    backend = _gateway(handler)
    try:
        result = await backend.chat_completions(
            {
                "model": "model-a",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert captured["url"] == "https://codex-gw.example.com/responses"
        assert captured["authorization"] == "Bearer gw-key-123"
        body = captured["body"]
        assert isinstance(body, dict)
        assert body["model"] == "model-a"
        assert body["instructions"] == "be brief"
        assert body["stream"] is True
        assert result["id"] == "resp-1"
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["usage"]["prompt_tokens"] == 10
    finally:
        await backend.aclose()


async def test_responses_collects_sse_into_dict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            {
                "id": "resp-2",
                "model": "model-a",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "world"}],
                    }
                ],
            }
        )

    backend = _gateway(handler)
    try:
        result = await backend.responses({"model": "model-a", "input": [], "stream": True})
        assert result["id"] == "resp-2"
    finally:
        await backend.aclose()


async def test_responses_stream_yields_sse_bytes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            {
                "id": "resp-3",
                "model": "model-a",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "streamed"}],
                    }
                ],
            }
        )

    backend = _gateway(handler)
    try:
        chunks = [c async for c in backend.responses_stream({"model": "model-a", "input": [], "stream": True})]
        assert b"response.completed" in b"".join(chunks)
    finally:
        await backend.aclose()


async def test_model_discovery_updates_advertised_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "model-a", "display_name": "Model A"},
                    {"slug": "model-b", "display_name": "Model B"},
                ]
            },
        )

    backend = _gateway(handler)
    try:
        assert backend.advertised_models == frozenset({"model-a"})
        await backend.refresh_advertised_models()
        assert backend.advertised_models == frozenset({"model-a", "model-b"})
    finally:
        await backend.aclose()


async def test_model_discovery_skips_on_4xx() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "expired"})

    backend = _gateway(handler)
    try:
        await backend.refresh_advertised_models()
        assert backend.advertised_models == frozenset({"model-a"})
    finally:
        await backend.aclose()


async def test_model_discovery_skips_on_network_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    backend = _gateway(handler)
    try:
        await backend.refresh_advertised_models()
        assert backend.advertised_models == frozenset({"model-a"})
    finally:
        await backend.aclose()


async def test_quota_headers_parsed_on_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            {
                "id": "resp-q",
                "model": "model-a",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
            headers={
                "x-codex-secondary-used-percent": "42",
                "x-codex-primary-used-percent": "10",
            },
        )

    from callosum.backend import CallHandle

    backend = _gateway(handler)
    try:
        h = CallHandle()
        await backend.responses({"model": "model-a", "input": [], "stream": True}, h)
        assert h.quota_after is not None
        assert h.quota_after.weekly_used_percent == 42.0
    finally:
        await backend.aclose()


async def test_rate_limited_records_cooldown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "2"}, json={"error": "rate"})

    backend = _gateway(handler)
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
        assert excinfo.value.classification == "rate_limited"
        snapshot = await backend.usage_snapshot()
        assert snapshot.cooldown_until_ts is not None
    finally:
        await backend.aclose()


async def test_401_raises_auth_invalid_no_retry() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(401, json={"error": {"code": "invalid_token"}})

    backend = _gateway(handler)
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "model-a", "messages": [{"role": "user", "content": "hi"}]})
        assert excinfo.value.classification == "auth_invalid"
        assert call_count["n"] == 1
    finally:
        await backend.aclose()


async def test_transport_failures_set_cooldown_after_threshold() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    backend = _gateway(handler)
    try:
        for _ in range(3):
            with pytest.raises(BackendError):
                await backend.responses({"model": "model-a", "input": [], "stream": True})
        snapshot = await backend.usage_snapshot()
        assert snapshot.cooldown_until_ts is not None
    finally:
        await backend.aclose()


async def test_health_returns_ok() -> None:
    backend = _gateway()
    try:
        result = await backend.health()
        assert result.available is True
    finally:
        await backend.aclose()


async def test_usage_snapshot_initial_state() -> None:
    backend = _gateway()
    try:
        snapshot = await backend.usage_snapshot()
        assert snapshot.cooldown_until_ts is None
        assert snapshot.weekly_exhausted is False
    finally:
        await backend.aclose()


async def test_construction_accepts_empty_advertised_models() -> None:
    backend = CodexGatewayBackend(
        id="gw-empty",
        api_key="key",
        base_url="https://gw.example.com",
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    try:
        assert backend.advertised_models == frozenset()
    finally:
        await backend.aclose()
