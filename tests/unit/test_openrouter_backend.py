"""Unit tests for Callosum's direct OpenRouter backend."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends.openrouter import OPENROUTER_PRIORITY_OFFSET, OpenRouterBackend
from callosum.errors import BackendError


def _models_payload(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"data": list(entries)}


def _model(slug: str, context_length: int | None = None) -> dict[str, Any]:
    entry: dict[str, Any] = {"id": slug, "name": slug}
    if context_length is not None:
        entry["context_length"] = context_length
    return entry


_DEFAULT_MODELS = _models_payload(
    _model("openai/model-a0f5", 128_000),
    _model("anthropic/model-a0aa", 200_000),
    _model("meta-model-a0g1/model-a0c2", 128_000),
    _model("model-a0g3/model-a0g3-2.5-72b", 131_072),
    _model("model-a0e2/model-a0c6", 65_536),
    _model("google/model-a0d5", 8_192),
)

_DEFAULT_PROVIDERS = {
    "data": [
        {"slug": "alibaba", "headquarters": "SG", "datacenters": ["SG", "CN"]},
        {"slug": "z-ai", "headquarters": "CN", "datacenters": ["CN"]},
        {"slug": "deepinfra", "headquarters": "US", "datacenters": ["US"]},
        {"slug": "nebius", "headquarters": "NL", "datacenters": None},
        {"slug": "unknown-co", "headquarters": None, "datacenters": None},
    ]
}


def _chat_reply(model: str) -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi there"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _direct_handler(
    *,
    models: dict[str, Any] | None = None,
    providers: dict[str, Any] | None = None,
    chat_json: dict[str, Any] | None = None,
    chat_sse: list[bytes] | None = None,
    chat_status: int = 200,
    chat_error: BaseException | None = None,
    providers_status: int = 200,
    models_status: int = 200,
) -> tuple[Callable[[httpx.Request], httpx.Response], SimpleNamespace]:
    """Build a MockTransport handler for OpenRouter's direct HTTP surface."""
    models_default = models if models is not None else _DEFAULT_MODELS
    providers_default = providers if providers is not None else _DEFAULT_PROVIDERS
    records = SimpleNamespace(models=[], providers=[], chat=[], stream=[])

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/models"):
            records.models.append(request)
            if models_status != 200:
                return httpx.Response(models_status, json={"error": "models failed"})
            return httpx.Response(200, json=models_default)
        if path.endswith("/providers"):
            records.providers.append(request)
            if providers_status != 200:
                return httpx.Response(providers_status, json={"error": "providers failed"})
            return httpx.Response(200, json=providers_default)
        if path.endswith("/chat/completions"):
            if request.headers.get("Accept") == "text/event-stream":
                records.stream.append(request)
            else:
                records.chat.append(request)
            if chat_error is not None:
                raise chat_error
            if path == "/api/v1/chat/completions" and request.method == "POST" and chat_status != 200:
                return httpx.Response(chat_status, json={"error": "chat failed"})
            if request.headers.get("Accept") == "text/event-stream":
                content = (
                    b"".join(chat_sse)
                    if chat_sse is not None
                    else b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\ndata: [DONE]\n\n'
                )
                return httpx.Response(chat_status, content=content, headers={"content-type": "text/event-stream"})
            return httpx.Response(
                200,
                json=chat_json or _chat_reply("openai/model-a0f5"),
            )
        return httpx.Response(404, json={"error": "unknown target"})

    return handler, records


def _backend(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    api_key: str = "test-key",
    **kwargs: Any,
) -> OpenRouterBackend:
    return OpenRouterBackend(
        id="openrouter",
        api_key=api_key,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_advertised_models_empty_before_first_poll() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler)
    assert backend.advertised_models == frozenset()
    health = await backend.health()
    assert health.available is True
    assert backend.advertised_models == frozenset({"openai/model-a0f5", "anthropic/model-a0aa"})
    await backend.aclose()


async def test_family_exclude_drops_ollama_cloud_overlap() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler)
    await backend.health()
    assert backend.advertised_models == frozenset({"openai/model-a0f5", "anthropic/model-a0aa"})
    assert "meta-model-a0g1/model-a0c2" not in backend.advertised_models
    assert "model-a0g3/model-a0g3-2.5-72b" not in backend.advertised_models
    assert "model-a0e2/model-a0c6" not in backend.advertised_models
    assert "google/model-a0d5" not in backend.advertised_models
    await backend.aclose()


async def test_family_exclude_disabled_keeps_all() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler, exclude_families=frozenset())
    await backend.health()
    assert backend.advertised_models == frozenset(
        {
            "openai/model-a0f5",
            "anthropic/model-a0aa",
            "meta-model-a0g1/model-a0c2",
            "model-a0g3/model-a0g3-2.5-72b",
            "model-a0e2/model-a0c6",
            "google/model-a0d5",
        }
    )
    await backend.aclose()


async def test_allowlist_filter() -> None:
    handler, _records = _direct_handler()
    backend = _backend(
        handler,
        model_filter="allowlist",
        allowlist=frozenset({"openai/model-a0f5", "model-a0g3/model-a0g3-2.5-72b"}),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"openai/model-a0f5"})
    await backend.aclose()


async def test_prefix_filter() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler, model_filter="prefix", model_prefix="anthropic/")
    await backend.health()
    assert backend.advertised_models == frozenset({"anthropic/model-a0aa"})
    await backend.aclose()


async def test_invalid_model_filter_raises() -> None:
    handler, _records = _direct_handler()
    with pytest.raises(ValueError):
        _backend(handler, model_filter="bogus")


async def test_model_metadata_remote_band_and_context_window() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler)
    await backend.health()
    metadata = backend.model_metadata
    assert set(metadata) == {"openai/model-a0f5", "anthropic/model-a0aa"}
    for entry in metadata.values():
        assert entry.supported_in_api is True
        assert entry.visibility == "list"
        assert entry.supported_reasoning_levels == ("default",)
        assert entry.priority is not None
        assert OPENROUTER_PRIORITY_OFFSET <= entry.priority < 10_000
    assert metadata["openai/model-a0f5"].priority == OPENROUTER_PRIORITY_OFFSET
    assert metadata["anthropic/model-a0aa"].priority == OPENROUTER_PRIORITY_OFFSET + 1
    assert metadata["openai/model-a0f5"].context_window == 128_000
    assert metadata["anthropic/model-a0aa"].context_window == 200_000
    await backend.aclose()


async def test_catalog_request_uses_direct_bearer_authentication() -> None:
    handler, records = _direct_handler()
    backend = _backend(handler)
    await backend.health()
    assert records.models[0].method == "GET"
    assert str(records.models[0].url) == "https://openrouter.ai/api/v1/models"
    assert records.models[0].headers["authorization"] == "Bearer test-key"
    await backend.aclose()


async def test_usage_snapshot_is_advisory() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler)
    await backend.health()
    snapshot = await backend.usage_snapshot()
    assert snapshot.remaining_fraction == 1.0
    assert snapshot.weekly_exhausted is False
    assert snapshot.cooldown_until_ts is None
    await backend.aclose()


async def test_quota_snapshot_is_none() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler)
    assert await backend.quota_snapshot() is None
    await backend.aclose()


async def test_402_sets_exhausted_cooldown_and_rate_limited() -> None:
    handler, _records = _direct_handler(chat_status=402)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions(
            {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]},
            handle,
        )
    assert exc_info.value.classification == "rate_limited"
    assert exc_info.value.status_code == 402
    snapshot = await backend.usage_snapshot()
    assert snapshot.weekly_exhausted is True
    assert snapshot.remaining_fraction == 0.0
    assert snapshot.cooldown_until_ts is not None
    await backend.aclose()


async def test_402_stream_sets_exhausted_cooldown() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _records = _direct_handler(chat_sse=chunks, chat_status=402)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _chunk in backend.chat_completions_stream(
            {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]},
            handle,
        ):
            pass
    assert exc_info.value.classification == "rate_limited"
    snapshot = await backend.usage_snapshot()
    assert snapshot.weekly_exhausted is True
    await backend.aclose()


async def test_402_responses_stream_sets_exhausted_cooldown() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _records = _direct_handler(chat_sse=chunks, chat_status=402)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _chunk in backend.responses_stream(
            {"model": "openai/model-a0f5", "input": "say hi", "stream": True},
            handle,
        ):
            pass
    assert exc_info.value.classification == "rate_limited"
    assert exc_info.value.status_code == 402
    snapshot = await backend.usage_snapshot()
    assert snapshot.weekly_exhausted is True
    await backend.aclose()


async def test_health_reports_network_when_request_fails() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = _backend(handler, catalog_refresh_s=0)
    health = await backend.health()
    assert health.available is False
    assert health.reason == "network"
    await backend.aclose()


async def test_health_reports_no_key_when_key_missing_or_empty() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request should be sent without an API key")

    backend = _backend(handler, api_key="", catalog_refresh_s=0)
    health = await backend.health()
    assert health.available is False
    assert health.reason == "no-key"
    await backend.aclose()


async def test_health_reports_auth_invalid_when_models_returns_401() -> None:
    handler, _records = _direct_handler(models_status=401)
    backend = _backend(handler, catalog_refresh_s=0)
    health = await backend.health()
    assert health.available is False
    assert health.reason == "auth_invalid"
    await backend.aclose()


async def test_health_unhealthy_after_poll_sets_cooldown() -> None:
    state = {"up": True}

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("down")
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json=_DEFAULT_MODELS)
        if request.url.path.endswith("/providers"):
            return httpx.Response(200, json=_DEFAULT_PROVIDERS)
        return httpx.Response(404)

    backend = _backend(handler, catalog_refresh_s=0)
    await backend.health()
    snapshot = await backend.usage_snapshot()
    assert snapshot.cooldown_until_ts is None
    state["up"] = False
    await backend.health()
    snapshot = await backend.usage_snapshot()
    assert snapshot.cooldown_until_ts is not None
    await backend.aclose()


async def test_refresh_advertised_models_forces_refetch() -> None:
    poll_count = {"value": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            poll_count["value"] += 1
            if poll_count["value"] == 1:
                return httpx.Response(200, json=_models_payload(_model("openai/model-a0f5", 128_000)))
            return httpx.Response(
                200,
                json=_models_payload(
                    _model("openai/model-a0f5", 128_000),
                    _model("anthropic/model-a0aa", 200_000),
                ),
            )
        if request.url.path.endswith("/providers"):
            return httpx.Response(200, json=_DEFAULT_PROVIDERS)
        return httpx.Response(404)

    backend = _backend(handler, catalog_refresh_s=3600)
    await backend.health()
    assert backend.advertised_models == frozenset({"openai/model-a0f5"})
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"openai/model-a0f5", "anthropic/model-a0aa"})
    await backend.aclose()


async def test_chat_completions_nonstream() -> None:
    handler, records = _direct_handler()
    backend = _backend(handler)
    handle = CallHandle()
    reply = await backend.chat_completions(
        {
            "model": "openai/model-a0f5",
            "reasoning": {"effort": "high"},
            "parallel_tool_calls": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        handle,
    )
    assert reply["choices"][0]["message"]["content"] == "hi there"
    assert handle.upstream_status == 200
    request = records.chat[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer test-key"
    sent = json.loads(request.content)
    assert sent["model"] == "openai/model-a0f5"
    assert sent["stream"] is False
    assert "reasoning" not in sent
    assert "parallel_tool_calls" not in sent
    await backend.aclose()


async def test_chat_completions_401_is_auth_invalid() -> None:
    handler, _records = _direct_handler(chat_status=401)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions(
            {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]},
            handle,
        )
    assert exc_info.value.classification == "auth_invalid"
    assert exc_info.value.status_code == 401
    snapshot = await backend.usage_snapshot()
    assert snapshot.cooldown_until_ts is None
    await backend.aclose()


async def test_chat_completions_429_is_rate_limited() -> None:
    handler, _records = _direct_handler(chat_status=429)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions(
            {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]},
            handle,
        )
    assert exc_info.value.classification == "rate_limited"
    assert exc_info.value.status_code == 429
    await backend.aclose()


async def test_chat_completions_5xx_is_transient() -> None:
    handler, _records = _direct_handler(chat_status=503)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions(
            {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]},
            handle,
        )
    assert exc_info.value.classification == "transient"
    await backend.aclose()


async def test_responses_nonstream() -> None:
    handler, _records = _direct_handler()
    backend = _backend(handler)
    response = await backend.responses({"model": "openai/model-a0f5", "input": "say hi"})
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["usage"]["input_tokens"] == 5
    assert response["usage"]["output_tokens"] == 2
    message = next(item for item in response["output"] if item["type"] == "message")
    assert message["content"][0]["text"] == "hi there"
    await backend.aclose()


async def test_responses_stream_401_is_auth_invalid() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"Hel"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, _records = _direct_handler(chat_sse=chunks, chat_status=401)
    backend = _backend(handler)
    await backend.health()
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _chunk in backend.responses_stream(
            {"model": "openai/model-a0f5", "input": "say hi", "stream": True},
            handle,
        ):
            pass
    assert exc_info.value.classification == "auth_invalid"
    assert handle.upstream_status == 401
    await backend.aclose()


async def test_chat_completions_stream_passthrough() -> None:
    chunks = [b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n', b"data: [DONE]\n\n"]
    handler, records = _direct_handler(chat_sse=chunks)
    backend = _backend(handler)
    handle = CallHandle()
    output = b"".join(
        [
            chunk
            async for chunk in backend.chat_completions_stream(
                {"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]},
                handle,
            )
        ]
    )
    assert b'data: {"id":"x"' in output
    assert b"[DONE]" in output
    assert handle.upstream_status == 200
    assert handle.stream_summary is None
    assert records.stream[0].headers["authorization"] == "Bearer test-key"
    await backend.aclose()


async def test_transport_error_marks_unhealthy() -> None:
    handler, _records = _direct_handler(chat_error=httpx.ConnectError("openrouter down"))
    backend = _backend(handler)
    await backend.health()
    snapshot = await backend.usage_snapshot()
    assert snapshot.cooldown_until_ts is None
    with pytest.raises(BackendError):
        await backend.chat_completions({"model": "openai/model-a0f5", "messages": [{"role": "user", "content": "hi"}]})
    snapshot = await backend.usage_snapshot()
    assert snapshot.cooldown_until_ts is not None
    await backend.aclose()


def test_backend_absent_from_build_when_env_unset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.delenv("CALLOSUM_OPENROUTER_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    configuration = Config()
    backends = build_runtime_backends(
        configuration,
        operator_state=OperatorState(tmp_path / "operator.sqlite"),
    )
    assert all(getattr(backend, "id", None) != "openrouter" for backend in backends)


def test_backend_present_when_env_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.backends.openrouter import OpenRouterBackend
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OPENROUTER_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    configuration = Config()
    backends = build_runtime_backends(
        configuration,
        operator_state=OperatorState(tmp_path / "operator.sqlite"),
    )
    identifiers = [getattr(backend, "id", None) for backend in backends]
    assert "openrouter" in identifiers
    backend = next(backend for backend in backends if getattr(backend, "id", None) == "openrouter")
    assert isinstance(backend, OpenRouterBackend)
    assert backend.kind == "openrouter"
    assert backend._api_key == "test-key"
    import asyncio

    asyncio.run(backend.aclose())


def test_env_exclude_families_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OPENROUTER_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_OPENROUTER_EXCLUDE_FAMILIES", "model-a0g3,model-a0g1")
    configuration = Config()
    backends = build_runtime_backends(
        configuration,
        operator_state=OperatorState(tmp_path / "operator.sqlite"),
    )
    backend = next(backend for backend in backends if getattr(backend, "id", None) == "openrouter")
    assert backend._exclude_families == frozenset({"model-a0g3", "model-a0g1"})
    asyncio.run(backend.aclose())
