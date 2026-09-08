"""Tests for the direct ollama.com Ollama Cloud backend."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends.ollama_cloud import CLOUD_PRIORITY_OFFSET, OllamaCloudBackend
from callosum.errors import BackendError

API_KEY = "test-key"


def _tags_payload(*names: str) -> dict[str, Any]:
    return {"models": [{"name": name} for name in names]}


def _show_payload(
    *,
    capabilities: list[str],
    context_length: int,
    parameter_count: int | None = None,
) -> dict[str, Any]:
    model_info: dict[str, Any] = {"gptoss.context_length": context_length}
    if parameter_count is not None:
        model_info["general.parameter_count"] = parameter_count
    return {"capabilities": capabilities, "model_info": model_info}


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
    tags: dict[str, Any] | Callable[[], dict[str, Any]] | None = None,
    shows: dict[str, dict[str, Any]] | None = None,
    chat_json: dict[str, Any] | None = None,
    chat_sse: list[bytes] | None = None,
    chat_status: int = 200,
    models_status: int = 200,
    chat_error: BaseException | None = None,
) -> tuple[Callable[[httpx.Request], httpx.Response], list[httpx.Request]]:
    tags_default = tags if tags is not None else _tags_payload("model-a0d2:cloud")
    shows = shows or {}
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if path == "/v1/models":
            return httpx.Response(models_status, json={"data": []})
        if path == "/api/tags":
            payload = tags_default() if callable(tags_default) else tags_default
            return httpx.Response(200, json=payload)
        if path == "/api/show":
            name = json.loads(request.content).get("name")
            if name in shows:
                return httpx.Response(200, json=shows[name])
            return httpx.Response(404, json={"error": "not found"})
        if path == "/v1/chat/completions":
            if chat_error is not None:
                raise chat_error
            if chat_status != 200:
                return httpx.Response(chat_status, json={"error": "invalid key"})
            if chat_sse is not None:
                content = b"".join(chat_sse)
                return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})
            return httpx.Response(200, json=chat_json or _chat_reply("model-a0d2:cloud"))
        return httpx.Response(404, json={"error": f"unexpected path {path}"})

    return handler, requests


def _backend(handler: Callable[[httpx.Request], httpx.Response], **kwargs: Any) -> OllamaCloudBackend:
    return OllamaCloudBackend(
        id="ollama-cloud",
        api_key=API_KEY,
        transport=httpx.MockTransport(handler),
        **kwargs,
    )


async def test_advertised_models_empty_before_first_refresh() -> None:
    handler, _ = _direct_handler()
    backend = _backend(handler)
    assert backend.advertised_models == frozenset()
    await backend.aclose()


async def test_default_empty_suffix_accepts_all_catalog_names() -> None:
    handler, _ = _direct_handler(tags=_tags_payload("model-a0d2", "model-a0g2", "model-a0e4"))
    backend = _backend(handler)
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2", "model-a0g2", "model-a0e4"})
    await backend.aclose()


async def test_suffix_filter_keeps_only_matching_models() -> None:
    handler, _ = _direct_handler(
        tags=_tags_payload("model-a0d2:cloud", "model-a0b4", "model-a0f3:cloud", "model-a0d5")
    )
    backend = _backend(handler, model_suffix=":cloud")
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


async def test_catalog_refresh_picks_up_new_models() -> None:
    poll = {"count": 0}

    def tags() -> dict[str, Any]:
        poll["count"] += 1
        if poll["count"] == 1:
            return _tags_payload("model-a0d2:cloud")
        return _tags_payload("model-a0d2:cloud", "model-a0f3:cloud")

    handler, _ = _direct_handler(tags=tags)
    backend = _backend(handler, catalog_refresh_s=0)
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


async def test_refresh_advertised_models_bypasses_ttl() -> None:
    poll = {"count": 0}

    def tags() -> dict[str, Any]:
        poll["count"] += 1
        if poll["count"] == 1:
            return _tags_payload("model-a0d2:cloud")
        return _tags_payload("model-a0d2:cloud", "model-a0f3:cloud")

    handler, _ = _direct_handler(tags=tags)
    backend = _backend(handler, catalog_refresh_s=3600)
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


async def test_model_metadata_marks_cloud_models_remote_band() -> None:
    handler, _ = _direct_handler(tags=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))
    backend = _backend(handler)
    await backend.refresh_advertised_models()
    metadata = backend.model_metadata
    assert set(metadata) == {"model-a0d2:cloud", "model-a0f3:cloud"}
    for item in metadata.values():
        assert item.supported_in_api is True
        assert item.visibility == "list"
        assert item.supported_reasoning_levels == ("default",)
        assert item.priority is not None
        assert CLOUD_PRIORITY_OFFSET <= item.priority < 10_000
    assert metadata["model-a0d2:cloud"].priority == CLOUD_PRIORITY_OFFSET
    assert metadata["model-a0f3:cloud"].priority == CLOUD_PRIORITY_OFFSET + 1
    await backend.aclose()


async def test_cell_capabilities_from_model_detail() -> None:
    handler, _ = _direct_handler(
        tags=_tags_payload("model-a0d2:cloud"),
        shows={
            "model-a0d2:cloud": _show_payload(
                capabilities=["completion", "tools", "vision"],
                context_length=128_000,
                parameter_count=20_000_000_000,
            )
        },
    )
    backend = _backend(handler)
    await backend.refresh_advertised_models()
    capabilities = backend.cell_capabilities("model-a0d2:cloud")
    assert capabilities.cost_rank == 10
    assert capabilities.supports_tools is True
    assert capabilities.modalities == frozenset({"text", "image"})
    assert capabilities.context_window == 128_000
    assert capabilities.parameter_count == 20_000_000_000
    await backend.aclose()


async def test_cell_capabilities_fallback_when_model_detail_missing() -> None:
    handler, _ = _direct_handler(tags=_tags_payload("model-a0d2:cloud"))
    backend = _backend(handler)
    await backend.refresh_advertised_models()
    capabilities = backend.cell_capabilities("model-a0d2:cloud")
    assert capabilities.cost_rank == 10
    assert capabilities.supports_tools is False
    assert capabilities.modalities == frozenset({"text"})
    await backend.aclose()


async def test_cell_capabilities_fallback_before_refresh() -> None:
    handler, _ = _direct_handler()
    backend = _backend(handler)
    capabilities = backend.cell_capabilities("model-a0d2:cloud")
    assert capabilities.cost_rank == 10
    assert capabilities.supports_tools is False
    await backend.aclose()


async def test_health_checks_models_directly_with_api_key() -> None:
    handler, requests = _direct_handler()
    backend = _backend(handler)
    health = await backend.health()
    assert health.available is True
    assert health.reason == "ok"
    assert len(requests) == 1
    assert requests[0].method == "GET"
    assert str(requests[0].url) == "https://ollama.com/v1/models"
    assert requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    await backend.aclose()


async def test_health_reports_no_key_when_api_key_is_empty() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("health must not make a request without an API key")

    backend = OllamaCloudBackend(id="ollama-cloud", api_key="", transport=httpx.MockTransport(handler))
    health = await backend.health()
    assert health.available is False
    assert health.reason == "no-key"
    await backend.aclose()


async def test_health_reports_network_when_request_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = _backend(handler)
    health = await backend.health()
    assert health.available is False
    assert health.reason == "network"
    await backend.aclose()


async def test_health_reports_invalid_key_on_401() -> None:
    handler, _ = _direct_handler(models_status=401)
    backend = _backend(handler)
    health = await backend.health()
    assert health.available is False
    assert health.reason == "auth_invalid"
    await backend.aclose()


async def test_usage_snapshot_is_advisory_when_healthy() -> None:
    handler, _ = _direct_handler()
    backend = _backend(handler)
    await backend.refresh_advertised_models()
    snapshot = await backend.usage_snapshot()
    assert snapshot.remaining_fraction == 1.0
    assert snapshot.weekly_exhausted is False
    assert snapshot.cooldown_until_ts is None
    assert await backend.quota_snapshot() is None
    await backend.aclose()


async def test_transport_failure_sets_short_cooldown_after_successful_refresh() -> None:
    state = {"up": True}
    tags = _tags_payload("model-a0d2:cloud")

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("cloud down")
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        if request.url.path == "/api/show":
            return httpx.Response(404)
        return httpx.Response(200, json=_chat_reply("model-a0d2:cloud"))

    backend = _backend(handler)
    await backend.refresh_advertised_models()
    state["up"] = False
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "model-a0d2:cloud"})
    assert exc_info.value.classification == "transient"
    snapshot = await backend.usage_snapshot()
    assert snapshot.cooldown_until_ts is not None
    await backend.aclose()


async def test_chat_completions_nonstream_direct_request() -> None:
    handler, requests = _direct_handler()
    backend = _backend(handler)
    handle = CallHandle()
    reply = await backend.chat_completions(
        {
            "model": "model-a0d2:cloud",
            "reasoning": {"effort": "high"},
            "parallel_tool_calls": True,
        },
        handle,
    )
    assert reply["choices"][0]["message"]["content"] == "hi there"
    assert handle.upstream_status == 200
    chat_requests = [request for request in requests if request.url.path == "/v1/chat/completions"]
    assert len(chat_requests) == 1
    assert chat_requests[0].method == "POST"
    assert str(chat_requests[0].url) == "https://ollama.com/v1/chat/completions"
    assert chat_requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert chat_requests[0].headers["content-type"] == "application/json"
    assert chat_requests[0].headers["accept"] == "application/json"
    sent = json.loads(chat_requests[0].content)
    assert sent["model"] == "model-a0d2:cloud"
    assert sent["stream"] is False
    assert "reasoning" not in sent
    assert "parallel_tool_calls" not in sent
    await backend.aclose()


async def test_chat_completions_401_is_auth_invalid() -> None:
    handler, _ = _direct_handler(chat_status=401)
    backend = _backend(handler)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        await backend.chat_completions({"model": "model-a0d2:cloud"}, handle)
    assert exc_info.value.classification == "auth_invalid"
    assert exc_info.value.status_code == 401
    assert handle.upstream_status == 401
    await backend.aclose()


async def test_responses_nonstream_translation() -> None:
    handler, _ = _direct_handler()
    backend = _backend(handler)
    response = await backend.responses({"model": "model-a0d2:cloud", "input": "say hi"})
    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert response["usage"]["input_tokens"] == 5
    assert response["usage"]["output_tokens"] == 2
    message = next(item for item in response["output"] if item["type"] == "message")
    assert message["content"][0]["text"] == "hi there"
    await backend.aclose()


async def test_responses_stream_emits_translated_events() -> None:
    chunks = [
        b'data: {"id":"x","model":"model-a0d2:cloud","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, requests = _direct_handler(chat_sse=chunks)
    backend = _backend(handler)
    handle = CallHandle()
    events: list[dict[str, Any]] = []
    async for raw in backend.responses_stream(
        {"model": "model-a0d2:cloud", "input": "say hi", "stream": True}, handle
    ):
        for event_block in raw.split(b"\n\n"):
            for line in event_block.split(b"\n"):
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip().decode()
                if payload and payload != "[DONE]":
                    with contextlib.suppress(json.JSONDecodeError):
                        events.append(json.loads(payload))
    assert [event["type"] for event in events][0] == "response.created"
    deltas = [event for event in events if event["type"] == "response.output_text.delta"]
    assert "".join(event["delta"] for event in deltas) == "Hello"
    completed = [event for event in events if event["type"] == "response.completed"]
    assert len(completed) == 1
    assert completed[0]["response"]["usage"]["input_tokens"] == 3
    assert completed[0]["response"]["usage"]["output_tokens"] == 2
    assert handle.stream_summary is not None
    assert handle.stream_summary.completed_response is not None
    assert handle.stream_summary.completed_response["usage"]["input_tokens"] == 3
    chat_requests = [request for request in requests if request.url.path == "/v1/chat/completions"]
    assert len(chat_requests) == 1
    assert chat_requests[0].headers["authorization"] == f"Bearer {API_KEY}"
    assert chat_requests[0].headers["accept"] == "text/event-stream"
    await backend.aclose()


async def test_responses_stream_401_is_auth_invalid() -> None:
    handler, _ = _direct_handler(chat_sse=[b'data: {"error":"auth"}\n\n'], chat_status=401)
    backend = _backend(handler)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.responses_stream(
            {"model": "model-a0d2:cloud", "input": "say hi", "stream": True}, handle
        ):
            pass
    assert exc_info.value.classification == "auth_invalid"
    assert handle.upstream_status == 401
    await backend.aclose()


async def test_chat_completions_stream_passthrough() -> None:
    chunks = [
        b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, _ = _direct_handler(chat_sse=chunks)
    backend = _backend(handler)
    handle = CallHandle()
    output = b"".join(
        [chunk async for chunk in backend.chat_completions_stream({"model": "model-a0d2:cloud"}, handle)]
    )
    assert b'data: {"id":"x"' in output
    assert b"[DONE]" in output
    assert handle.upstream_status == 200
    assert handle.stream_summary is None
    await backend.aclose()


async def test_chat_completions_stream_401_is_auth_invalid() -> None:
    handler, _ = _direct_handler(chat_sse=[b'data: {"error":"auth"}\n\n'], chat_status=401)
    backend = _backend(handler)
    handle = CallHandle()
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.chat_completions_stream({"model": "model-a0d2:cloud"}, handle):
            pass
    assert exc_info.value.classification == "auth_invalid"
    assert handle.upstream_status == 401
    await backend.aclose()


def test_backend_absent_from_build_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    backends = build_runtime_backends(Config(), operator_state=OperatorState(tmp_path / "state.sqlite"))
    assert all(backend.id != "ollama-cloud" for backend in backends)


def test_backend_present_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_OLLAMA_CLOUD_API_KEY", API_KEY)
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    backends = build_runtime_backends(Config(), operator_state=OperatorState(tmp_path / "state.sqlite"))
    cloud = next(backend for backend in backends if backend.id == "ollama-cloud")
    assert isinstance(cloud, OllamaCloudBackend)
    assert cloud.kind == "ollama_cloud"
    assert cloud._api_key == API_KEY
