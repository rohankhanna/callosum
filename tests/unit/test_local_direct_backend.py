"""Tests for LocalModelRegistryBackend's filtering of chat-only cells.

Background: callosum's primary inbound path is /v1/responses (Codex
CLI's protocol). Chat-only cells on this backend can't serve streaming
responses because no chat→responses stream translator is wired here.
Advertising them anyway causes wasted dispatch retries — the router
picks them as primary, the request 502s instantly, then dispatch
falls through to a `-responses-proxy` sibling on the same backend.
The filter at the source side removes the noise without losing any
functional capability.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends.local_direct import LocalModelRegistryBackend
from callosum.errors import BackendError
from callosum.local import ModelEntry
from callosum.routing.local_performance import LocalPerformanceRegime

pytestmark = pytest.mark.anyio


class _FakeSource:
    """LocalModelRegistrySource stub. The backend only consumes .models(force)
    so a tiny shim suffices."""

    def __init__(self, entries: list[ModelEntry]) -> None:
        self._entries = entries

    def models(self, *, force: bool = False) -> list[ModelEntry]:
        return list(self._entries)


def _entry(model_id: str, surfaces: tuple[str, ...]) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        endpoint="http://127.0.0.1:0",
        runtime="ollama",
        runtime_model=model_id,
        family="test",
        context_window=128_000,
        api_surfaces=surfaces,
        enabled=True,
    )


def _catalog_entry(
    model_id: str,
    *,
    quantization: str | None = "bf16",
    runnable_on_host: bool | None = True,
    status: str | None = "working",
    throughput: float | None = 20.0,
    surfaces: tuple[str, ...] = ("responses",),
) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        endpoint="http://127.0.0.1:0",
        runtime="ollama",
        runtime_model=model_id,
        family="test",
        context_window=128_000,
        api_surfaces=surfaces,
        enabled=True,
        local_quantization=quantization,
        local_runnable_on_host=runnable_on_host,
        local_status=status,
        estimated_tokens_per_second=throughput,
    )


def test_advertised_models_drops_chat_only_cells() -> None:
    """A chat-only sibling (api_surfaces == ("chat",)) is filtered out
    even when present in the source — the router never picks it, so
    the 502-and-retry waste is avoided."""
    src = _FakeSource(
        [
            _entry("model-a0a9", ("chat",)),
            _entry("model-a0a1", ("responses",)),
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    advertised = backend.advertised_models
    assert "model-a0a1" in advertised
    assert "model-a0a9" not in advertised


def test_advertised_models_keeps_cells_with_both_surfaces() -> None:
    """A cell that advertises both surfaces (chat AND responses) is
    serveable on the /v1/responses path and stays advertised."""
    src = _FakeSource(
        [
            _entry("dual-surface", ("chat", "responses")),
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    assert "dual-surface" in backend.advertised_models


def test_advertised_models_drops_cells_with_unknown_surfaces() -> None:
    """Defensive: a cell whose api_surfaces doesn't include 'responses'
    (e.g. empty tuple from a malformed registry entry) is filtered out
    rather than silently routed to."""
    src = _FakeSource(
        [
            _entry("malformed", ()),
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    assert backend.advertised_models == frozenset()


def test_model_metadata_filtered_same_as_advertised_models() -> None:
    """model_metadata is the cell-grid merger's hook. It must match
    advertised_models in filter behavior — otherwise the grid would
    include cells the router would then refuse to advertise."""
    src = _FakeSource(
        [
            _entry("chat-only", ("chat",)),
            _entry("responses-capable", ("responses",)),
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    meta = backend.model_metadata
    assert "responses-capable" in meta
    assert "chat-only" not in meta


def test_curated_local_models_exposes_admitted_and_rejected_view() -> None:
    src = _FakeSource(
        [
            _catalog_entry("good"),
            _catalog_entry("quantized", quantization="q4_k_m"),
            _catalog_entry("chat-only", surfaces=("chat",)),
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)

    curated = {item.id: item for item in backend.curated_local_models()}

    assert curated["good"].admitted is True
    assert curated["good"].reasons == ()
    assert curated["quantized"].admitted is False
    assert curated["quantized"].reasons == ("non-training-precision:q4_k_m",)
    assert curated["chat-only"].admitted is False
    assert curated["chat-only"].reasons == ("no-responses-surface",)


def test_admitted_local_model_ids_and_reasons_honor_throughput_floor() -> None:
    src = _FakeSource(
        [
            _catalog_entry("fast", throughput=22.0),
            _catalog_entry("slow", throughput=4.0),
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)

    assert backend.admitted_local_model_ids(min_tokens_per_second=10.0) == frozenset({"fast"})
    assert backend.local_model_admission_reasons("fast", min_tokens_per_second=10.0) == ()
    assert backend.local_model_admission_reasons("slow", min_tokens_per_second=10.0) == (
        "throughput-below-floor:4.000",
    )
    assert backend.local_model_admission_reasons("missing") == ("unknown-model",)


def test_local_performance_model_uses_hub_evidence() -> None:
    src = _FakeSource(
        [
            ModelEntry(
                id="responses-capable",
                endpoint="http://127.0.0.1:0",
                runtime="ollama",
                runtime_model="responses-capable",
                family="test",
                context_window=128_000,
                api_surfaces=("responses",),
                enabled=True,
                local_quantization="bf16",
                local_runnable_on_host=True,
                local_status="working",
                estimated_tokens_per_second=20.0,
                local_pool_bytes=24 * 1024**3,
                local_fit_limit_tokens=8192,
                local_prefill_ms_per_token=2.0,
                local_decode_bandwidth_kappa=1.25,
            )
        ]
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    model = backend.local_performance_model("responses-capable")
    assert model is not None
    assert model.fit_limit_tokens() == 8192
    assert model.regime_for(1_000) == LocalPerformanceRegime.UNDERUTILIZED
    assert model.regime_for(8_150) == LocalPerformanceRegime.POOL_EDGE


def test_cell_capabilities_expose_local_gpu_opportunity_cost() -> None:
    src = _FakeSource([_catalog_entry("fast", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)

    caps = backend.cell_capabilities("fast")

    assert caps.local_throughput_tps == 25.0
    assert caps.local_gpu_seconds_per_token == 0.04


async def test_responses_stream_forwards_large_tool_request_no_size_cap() -> None:
    # A large tool request is no longer pre-rejected on a byte count — whether
    # it fits is the model's context window's call, and a stall is caught
    # behaviorally by stall_guarded, not by guessing at size. So a big body
    # must actually reach the upstream (here a 500), surfacing as transient
    # rather than a 413 raised before send.
    src = _FakeSource([_entry("responses-capable", ("responses",))])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    backend = LocalModelRegistryBackend(
        id="test",
        source=src,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendError) as exc_info:
        async for _ in backend.responses_stream(
            {
                "model": "responses-capable",
                "input": "x" * 300_000,
                "stream": True,
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            }
        ):
            pass
    assert exc_info.value.status_code != 413
    assert "/v1/responses" in calls
    await backend.aclose()


async def test_chat_completions_stream_translates_responses_native_text_sse() -> None:
    src = _FakeSource([_entry("responses-capable", ("responses", "chat"))])
    captured: dict[str, Any] = {}
    upstream = b"".join(
        [
            _responses_event(
                "response.created",
                {"response": {"id": "resp_1", "model": "runtime-model"}},
            ),
            _responses_event("response.output_text.delta", {"delta": "Hel"}),
            _responses_event("response.output_text.delta", {"delta": "lo"}),
            _responses_event(
                "response.completed",
                {
                    "response": {
                        "id": "resp_1",
                        "model": "runtime-model",
                        "status": "completed",
                        "usage": {
                            "input_tokens": 3,
                            "output_tokens": 2,
                            "total_tokens": 5,
                        },
                    }
                },
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=upstream,
            headers={"content-type": "text/event-stream"},
        )

    backend = LocalModelRegistryBackend(
        id="test",
        source=src,
        transport=httpx.MockTransport(handler),
    )
    handle = CallHandle()
    chunks = [
        c
        async for c in backend.chat_completions_stream(
            {
                "model": "responses-capable",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
            handle,
        )
    ]
    events = _chat_events(chunks)
    assert captured["path"] == "/v1/responses"
    assert captured["body"]["stream"] is True
    assert captured["body"]["model"] == "responses-capable"
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert [e["choices"][0]["delta"] for e in events[:3]] == [
        {"role": "assistant"},
        {"content": "Hel"},
        {"content": "lo"},
    ]
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    assert events[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert handle.stream_summary is not None
    completed = handle.stream_summary.completed_response
    assert completed is not None
    assert completed["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    await backend.aclose()


async def test_chat_completions_stream_translates_responses_tool_arguments() -> None:
    src = _FakeSource([_entry("responses-capable", ("responses", "chat"))])
    upstream = b"".join(
        [
            _responses_event(
                "response.output_item.added",
                {
                    "output_index": 0,
                    "item": {
                        "type": "function_call",
                        "call_id": "call_1",
                        "name": "shell",
                    },
                },
            ),
            _responses_event(
                "response.function_call_arguments.delta",
                {"output_index": 0, "delta": '{"cm'},
            ),
            _responses_event(
                "response.function_call_arguments.delta",
                {"output_index": 0, "delta": 'd":"ls"}'},
            ),
            _responses_event(
                "response.completed",
                {"response": {"id": "resp_1", "status": "completed"}},
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=upstream,
            headers={"content-type": "text/event-stream"},
        )

    backend = LocalModelRegistryBackend(
        id="test",
        source=src,
        transport=httpx.MockTransport(handler),
    )
    events = _chat_events(
        [
            c
            async for c in backend.chat_completions_stream(
                {
                    "model": "responses-capable",
                    "messages": [{"role": "user", "content": "run ls"}],
                    "stream": True,
                }
            )
        ]
    )
    tool_deltas = [
        e["choices"][0]["delta"]["tool_calls"][0] for e in events if "tool_calls" in e["choices"][0]["delta"]
    ]
    assert tool_deltas[0]["function"] == {"name": "shell", "arguments": ""}
    assert tool_deltas[1]["function"]["arguments"] == '{"cm'
    assert tool_deltas[2]["function"]["arguments"] == 'd":"ls"}'
    assert events[-1]["choices"][0]["finish_reason"] == "stop"
    await backend.aclose()


async def test_chat_completions_stream_requests_usage_from_chat_surface() -> None:
    src = _FakeSource([_entry("chat-only", ("chat",))])
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=(
                b'data: {"id":"cmpl_1","choices":[{"delta":{"content":"hi"}}]}\n\n'
                b'data: {"id":"cmpl_1","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
                b"data: [DONE]\n\n"
            ),
            headers={"content-type": "text/event-stream"},
        )

    backend = LocalModelRegistryBackend(
        id="test",
        source=src,
        transport=httpx.MockTransport(handler),
    )
    handle = CallHandle()
    events = _chat_events(
        [
            c
            async for c in backend.chat_completions_stream(
                {
                    "model": "chat-only",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
                handle,
            )
        ]
    )
    assert captured["path"] == "/v1/chat/completions"
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert events[-1]["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    assert handle.stream_summary is not None
    completed = handle.stream_summary.completed_response
    assert completed is not None
    assert completed["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    await backend.aclose()


def _responses_event(event_type: str, payload: dict[str, Any]) -> bytes:
    data = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode()


def _chat_events(chunks: list[bytes]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for raw_event in b"".join(chunks).split(b"\n\n"):
        for line in raw_event.split(b"\n"):
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:") :].strip()
            if not data or data == b"[DONE]":
                continue
            events.append(json.loads(data))
    return events


async def test_responses_forwards_large_tool_request_no_size_cap() -> None:
    # Non-stream counterpart: a large tool request reaches upstream instead of
    # being pre-rejected with 413 on a byte count.
    src = _FakeSource([_entry("responses-capable", ("responses",))])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500)

    backend = LocalModelRegistryBackend(
        id="test",
        source=src,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendError) as exc_info:
        await backend.responses(
            {
                "model": "responses-capable",
                "input": "x" * 300_000,
                "tools": [{"type": "function", "function": {"name": "shell"}}],
            }
        )
    assert exc_info.value.status_code != 413
    assert "/v1/responses" in calls
    await backend.aclose()
