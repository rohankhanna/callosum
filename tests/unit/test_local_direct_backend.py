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
from callosum.backends._ollama_capabilities import OllamaShowCapabilities
from callosum.backends.local_direct import LocalModelRegistryBackend
from callosum.cell_grid import Cell
from callosum.errors import BackendError
from callosum.local import CapabilityRow, ModelEntry
from callosum.routing.capability import CapabilityFilter
from callosum.routing.local_performance import LocalPerformanceRegime
from callosum.routing.protocols import PromptFeatures

pytestmark = pytest.mark.anyio


class _FakeSource:
    """LocalModelRegistrySource stub. The backend only consumes .models(force)
    and .capabilities(force) so a tiny shim suffices."""

    def __init__(
        self,
        entries: list[ModelEntry],
        caps: dict[str, CapabilityRow] | None = None,
    ) -> None:
        self._entries = entries
        self._caps = caps or {}

    def models(self, *, force: bool = False) -> list[ModelEntry]:
        return list(self._entries)

    def capabilities(self, *, force: bool = False) -> dict[str, CapabilityRow]:
        return dict(self._caps)


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


# ---------------------------------------------------------------------------
# cell_capabilities — three-tier, per-field precedence
# (tier 1 hub-canonical > tier 2 ollama /api/show stopgap > tier 3 defaults)
# ---------------------------------------------------------------------------


def _hub_row(
    model_id: str,
    *,
    modalities: frozenset[str] | None = None,
    supports_tools: bool | None = None,
) -> CapabilityRow:
    return CapabilityRow(
        model_id=model_id,
        quantization_label=None,
        runnable_on_host=None,
        status=None,
        estimated_tokens_per_second=None,
        pool_bytes=None,
        fit_limit_tokens=None,
        prefill_ms_per_token=None,
        decode_bandwidth_kappa=None,
        modalities=modalities,
        supports_tools=supports_tools,
    )


def _ollama_stopgap(
    *,
    modalities: frozenset[str] = frozenset({"text"}),
    supports_tools: bool = True,
    context_window: int = 128_000,
    parameter_count: int | None = None,
) -> OllamaShowCapabilities:
    return OllamaShowCapabilities(
        modalities=modalities,
        supports_tools=supports_tools,
        context_window=context_window,
        parameter_count=parameter_count,
    )


def test_cell_capabilities_tier3_defaults_when_no_stopgap_no_hub() -> None:
    """With neither a tier-2 stopgap cache entry nor a tier-1 hub row, the
    cell reports the conservative defaults — text-only, optimistic tools,
    the roster's context window — and local-perf fields flow from the entry."""
    src = _FakeSource([_catalog_entry("m", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)
    caps = backend.cell_capabilities("m")
    assert caps.modalities == frozenset({"text"})
    assert caps.supports_tools is True
    assert caps.context_window == 128_000
    assert caps.local_throughput_tps == 25.0
    assert caps.local_catalog_admitted is True


def test_cell_capabilities_tier2_modalities_stopgap_default_mode() -> None:
    """Default env (modalities): a tier-2 /api/show vision self-report is
    applied to modalities (strictly additive), but a tools=False claim is
    NOT applied — tools stay optimistic (probe-revocable). The roster's
    context window is not clobbered by the self-report."""
    src = _FakeSource([_catalog_entry("m", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["m"] = _ollama_stopgap(
        modalities=frozenset({"text", "image"}), supports_tools=False, context_window=200_000
    )
    caps = backend.cell_capabilities("m")
    assert caps.modalities == frozenset({"text", "image"})
    assert caps.supports_tools is True  # tools NOT applied in modalities mode
    assert caps.context_window == 128_000  # roster value preserved (risk-#2 guard)


def test_cell_capabilities_tier2_all_mode_applies_tool_accuracy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env `all`: the tier-2 tools claim is applied too — a self-reported
    supports_tools=False flows through. This is operator-gated because the
    probe is one-directional (probe-fail revokes, probe-pass cannot grant)."""
    monkeypatch.setenv("CALLOSUM_LOCAL_CAPABILITIES_STOPGAP", "all")
    src = _FakeSource([_catalog_entry("m", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["m"] = _ollama_stopgap(
        modalities=frozenset({"text", "image"}), supports_tools=False, context_window=200_000
    )
    caps = backend.cell_capabilities("m")
    assert caps.supports_tools is False
    assert caps.modalities == frozenset({"text", "image"})


def test_cell_capabilities_tier2_off_mode_ignores_stopgap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Env `off`: the tier-2 stopgap is ignored entirely — modalities and tools
    fall back to tier-3 defaults regardless of what the cache holds."""
    monkeypatch.setenv("CALLOSUM_LOCAL_CAPABILITIES_STOPGAP", "off")
    src = _FakeSource([_catalog_entry("m", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["m"] = _ollama_stopgap(modalities=frozenset({"text", "image"}), supports_tools=False)
    caps = backend.cell_capabilities("m")
    assert caps.modalities == frozenset({"text"})
    assert caps.supports_tools is True


def test_cell_capabilities_tier2_context_window_fills_only_when_entry_lacks_it() -> None:
    """Risk-#2 guard: an ollama self-reported context window never clobbers a
    measured roster value; it fills only when the roster entry is None."""
    # Roster has a value -> self-report ignored.
    src_known = _FakeSource([_catalog_entry("m-known", throughput=25.0)])
    backend_known = LocalModelRegistryBackend(id="test", source=src_known)
    backend_known._capabilities_cache["m-known"] = _ollama_stopgap(context_window=200_000)
    assert backend_known.cell_capabilities("m-known").context_window == 128_000

    # Roster lacks a value -> self-report fills it.
    src_unknown = _FakeSource(
        [
            ModelEntry(
                id="m-unknown",
                endpoint="http://127.0.0.1:0",
                runtime="ollama",
                runtime_model="m-unknown",
                family="test",
                context_window=None,
                api_surfaces=("responses",),
                enabled=True,
            )
        ]
    )
    backend_unknown = LocalModelRegistryBackend(id="test", source=src_unknown)
    backend_unknown._capabilities_cache["m-unknown"] = _ollama_stopgap(context_window=200_000)
    assert backend_unknown.cell_capabilities("m-unknown").context_window == 200_000


def test_cell_capabilities_tier1_hub_canonical_wins_per_field_over_stopgap() -> None:
    """When the hub emits a field, it is canonical truth and wins over the
    tier-2 stopgap PER FIELD. Hub says text-only + no-tools; stopgap says
    vision + tools -> hub wins both (text, tools=False)."""
    src = _FakeSource(
        [_catalog_entry("m", throughput=25.0)],
        caps={"m": _hub_row("m", modalities=frozenset({"text"}), supports_tools=False)},
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["m"] = _ollama_stopgap(modalities=frozenset({"text", "image"}), supports_tools=True)
    caps = backend.cell_capabilities("m")
    assert caps.modalities == frozenset({"text"})
    assert caps.supports_tools is False


def test_cell_capabilities_tier1_hub_silent_field_falls_through_to_stopgap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-field precedence: when the hub emits modalities but is silent on
    supports_tools (None), tools come from the tier-2 stopgap (which only
    applies under `all`). Hub modalities win; stopgap tools fill the gap."""
    monkeypatch.setenv("CALLOSUM_LOCAL_CAPABILITIES_STOPGAP", "all")
    src = _FakeSource(
        [_catalog_entry("m", throughput=25.0)],
        caps={"m": _hub_row("m", modalities=frozenset({"text"}), supports_tools=None)},
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["m"] = _ollama_stopgap(modalities=frozenset({"text", "image"}), supports_tools=True)
    caps = backend.cell_capabilities("m")
    assert caps.modalities == frozenset({"text"})  # hub canonical
    assert caps.supports_tools is True  # tier-2 stopgap fills the hub-silent field


def test_cell_capabilities_preserves_local_perf_fields_across_tiers() -> None:
    """Local-performance fields (throughput/quant/admission) ALWAYS come from the
    roster entry, never from tiers 1 or 2 — even when both tiers are active."""
    src = _FakeSource(
        [_catalog_entry("m", throughput=25.0, quantization="bf16", status="working")],
        caps={"m": _hub_row("m", modalities=frozenset({"text"}), supports_tools=False)},
    )
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["m"] = _ollama_stopgap(modalities=frozenset({"text", "image"}), supports_tools=False)
    caps = backend.cell_capabilities("m")
    assert caps.local_throughput_tps == 25.0
    assert caps.local_gpu_seconds_per_token == 0.04
    assert caps.local_quantization == "bf16"
    assert caps.local_status == "working"
    assert caps.local_catalog_admitted is True


def test_cell_capabilities_unknown_model_returns_conservative_fallback() -> None:
    """A model id absent from the roster gets the unknown-model fallback —
    text-only, supports_tools=False (not routable), flagged unknown."""
    src = _FakeSource([_catalog_entry("m", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)
    caps = backend.cell_capabilities("missing")
    assert caps.modalities == frozenset({"text"})
    assert caps.supports_tools is False
    assert caps.local_catalog_admitted is False
    assert "unknown-model" in caps.local_admission_reasons


async def test_refresh_capabilities_populates_cache_for_ollama_only() -> None:
    """_refresh_capabilities probes only ollama runtimes (only ollama exposes
    /api/show). A vllm cell on the same backend is skipped and falls to tier 1
    or 3. The cache is keyed by model id."""
    entries = [
        ModelEntry(
            id="ollama-vision",
            endpoint="http://127.0.0.1:11434",
            runtime="ollama",
            runtime_model="model-a0g1-vision",
            family="test",
            context_window=128_000,
            api_surfaces=("responses",),
            enabled=True,
        ),
        ModelEntry(
            id="vllm-cell",
            endpoint="http://127.0.0.1:8000",
            runtime="vllm",
            runtime_model="vllm-cell",
            family="test",
            context_window=128_000,
            api_surfaces=("responses",),
            enabled=True,
        ),
    ]
    src = _FakeSource(entries)

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("name") == "model-a0g1-vision":
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "tools", "vision"],
                    "model_info": {"model-a0g1.context_length": 131072, "general.parameter_count": 8000000000},
                },
            )
        # A vllm cell should never be probed; if it is, surface it loudly.
        if body.get("name") == "vllm-cell":
            return httpx.Response(200, json={"capabilities": ["completion"]})
        return httpx.Response(404)

    backend = LocalModelRegistryBackend(id="test", source=src, transport=httpx.MockTransport(handler))
    await backend.refresh_advertised_models()
    assert "ollama-vision" in backend._capabilities_cache
    assert "vllm-cell" not in backend._capabilities_cache
    cached = backend._capabilities_cache["ollama-vision"]
    assert cached.modalities == frozenset({"text", "image"})
    assert cached.supports_tools is True
    assert cached.context_window == 131_072
    await backend.aclose()


async def test_refresh_capabilities_failure_yields_tier3_defaults() -> None:
    """When /api/show fails (non-200 / bad JSON / transport error), the cache
    stays empty for that cell and cell_capabilities falls back to tier-3
    defaults — never raises, never blocks the refresh tick."""
    src = _FakeSource(
        [
            ModelEntry(
                id="bad-cell",
                endpoint="http://127.0.0.1:11434",
                runtime="ollama",
                runtime_model="bad-cell",
                family="test",
                context_window=128_000,
                api_surfaces=("responses",),
                enabled=True,
            )
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    backend = LocalModelRegistryBackend(id="test", source=src, transport=httpx.MockTransport(handler))
    await backend.refresh_advertised_models()
    assert "bad-cell" not in backend._capabilities_cache
    caps = backend.cell_capabilities("bad-cell")
    assert caps.modalities == frozenset({"text"})  # tier-3 default
    assert caps.supports_tools is True
    await backend.aclose()


def test_vision_request_routes_to_local_vision_cell_through_capability_filter() -> None:
    """Headline behavior: a vision request (modalities={text,image}) against a
    local ollama cell whose tier-2 /api/show stopgap reports vision PASSES the
    hard CapabilityFilter. Today (no stopgap) such a cell reports text-only and
    is dropped -> empty pool -> HTTP 400. The modality stopgap is strictly
    additive: zero regression for the text-only routing that works today."""
    src = _FakeSource([_catalog_entry("vision-cell", throughput=25.0)])
    backend = LocalModelRegistryBackend(id="test", source=src)
    backend._capabilities_cache["vision-cell"] = _ollama_stopgap(modalities=frozenset({"text", "image"}))
    cell = Cell(model="vision-cell", reasoning_effort="default")

    capabilities_of = {cell: backend.cell_capabilities(cell.model)}
    filt = CapabilityFilter(capabilities_of=capabilities_of.__getitem__)

    vision_features = PromptFeatures(
        text="describe this image",
        tokens=10,
        modalities=frozenset({"text", "image"}),
        needs_tools=False,
    )
    assert filt.filter([cell], vision_features) == [cell]

    # A text-only cell (no stopgap) is still dropped for a vision request.
    text_only = Cell(model="text-cell", reasoning_effort="default")
    caps_of = {text_only: backend.cell_capabilities("text-cell")}
    assert caps_of[text_only].modalities == frozenset({"text"})
    filt_text = CapabilityFilter(capabilities_of=caps_of.__getitem__)
    assert filt_text.filter([text_only], vision_features) == []


async def test_health_ok_when_models_present() -> None:
    src = _FakeSource([_entry("m1", ("responses",))])
    backend = LocalModelRegistryBackend(id="test", source=src)
    h = await backend.health()
    assert h.available is True
    assert h.reason == "ok"
    await backend.aclose()


async def test_health_catalog_empty_when_cli_healthy_but_garage_empty() -> None:
    # CLI ran clean (ok) but listed 0 models — the garage is empty, not the CLI.
    src = _FakeSource([])
    src.last_fetch_reason = "ok"  # type: ignore[attr-defined]
    backend = LocalModelRegistryBackend(id="test", source=src)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "catalog_empty"
    await backend.aclose()


async def test_health_catalog_cli_broken_when_fetch_failed() -> None:
    # The catalog CLI failed (broken) — surface the real cause, not "unknown".
    src = _FakeSource([])
    src.last_fetch_reason = "broken"  # type: ignore[attr-defined]
    backend = LocalModelRegistryBackend(id="test", source=src)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "catalog_cli_broken"
    await backend.aclose()


async def test_health_unknown_when_source_has_no_reason_attr() -> None:
    # A source stub without last_fetch_reason (legacy/other impls) falls back
    # to the opaque "unknown" rather than claiming the garage is empty.
    src = _FakeSource([])
    backend = LocalModelRegistryBackend(id="test", source=src)
    h = await backend.health()
    assert h.available is False
    assert h.reason == "unknown"
    await backend.aclose()
