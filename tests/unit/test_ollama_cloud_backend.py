"""Tests for the Ollama Cloud backend (sub-slices 1 + 2).

The backend talks to the local ollama daemon (localhost:11434) that holds the
Ollama Cloud auth under `ollama signin`. Callosum holds NO credential. Sub-slice
1 covers catalog discovery from `/api/tags` (with the `:cloud` suffix filter
that partitions local vs. cloud), `ModelMetadata` synthesis (so cloud models
join the cell grid in the REMOTE band), per-model capabilities from `/api/show`,
honest-advisory `usage_snapshot` (NOT the local free-stub), health, and refresh.
Sub-slice 2 covers the four dispatch methods — chat-completions (non-stream +
raw stream) and responses (non-stream + translated SSE stream) — which hit the
daemon's OpenAI-compatible `/v1/chat/completions` with no `Authorization` header
(the daemon injects cloud auth upstream), reusing the shared Responses↔Chat
translators + chat→Responses streaming generator extracted to
`callosum.backends._responses_chat`.

Routing-classification is asserted here at the catalog/metadata level
(remote-band priority, remote `cost_rank`) and in the integration fixtures
(`test_routing_mode_matrix`, `test_selector_routing`) at the lane level.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends.ollama_cloud import (
    CLOUD_PRIORITY_OFFSET,
    OllamaCloudBackend,
)
from callosum.errors import BackendError


def _tags_payload(*names: str) -> dict[str, Any]:
    return {"models": [{"name": n} for n in names]}


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


def _daemon_handler(
    tags: dict[str, Any],
    shows: dict[str, dict[str, Any]] | None = None,
) -> Any:
    """Route /api/tags and /api/show to canned payloads. `shows` maps model
    name → /api/show response; a model absent from `shows` gets a 404 so the
    capabilities fallback path is exercised."""
    shows = shows or {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/tags":
            return httpx.Response(200, json=tags)
        if request.url.path == "/api/show":
            try:
                body = request.read()
                import json

                name = json.loads(body).get("name")
            except Exception:
                return httpx.Response(400)
            if name in shows:
                return httpx.Response(200, json=shows[name])
            return httpx.Response(404)
        return httpx.Response(404)

    return handler


def _chat_reply(model: str) -> dict[str, Any]:
    """A minimal chat-completions non-stream reply with `usage` in the
    prompt_tokens/completion_tokens shape the ollama daemon returns (Approach A
    — OpenAI-compatible /v1/chat/completions), which `_extract_tokens` parses
    directly and `_chat_to_responses_response` maps to Responses shape."""
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


def _dispatch_handler(
    *,
    tags: dict[str, Any] | None = None,
    shows: dict[str, dict[str, Any]] | None = None,
    chat_json: dict[str, Any] | None = None,
    chat_sse: list[bytes] | None = None,
    chat_error: BaseException | None = None,
) -> tuple[Any, list[httpx.Request]]:
    """Route /api/tags, /api/show, AND /v1/chat/completions to canned payloads.

    Captures every /v1/chat/completions request so dispatch tests can assert the
    `:cloud` model was forwarded and Codex-only fields were stripped before the
    POST. `chat_json` is the non-stream reply (defaults to a canned `_chat_reply`);
    `chat_sse` is the streamed reply (chunks joined into one SSE body);
    `chat_error` is raised on the chat endpoint to exercise the transport-error
    path (the `/api/tags` poll still succeeds so the backend starts healthy).
    """
    tags = tags or _tags_payload("model-a0d2:cloud")
    shows = shows or {}
    chat_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/tags":
            return httpx.Response(200, json=tags)
        if path == "/api/show":
            try:
                name = json.loads(request.read()).get("name")
            except Exception:
                return httpx.Response(400)
            if name in shows:
                return httpx.Response(200, json=shows[name])
            return httpx.Response(404)
        if path == "/v1/chat/completions":
            chat_requests.append(request)
            if chat_error is not None:
                raise chat_error
            if chat_sse is not None:
                return httpx.Response(
                    200,
                    content=b"".join(chat_sse),
                    headers={"content-type": "text/event-stream"},
                )
            return httpx.Response(200, json=chat_json or _chat_reply("model-a0d2:cloud"))
        return httpx.Response(404)

    return handler, chat_requests


# ---------- catalog discovery + suffix filter ------------------------------


async def test_advertised_models_empty_before_first_poll() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    assert backend.advertised_models == frozenset()
    h = await backend.health()  # forces a refresh
    assert h.available is True
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.aclose()


async def test_suffix_filter_keeps_only_cloud_models() -> None:
    """The local/cloud partition: only `:cloud`-suffixed models are cloud-
    metered; bare local models stay on the LocalModelRegistry / litellm_gateway path
    and must NOT appear in this backend's catalog."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200,
                json=_tags_payload("model-a0d2:cloud", "model-a0b4", "model-a0f3:cloud", "model-a0d5"),
            )
        ),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


async def test_custom_suffix_filter() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        model_suffix="-remote",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(
                200, json=_tags_payload("model-a0d2-remote", "model-a0e4:cloud")
            )
        ),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2-remote"})
    await backend.aclose()


async def test_catalog_picks_up_new_models_on_refresh() -> None:
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))
        return httpx.Response(200, json=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=0,  # always refresh
        transport=httpx.MockTransport(handler),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


# ---------- model_metadata synthesis (remote band) --------------------------


async def test_model_metadata_marks_cloud_models_remote_band() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))
        ),
    )
    await backend.health()
    md = backend.model_metadata
    assert set(md.keys()) == {"model-a0d2:cloud", "model-a0f3:cloud"}
    for _slug, m in md.items():
        assert m.supported_in_api is True
        assert m.visibility == "list"
        assert m.supported_reasoning_levels == ("default",)
        # Remote band: above curated Codex (tens), below free local (10_000).
        assert m.priority is not None and m.priority >= CLOUD_PRIORITY_OFFSET
        assert m.priority < 10_000
    # Ordering preserved: first cataloged model gets the base offset.
    assert md["model-a0d2:cloud"].priority == CLOUD_PRIORITY_OFFSET
    assert md["model-a0f3:cloud"].priority == CLOUD_PRIORITY_OFFSET + 1
    await backend.aclose()


# ---------- /api/show → cell_capabilities -----------------------------------


async def test_cell_capabilities_from_show() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            _daemon_handler(
                _tags_payload("model-a0d2:cloud"),
                shows={
                    "model-a0d2:cloud": _show_payload(
                        capabilities=["completion", "tools", "vision"],
                        context_length=128_000,
                        parameter_count=20_000_000_000,
                    )
                },
            )
        ),
    )
    await backend.health()  # triggers _refresh_capabilities
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10  # remote default
    assert caps.supports_tools is True
    assert "text" in caps.modalities
    assert "image" in caps.modalities
    assert caps.context_window == 128_000
    assert caps.parameter_count == 20_000_000_000
    await backend.aclose()


async def test_cell_capabilities_fallback_when_show_missing() -> None:
    """A model in the catalog whose /api/show 404s (or hasn't been probed yet)
    falls back to conservative text-only/no-tools defaults rather than
    erroring — the request can still dispatch."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(
            _daemon_handler(_tags_payload("model-a0d2:cloud"), shows={})
        ),
    )
    await backend.health()
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10
    assert caps.supports_tools is False
    assert caps.modalities == frozenset({"text"})
    await backend.aclose()


async def test_cell_capabilities_fallback_before_first_refresh() -> None:
    """Synchronous cell_capabilities called before any catalog refresh
    returns the safe default (no crash, no stale state)."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    caps = backend.cell_capabilities("model-a0d2:cloud")
    assert caps.cost_rank == 10
    assert caps.supports_tools is False
    await backend.aclose()


# ---------- honest-advisory usage_snapshot ----------------------------------


async def test_usage_snapshot_is_honest_not_local_free_stub() -> None:
    """Cloud is NOT free: usage_snapshot reports remaining_fraction=1.0
    ("full, eligible, no signal yet"), NOT the local free-stub's 0.001 that
    suppresses primary selection. weekly_exhausted is False (we genuinely
    don't know from headers), and there is no cooldown on a healthy daemon."""
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    await backend.health()
    snap = await backend.usage_snapshot()
    assert snap.remaining_fraction == 1.0
    assert snap.weekly_exhausted is False
    assert snap.cooldown_until_ts is None
    await backend.aclose()


async def test_quota_snapshot_is_none() -> None:
    backend = OllamaCloudBackend(
        id="ollama-cloud",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))),
    )
    assert await backend.quota_snapshot() is None
    await backend.aclose()


# ---------- health ----------------------------------------------------------


async def test_health_reports_network_when_daemon_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=0,
        transport=httpx.MockTransport(handler),
    )
    h = await backend.health()
    assert h.available is False
    assert h.reason == "network"
    await backend.aclose()


async def test_health_unhealthy_after_poll_sets_cooldown() -> None:
    """Once we've polled successfully, a subsequent outage sets a short
    cooldown so the backend is excluded from routable backends. On cold
    start (never polled) there is no cooldown so the fleet isn't empty."""
    state = {"up": True}
    tags = _tags_payload("model-a0d2:cloud")

    def handler(request: httpx.Request) -> httpx.Response:
        if not state["up"]:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json=tags)

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=0,
        transport=httpx.MockTransport(handler),
    )
    await backend.health()  # first poll succeeds
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None  # healthy, no cooldown
    state["up"] = False
    await backend.health()  # outage
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None  # transient outage → cooldown
    await backend.aclose()


# ---------- refresh_advertised_models ---------------------------------------


async def test_refresh_advertised_models_forces_refetch() -> None:
    """refresh_advertised_models bypasses the TTL gate so the lifespan loop
    can refresh all backends uniformly and callers see a current catalog
    immediately after it returns."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_tags_payload("model-a0d2:cloud"))
        return httpx.Response(200, json=_tags_payload("model-a0d2:cloud", "model-a0f3:cloud"))

    backend = OllamaCloudBackend(
        id="ollama-cloud",
        catalog_refresh_s=3600,  # long TTL so normal access won't refresh
        transport=httpx.MockTransport(handler),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud"})
    # Within TTL, a plain health() won't re-fetch; refresh_advertised_models must.
    await backend.refresh_advertised_models()
    assert backend.advertised_models == frozenset({"model-a0d2:cloud", "model-a0f3:cloud"})
    await backend.aclose()


# ---------- dispatch (sub-slice 2: chat + responses, stream + non-stream) ----


async def test_chat_completions_nonstream() -> None:
    """Non-stream chat dispatch POSTs the codex-stripped body to the daemon's
    /v1/chat/completions and surfaces the raw chat reply; upstream_status lands
    on the handle; Codex-only fields (`reasoning`, `parallel_tool_calls`) that
    the daemon rejects are stripped before forwarding, and `stream` is forced
    False."""
    handler, chat_requests = _dispatch_handler()
    backend = OllamaCloudBackend(id="ollama-cloud", transport=httpx.MockTransport(handler))
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
    # Exactly one /v1/chat/completions POST, carrying the cloud model + stripped body.
    assert len(chat_requests) == 1
    sent = json.loads(chat_requests[0].content)
    assert sent["model"] == "model-a0d2:cloud"
    assert sent["stream"] is False
    assert "reasoning" not in sent
    assert "parallel_tool_calls" not in sent
    await backend.aclose()


async def test_responses_nonstream() -> None:
    """The Responses path translates request→chat, dispatches, then translates
    the chat reply back to Responses shape: usage mapped from
    prompt_tokens/completion_tokens to input_tokens/output_tokens, and a
    message item carrying the assistant text in output[]."""
    handler, _ = _dispatch_handler()
    backend = OllamaCloudBackend(id="ollama-cloud", transport=httpx.MockTransport(handler))
    resp = await backend.responses({"model": "model-a0d2:cloud", "input": "say hi"})
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["usage"]["input_tokens"] == 5
    assert resp["usage"]["output_tokens"] == 2
    msg = next(o for o in resp["output"] if o["type"] == "message")
    assert msg["content"][0]["text"] == "hi there"
    await backend.aclose()


async def test_responses_stream_emits_responses_sse() -> None:
    """responses_stream accepts a Responses-shape body (`input`, `stream:True`)
    — the smoke-route requirement — and emits Responses-API SSE translated from
    the chat-completions stream: response.created first, incremental
    output_text.delta events, then response.completed carrying mapped usage.
    The ResponsesStreamCollector tees the completed usage onto
    handle.stream_summary so per-request metering populates automatically."""
    chunks = [
        b'data: {"id":"x","model":"model-a0d2:cloud","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, _ = _dispatch_handler(chat_sse=chunks)
    backend = OllamaCloudBackend(id="ollama-cloud", transport=httpx.MockTransport(handler))
    handle = CallHandle()
    events: list[dict[str, Any]] = []
    async for raw in backend.responses_stream(
        {"model": "model-a0d2:cloud", "input": "say hi", "stream": True}, handle
    ):
        for ev_chunk in raw.split(b"\n\n"):
            for line in ev_chunk.split(b"\n"):
                if line.startswith(b"data:"):
                    s = line[5:].strip().decode()
                    if s and s != "[DONE]":
                        with contextlib.suppress(json.JSONDecodeError):
                            events.append(json.loads(s))
    types = [e["type"] for e in events]
    assert types[0] == "response.created"
    text_deltas = [e for e in events if e["type"] == "response.output_text.delta"]
    assert "".join(e["delta"] for e in text_deltas) == "Hello"
    completed = [e for e in events if e["type"] == "response.completed"]
    assert len(completed) == 1
    assert completed[0]["response"]["usage"]["input_tokens"] == 3
    assert completed[0]["response"]["usage"]["output_tokens"] == 2
    # The collector teed the completed usage onto the handle → metering.
    assert handle.stream_summary is not None
    assert handle.stream_summary.completed_response is not None
    assert handle.stream_summary.completed_response["usage"]["input_tokens"] == 3
    await backend.aclose()


async def test_chat_completions_stream_passthrough() -> None:
    """chat_completions_stream is a raw byte passthrough — chat-completions SSE
    bytes flow through unchanged (no Responses translation). By design it does
    NOT set handle.stream_summary (mirrors litellm_gateway's chat-native gap;
    codex traffic uses /v1/responses → responses_stream, which does)."""
    chunks = [
        b'data: {"id":"x","choices":[{"delta":{"content":"hi"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    handler, _ = _dispatch_handler(chat_sse=chunks)
    backend = OllamaCloudBackend(id="ollama-cloud", transport=httpx.MockTransport(handler))
    handle = CallHandle()
    out = b"".join([c async for c in backend.chat_completions_stream({"model": "model-a0d2:cloud"}, handle)])
    assert b'data: {"id":"x"' in out
    assert b"[DONE]" in out
    assert handle.stream_summary is None
    await backend.aclose()


async def test_transport_error_marks_unhealthy() -> None:
    """A transport error on /v1/chat/completions flips the backend unhealthy so
    usage_snapshot reports a cooldown immediately (without waiting for the
    catalog TTL to drive a refresh), and surfaces as a transient BackendError so
    the dispatch layer retries a different backend."""
    handler, _ = _dispatch_handler(chat_error=httpx.ConnectError("daemon down"))
    backend = OllamaCloudBackend(id="ollama-cloud", transport=httpx.MockTransport(handler))
    await backend.health()  # first /api/tags poll succeeds → healthy, catalog cached
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is None
    with pytest.raises(BackendError):
        await backend.chat_completions({"model": "model-a0d2:cloud"})
    snap = await backend.usage_snapshot()
    assert snap.cooldown_until_ts is not None  # transient outage → cooldown
    await backend.aclose()


# ---------- default-OFF registration ----------------------------------------


def test_backend_absent_from_build_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CALLOSUM_OLLAMA_CLOUD_ENABLED unset → build_runtime_backends does NOT
    append the backend. Default-OFF is a no-op for live routing."""
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    # Ensure other optional backends don't add noise to the assertion.
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    assert all(getattr(b, "id", None) != "ollama-cloud" for b in backends)


def test_backend_present_when_env_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from callosum.__main__ import build_runtime_backends
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    monkeypatch.setenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", "1")
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "0")
    cfg = Config()
    backends = build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))
    ids = [getattr(b, "id", None) for b in backends]
    assert "ollama-cloud" in ids
    cloud = next(b for b in backends if getattr(b, "id", None) == "ollama-cloud")
    assert cloud.kind == "ollama_cloud"