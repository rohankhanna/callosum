"""Tests for the LiteLLM gateway backend.

The backend is OPTIONAL plumbing for local models behind local LLM gateway's
LiteLLM gateway. These tests cover:

- Catalog discovery from /v1/models (the gateway's curated model_list).
- ModelMetadata synthesis (so local models join the cell grid with the
  right shape — single "default" effort, high priority sorts after
  Codex cells, included in the grid).
- Health: gateway unreachable → unavailable; gateway up → available.
- chat_completions passthrough.
- Responses-API translation (/v1/responses → /v1/chat/completions on
  the wire, translated back into Responses-API shape on the way out).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from callosum.backends.litellm_gateway import (
    DEFAULT_CALL_TIMEOUT_S,
    LOCAL_PRIORITY_OFFSET,
    LiteLLMGatewayBackend,
    _chat_to_responses_response,
    _responses_to_chat_request,
)


def _models_payload(*slugs: str) -> dict[str, Any]:
    return {"data": [{"id": s, "object": "model"} for s in slugs]}


# ---------- catalog discovery -----------------------------------------------


async def test_advertised_models_empty_before_first_poll() -> None:
    """No /v1/models call yet → catalog empty, backend reports unhealthy."""
    backend = LiteLLMGatewayBackend(
        id="local",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_models_payload("model-a0d3"))
        ),
    )
    assert backend.advertised_models == frozenset()
    h = await backend.health()
    # After health() forces a refresh, catalog populates.
    assert h.available is True
    assert backend.advertised_models == frozenset({"model-a0d3"})
    await backend.aclose()


async def test_catalog_picks_up_new_models_on_refresh() -> None:
    """A second /v1/models call sees an expanded model_list."""
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        if call_count["n"] == 1:
            return httpx.Response(200, json=_models_payload("model-a0d3"))
        return httpx.Response(200, json=_models_payload("model-a0d3", "model-a0d9"))

    backend = LiteLLMGatewayBackend(
        id="local",
        catalog_refresh_s=0,  # always refresh
        transport=httpx.MockTransport(handler),
    )
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d3"})
    await backend.health()
    assert backend.advertised_models == frozenset({"model-a0d3", "model-a0d9"})
    await backend.aclose()


# ---------- model_metadata synthesis ----------------------------------------


async def test_model_metadata_marks_local_models_as_listable_default_effort() -> None:
    """Synthesized metadata lets the cell-grid merge include local models
    without changing the filter logic in cell_grid.py — they look like any
    other API-listed model from the grid's perspective."""
    backend = LiteLLMGatewayBackend(
        id="local",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_models_payload("model-a0d3", "model-a0d9"))
        ),
    )
    await backend.health()  # populate catalog
    md = backend.model_metadata
    assert set(md.keys()) == {"model-a0d3", "model-a0d9"}
    for slug, m in md.items():
        assert m.supported_in_api is True
        assert m.visibility == "list"
        assert m.supported_reasoning_levels == ("default",)
        # Local priority is offset above the Codex-range — sorts after remote
        # in the recommender's ranking, so the cheap classifier sees Codex
        # cells first by default.
        assert m.priority is not None and m.priority >= LOCAL_PRIORITY_OFFSET
    await backend.aclose()


# ---------- health ----------------------------------------------------------


async def test_health_reports_network_when_gateway_unreachable() -> None:
    """Gateway down → backend.health() returns unavailable; dispatch skips
    this backend without erroring out the proxy.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    backend = LiteLLMGatewayBackend(
        id="local",
        catalog_refresh_s=0,
        transport=httpx.MockTransport(handler),
    )
    h = await backend.health()
    assert h.available is False
    assert h.reason == "network"
    await backend.aclose()


async def test_health_reports_ok_after_successful_refresh() -> None:
    backend = LiteLLMGatewayBackend(
        id="local",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_models_payload("model-a0d3"))
        ),
    )
    h = await backend.health()
    assert h.available is True
    assert h.reason == "ok"
    await backend.aclose()


# ---------- chat_completions passthrough ------------------------------------


async def test_chat_completions_posts_to_gateway_and_returns_payload() -> None:
    received: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d3"))
        received["url"] = str(request.url)
        received["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "model": "model-a0d3",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "hi back"}}
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
            },
        )

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    out = await backend.chat_completions(
        {"model": "model-a0d3", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert out["choices"][0]["message"]["content"] == "hi back"
    assert received["url"].endswith("/v1/chat/completions")
    assert received["body"]["model"] == "model-a0d3"
    # Backend always forces stream=False on this code path.
    assert received["body"]["stream"] is False
    await backend.aclose()


async def test_master_key_added_as_bearer_when_set() -> None:
    seen_auth: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append(request.headers.get("Authorization"))
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d3"))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "model-a0d3",
                "choices": [{"message": {"role": "assistant", "content": ""}}],
            },
        )

    backend = LiteLLMGatewayBackend(
        id="local", master_key="sk-master-test", transport=httpx.MockTransport(handler)
    )
    await backend.chat_completions(
        {"model": "model-a0d3", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert all(a == "Bearer sk-master-test" for a in seen_auth)
    await backend.aclose()


# ---------- /v1/responses translation ---------------------------------------


def test_responses_to_chat_request_collapses_input_list_to_messages() -> None:
    body = {
        "model": "model-a0d3",
        "instructions": "be terse",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "what is 2+2?"}],
            }
        ],
    }
    chat = _responses_to_chat_request(body)
    assert chat["model"] == "model-a0d3"
    assert chat["messages"] == [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "what is 2+2?"},
    ]


def test_chat_to_responses_response_wraps_assistant_text() -> None:
    chat = {
        "id": "cmpl-42",
        "model": "model-a0d3",
        "choices": [{"message": {"role": "assistant", "content": "4"}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
    }
    resp = _chat_to_responses_response(chat)
    assert resp["object"] == "response"
    assert resp["model"] == "model-a0d3"
    assert resp["output"][0]["content"][0]["text"] == "4"
    # Usage is translated from chat shape (prompt/completion) to
    # Responses-API shape (input/output). Codex CLI's stream parser
    # hard-fails on missing input_tokens.
    assert resp["usage"]["input_tokens"] == 5
    assert resp["usage"]["output_tokens"] == 1
    assert resp["usage"]["total_tokens"] == 6
    assert resp["status"] == "completed"


def test_chat_to_responses_response_translates_tool_calls() -> None:
    """When the model returns tool_calls (model-a0d5/model-a0g1/etc. doing function
    calling), the translator must emit Responses-API function_call output
    items. Earlier versions dropped tool_calls on the floor → Codex CLI
    received empty output and showed nothing."""
    chat = {
        "id": "chatcmpl-abc",
        "model": "model-a0a9",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_xyz",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": '{"cmd":"git log"}',
                        },
                    }
                ],
            }
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 17, "total_tokens": 117},
    }
    resp = _chat_to_responses_response(chat)
    # Output should contain a function_call item with the right shape.
    fc = [o for o in resp["output"] if o["type"] == "function_call"]
    assert len(fc) == 1
    assert fc[0]["name"] == "shell"
    assert fc[0]["arguments"] == '{"cmd":"git log"}'
    assert fc[0]["call_id"] == "call_xyz"
    assert fc[0]["status"] == "completed"


def test_chat_to_responses_response_translates_dict_arguments_to_json_string() -> None:
    """Some serving stacks (ollama at certain versions) return tool_call
    arguments as a JSON object rather than a string. The Responses-API
    spec requires a string."""
    chat = {
        "id": "x",
        "model": "m",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": {"k": "v"}},
                }],
            }
        }],
    }
    resp = _chat_to_responses_response(chat)
    fc = [o for o in resp["output"] if o["type"] == "function_call"][0]
    assert isinstance(fc["arguments"], str)
    assert json.loads(fc["arguments"]) == {"k": "v"}


def test_chat_to_responses_response_preserves_thinking_as_reasoning_item() -> None:
    """When a thinking model emits `message.thinking`, the translator
    must surface it in the Responses-API output as a reasoning item.
    Earlier versions dropped it on the floor — operators who want to
    SEE what the model was thinking (or use it for debugging /
    training-data extraction) were left blind."""
    chat = {
        "id": "chatcmpl-think",
        "model": "model-a0b0",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "OK",
                "thinking": "The user asked for OK. Reply with OK.",
            }
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17},
    }
    resp = _chat_to_responses_response(chat)
    reasoning_items = [o for o in resp["output"] if o["type"] == "reasoning"]
    assert len(reasoning_items) == 1
    assert reasoning_items[0]["summary"][0]["text"] == (
        "The user asked for OK. Reply with OK."
    )
    # Message text item should also be present, AFTER the reasoning item.
    msg_items = [o for o in resp["output"] if o["type"] == "message"]
    assert len(msg_items) == 1
    assert msg_items[0]["content"][0]["text"] == "OK"


def test_chat_to_responses_response_no_reasoning_item_when_thinking_absent() -> None:
    """No `thinking` field → no reasoning item. Output is just the
    message — same as before the thinking-preservation change."""
    chat = {
        "id": "x",
        "choices": [{"message": {"role": "assistant", "content": "OK"}}],
    }
    resp = _chat_to_responses_response(chat)
    reasoning = [o for o in resp["output"] if o["type"] == "reasoning"]
    assert reasoning == []


def test_chat_to_responses_response_emits_zero_usage_when_absent() -> None:
    """When ollama omits usage (some local serving paths do), still emit
    input_tokens/output_tokens=0 so the Codex CLI stream parser doesn't
    fail with 'missing field input_tokens'."""
    chat = {
        "id": "cmpl-x",
        "model": "model-a0d3",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        # No usage field at all.
    }
    resp = _chat_to_responses_response(chat)
    assert resp["usage"]["input_tokens"] == 0
    assert resp["usage"]["output_tokens"] == 0
    assert resp["usage"]["total_tokens"] == 0


async def test_responses_end_to_end_translates_through_chat_completions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d3"))
        # Confirm we got the translated body (messages array, not input list).
        body = json.loads(request.content)
        assert "messages" in body and "input" not in body
        return httpx.Response(
            200,
            json={
                "id": "cmpl-1",
                "model": "model-a0d3",
                "choices": [
                    {"message": {"role": "assistant", "content": "translated reply"}}
                ],
            },
        )

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    resp = await backend.responses(
        {
            "model": "model-a0d3",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                }
            ],
        }
    )
    assert resp["output"][0]["content"][0]["text"] == "translated reply"
    await backend.aclose()


# ---------- error path ------------------------------------------------------


def test_default_call_timeout_is_generous_for_cold_loads() -> None:
    """Cold-loading a 26B/31B model from disk into VRAM can take 60-120s on
    typical hardware. The backend's default timeout must accommodate that
    without slicing the request mid-load; operators with fast hardware can
    override via CALLOSUM_LITELLM_TIMEOUT_S in __main__.py.
    """
    # Lower bound check rather than an exact-equality assertion so we don't
    # have to update the test if we tune the constant later. Below 60s
    # would be too tight for cold-loads we've actually measured (model-a0d6
    # took 72s on this machine), so guard at 60.
    assert DEFAULT_CALL_TIMEOUT_S >= 60.0


async def test_codex_only_fields_stripped_before_send() -> None:
    """The recommender writes `reasoning.effort` on every routed request,
    including local cells. Local backends don't take that field; the
    gateway backend must strip it before forwarding so ollama/vllm/etc.
    don't choke on an unknown key."""
    seen_body: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d3"))
        seen_body.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": "model-a0d3",
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
            },
        )

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    await backend.chat_completions(
        {
            "model": "model-a0d3",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning": {"effort": "default"},  # would normally land here from recommender
        }
    )
    assert "reasoning" not in seen_body
    # Other keys still make it through untouched.
    assert seen_body["model"] == "model-a0d3"
    assert seen_body["messages"] == [{"role": "user", "content": "hi"}]
    await backend.aclose()


async def test_chat_completions_raises_backenderror_on_4xx() -> None:
    from callosum.errors import BackendError

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d3"))
        return httpx.Response(429, json={"error": {"message": "rate limit"}})

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    with pytest.raises(BackendError):
        await backend.chat_completions(
            {"model": "model-a0d3", "messages": [{"role": "user", "content": "hi"}]}
        )
    await backend.aclose()


async def test_responses_stream_translates_chat_deltas_as_they_arrive() -> None:
    """Stream-through: as each chat-completions delta arrives, the
    translator emits the corresponding Responses-API event. Asserts:
    response.created arrives BEFORE the content fully accumulates."""

    chunks = [
        b'data: {"id":"x","model":"model-a0d5","choices":[{"delta":{"content":"Hel"}}]}\n\n',
        b'data: {"id":"x","model":"model-a0d5","choices":[{"delta":{"content":"lo"}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d5"))
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=b"".join(chunks),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    events: list[dict[str, Any]] = []
    async for raw in backend.responses_stream(
        {"model": "model-a0d5", "input": "say hi", "stream": True}
    ):
        for ev_chunk in raw.split(b"\n\n"):
            if not ev_chunk:
                continue
            for line in ev_chunk.split(b"\n"):
                if line.startswith(b"data:"):
                    payload_str = line[5:].strip().decode()
                    if payload_str == "[DONE]":
                        continue
                    try:
                        events.append(json.loads(payload_str))
                    except json.JSONDecodeError:
                        pass
    types = [e["type"] for e in events]
    # Order invariants the Codex CLI parser depends on:
    assert types[0] == "response.created"
    assert "response.in_progress" in types
    assert "response.output_item.added" in types
    # Text deltas were emitted INCREMENTALLY (we sent 2 content chunks).
    text_deltas = [e for e in events if e["type"] == "response.output_text.delta"]
    assert len(text_deltas) == 2
    assert text_deltas[0]["delta"] == "Hel"
    assert text_deltas[1]["delta"] == "lo"
    # response.completed at the end carries assembled content + usage.
    completed = [e for e in events if e["type"] == "response.completed"]
    assert len(completed) == 1
    full = completed[0]["response"]
    assert full["output"][0]["content"][0]["text"] == "Hello"
    assert full["usage"]["input_tokens"] == 3
    assert full["usage"]["output_tokens"] == 2
    await backend.aclose()


async def test_responses_stream_translates_tool_call_argument_chunks() -> None:
    """Tool-call arguments arrive in chunks (`{"`, then `"cmd":"ls"}`);
    the translator should accumulate them and emit
    function_call_arguments.delta for each chunk + a .done event at the
    end with the full string."""

    chunks = [
        b'data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"shell","arguments":""}}]}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"cm"}}]}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"d\\":\\"ls\\"}"}}]}}]}\n\n',
        b'data: {"id":"x","choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json=_models_payload("model-a0d5"))
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                content=b"".join(chunks),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    events: list[dict[str, Any]] = []
    async for raw in backend.responses_stream(
        {"model": "model-a0d5", "input": "x", "stream": True}
    ):
        for ev_chunk in raw.split(b"\n\n"):
            for line in ev_chunk.split(b"\n"):
                if line.startswith(b"data:"):
                    s = line[5:].strip().decode()
                    if s and s != "[DONE]":
                        try:
                            events.append(json.loads(s))
                        except json.JSONDecodeError:
                            pass
    arg_deltas = [e for e in events if e["type"] == "response.function_call_arguments.delta"]
    assert len(arg_deltas) == 2
    arg_done = [e for e in events if e["type"] == "response.function_call_arguments.done"]
    assert len(arg_done) == 1
    assert arg_done[0]["arguments"] == '{"cmd":"ls"}'
    completed = [e for e in events if e["type"] == "response.completed"]
    full = completed[0]["response"]
    fc = [o for o in full["output"] if o["type"] == "function_call"][0]
    assert fc["name"] == "shell"
    assert fc["arguments"] == '{"cmd":"ls"}'
    assert fc["call_id"] == "call_1"
    await backend.aclose()


# ---------- cell_capabilities --------------------------------------------


async def test_litellm_cell_capabilities_falls_back_when_cache_empty() -> None:
    """Before discovery runs (or for non-ollama upstreams),
    cell_capabilities returns conservative defaults — text-only,
    no tools — so requests can still dispatch but tool/vision-needing
    ones drop the cell from the capability filter."""
    backend = LiteLLMGatewayBackend(
        id="local",
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=_models_payload("never-discovered"))
        ),
    )
    try:
        caps = backend.cell_capabilities("never-discovered")
        assert caps.context_window == 128_000
        assert caps.modalities == frozenset({"text"})
        assert caps.supports_tools is False
        assert caps.cost_rank == 0
    finally:
        await backend.aclose()


async def test_litellm_cell_capabilities_discovered_from_ollama() -> None:
    """When ollama's /api/show reports `tools`, `vision`, and a
    large context_length, cell_capabilities surfaces the real values
    — no hardcoded conservative ceiling drops the cell from the filter."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/v1/models"):
            return httpx.Response(200, json=_models_payload("model-a0b0"))
        if url.endswith("/model/info"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "model-a0b0",
                            "litellm_params": {"model": "ollama/model-a0d7"},
                        }
                    ]
                },
            )
        if url.endswith("/api/show"):
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion", "tools", "vision", "thinking"],
                    "model_info": {"model-a0f4.context_length": 262_144},
                },
            )
        return httpx.Response(404)

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    try:
        await backend.health()  # triggers catalog refresh + capability discovery
        caps = backend.cell_capabilities("model-a0b0")
        assert caps.context_window == 262_144
        assert "image" in caps.modalities
        assert "text" in caps.modalities
        assert caps.supports_tools is True
        assert caps.cost_rank == 0
    finally:
        await backend.aclose()


async def test_litellm_cell_capabilities_handles_text_only_local_model() -> None:
    """ollama's /api/show capabilities=[completion] only → no vision,
    no tools. cell_capabilities reflects that — request-routing will
    correctly drop this cell for tool-use or image prompts."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith("/v1/models"):
            return httpx.Response(200, json=_models_payload("text-only-local"))
        if url.endswith("/model/info"):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "text-only-local",
                            "litellm_params": {"model": "ollama/text-only:7b"},
                        }
                    ]
                },
            )
        if url.endswith("/api/show"):
            return httpx.Response(
                200,
                json={
                    "capabilities": ["completion"],
                    "model_info": {"model-a0g1.context_length": 8_192},
                },
            )
        return httpx.Response(404)

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    try:
        await backend.health()
        caps = backend.cell_capabilities("text-only-local")
        assert caps.context_window == 8_192
        assert caps.modalities == frozenset({"text"})
        assert caps.supports_tools is False
    finally:
        await backend.aclose()
