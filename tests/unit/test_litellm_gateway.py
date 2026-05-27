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
    assert resp["usage"]["total_tokens"] == 6


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


# ---------- VRAM safety: unload_model ---------------------------------------


async def test_unload_model_hits_ollama_direct_with_keep_alive_zero() -> None:
    """unload_model translates the LiteLLM name to its ollama upstream via
    /model/info, then POSTs directly to ollama /api/generate with
    keep_alive=0. LiteLLM does NOT pass keep_alive through (verified
    empirically), so the eviction MUST bypass LiteLLM."""
    captured: list[tuple[str, dict[str, Any]]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = json.loads(request.content) if request.content else {}
        captured.append((url, body))
        if "/model/info" in url:
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "model-a0b0",
                            "litellm_params": {"model": "ollama/model-a0d7"},
                        },
                    ]
                },
            )
        if "/api/generate" in url:
            return httpx.Response(200, json={"done_reason": "length"})
        return httpx.Response(404)

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    await backend.unload_model("model-a0b0")
    urls = [u for u, _ in captured]
    assert any("/model/info" in u for u in urls), urls
    unload_calls = [(u, b) for u, b in captured if "/api/generate" in u]
    assert len(unload_calls) == 1
    body = unload_calls[0][1]
    assert body["model"] == "model-a0d7"
    assert body["keep_alive"] == 0
    await backend.aclose()


async def test_unload_model_skips_when_upstream_not_ollama() -> None:
    """If the LiteLLM entry isn't ollama-backed, skip the unload — the
    direct-ollama eviction wouldn't reach it. /model/info gets fetched
    but no /api/generate call is made."""
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(str(request.url))
        if "/model/info" in str(request.url):
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "model_name": "claude-3-5",
                            "litellm_params": {"model": "anthropic/claude-3-5-sonnet"},
                        }
                    ]
                },
            )
        return httpx.Response(404)

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    await backend.unload_model("claude-3-5")
    assert any("/model/info" in u for u in captured)
    assert not any("/api/generate" in u for u in captured)
    await backend.aclose()


async def test_unload_model_swallows_errors() -> None:
    """Network failures during unload must not raise — the caller (dispatch)
    treats VRAM eviction as best-effort. A missed eviction degrades
    headroom but the user request must continue."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("gateway dead")

    backend = LiteLLMGatewayBackend(
        id="local", transport=httpx.MockTransport(handler)
    )
    # Must NOT raise.
    await backend.unload_model("model-a0b0")
    await backend.aclose()
