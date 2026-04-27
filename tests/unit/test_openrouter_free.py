from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from codex_proxy.backends.openrouter_free import (
    OpenRouterFreeBackend,
    _approx_token_budget,
    _code_score,
    _is_blocked_provider,
    _is_free,
)
from codex_proxy.errors import BackendError


def _catalog_payload(*entries: dict[str, Any]) -> dict[str, Any]:
    return {"data": list(entries)}


def _entry(
    *,
    id: str,
    context_length: int = 8192,
    free: bool = True,
    supports_tools: bool = False,
) -> dict[str, Any]:
    pricing = (
        {"prompt": "0", "completion": "0"} if free else {"prompt": "0.001", "completion": "0.002"}
    )
    sup = ["tools", "tool_choice"] if supports_tools else ["max_tokens"]
    return {
        "id": id,
        "context_length": context_length,
        "pricing": pricing,
        "supported_parameters": sup,
    }


# ---------- helpers ---------------------------------------------------------


def test_is_free_true_when_both_zero() -> None:
    assert _is_free({"prompt": "0", "completion": "0"}) is True
    assert _is_free({"prompt": 0.0, "completion": 0.0}) is True


def test_is_free_false_when_either_nonzero() -> None:
    assert _is_free({"prompt": "0.001", "completion": "0"}) is False
    assert _is_free({"prompt": "0", "completion": "0.001"}) is False
    assert _is_free({}) is False
    assert _is_free(None) is False


def test_code_score_picks_highest_matching_pattern() -> None:
    assert _code_score("meta-model-a0g1/model-a0a5:free") >= 100
    assert _code_score("meta-model-a0g1/model-a0g1-3.1-405b-instruct:free") >= 95
    assert _code_score("nousresearch/hermes-3-model-a0g1-3.1-70b:free") >= 70
    assert _code_score("totally-unknown/random:free") == 0


def test_blocked_providers_filtered_from_catalog() -> None:
    """Operator preference: no Chinese-origin cloud models in the free pool."""
    assert _is_blocked_provider("model-a0g3/model-a0f3-coder-32b-instruct:free") is True
    assert _is_blocked_provider("model-a0e2/model-a0e2-coder-v2:free") is True
    assert _is_blocked_provider("01-ai/yi-large:free") is True
    assert _is_blocked_provider("thudm/glm-4-9b:free") is True
    assert _is_blocked_provider("bytedance/doubao-pro:free") is True
    # Non-Chinese providers must NOT be blocked.
    assert _is_blocked_provider("meta-model-a0g1/model-a0a5:free") is False
    assert _is_blocked_provider("mistralai/mistral-large:free") is False
    assert _is_blocked_provider("google/model-a0d5-2-27b:free") is False
    assert _is_blocked_provider("nousresearch/hermes-3:free") is False


async def test_blocked_provider_excluded_from_picker() -> None:
    """Even if Qwen2.5-Coder is in the OpenRouter free catalog, the picker
    must never select it — the operator preference excludes it at parse time.
    """
    catalog = _catalog_payload(
        _entry(id="model-a0g3/model-a0f3-coder-32b:free", context_length=128_000, supports_tools=True),
        _entry(
            id="meta-model-a0g1/model-a0a5:free",
            context_length=32_000,
            supports_tools=True,
        ),
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=catalog)
        body = json.loads(req.content)
        # Even though model-a0g3 has bigger context, it's blocked; model-a0g1 wins.
        assert body["model"] == "meta-model-a0g1/model-a0a5:free"
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = OpenRouterFreeBackend(
        id="or", api_key="sk-or-test", transport=httpx.MockTransport(handler)
    )
    try:
        await backend.chat_completions(
            {"model": "model-a0e7", "messages": [{"role": "user", "content": "hi"}]}
        )
        # Confirm the catalog filter actually dropped the blocked entry.
        assert "model-a0g3/model-a0f3-coder-32b:free" not in backend.advertised_models
    finally:
        await backend.aclose()


def test_approx_token_budget_floors_at_1024() -> None:
    assert _approx_token_budget({}) == 4096
    assert _approx_token_budget({"messages": []}) == 1024
    assert (
        _approx_token_budget({"messages": [{"role": "user", "content": "x" * 8000}]}) == 4000
    )  # 8000 // 2


# ---------- backend tests ---------------------------------------------------


async def test_advertised_models_is_empty_until_catalog_fetched() -> None:
    backend = OpenRouterFreeBackend(
        id="or",
        api_key="sk-or-test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})),
    )
    try:
        # Before any call, advertised is just the virtual selector.
        assert backend.advertised_models == frozenset({"auto-fallback"})
    finally:
        await backend.aclose()


async def test_picks_highest_code_score_when_multiple_free_models_available() -> None:
    catalog = _catalog_payload(
        _entry(
            id="meta-model-a0g1/model-a0a5:free", context_length=32_000, supports_tools=True
        ),
        _entry(id="meta-model-a0g1/model-a0g1-3.1-8b:free", context_length=8000, supports_tools=True),
        _entry(id="bad-pricing/expensive", context_length=128_000, supports_tools=True, free=False),
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=catalog)
        body = json.loads(req.content)
        # The picker should have rewritten the model to model-a0g1-3.3 (highest score).
        assert body["model"] == "meta-model-a0g1/model-a0a5:free"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-or",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = OpenRouterFreeBackend(
        id="or", api_key="sk-or-test", transport=httpx.MockTransport(handler)
    )
    try:
        result = await backend.chat_completions(
            {"model": "model-a0e7", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert result["model"] == "meta-model-a0g1/model-a0a5:free"
    finally:
        await backend.aclose()


async def test_filters_out_non_tool_supporting_models_when_request_has_tools() -> None:
    catalog = _catalog_payload(
        _entry(id="big-model-no-tools/free:free", context_length=200_000, supports_tools=False),
        _entry(
            id="meta-model-a0g1/model-a0a5:free", context_length=32_000, supports_tools=True
        ),
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=catalog)
        body = json.loads(req.content)
        # Even though the no-tools model has bigger context, the picker must
        # exclude it because the request has tools.
        assert body["model"] == "meta-model-a0g1/model-a0a5:free"
        return httpx.Response(
            200,
            json={
                "id": "x",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = OpenRouterFreeBackend(
        id="or", api_key="sk-or-test", transport=httpx.MockTransport(handler)
    )
    try:
        await backend.chat_completions(
            {
                "model": "model-a0e7",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "x"}}],
            }
        )
    finally:
        await backend.aclose()


async def test_advertised_models_includes_shadow_set() -> None:
    catalog = _catalog_payload(_entry(id="meta-model-a0g1/model-a0a5:free"))
    backend = OpenRouterFreeBackend(
        id="or",
        api_key="sk-or-test",
        shadow_models=frozenset({"model-a0e7", "model-a0c3"}),
        transport=httpx.MockTransport(
            lambda r: (
                httpx.Response(200, json=catalog)
                if r.url.path.endswith("/models")
                else httpx.Response(200, json={})
            )
        ),
    )
    try:
        # Trigger a catalog fetch by making a chat call. (Or refresh directly.)
        await backend._refresh_catalog_if_stale()
        adv = backend.advertised_models
        assert "model-a0e7" in adv
        assert "model-a0c3" in adv
        assert "meta-model-a0g1/model-a0a5:free" in adv
        assert "auto-fallback" in adv
    finally:
        await backend.aclose()


async def test_pick_raises_when_catalog_is_empty() -> None:
    backend = OpenRouterFreeBackend(
        id="or",
        api_key="sk-or-test",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"data": []})),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions(
                {"model": "model-a0e7", "messages": [{"role": "user", "content": "hi"}]}
            )
        assert "catalog" in excinfo.value.message.lower()
    finally:
        await backend.aclose()


async def test_responses_translates_to_chat_and_back() -> None:
    """Hermes/codex CLI calls /v1/responses; OpenRouter doesn't support it.
    The backend must translate request → chat-completions, response → Responses-API.
    """
    catalog = _catalog_payload(
        _entry(id="meta-model-a0g1/model-a0a5:free", supports_tools=True)
    )

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/models"):
            return httpx.Response(200, json=catalog)
        body = json.loads(req.content)
        # System instruction came through as a system message.
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "be brief"
        # The user input survived the translation.
        assert body["messages"][1]["role"] == "user"
        assert body["messages"][1]["content"] == "hi"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    backend = OpenRouterFreeBackend(
        id="or", api_key="sk-or-test", transport=httpx.MockTransport(handler)
    )
    try:
        result = await backend.responses(
            {
                "model": "model-a0e7",
                "instructions": "be brief",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        )
        # Response shape: Responses-API output array with output_text.
        assert result["object"] == "response"
        output = result["output"][0]
        assert output["role"] == "assistant"
        assert output["content"][0]["text"] == "hello"
    finally:
        await backend.aclose()
