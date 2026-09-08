"""Hermetic regression: local LLM gateway / LiteLLM-gateway mutual exclusivity +
chat→responses translation for responses-native entries (merge-blocking).
Pins fix 993e82a — "fix(routing): de-duplicate local LLM gateway backend +
translate chat for responses-native entries".

The original live-use bug: callosum's /v1/chat/completions hung ~240s on
local-model targets in local-only routing mode. Two independent root causes
landed in one commit, each guarded here in isolation so a single-cause
reintroduction turns the matching test red:

1. **Mutual-exclusivity (__main__.build_runtime_backends)** — the
   docstring says the LiteLLM gateway is a FALLBACK, but the code registered
   BOTH LocalModelRegistryBackend and LiteLLMGatewayBackend when both env
   conditions were true. Both advertised the same models with the same
   backend id; the router could pick either. When it picked the gateway path
   for a *-responses-proxy model, the request hit the gateway's
   chat-completions, which the upstream proxy doesn't implement → hang.
   The fix added a local_added flag and gated the gateway on
   ... and not local_added.

   test_litellm_gateway_not_registered_when_local_added: with
   both env flags set and the hub probe stubbed available, assert NO
   LiteLLMGatewayBackend is constructed. Reverting the guard (drop
   and not local_added) → the gateway IS constructed → red.
   test_litellm_gateway_registered_when_hub_disabled pins the other
   side — the fallback still fires when the hub is disabled, so the guard
   isn't over-skipping.

2. **chat→responses translation (LocalModelRegistryBackend.chat_completions)** —
   chat_completions() blindly POSTed to /v1/chat/completions even
   when the entry's runtime is responses-native (api_surfaces contains
   "responses"); the proxy process doesn't implement chat-completions →
   same hang. The fix mirrors responses(): when the entry advertises
   "responses", translate the chat body to Responses shape, POST to
   /v1/responses, translate the payload back.

   test_chat_completions_routes_to_responses_endpoint_for_native_entry:
   a stub source returns one entry with api_surfaces=("responses","chat");
   a httpx.MockTransport records the request URL and returns a synthetic
   Responses payload. Assert the POST went to /v1/responses (not
   /v1/chat/completions) and the response was translated back into a
   chat.completion. Reverting the if "responses" in entry.api_surfaces
   branch → the POST goes to /v1/chat/completions → the URL assertion
   fails → red.

Hermetic: build_runtime_backends is driven with stub sources/backends
(no real CLI, no network); chat_completions is driven through a
httpx.MockTransport (in-process, no socket). Runs in the Tier 1 gate's
unit suite as a merge-blocking regression.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from callosum.backend import CallHandle
from callosum.backends.local_direct import LocalModelRegistryBackend
from callosum.local import ModelEntry

# --------------------------------------------------------------------------- #
# Part 1 — mutual exclusivity in build_runtime_backends.                      #
# --------------------------------------------------------------------------- #


class _StubHubSource:
    """Stands in for LocalModelRegistrySource: probe_availability reports
    available without running the real local-llm CLI subprocess, and the
    constructor is a no-op (the real constructor stores CLI config only, but
    stubbing removes any doubt about side effects)."""

    def __init__(self, *, cli_command: list[str] | None = None, **_kw: Any) -> None:
        self._cli = cli_command

    @staticmethod
    def probe_availability(cli_command: list[str] | None = None) -> tuple[bool, str]:
        return True, "ok"


class _StubHubBackend:
    """Stands in for LocalModelRegistryBackend so no real catalog/CLI work runs.
    Records that it was constructed so the test can confirm the hub path ran."""

    def __init__(self, **kw: Any) -> None:
        self.kwargs = kw
        self.id = kw.get("id")


class _LitellmSpy:
    """Stands in for LiteLLMGatewayBackend so the test can detect the
    gateway was (or wasn't) registered WITHOUT constructing a real
    httpx.AsyncClient (which would emit ResourceWarnings if unclosed)."""

    def __init__(self, **kw: Any) -> None:
        self.kwargs = kw
        self.id = kw.get("id")


def _import_main(monkeypatch: pytest.MonkeyPatch):
    """Import callosum.__main__ fresh and stub its local LLM gateway + litellm
    classes so build_runtime_backends runs hermetically."""
    import callosum.__main__ as main_mod

    monkeypatch.setattr(main_mod, "LocalModelRegistrySource", _StubHubSource)
    monkeypatch.setattr(main_mod, "LocalModelRegistryBackend", _StubHubBackend)
    monkeypatch.setattr(main_mod, "LiteLLMGatewayBackend", _LitellmSpy)
    return main_mod


def test_litellm_gateway_not_registered_when_local_added(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the local LLM gateway backend is available + registered, the LiteLLM
    gateway MUST NOT also be registered (mutual exclusivity).

    Reverting __main__.build_runtime_backends ~L108 (dropping
    and not local_added from the litellm-gateway guard) makes the
    gateway get constructed alongside the hub → a _LitellmSpy appears in
    the returned backends → this assertion fails.
    """
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    main_mod = _import_main(monkeypatch)
    monkeypatch.delenv("CALLOSUM_LOCAL_DISABLED", raising=False)
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "1")

    cfg = Config()
    backends = main_mod.build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))

    # The hub path ran (the stub hub backend is present).
    assert any(isinstance(b, _StubHubBackend) for b in backends), (
        "LocalModelRegistryBackend stub not registered — the hub path didn't run"
    )
    # The critical regression: NO litellm gateway alongside the hub.
    assert not any(isinstance(b, _LitellmSpy) for b in backends), (
        "LiteLLMGatewayBackend registered alongside LocalModelRegistryBackend — "
        "the mutual-exclusivity guard regressed (993e82a part 1)"
    )


def test_litellm_gateway_registered_when_hub_disabled(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the local LLM gateway is disabled, the LiteLLM gateway fallback MUST
    still register. Pins that the and not local_added guard
    doesn't over-skip the legitimate fallback path."""
    from callosum.config import Config
    from callosum.operator_state import OperatorState

    main_mod = _import_main(monkeypatch)
    monkeypatch.setenv("CALLOSUM_LOCAL_DISABLED", "1")
    monkeypatch.delenv("CALLOSUM_OLLAMA_CLOUD_ENABLED", raising=False)
    monkeypatch.setenv("CALLOSUM_LITELLM_GATEWAY_ENABLED", "1")

    cfg = Config()
    backends = main_mod.build_runtime_backends(cfg, operator_state=OperatorState(tmp_path / "op.sqlite"))

    assert not any(isinstance(b, _StubHubBackend) for b in backends), (
        "hub backend registered despite CALLOSUM_LOCAL_DISABLED=1"
    )
    assert any(isinstance(b, _LitellmSpy) for b in backends), (
        "LiteLLMGatewayBackend fallback NOT registered when hub disabled — "
        "the guard is over-skipping the legitimate fallback (993e82a part 1)"
    )


# --------------------------------------------------------------------------- #
# Part 2 — chat→responses translation in LocalModelRegistryBackend.chat_completions. #
# --------------------------------------------------------------------------- #


class _SingleEntrySource:
    """A LocalModelSource stub that returns one ModelEntry — the
    minimum the backend's _resolve needs to route a request."""

    def __init__(self, entry: ModelEntry) -> None:
        self._entry = entry

    def models(self, *, force: bool = False) -> list[ModelEntry]:
        return [self._entry]


def _entry(api_surfaces: tuple[str, ...]) -> ModelEntry:
    return ModelEntry(
        id="proxy-model",
        endpoint="http://runtime.test:1234",
        runtime="responses_proxy",
        runtime_model="runtime-model",
        family="test",
        context_window=4096,
        api_surfaces=api_surfaces,
        enabled=True,
    )


def _responses_payload() -> dict[str, Any]:
    """A minimal Responses-API payload that _responses_to_chat_response
    can translate into a chat.completion (one output_text block)."""
    return {
        "id": "resp_1",
        "created_at": 1700000000,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi"}],
            }
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.asyncio
async def test_chat_completions_routes_to_responses_endpoint_for_native_entry() -> None:
    """For an entry advertising responses in api_surfaces,
    chat_completions MUST POST to /v1/responses (translating the chat
    body) and return a chat.completion — NOT POST to
    /v1/chat/completions (which the responses-native proxy doesn't
    implement and would hang).

    Reverting local_direct.py ~L520 (removing the
    if "responses" in entry.api_surfaces branch) makes the code fall
    through to the /v1/chat/completions POST → the recorded URL no longer
    ends with /v1/responses → this assertion fails.
    """
    recorded: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        recorded["url"] = str(request.url)
        return httpx.Response(200, json=_responses_payload())

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = LocalModelRegistryBackend(
        id="hub",
        source=_SingleEntrySource(_entry(("responses", "chat"))),
        client=client,
    )
    try:
        result = await backend.chat_completions(
            {"model": "proxy-model", "messages": [{"role": "user", "content": "hi"}]},
            CallHandle(),
        )
    finally:
        await backend.aclose()

    # The core regression: the request went to the Responses endpoint.
    assert recorded["url"].endswith("/v1/responses"), (
        f"chat_completions POSTed to {recorded['url']!r} instead of "
        f"/v1/responses — the responses-native translation regressed "
        f"(993e82a part 2)"
    )
    # And the response was translated back into a chat.completion.
    assert result["object"] == "chat.completion", f"expected chat.completion object, got {result.get('object')!r}"
    assert result["choices"][0]["message"]["content"] == "hi"
    # The public model id (what the client asked for) is preserved, not the
    # runtime alias.
    assert result["model"] == "proxy-model"


@pytest.mark.asyncio
async def test_chat_completions_routes_to_chat_endpoint_for_chat_only_entry() -> None:
    """For an entry advertising ONLY chat (a bare chat-completions
    runtime), chat_completions MUST POST to /v1/chat/completions
    directly — the translation branch must not fire for chat-only cells.
    Pins that the if "responses" in entry.api_surfaces guard isn't
    over-triggering."""
    recorded: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        recorded["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "runtime-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = LocalModelRegistryBackend(
        id="hub",
        source=_SingleEntrySource(_entry(("chat",))),
        client=client,
    )
    try:
        result = await backend.chat_completions(
            {"model": "proxy-model", "messages": [{"role": "user", "content": "hi"}]},
            CallHandle(),
        )
    finally:
        await backend.aclose()

    assert recorded["url"].endswith("/v1/chat/completions"), (
        f"chat-only entry POSTed to {recorded['url']!r} — the translation "
        f"branch fired for a chat-only cell (over-trigger)"
    )
    assert result["object"] == "chat.completion"
