from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from callosum.auth_vault import AuthVault
from callosum.backends.codex_auth_vault import (
    _DEFAULT_CLIENT_VERSION,
    CodexAuthVaultBackend,
    _resolve_codex_client_version,
    _responses_to_chat_finish_reason,
    _responses_to_chat_response,
)
from callosum.codex_quota import CodexQuotaSnapshot
from callosum.errors import BackendError


def _write_auth_json(path: Path, *, access_token: str = "access-codex") -> None:
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": access_token,
                    "refresh_token": "refresh-codex",
                    "account_id": "acct-codex",
                },
                "last_refresh": "2024-01-01T00:00:00Z",
            }
        )
    )


def _make_vault(path: Path, *, transport: httpx.MockTransport | None = None) -> AuthVault:
    if transport is None:

        def default_handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("vault transport should not be used")

        transport = httpx.MockTransport(default_handler)
    return AuthVault(path=path, transport=transport)


def _sse_response(response_payload: dict, *, status: int = 200, headers: dict | None = None) -> httpx.Response:
    """Build an SSE response equivalent to a non-streaming JSON 200.

    Codex Responses API now requires `stream: true` always, so the backend
    sends streaming requests upstream and buffers the resulting SSE into a
    dict. Test handlers simulate that by returning a `response.created` event
    followed by a `response.completed` event whose `response` field is the
    payload the test expects to come back from `backend.responses(...)`.
    """
    events = [
        ("response.created", {"type": "response.created", "id": response_payload.get("id", "r1")}),
        (
            "response.completed",
            {"type": "response.completed", "response": response_payload},
        ),
    ]
    body = "".join(f"event: {n}\ndata: {json.dumps(p)}\n\n" for n, p in events).encode()
    return httpx.Response(
        status,
        content=body,
        headers={"content-type": "text/event-stream", **(headers or {})},
    )


async def test_chat_completions_translates_and_forwards_to_responses_endpoint(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["account_id"] = request.headers.get("chatgpt-account-id")
        captured["body"] = json.loads(request.content)
        return _sse_response(
            {
                "id": "resp-1",
                "model": "model-a0d0",
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

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        base_url="https://chatgpt.example.com/backend-api/codex",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await backend.chat_completions(
            {
                "model": "model-a0d0",
                "messages": [
                    {"role": "system", "content": "be brief"},
                    {"role": "user", "content": "hi"},
                ],
            }
        )
        assert captured["url"] == "https://chatgpt.example.com/backend-api/codex/responses"
        assert captured["authorization"] == "Bearer access-codex"
        assert captured["account_id"] == "acct-codex"
        body = captured["body"]
        assert isinstance(body, dict)
        assert body["model"] == "model-a0d0"
        assert body["instructions"] == "be brief"
        # Codex Responses API now requires stream=true ALWAYS — even when the
        # caller wanted a non-streaming dict response, backend internally
        # promotes to streaming and collects the SSE.
        assert body["stream"] is True
        assert body["input"] == [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ]
        assert result["id"] == "resp-1"
        assert result["choices"][0]["message"]["content"] == "hello"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 2
    finally:
        await backend.aclose()


async def test_chat_translation_defaults_instructions_and_store(tmp_path: Path) -> None:
    """Codex Responses API rejects requests without `instructions` (non-empty)
    and `store: False`. When a chat-completions caller (e.g. hermes) sends a
    body lacking either, the translation must backfill defaults so upstream
    accepts it.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _sse_response(
            {
                "id": "resp-x",
                "model": "model-a0e7",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        # Chat body with NO system message and NO store field — the two
        # missing pieces that used to 400 on real hermes traffic.
        await backend.chat_completions(
            {
                "model": "model-a0e7",
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        body = captured["body"]
        assert isinstance(body, dict)
        assert isinstance(body.get("instructions"), str)
        assert body["instructions"]  # non-empty fallback
        assert body.get("store") is False
        # Stream is forced to true upstream (Codex Responses API requires it).
        assert body.get("stream") is True
    finally:
        await backend.aclose()


async def test_chat_translation_preserves_explicit_store(tmp_path: Path) -> None:
    """If the caller DID provide `store`, don't override it. Backfill is only
    for the missing-default case.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _sse_response(
            {
                "id": "x",
                "model": "model-a0e7",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }
        )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.chat_completions(
            {
                "model": "model-a0e7",
                "messages": [{"role": "user", "content": "hi"}],
                "store": True,  # explicit; must survive translation unchanged
            }
        )
        body = captured["body"]
        assert isinstance(body, dict)
        assert body.get("store") is True
    finally:
        await backend.aclose()


async def test_clear_cooldown_resets_snapshot_and_persists(tmp_path: Path) -> None:
    """clear_cooldown wipes cooldown_until_ts + weekly_exhausted and the
    state_store gets the cleared snapshot so a restart doesn't resurrect it.

    Regression: without this, a stale cooldown set from a transient or
    mis-classified 429 (or an upstream weekly_reset_at that overshot) had
    no in-process recovery path — the chicken-and-egg with the dispatcher's
    cooldown-skip guard locked the backend out until the timestamp finally
    expired or the operator manually edited the JSON on disk.
    """
    from callosum.backend import UsageSnapshot
    from callosum.state import StateStore

    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    state_dir = tmp_path / "state"
    state_store = StateStore(state_dir)
    # Seed a stuck cooldown on disk (mimics the production failure mode).
    state_store.save_usage(
        "vault-stuck",
        UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=time.time() + 7 * 86400,
            weekly_exhausted=True,
            probed_at_ts=time.time() - 2 * 86400,
        ),
    )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-stuck",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        state_store=state_store,
    )
    try:
        before = await backend.usage_snapshot()
        assert before.cooldown_until_ts is not None
        assert before.weekly_exhausted is True

        cleared = backend.clear_cooldown()
        assert cleared.cooldown_until_ts is None
        assert cleared.weekly_exhausted is False

        after = await backend.usage_snapshot()
        assert after.cooldown_until_ts is None
        assert after.weekly_exhausted is False

        # Persisted: a fresh StateStore reading the same dir sees cleared state.
        persisted = StateStore(state_dir).load_usage("vault-stuck")
        assert persisted is not None
        assert persisted.cooldown_until_ts is None
        assert persisted.weekly_exhausted is False
    finally:
        await backend.aclose()


async def test_rate_limited_response_classifies_and_records_cooldown(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"retry-after": "2"}, json={"error": "rate"})

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]})
        assert excinfo.value.classification == "rate_limited"
        snapshot = await backend.usage_snapshot()
        assert snapshot.cooldown_until_ts is not None
    finally:
        await backend.aclose()


async def test_auth_invalid_triggers_when_upstream_401(tmp_path: Path) -> None:
    """Upstream 401 → backend force-refreshes the vault. If the refresh ALSO
    fails (refresh_token revoked), surface auth_invalid. The test models the
    end-state worst case: both upstream and refresh return 401.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "invalid_token"}})

    def vault_handler(request: httpx.Request) -> httpx.Response:
        # Refresh endpoint also rejects — refresh_token is dead.
        return httpx.Response(401, json={"error": {"message": "refresh invalid"}})

    vault = AuthVault(path=auth_path, transport=httpx.MockTransport(vault_handler))
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(upstream_handler),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions({"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]})
        assert excinfo.value.classification == "auth_invalid"
    finally:
        await backend.aclose()


async def test_upstream_401_triggers_force_refresh_and_retries(tmp_path: Path) -> None:
    """Healthy refresh path: upstream returns 401 once → backend force-refreshes
    the vault (gets new access_token) → retries upstream → 200. The retry path
    is what unblocks server-side token revocations.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path, access_token="stale-access")

    upstream_calls: list[str] = []

    def upstream_handler(request: httpx.Request) -> httpx.Response:
        upstream_calls.append(request.headers.get("Authorization", ""))
        if len(upstream_calls) == 1:
            return httpx.Response(401, json={"error": {"code": "invalid_token"}})
        return _sse_response(
            {
                "id": "resp-ok",
                "model": "model-a0d0",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            }
        )

    def vault_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": "fresh-access",
                "refresh_token": "fresh-refresh",
                "id_token": None,
            },
        )

    vault = AuthVault(path=auth_path, transport=httpx.MockTransport(vault_handler))
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(upstream_handler),
    )
    try:
        result = await backend.chat_completions(
            {"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert "choices" in result
        # First call used the stale token; second used the fresh one.
        assert upstream_calls[0] == "Bearer stale-access"
        assert upstream_calls[1] == "Bearer fresh-access"
    finally:
        await backend.aclose()


async def test_chat_completions_stream_emits_valid_sse_chunks(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return _sse_response(
            {
                "id": "resp-stream",
                "model": "model-a0d0",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "streamed"}],
                    }
                ],
            }
        )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks = [
            c
            async for c in backend.chat_completions_stream(
                {
                    "model": "model-a0d0",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                }
            )
        ]
        assert chunks[-1] == b"data: [DONE]\n\n"
        joined = b"".join(chunks).decode()
        assert '"role": "assistant"' in joined
        assert '"content": "streamed"' in joined
        assert '"finish_reason": "stop"' in joined
    finally:
        await backend.aclose()


async def test_responses_forwards_body_verbatim_with_vault_headers(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["authorization"] = request.headers.get("Authorization")
        captured["account_id"] = request.headers.get("chatgpt-account-id")
        captured["beta"] = request.headers.get("OpenAI-Beta")
        captured["originator"] = request.headers.get("originator")
        captured["version"] = request.headers.get("version")
        captured["body"] = json.loads(request.content)
        return _sse_response({"id": "resp-xyz", "object": "response"})

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        base_url="https://chatgpt.example.com/backend-api/codex",
        transport=httpx.MockTransport(handler),
    )
    request_body = {
        "model": "model-a0d0",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "instructions": "be brief",
        "store": False,
    }
    try:
        result = await backend.responses(request_body)
        assert captured["url"] == "https://chatgpt.example.com/backend-api/codex/responses"
        assert captured["authorization"] == "Bearer access-codex"
        assert captured["account_id"] == "acct-codex"
        assert captured["beta"] == "responses=v1"
        assert captured["originator"] == "codex_cli_rs"
        # model-a0c4's backend routing requires a `version` header
        # alongside originator (openai/codex#31967); without it luna 404s
        # while sol/terra serve. Pin the header is wired to the resolver.
        assert captured["version"] == _resolve_codex_client_version()
        assert captured["version"]
        # Body forwarded with one mutation: stream is forced to true (Codex
        # Responses API now requires it). Other fields pass through unchanged.
        forwarded = captured["body"]
        assert isinstance(forwarded, dict)
        assert forwarded["stream"] is True
        for key, expected in request_body.items():
            assert forwarded.get(key) == expected
        assert result["id"] == "resp-xyz"
    finally:
        await backend.aclose()


async def test_responses_strips_custom_tool_call_namespace(tmp_path: Path) -> None:
    """Parity with the active credential_proxy path: the Codex CLI emits
    `custom_tool_call` input items with a top-level `namespace` field, and
    ChatGPT's `/codex/responses` rejects `namespace` as an unknown parameter
    (HTTP 400 "Unknown parameter: 'input[N].namespace'"). This inert
    fallback must apply the same strip so activating it doesn't regress."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return _sse_response({"id": "resp-ns", "object": "response"})

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    request_body = {
        "model": "model-a0d0",
        "input": [
            {"type": "custom_tool_call", "status": "completed", "call_id": "call_n",
             "name": "exec", "namespace": "exec", "input": "await tools.exec_command({})"},
        ],
    }
    try:
        await backend.responses(request_body)
        forwarded = captured["body"]
        assert isinstance(forwarded, dict)
        ctc = next(it for it in forwarded["input"] if it.get("type") == "custom_tool_call")
        assert "namespace" not in ctc
        assert ctc["name"] == "exec"
        assert ctc["call_id"] == "call_n"
    finally:
        await backend.aclose()


async def test_responses_stream_strips_custom_tool_call_namespace(tmp_path: Path) -> None:
    """Same strip on the streaming path."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=b'data: {"type":"response.completed"}\n\n',
        )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    request_body = {
        "model": "model-a0d0",
        "input": [
            {"type": "custom_tool_call", "call_id": "call_s", "name": "exec",
             "namespace": "exec", "input": "x"},
        ],
    }
    try:
        chunks: list[bytes] = []
        async for chunk in backend.responses_stream(request_body):
            chunks.append(chunk)
        forwarded = captured["body"]
        assert isinstance(forwarded, dict)
        ctc = next(it for it in forwarded["input"] if it.get("type") == "custom_tool_call")
        assert "namespace" not in ctc
    finally:
        await backend.aclose()


async def test_responses_stream_strips_bogus_custom_tool_call_namespace(tmp_path: Path) -> None:
    """Parity with the active credential_proxy path: the upstream returns
    `custom_tool_call` items with a `namespace` for tools the client declared
    WITHOUT one; the codex CLI then mis-dispatches `namespace+name`. The
    inert fallback must strip it from the streamed response too, while
    leaving legitimately-namespaced tools intact."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    sse = (
        b'event: response.output_item.added\n'
        b'data: {"type":"response.output_item.added","item":{"id":"ctc_1",'
        b'"type":"custom_tool_call","call_id":"call_e","name":"exec",'
        b'"namespace":"exec","input":"await tools.exec_command({})"}}\n\n'
        b'event: response.output_item.added\n'
        b'data: {"type":"response.output_item.added","item":{"id":"ctc_2",'
        b'"type":"custom_tool_call","call_id":"call_f","name":"followup_task",'
        b'"namespace":"collaboration","input":"{}"}}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks: list[bytes] = []
        async for chunk in backend.responses_stream({
            "model": "model-a0d0",
            "input": [
                {"type": "additional_tools", "role": "developer", "tools": [
                    {"type": "custom", "name": "exec"},
                    {"type": "namespace", "name": "collaboration", "tools": [
                        {"type": "function", "name": "followup_task"},
                    ]},
                ]},
            ],
        }):
            chunks.append(chunk)
        out = b"".join(chunks).decode("utf-8")
        items = []
        for ev in out.split("\n\n"):
            if "data: " in ev:
                obj = json.loads(ev.split("data: ", 1)[1])
                if obj.get("type") == "response.output_item.added":
                    items.append(obj["item"])
        by_name = {it["name"]: it for it in items}
        assert "namespace" not in by_name["exec"]
        assert by_name["exec"]["name"] == "exec"
        assert by_name["followup_task"]["namespace"] == "collaboration"
    finally:
        await backend.aclose()


async def test_responses_stream_passes_upstream_sse_through_byte_for_byte(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    sse_body = (
        b'data: {"type":"response.created","id":"resp-s"}\n\n'
        b'data: {"type":"response.output_text.delta","delta":"hel"}\n\n'
        b'data: {"type":"response.output_text.delta","delta":"lo"}\n\n'
        b'data: {"type":"response.completed"}\n\n'
        b"data: [DONE]\n\n"
    )
    captured_accept: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_accept["accept"] = request.headers.get("Accept")
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse_body,
        )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        chunks: list[bytes] = []
        async for chunk in backend.responses_stream({"model": "model-a0d0", "input": []}):
            chunks.append(chunk)
        assert b"".join(chunks) == sse_body
        assert captured_accept["accept"] == "text/event-stream"
    finally:
        await backend.aclose()


async def test_construction_accepts_empty_advertised_models(tmp_path: Path) -> None:
    """Empty advertised_models is LEGAL — the backend discovers its
    catalog dynamically via refresh_advertised_models. Hard-coding a
    static list in config defeated dynamic routing and forced operator
    churn each time OpenAI shipped a model. The constructor used to
    reject empty; this test pins the relaxation. Until discovery
    succeeds the backend's advertised set is empty and the router
    skips it — correct behavior for a backend whose served models
    aren't yet known."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    try:
        backend = CodexAuthVaultBackend(
            id="vault-a",
            vault=vault,
            advertised_models=frozenset(),
        )
        assert backend.advertised_models == frozenset()
    finally:
        await vault.aclose()


# ---------- dynamic model discovery -----------------------------------------


async def test_refresh_advertised_models_updates_cache_from_200(tmp_path: Path) -> None:
    """A successful upstream /models response replaces the static set."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "model-a0e7", "display_name": "MODEL-A0E7"},
                    {"slug": "model-a0c3", "display_name": "MODEL-A0G4"},
                    {"slug": "model-a0b3", "display_name": "Brand New Model"},
                ]
            },
        )

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"stale-cold-start-model"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        # Before refresh: static fallback is what's advertised.
        assert backend.advertised_models == frozenset({"stale-cold-start-model"})
        await backend.refresh_advertised_models()
        # After refresh: dynamic set replaces static. Includes a model the
        # operator never wrote into TOML — that's the whole point.
        assert backend.advertised_models == frozenset({"model-a0e7", "model-a0c3", "model-a0b3"})
    finally:
        await backend.aclose()


async def test_refresh_advertised_models_silently_skips_on_4xx(tmp_path: Path) -> None:
    """A 4xx from the upstream /models endpoint must NOT raise — backend
    silently keeps using its static fallback set so startup doesn't crash.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "expired"}})

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.refresh_advertised_models()  # must not raise
        # Cache untouched → static fallback still in use.
        assert backend.advertised_models == frozenset({"model-a0e7"})
    finally:
        await backend.aclose()


async def test_refresh_advertised_models_silently_skips_on_network_error(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("network down")

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.refresh_advertised_models()  # must not raise
        assert backend.advertised_models == frozenset({"model-a0e7"})
    finally:
        await backend.aclose()


async def test_refresh_advertised_models_respects_ttl(tmp_path: Path) -> None:
    """Successive calls within models_refresh_s should NOT hit upstream
    again — the cache TTL throttles the refresh rate.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        return httpx.Response(200, json={"models": [{"slug": "model-a0e7"}]})

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"cold"}),
        models_refresh_s=3600.0,
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.refresh_advertised_models(now=1000.0)
        await backend.refresh_advertised_models(now=1500.0)  # within TTL
        assert call_count["n"] == 1
        await backend.refresh_advertised_models(now=1000.0 + 4000.0)  # past TTL
        assert call_count["n"] == 2
    finally:
        await backend.aclose()


async def test_refresh_advertised_models_handles_data_shape(tmp_path: Path) -> None:
    """Tolerate the OpenAI-compatible {"data": [{"id": "..."}]} shape too,
    since the upstream surface has shipped both at different times.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": [{"id": "model-a0d8"}]})

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"cold"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.refresh_advertised_models()
        assert backend.advertised_models == frozenset({"model-a0d8"})
    finally:
        await backend.aclose()


async def test_refresh_advertised_models_extracts_full_metadata(tmp_path: Path) -> None:
    """When the upstream `/models` response includes the rich ModelInfo shape
    (supported_reasoning_levels, supported_in_api, priority, visibility, etc.),
    the backend captures it all under `model_metadata`. Removes the need to
    hardcode reasoning levels / completion-shape heuristics on this side.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)

    rich_payload = {
        "models": [
            {
                "slug": "model-a0e7",
                "display_name": "model-a0e7",
                "description": "Strong model for everyday coding.",
                "context_window": 272000,
                "supported_in_api": True,
                "visibility": "list",
                "priority": 16,
                "default_reasoning_level": "medium",
                "supported_reasoning_levels": [
                    {"effort": "low"},
                    {"effort": "medium"},
                    {"effort": "high"},
                    {"effort": "xhigh"},
                ],
                "input_modalities": ["text", "image"],
            },
            {
                "slug": "codex-auto-review",
                "supported_in_api": True,
                "visibility": "hide",  # internal helper; should still be captured
                "priority": 43,
                "supported_reasoning_levels": [{"effort": "low"}],
            },
        ]
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=rich_payload)

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"cold"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.refresh_advertised_models()
        # advertised set still has both (filtering happens at the cell grid)
        assert backend.advertised_models == frozenset({"model-a0e7", "codex-auto-review"})
        md = backend.model_metadata
        assert set(md.keys()) == {"model-a0e7", "codex-auto-review"}
        gpt = md["model-a0e7"]
        assert gpt.display_name == "model-a0e7"
        assert gpt.context_window == 272000
        assert gpt.supported_in_api is True
        assert gpt.visibility == "list"
        assert gpt.priority == 16
        assert gpt.default_reasoning_level == "medium"
        assert gpt.supported_reasoning_levels == ("low", "medium", "high", "xhigh")
        assert gpt.input_modalities == ("text", "image")
        # Hidden model is captured with full metadata too — filter is downstream.
        hidden = md["codex-auto-review"]
        assert hidden.visibility == "hide"
        assert hidden.priority == 43
    finally:
        await backend.aclose()


async def test_refresh_advertised_models_minimal_legacy_shape_still_works(
    tmp_path: Path,
) -> None:
    """When the upstream returns only the legacy minimal shape (slug + maybe
    context_length, no ModelInfo extras), advertised_models still populates
    and model_metadata records exist with defensive None fields. The cell
    grid then falls back to the regex + compatibility-effort path.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"slug": "model-a0d8", "context_length": 100000}]})

    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"cold"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        await backend.refresh_advertised_models()
        assert backend.advertised_models == frozenset({"model-a0d8"})
        md = backend.model_metadata
        assert "model-a0d8" in md
        assert md["model-a0d8"].context_window == 100000
        # All the ModelInfo extras are None / empty.
        assert md["model-a0d8"].priority is None
        assert md["model-a0d8"].supported_in_api is None
        assert md["model-a0d8"].supported_reasoning_levels == ()
    finally:
        await backend.aclose()


# ---------- weekly-exhaustion derivation from quota snapshot ----------------


def _make_quota(weekly_used_percent: int | None) -> CodexQuotaSnapshot:
    """Construct a minimal CodexQuotaSnapshot with the field that drives the
    weekly-exhausted derivation; other fields are best-effort defaults.
    """
    return CodexQuotaSnapshot(
        plan_type="plus",
        active_limit="premium",
        five_hourly_used_percent=10,
        weekly_used_percent=weekly_used_percent,
        five_hourly_window_minutes=300,
        weekly_window_minutes=10080,
        five_hourly_reset_at=None,
        weekly_reset_at=None,
        five_hourly_reset_after_seconds=None,
        weekly_reset_after_seconds=None,
        five_hourly_over_weekly_limit_percent=None,
        credits_balance=None,
        credits_has_credits=False,
        credits_unlimited=False,
        observed_at=0.0,
    )


async def test_usage_snapshot_marks_weekly_exhausted_from_quota(tmp_path: Path) -> None:
    """When upstream quota reports weekly_used_percent >= 99, usage_snapshot
    must surface weekly_exhausted=True so the selector demotes this backend
    before spending another real request to learn it the hard way.
    """
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})),
    )
    try:
        # No quota seen yet: weekly_exhausted defaults to False.
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is False

        # Simulate a recent upstream call that came back with weekly=99.
        backend._last_quota = _make_quota(weekly_used_percent=99)  # type: ignore[attr-defined]
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is True

        # 100 should also be marked exhausted.
        backend._last_quota = _make_quota(weekly_used_percent=100)  # type: ignore[attr-defined]
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is True

        # Below the threshold: not exhausted.
        backend._last_quota = _make_quota(weekly_used_percent=85)  # type: ignore[attr-defined]
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is False

        # Missing percent value: don't infer either way.
        backend._last_quota = _make_quota(weekly_used_percent=None)  # type: ignore[attr-defined]
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is False
    finally:
        await backend.aclose()


async def test_weekly_exhausted_unsets_when_current_quota_low(tmp_path: Path) -> None:
    """Regression: previously the flag was sticky — once set True by a 429 it
    stayed True for the rest of the proxy run even if the next quota
    snapshot showed weekly usage well below the threshold. That demoted
    backends for an entire week on the strength of one momentarily-high
    reading. Now the flag derives PURELY from the current quota.
    """
    from callosum.backend import UsageSnapshot
    from callosum.state import StateStore

    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    state_dir = tmp_path / "state"
    state_store = StateStore(state_dir)
    # Persist a "stuck True" snapshot from a prior session.
    state_store.save_usage(
        "vault-stuck-flag",
        UsageSnapshot(
            remaining_fraction=None,
            cooldown_until_ts=time.time() - 3600,  # already expired
            weekly_exhausted=True,
            probed_at_ts=time.time() - 86400,
        ),
    )

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-stuck-flag",
        vault=vault,
        advertised_models=frozenset({"model-a0e7"}),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        state_store=state_store,
    )
    try:
        # With no fresh quota seen yet, the persisted True is honored
        # (cold-start safety: better to demote until proven otherwise).
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is True

        # First real upstream call observes the actual quota: 44%.
        backend._last_quota = _make_quota(weekly_used_percent=44)  # type: ignore[attr-defined]
        snap = await backend.usage_snapshot()
        assert snap.weekly_exhausted is False, "weekly_exhausted should clear when current quota shows low usage"
    finally:
        await backend.aclose()


# ---------- dynamic codex client_version resolution -------------------------


def test_client_version_reads_from_codex_version_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ~/.codex/version.json exists with a `latest_version` string, the
    helper returns that value (not the hardcoded default).
    """
    monkeypatch.delenv("CODEX_CLIENT_VERSION", raising=False)
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "version.json").write_text(
        json.dumps({"latest_version": "0.999.0", "last_checked_at": "2026-04-29T00:00:00Z"})
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert _resolve_codex_client_version() == "0.999.0"


def test_client_version_falls_back_when_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No ~/.codex/version.json (and no env override) → hardcoded default."""
    monkeypatch.delenv("CODEX_CLIENT_VERSION", raising=False)
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert _resolve_codex_client_version() == _DEFAULT_CLIENT_VERSION


def test_client_version_env_override_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env var beats both the file and the default — operator can pin."""
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "version.json").write_text(json.dumps({"latest_version": "0.999.0"}))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("CODEX_CLIENT_VERSION", "9.9.9-pinned")
    assert _resolve_codex_client_version() == "9.9.9-pinned"


def test_client_version_falls_back_when_file_malformed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Malformed JSON or missing `latest_version` field → hardcoded default."""
    monkeypatch.delenv("CODEX_CLIENT_VERSION", raising=False)
    home = tmp_path / "home"
    (home / ".codex").mkdir(parents=True)
    (home / ".codex" / "version.json").write_text("{not valid json")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert _resolve_codex_client_version() == _DEFAULT_CLIENT_VERSION


# ---------- offline / transport-failure tracking ---------------------------


async def test_transport_failures_set_cooldown_after_threshold(tmp_path: Path) -> None:
    """Three consecutive transport-level errors (ConnectError, DNS failure,
    etc.) flip the backend into cooldown so `_filter_cells_to_routable`
    excludes it. Critical for offline-failover: without this, the
    recommender keeps choosing remote cells the request can never reach."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    call_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        call_count["n"] += 1
        raise httpx.ConnectError("offline")

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        # First 2 failures: no cooldown yet.
        for _ in range(2):
            with pytest.raises(BackendError):
                await backend.responses({"model": "model-a0d0", "input": "hi"})
        snap = await backend.usage_snapshot()
        assert snap.cooldown_until_ts is None
        # 3rd failure crosses the threshold → cooldown set.
        with pytest.raises(BackendError):
            await backend.responses({"model": "model-a0d0", "input": "hi"})
        snap = await backend.usage_snapshot()
        assert snap.cooldown_until_ts is not None
        assert snap.cooldown_until_ts > time.time()
    finally:
        await backend.aclose()


async def test_transport_success_clears_offline_cooldown(tmp_path: Path) -> None:
    """One successful round-trip wipes the consecutive-failures counter and
    clears the transport cooldown. Otherwise the backend would stay
    excluded long after the network comes back."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    failure_count = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if failure_count["n"] < 3:
            failure_count["n"] += 1
            raise httpx.ConnectError("offline")
        # 4th call: succeed. Return a minimal Responses-API SSE stream.
        return _sse_response({"id": "r1", "output": []})

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        # Trip the offline cooldown.
        for _ in range(3):
            with pytest.raises(BackendError):
                await backend.responses({"model": "model-a0d0", "input": "hi"})
        snap = await backend.usage_snapshot()
        assert snap.cooldown_until_ts is not None
        # One successful call → counter resets, cooldown cleared.
        await backend.responses({"model": "model-a0d0", "input": "hi"})
        snap = await backend.usage_snapshot()
        assert snap.cooldown_until_ts is None
    finally:
        await backend.aclose()


# ---------- cell_capabilities --------------------------------------------


async def test_cell_capabilities_uses_model_metadata_when_present(tmp_path: Path) -> None:
    """When the upstream catalog included context_window + input_modalities,
    cell_capabilities surfaces them so the routing capability filter
    sees the real values, not Codex defaults."""
    from callosum.cell_grid import ModelMetadata

    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e8"}),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    try:
        # Inject explicit metadata as if the catalog had populated it.
        backend._model_metadata["model-a0e8"] = ModelMetadata(
            slug="model-a0e8",
            context_window=400_000,
            input_modalities=("text", "image"),
        )
        caps = backend.cell_capabilities("model-a0e8")
        assert caps.context_window == 400_000
        assert caps.modalities == frozenset({"text", "image"})
        assert caps.supports_tools is True
        assert caps.cost_rank == 10
    finally:
        await backend.aclose()


async def test_cell_capabilities_falls_back_to_codex_defaults(tmp_path: Path) -> None:
    """Cold start (no catalog poll yet) → no per-model metadata. Return
    conservative defaults so the request can still be routed."""
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0e8"}),
        transport=httpx.MockTransport(lambda r: httpx.Response(200)),
    )
    try:
        caps = backend.cell_capabilities("model-a0e8")
        assert caps.context_window == 256_000
        assert caps.modalities == frozenset({"text"})
        assert caps.supports_tools is True
        assert caps.cost_rank == 10
    finally:
        await backend.aclose()


# ---------- finish_reason translation ----------------------------------


def test_responses_to_chat_finish_reason_completed_maps_to_stop() -> None:
    assert _responses_to_chat_finish_reason({"status": "completed"}) == "stop"


def test_responses_to_chat_finish_reason_incomplete_max_tokens_maps_to_length() -> None:
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
    }
    assert _responses_to_chat_finish_reason(payload) == "length"


def test_responses_to_chat_finish_reason_incomplete_content_filter() -> None:
    payload = {
        "status": "incomplete",
        "incomplete_details": {"reason": "content_filter"},
    }
    assert _responses_to_chat_finish_reason(payload) == "content_filter"


def test_responses_to_chat_finish_reason_incomplete_without_details_maps_to_length() -> None:
    """Truncation occurred but the upstream didn't say why. Safer to
    surface 'length' than to silently report 'stop' — clients use the
    distinction to decide whether the response is reliable."""
    assert _responses_to_chat_finish_reason({"status": "incomplete"}) == "length"


def test_responses_to_chat_finish_reason_unknown_status_falls_back_to_stop() -> None:
    """Unknown/missing status: 'stop' is the safest default. Mapping
    unknowns to 'length' would cause spurious 'truncated' framing in
    client UIs."""
    assert _responses_to_chat_finish_reason({}) == "stop"
    assert _responses_to_chat_finish_reason({"status": "failed"}) == "stop"


def test_responses_to_chat_response_propagates_length_finish_reason() -> None:
    """The bug this fix addresses: a budget-eaten response (status
    incomplete + max_output_tokens) was reported with finish_reason
    'stop' in the chat-completions translation, masking the
    truncation. Verifies the helper now plumbs the right value
    end-to-end."""
    payload = {
        "id": "resp_abc",
        "model": "model-a0a6",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "partial"}],
            }
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 100,
            "total_tokens": 110,
        },
    }
    chat = _responses_to_chat_response(payload, model="model-a0a6")
    assert chat["choices"][0]["finish_reason"] == "length"
    assert chat["choices"][0]["message"]["content"] == "partial"


def test_responses_to_chat_response_preserves_completed_as_stop() -> None:
    payload = {
        "id": "resp_xyz",
        "model": "model-a0e8",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "done."}],
            }
        ],
    }
    chat = _responses_to_chat_response(payload, model="model-a0e8")
    assert chat["choices"][0]["finish_reason"] == "stop"


# ---------- catalog persistence / warm-start (parity with credential_proxy) --


async def test_refresh_persists_catalog_to_state_store(tmp_path: Path) -> None:
    """A successful /models refresh persists the catalog so the next cold boot
    warm-starts from it. Parity with CredentialProxyBackend."""
    from callosum.state import StateStore

    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "slug": "model-a0e7",
                        "context_window": 200000,
                        "default_reasoning_level": "xhigh",
                        "supported_reasoning_levels": [{"effort": "low"}, {"effort": "high"}],
                    },
                    {"slug": "model-a0c3", "context_window": 200000},
                ]
            },
        )

    backend = CodexAuthVaultBackend(
        id="persist-vault",
        vault=vault,
        advertised_models=frozenset({"placeholder"}),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )
    try:
        await backend.refresh_advertised_models(now=1000.0)
        assert backend.advertised_models == frozenset({"model-a0e7", "model-a0c3"})
    finally:
        await backend.aclose()

    persisted = store.load_catalog("persist-vault")
    assert persisted is not None
    assert sorted(persisted["advertised_models"]) == ["model-a0e7", "model-a0c3"]
    assert persisted["context_windows"]["model-a0e7"] == 200000
    assert persisted["model_metadata"]["model-a0e7"]["default_reasoning_level"] == "xhigh"
    assert persisted["model_metadata"]["model-a0e7"]["supported_reasoning_levels"] == ["low", "high"]
    assert persisted["fetched_at"] == 1000.0


async def test_warm_start_serves_persisted_catalog_when_refresh_fails(
    tmp_path: Path,
) -> None:
    """Cold-boot empty-catalog fix parity: a fresh backend whose refresh fails
    still reports a non-empty advertised_models warm-started from disk."""
    from callosum.state import StateStore

    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)

    # First boot: refresh succeeds, persists catalog.
    vault1 = _make_vault(auth_path)

    def ok_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "models": [
                    {"slug": "model-a0e7", "context_window": 200000},
                    {"slug": "model-a0e8", "context_window": 400000},
                ]
            },
        )

    b1 = CodexAuthVaultBackend(
        id="warm-vault",
        vault=vault1,
        advertised_models=frozenset(),
        transport=httpx.MockTransport(ok_handler),
        state_store=store,
    )
    try:
        await b1.refresh_advertised_models(now=1000.0)
        assert b1.advertised_models == frozenset({"model-a0e7", "model-a0e8"})
    finally:
        await b1.aclose()

    # Second boot: refresh fails (network down). Warm-start from disk.
    vault2 = _make_vault(auth_path)

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("upstream down at boot", request=request)

    b2 = CodexAuthVaultBackend(
        id="warm-vault",  # same id → same persisted catalog
        vault=vault2,
        advertised_models=frozenset({"static-only-fallback"}),
        transport=httpx.MockTransport(fail_handler),
        state_store=store,
    )
    try:
        assert b2.advertised_models == frozenset({"model-a0e7", "model-a0e8"})
        assert b2.model_context_windows == {"model-a0e7": 200000, "model-a0e8": 400000}
        assert set(b2.model_metadata) == {"model-a0e7", "model-a0e8"}
        await b2.refresh_advertised_models()  # fails, must not clear warm-start
        assert b2.advertised_models == frozenset({"model-a0e7", "model-a0e8"})
    finally:
        await b2.aclose()


async def test_warm_start_corrupted_blob_falls_back_to_static(tmp_path: Path) -> None:
    """A corrupted persisted catalog must not crash startup — fall back to the
    static hint. Untrusted disk state."""
    from callosum.state import StateStore

    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)
    store.save_catalog("bad-vault", {"advertised_models": "not-a-list"})

    vault = _make_vault(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    b = CodexAuthVaultBackend(
        id="bad-vault",
        vault=vault,
        advertised_models=frozenset({"static-fallback"}),
        transport=httpx.MockTransport(handler),
        state_store=store,
    )
    try:
        assert b.advertised_models == frozenset({"static-fallback"})
    finally:
        await b.aclose()
