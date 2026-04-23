from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from codex_proxy.auth_vault import AuthVault
from codex_proxy.backends.codex_auth_vault import CodexAuthVaultBackend
from codex_proxy.errors import BackendError


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
        return httpx.Response(
            200,
            json={
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
            },
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
        assert body["stream"] is False
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
            await backend.chat_completions(
                {"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]}
            )
        assert excinfo.value.classification == "rate_limited"
        snapshot = await backend.usage_snapshot()
        assert snapshot.cooldown_until_ts is not None
    finally:
        await backend.aclose()


async def test_auth_invalid_triggers_when_upstream_401(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"code": "invalid_token"}})

    vault = _make_vault(auth_path)
    backend = CodexAuthVaultBackend(
        id="vault-a",
        vault=vault,
        advertised_models=frozenset({"model-a0d0"}),
        transport=httpx.MockTransport(handler),
    )
    try:
        with pytest.raises(BackendError) as excinfo:
            await backend.chat_completions(
                {"model": "model-a0d0", "messages": [{"role": "user", "content": "hi"}]}
            )
        assert excinfo.value.classification == "auth_invalid"
    finally:
        await backend.aclose()


async def test_chat_completions_stream_emits_valid_sse_chunks(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "resp-stream",
                "model": "model-a0d0",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "streamed"}],
                    }
                ],
            },
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
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"id": "resp-xyz", "object": "response"})

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
        # Request passes through unchanged — no chat-to-responses translation.
        assert captured["body"] == request_body
        assert result["id"] == "resp-xyz"
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


async def test_construction_rejects_empty_advertised_models(tmp_path: Path) -> None:
    auth_path = tmp_path / "auth.json"
    _write_auth_json(auth_path)
    vault = _make_vault(auth_path)
    try:
        with pytest.raises(ValueError, match="advertised_models"):
            CodexAuthVaultBackend(
                id="vault-a",
                vault=vault,
                advertised_models=frozenset(),
            )
    finally:
        await vault.aclose()
