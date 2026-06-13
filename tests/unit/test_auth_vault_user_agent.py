"""Tests for the User-Agent header on callosum's codex-flow HTTP clients.

httpx's default `python-httpx/X.Y.Z` UA is a strong "automation"
signal to OpenAI's anti-abuse layer. Callosum sets an honest UA that
identifies itself + the codex flow context. These tests pin:

  * The UA names callosum first (we never impersonate).
  * Codex client version is included (matches what `client_version=`
    query param sends, so the picture is consistent).
  * AuthVault and CodexAuthVaultBackend both wire the UA into their
    httpx.AsyncClient.

When this drifts (someone constructs a new httpx client without the
default headers), the test fails. That's the point.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from callosum.auth_vault import AuthVault, _default_headers, _user_agent
from callosum.backends.codex_auth_vault import CodexAuthVaultBackend


def _write_minimal_auth(path: Path) -> None:
    """Bare-minimum auth.json so AuthVault doesn't refuse to construct.
    Values are clearly bogus — they're never used; the test only cares
    about HTTP client construction."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "test-access-not-real",
                    "refresh_token": "test-refresh-not-real",
                    "account_id": "test@example.invalid",
                }
            }
        )
    )


def test_user_agent_names_callosum_first() -> None:
    """We never impersonate codex CLI. The UA must lead with the
    callosum name so that an upstream observer reading the header
    knows where the request is actually from."""
    ua = _user_agent()
    assert ua.startswith("callosum/"), f"UA must start with 'callosum/<version>'; got {ua!r}"


def test_user_agent_includes_codex_flow_context() -> None:
    """The request IS part of the codex auth flow (we hit the same
    endpoints codex CLI hits). Including `codex-flow` and the codex
    client version makes the UA an accurate description, not a
    surface-level identifier."""
    ua = _user_agent()
    assert "codex-flow" in ua, f"UA must include 'codex-flow'; got {ua!r}"
    assert "codex_cli_rs/" in ua, f"UA must include codex_cli_rs version; got {ua!r}"


def test_user_agent_includes_python_runtime() -> None:
    """The python runtime is part of the request's true nature.
    Recording it in the UA helps callosum's own diagnostics later
    (you can grep server-side logs by python version) and matches the
    convention many SDK UAs follow."""
    ua = _user_agent()
    assert "python/" in ua, f"UA must include python/<ver>; got {ua!r}"


def test_user_agent_is_NOT_default_httpx() -> None:
    """The whole reason this code exists: avoid the
    `python-httpx/X.Y.Z` default UA. If someone refactors and the UA
    regresses to httpx's default, this test catches it immediately."""
    ua = _user_agent()
    assert not ua.startswith("python-httpx/"), f"UA must NOT be httpx default; got {ua!r}"


def test_default_headers_carries_user_agent() -> None:
    """The default-headers helper is what httpx client constructors
    consume. Pin that it produces a `User-Agent` key with the same
    string `_user_agent()` returns."""
    headers = _default_headers()
    assert "User-Agent" in headers
    assert headers["User-Agent"] == _user_agent()


def test_auth_vault_sets_user_agent_on_client(tmp_path: Path) -> None:
    """AuthVault's constructor builds an httpx.AsyncClient when no
    explicit client is injected. That auto-built client must carry
    the callosum UA. Test by reading the client's default headers
    after construction."""
    auth_path = tmp_path / "auth.json"
    _write_minimal_auth(auth_path)
    vault = AuthVault(path=auth_path)
    try:
        assert vault._client.headers.get("User-Agent") == _user_agent()
        # Quick negative check: the original httpx default is not present.
        ua = str(vault._client.headers.get("User-Agent"))
        assert not ua.startswith("python-httpx/")
    finally:
        # AuthVault owns the client when it built it; close to avoid
        # leaving an event loop / connection pool behind.
        import asyncio

        asyncio.run(vault.aclose() if hasattr(vault, "aclose") else asyncio.sleep(0))


def test_codex_auth_vault_backend_sets_user_agent(tmp_path: Path) -> None:
    """The CodexAuthVaultBackend has its OWN httpx client (separate
    from AuthVault's, because it talks to a different endpoint —
    /codex/responses vs auth.openai.com/oauth/token). It must ALSO
    carry the UA, otherwise the upstream API calls would still leak
    the python-httpx default."""
    auth_path = tmp_path / "auth.json"
    _write_minimal_auth(auth_path)
    vault = AuthVault(path=auth_path)
    backend = CodexAuthVaultBackend(
        id="test-backend",
        vault=vault,
        advertised_models=frozenset({"model-a0e8"}),
    )
    try:
        assert backend._client.headers.get("User-Agent") == _user_agent()
        ua = str(backend._client.headers.get("User-Agent"))
        assert not ua.startswith("python-httpx/")
    finally:
        import asyncio

        with pytest.MonkeyPatch.context():
            asyncio.run(backend.aclose())


def test_user_agent_falls_back_when_codex_version_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """If the codex CLI install isn't reachable, the UA should still
    be built (with a known fallback). This is the cron / minimal
    environment case — we shouldn't crash just because the version
    file doesn't exist. The UA stays well-formed."""
    monkeypatch.delenv("CODEX_CLIENT_VERSION", raising=False)
    # Point HOME at an empty dir so ~/.codex/version.json is missing.
    monkeypatch.setenv("HOME", str(tmp_path))
    ua = _user_agent()
    assert ua.startswith("callosum/")
    assert "codex_cli_rs/" in ua
