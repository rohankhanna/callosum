from __future__ import annotations

import time
from pathlib import Path

import pytest

from codex_proxy.auth import (
    ApiKeyInvalidError,
    AuthService,
    InvalidCredentialsError,
    SessionInvalidError,
)
from codex_proxy.auth_db import AuthDB


def _service(tmp_path: Path, *, ttl: int = 1800) -> AuthService:
    return AuthService(AuthDB(tmp_path / "auth.sqlite"), session_ttl_seconds=ttl)


def test_register_then_login_returns_session_token(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    user = svc.register(username="alice", password="hunter2")
    assert user.username == "alice"
    issued = svc.login(username="alice", password="hunter2")
    assert issued.plaintext  # plaintext is returned to the caller
    # plaintext is NOT the same as the stored token_hash (sanity check that we
    # store hashed, not raw).
    assert issued.session.token_hash != issued.plaintext
    svc.db.close()


def test_register_rejects_duplicate_username(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.register(username="alice", password="x")
    with pytest.raises(InvalidCredentialsError):
        svc.register(username="alice", password="y")
    svc.db.close()


def test_login_rejects_bad_password(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.register(username="alice", password="hunter2")
    with pytest.raises(InvalidCredentialsError):
        svc.login(username="alice", password="wrong")
    svc.db.close()


def test_login_rejects_unknown_user_with_same_error(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    with pytest.raises(InvalidCredentialsError):
        svc.login(username="ghost", password="anything")
    svc.db.close()


def test_resolve_session_returns_user_id(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    user = svc.register(username="alice", password="x")
    issued = svc.login(username="alice", password="x")
    session = svc.resolve_session(issued.plaintext)
    assert session.user_id == user.id
    svc.db.close()


def test_resolve_session_rejects_expired(tmp_path: Path) -> None:
    svc = _service(tmp_path, ttl=-1)  # any session born already expired
    svc.register(username="alice", password="x")
    issued = svc.login(username="alice", password="x")
    with pytest.raises(SessionInvalidError):
        svc.resolve_session(issued.plaintext)
    svc.db.close()


def test_logout_invalidates_session(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.register(username="alice", password="x")
    issued = svc.login(username="alice", password="x")
    svc.logout(issued.plaintext)
    with pytest.raises(SessionInvalidError):
        svc.resolve_session(issued.plaintext)
    svc.db.close()


def test_create_and_resolve_api_key(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    user = svc.register(username="alice", password="x")
    issued = svc.create_api_key(user_id=user.id, label="laptop")
    assert issued.plaintext
    resolved = svc.resolve_api_key(issued.plaintext)
    assert resolved.user_id == user.id
    assert resolved.label == "laptop"
    # touch_api_key updates last_used_at as a side effect.
    fresh = svc.list_api_keys(user_id=user.id)[0]
    assert fresh.last_used_at is not None
    svc.db.close()


def test_revoked_api_key_is_rejected(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    user = svc.register(username="alice", password="x")
    issued = svc.create_api_key(user_id=user.id)
    svc.revoke_api_key(key_id=issued.api_key.id, user_id=user.id)
    with pytest.raises(ApiKeyInvalidError):
        svc.resolve_api_key(issued.plaintext)
    svc.db.close()


def test_unknown_api_key_is_rejected(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    with pytest.raises(ApiKeyInvalidError):
        svc.resolve_api_key("fake-token-that-was-never-issued")
    svc.db.close()


def test_passwords_are_not_stored_plaintext(tmp_path: Path) -> None:
    svc = _service(tmp_path)
    svc.register(username="alice", password="hunter2")
    user = svc.db.get_user_by_username("alice")
    assert user is not None
    # password_hash should be argon2id format ($argon2id$...) not raw.
    assert user.password_hash.startswith("$argon2")
    assert "hunter2" not in user.password_hash
    svc.db.close()


def test_session_ttl_is_honored(tmp_path: Path) -> None:
    svc = _service(tmp_path, ttl=60)
    svc.register(username="alice", password="x")
    issued = svc.login(username="alice", password="x")
    # expires_at should be ~60s ahead of now.
    drift = abs(issued.session.expires_at - (time.time() + 60))
    assert drift < 5
    svc.db.close()
