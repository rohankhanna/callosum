from __future__ import annotations

from pathlib import Path

import pytest

from callosum.auth_db import AuthDB, UsernameTakenError


def _open(tmp_path: Path) -> AuthDB:
    return AuthDB(tmp_path / "auth.sqlite")


def test_insert_and_lookup_user(tmp_path: Path) -> None:
    db = _open(tmp_path)
    user_id = db.insert_user(username="alice", password_hash="$argon2$x", created_at=1.0)
    assert user_id >= 1
    user = db.get_user_by_username("alice")
    assert user is not None and user.username == "alice" and user.id == user_id
    assert db.get_user_by_id(user_id) == user
    db.close()


def test_duplicate_username_raises(tmp_path: Path) -> None:
    db = _open(tmp_path)
    db.insert_user(username="alice", password_hash="x", created_at=1.0)
    with pytest.raises(UsernameTakenError):
        db.insert_user(username="alice", password_hash="y", created_at=2.0)
    db.close()


def test_session_round_trip(tmp_path: Path) -> None:
    db = _open(tmp_path)
    user_id = db.insert_user(username="bob", password_hash="x", created_at=1.0)
    db.insert_session(token_hash="hash-1", user_id=user_id, created_at=1.0, expires_at=100.0)
    session = db.get_session("hash-1")
    assert session is not None and session.user_id == user_id
    db.delete_session("hash-1")
    assert db.get_session("hash-1") is None
    db.close()


def test_delete_expired_sessions_drops_only_old(tmp_path: Path) -> None:
    db = _open(tmp_path)
    user_id = db.insert_user(username="bob", password_hash="x", created_at=1.0)
    db.insert_session(token_hash="old", user_id=user_id, created_at=1.0, expires_at=10.0)
    db.insert_session(token_hash="fresh", user_id=user_id, created_at=1.0, expires_at=1000.0)
    deleted = db.delete_expired_sessions(now=100.0)
    assert deleted == 1
    assert db.get_session("old") is None
    assert db.get_session("fresh") is not None
    db.close()


def test_api_key_create_list_and_revoke(tmp_path: Path) -> None:
    db = _open(tmp_path)
    user_id = db.insert_user(username="bob", password_hash="x", created_at=1.0)
    key_id = db.insert_api_key(
        user_id=user_id,
        key_hash="kh-1",
        key_prefix="abcd1234",
        label="laptop",
        created_at=2.0,
    )
    keys = db.list_api_keys(user_id)
    assert [k.id for k in keys] == [key_id]
    assert keys[0].label == "laptop"
    assert keys[0].revoked_at is None
    revoked = db.revoke_api_key(key_id=key_id, user_id=user_id, revoked_at=5.0)
    assert revoked is True
    again = db.list_api_keys(user_id)
    assert again[0].revoked_at == 5.0
    # Revoking twice is a no-op (returns False).
    assert db.revoke_api_key(key_id=key_id, user_id=user_id, revoked_at=6.0) is False
    db.close()


def test_revoke_other_users_key_is_rejected(tmp_path: Path) -> None:
    db = _open(tmp_path)
    alice = db.insert_user(username="alice", password_hash="x", created_at=1.0)
    mallory = db.insert_user(username="mallory", password_hash="x", created_at=1.0)
    key_id = db.insert_api_key(
        user_id=alice, key_hash="kh", key_prefix="zzzz1111", label=None, created_at=2.0
    )
    assert db.revoke_api_key(key_id=key_id, user_id=mallory, revoked_at=5.0) is False
    # Alice's key is still active.
    assert db.list_api_keys(alice)[0].revoked_at is None
    db.close()


def test_touch_api_key_updates_last_used_at(tmp_path: Path) -> None:
    db = _open(tmp_path)
    user_id = db.insert_user(username="bob", password_hash="x", created_at=1.0)
    key_id = db.insert_api_key(
        user_id=user_id, key_hash="kh", key_prefix="xxxx0000", label=None, created_at=2.0
    )
    assert db.list_api_keys(user_id)[0].last_used_at is None
    db.touch_api_key(key_id=key_id, now=42.0)
    assert db.list_api_keys(user_id)[0].last_used_at == 42.0
    db.close()


def test_get_api_key_by_hash(tmp_path: Path) -> None:
    db = _open(tmp_path)
    user_id = db.insert_user(username="bob", password_hash="x", created_at=1.0)
    db.insert_api_key(
        user_id=user_id,
        key_hash="lookup-me",
        key_prefix="abcdwxyz",
        label=None,
        created_at=2.0,
    )
    got = db.get_api_key_by_hash("lookup-me")
    assert got is not None and got.user_id == user_id
    assert db.get_api_key_by_hash("nope") is None
    db.close()
