from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash TEXT NOT NULL UNIQUE,
    key_prefix TEXT NOT NULL,
    label TEXT,
    created_at REAL NOT NULL,
    last_used_at REAL,
    revoked_at REAL
);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
"""


@dataclass(frozen=True, slots=True)
class User:
    id: int
    username: str
    password_hash: str
    created_at: float


@dataclass(frozen=True, slots=True)
class Session:
    token_hash: str
    user_id: int
    created_at: float
    expires_at: float


@dataclass(frozen=True, slots=True)
class ApiKey:
    id: int
    user_id: int
    key_hash: str
    key_prefix: str
    label: str | None
    created_at: float
    last_used_at: float | None
    revoked_at: float | None


class UsernameTakenError(Exception):
    pass


class AuthDB:
    """SQLite-backed store for users, sessions, and API keys.

    Mirrors the UsageLog pattern: WAL mode, single shared connection,
    lock-guarded writes. Hot path (api-key lookup on every /v1/* call) is
    indexed by `key_hash` UNIQUE so it's an O(log n) point lookup.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- users ---------------------------------------------------------

    def insert_user(self, *, username: str, password_hash: str, created_at: float) -> int:
        with self._lock:
            try:
                cursor = self._conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, password_hash, created_at),
                )
            except sqlite3.IntegrityError as exc:
                if "UNIQUE" in str(exc):
                    raise UsernameTakenError(username) from exc
                raise
            user_id = cursor.lastrowid
            if user_id is None:
                raise RuntimeError("sqlite3 did not return a rowid for the inserted user")
            return int(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        row = self._conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None:
            return None
        return User(id=row[0], username=row[1], password_hash=row[2], created_at=row[3])

    def get_user_by_id(self, user_id: int) -> User | None:
        row = self._conn.execute(
            "SELECT id, username, password_hash, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        if row is None:
            return None
        return User(id=row[0], username=row[1], password_hash=row[2], created_at=row[3])

    # ---- sessions ------------------------------------------------------

    def insert_session(self, *, token_hash: str, user_id: int, created_at: float, expires_at: float) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (token_hash, user_id, created_at, expires_at),
            )

    def get_session(self, token_hash: str) -> Session | None:
        row = self._conn.execute(
            "SELECT token_hash, user_id, created_at, expires_at FROM sessions WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        return Session(token_hash=row[0], user_id=row[1], created_at=row[2], expires_at=row[3])

    def delete_session(self, token_hash: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))

    def delete_expired_sessions(self, *, now: float) -> int:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
            return cursor.rowcount

    # ---- api keys ------------------------------------------------------

    def insert_api_key(
        self,
        *,
        user_id: int,
        key_hash: str,
        key_prefix: str,
        label: str | None,
        created_at: float,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO api_keys (user_id, key_hash, key_prefix, label, created_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, key_hash, key_prefix, label, created_at),
            )
            key_id = cursor.lastrowid
            if key_id is None:
                raise RuntimeError("sqlite3 did not return a rowid for the inserted api_key")
            return int(key_id)

    def get_api_key_by_hash(self, key_hash: str) -> ApiKey | None:
        row = self._conn.execute(
            "SELECT id, user_id, key_hash, key_prefix, label, created_at,"
            " last_used_at, revoked_at"
            " FROM api_keys WHERE key_hash = ?",
            (key_hash,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_api_key(row)

    def list_api_keys(self, user_id: int) -> list[ApiKey]:
        rows = self._conn.execute(
            "SELECT id, user_id, key_hash, key_prefix, label, created_at,"
            " last_used_at, revoked_at"
            " FROM api_keys WHERE user_id = ? ORDER BY id",
            (user_id,),
        ).fetchall()
        return [_row_to_api_key(r) for r in rows]

    def count_active_api_keys(self) -> int:
        """Number of non-revoked API keys across all users. Used by the auth
        middleware's 401 diagnostics to distinguish 'wrong key' from
        'empty auth store'."""
        row = self._conn.execute("SELECT COUNT(*) FROM api_keys WHERE revoked_at IS NULL").fetchone()
        return int(row[0]) if row else 0

    def revoke_api_key(self, *, key_id: int, user_id: int, revoked_at: float) -> bool:
        """Revoke a key. Returns True if a row was actually updated.

        Scoped to user_id so a user can only revoke their own keys.
        """
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE api_keys SET revoked_at = ? WHERE id = ? AND user_id = ? AND revoked_at IS NULL",
                (revoked_at, key_id, user_id),
            )
            return cursor.rowcount > 0

    def touch_api_key(self, *, key_id: int, now: float) -> None:
        """Update last_used_at on a key. Best-effort, swallow errors silently
        so a logging side effect cannot break the request flow."""
        try:
            with self._lock:
                self._conn.execute("UPDATE api_keys SET last_used_at = ? WHERE id = ?", (now, key_id))
        except sqlite3.Error:
            pass


def _row_to_api_key(row: tuple) -> ApiKey:  # type: ignore[type-arg]
    return ApiKey(
        id=row[0],
        user_id=row[1],
        key_hash=row[2],
        key_prefix=row[3],
        label=row[4],
        created_at=row[5],
        last_used_at=row[6],
        revoked_at=row[7],
    )
