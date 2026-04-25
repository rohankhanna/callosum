from __future__ import annotations

import json
import sqlite3
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path

from codex_proxy.codex_quota import CodexQuotaSnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    latency_ms INTEGER NOT NULL,
    route TEXT NOT NULL,
    stream INTEGER NOT NULL,
    session_id TEXT,
    user_id INTEGER,
    api_key_id INTEGER,
    backend_id TEXT NOT NULL,
    model TEXT,
    reasoning_effort TEXT,
    status INTEGER NOT NULL,
    classification TEXT,
    request_bytes INTEGER,
    response_bytes INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    cached_tokens INTEGER,
    reasoning_tokens INTEGER,
    plan_type TEXT,
    active_limit TEXT,
    primary_used_percent_before INTEGER,
    primary_used_percent_after INTEGER,
    secondary_used_percent_before INTEGER,
    secondary_used_percent_after INTEGER,
    primary_reset_at INTEGER,
    secondary_reset_at INTEGER,
    primary_over_secondary_limit_percent INTEGER,
    credits_balance TEXT,
    credits_has_credits INTEGER,
    credits_unlimited INTEGER,
    quota_reset_crossover INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_requests_ts_start ON requests(ts_start);
CREATE INDEX IF NOT EXISTS idx_requests_backend_id ON requests(backend_id);
CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id);
CREATE INDEX IF NOT EXISTS idx_requests_api_key_id ON requests(api_key_id);

CREATE TABLE IF NOT EXISTS request_bodies (
    request_id INTEGER PRIMARY KEY REFERENCES requests(id) ON DELETE CASCADE,
    req_payload BLOB,
    resp_payload BLOB,
    upstream_headers BLOB
);
"""

# Columns added after the initial v1 schema. ALTER TABLE on each one (guarded
# against "duplicate column" errors) means existing usage_log databases gain
# the new columns automatically on next startup.
_MIGRATIONS = [
    "ALTER TABLE requests ADD COLUMN user_id INTEGER",
    "ALTER TABLE requests ADD COLUMN api_key_id INTEGER",
    "CREATE INDEX IF NOT EXISTS idx_requests_user_id ON requests(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_requests_api_key_id ON requests(api_key_id)",
]


@dataclass(slots=True)
class UsageLogEntry:
    ts_start: float
    ts_end: float
    route: str
    stream: bool
    session_id: str | None
    backend_id: str
    model: str | None
    reasoning_effort: str | None
    status: int
    classification: str | None
    request_bytes: int | None = None
    response_bytes: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    quota_before: CodexQuotaSnapshot | None = None
    quota_after: CodexQuotaSnapshot | None = None
    # When body capture is on, these carry the full payloads (already serialized
    # to bytes; the log compresses them before writing).
    req_payload: bytes | None = None
    resp_payload: bytes | None = None
    upstream_headers: dict[str, str] | None = None
    # Multi-tenant attribution. Both null in single-operator mode (no auth db).
    user_id: int | None = None
    api_key_id: int | None = None


class UsageLog:
    """SQLite-backed request log.

    Writes are synchronous but fast (single-digit ms per insert on a local
    SSD). All operations guarded by an internal lock so the log is safe even
    if uvicorn is ever configured with multiple workers — though it is not,
    by default, for this proxy.
    """

    def __init__(self, path: Path, *, capture_bodies: bool = True) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._capture_bodies = capture_bodies
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we lock manually
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._apply_migrations()

    def _apply_migrations(self) -> None:
        """Apply post-v1 ALTER TABLE migrations idempotently.

        Each statement is wrapped in its own try/except so re-running on a
        DB that already has the column is a no-op. SQLite has no
        IF NOT EXISTS for ADD COLUMN, hence the catch.
        """
        for stmt in _MIGRATIONS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                # "duplicate column name" means the migration was already
                # applied — that's fine. Anything else is a real error.
                if "duplicate column" not in str(exc).lower():
                    raise

    @property
    def path(self) -> Path:
        return self._path

    @property
    def capture_bodies(self) -> bool:
        return self._capture_bodies

    def record(self, entry: UsageLogEntry) -> int:
        """Insert one row; return its rowid.

        When body capture is on and any of req/resp/headers is set, also
        writes a companion row into `request_bodies` with zlib-compressed
        blobs.
        """
        latency_ms = int(round((entry.ts_end - entry.ts_start) * 1000))
        crossover = _is_reset_crossover(entry.quota_before, entry.quota_after)
        qb = entry.quota_before
        qa = entry.quota_after
        row = (
            entry.ts_start,
            entry.ts_end,
            latency_ms,
            entry.route,
            1 if entry.stream else 0,
            entry.session_id,
            entry.user_id,
            entry.api_key_id,
            entry.backend_id,
            entry.model,
            entry.reasoning_effort,
            entry.status,
            entry.classification,
            entry.request_bytes,
            entry.response_bytes,
            entry.prompt_tokens,
            entry.completion_tokens,
            entry.total_tokens,
            entry.cached_tokens,
            entry.reasoning_tokens,
            _pick(qa, qb, "plan_type"),
            _pick(qa, qb, "active_limit"),
            qb.primary_used_percent if qb is not None else None,
            qa.primary_used_percent if qa is not None else None,
            qb.secondary_used_percent if qb is not None else None,
            qa.secondary_used_percent if qa is not None else None,
            qa.primary_reset_at if qa is not None else None,
            qa.secondary_reset_at if qa is not None else None,
            qa.primary_over_secondary_limit_percent if qa is not None else None,
            _pick(qa, qb, "credits_balance"),
            _bool_to_int(qa.credits_has_credits if qa is not None else None),
            _bool_to_int(qa.credits_unlimited if qa is not None else None),
            1 if crossover else 0,
        )
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT INTO requests (
                    ts_start, ts_end, latency_ms, route, stream,
                    session_id, user_id, api_key_id,
                    backend_id, model, reasoning_effort,
                    status, classification,
                    request_bytes, response_bytes,
                    prompt_tokens, completion_tokens, total_tokens,
                    cached_tokens, reasoning_tokens,
                    plan_type, active_limit,
                    primary_used_percent_before, primary_used_percent_after,
                    secondary_used_percent_before, secondary_used_percent_after,
                    primary_reset_at, secondary_reset_at,
                    primary_over_secondary_limit_percent,
                    credits_balance, credits_has_credits, credits_unlimited,
                    quota_reset_crossover
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?
                )
                """,
                row,
            )
            request_id = cursor.lastrowid
            if request_id is None:
                raise RuntimeError("sqlite3 did not return a rowid for the inserted request")
            if self._capture_bodies and (
                entry.req_payload is not None
                or entry.resp_payload is not None
                or entry.upstream_headers is not None
            ):
                self._conn.execute(
                    "INSERT INTO request_bodies"
                    " (request_id, req_payload, resp_payload, upstream_headers)"
                    " VALUES (?, ?, ?, ?)",
                    (
                        request_id,
                        _compress(entry.req_payload),
                        _compress(entry.resp_payload),
                        _compress(
                            json.dumps(entry.upstream_headers).encode()
                            if entry.upstream_headers is not None
                            else None
                        ),
                    ),
                )
            return int(request_id)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _compress(data: bytes | None) -> bytes | None:
    if data is None:
        return None
    return zlib.compress(data, level=6)


def decompress(blob: bytes | None) -> bytes | None:
    """Inverse of `_compress`. Public so analysis tools can read bodies back."""
    if blob is None:
        return None
    return zlib.decompress(blob)


def _is_reset_crossover(
    before: CodexQuotaSnapshot | None, after: CodexQuotaSnapshot | None
) -> bool:
    if before is None or after is None:
        return False
    # Either window resetting during the request looks like post < pre.
    return (
        before.primary_used_percent is not None
        and after.primary_used_percent is not None
        and after.primary_used_percent < before.primary_used_percent
    ) or (
        before.secondary_used_percent is not None
        and after.secondary_used_percent is not None
        and after.secondary_used_percent < before.secondary_used_percent
    )


def _pick(primary: object | None, fallback: object | None, attr: str) -> object:
    if primary is not None:
        value = getattr(primary, attr, None)
        if value is not None:
            return value
    if fallback is not None:
        return getattr(fallback, attr, None)
    return None


def _bool_to_int(value: bool | None) -> int | None:
    if value is None:
        return None
    return 1 if value else 0
