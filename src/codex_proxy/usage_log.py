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
    five_hourly_used_percent_before INTEGER,
    five_hourly_used_percent_after INTEGER,
    weekly_used_percent_before INTEGER,
    weekly_used_percent_after INTEGER,
    five_hourly_reset_at INTEGER,
    weekly_reset_at INTEGER,
    five_hourly_over_weekly_limit_percent INTEGER,
    credits_balance TEXT,
    credits_has_credits INTEGER,
    credits_unlimited INTEGER,
    quota_reset_crossover INTEGER NOT NULL DEFAULT 0,
    requested_model TEXT,
    requested_reasoning_effort TEXT,
    routing_mode TEXT
);
CREATE INDEX IF NOT EXISTS idx_requests_ts_start ON requests(ts_start);
CREATE INDEX IF NOT EXISTS idx_requests_backend_id ON requests(backend_id);
-- user_id, api_key_id, requested_*, routing_mode indexes live in _MIGRATIONS so
-- they run AFTER the ALTER-TABLE that adds the columns on pre-existing databases.

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
    # Router (auto-learning / auto) columns. requested_* is what the client asked for
    # before any router rewrote the body; the existing model + reasoning_effort columns
    # continue to mean what was actually served upstream.
    "ALTER TABLE requests ADD COLUMN requested_model TEXT",
    "ALTER TABLE requests ADD COLUMN requested_reasoning_effort TEXT",
    "ALTER TABLE requests ADD COLUMN routing_mode TEXT",
    # Backfill: pre-router rows had no rewriting, so served == requested.
    "UPDATE requests SET requested_model = model WHERE requested_model IS NULL",
    "UPDATE requests SET requested_reasoning_effort = reasoning_effort"
    " WHERE requested_reasoning_effort IS NULL",
    "UPDATE requests SET routing_mode = 'pass-through' WHERE routing_mode IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_requests_routing_mode ON requests(routing_mode)",
    "CREATE INDEX IF NOT EXISTS idx_requests_served_cell ON requests(model, reasoning_effort)",
    # Quota-window column renames: upstream's x-codex-primary-* (5h window) and
    # x-codex-secondary-* (weekly window) are preserved as-is over the wire, but
    # we name our own columns by what they actually mean. RENAME COLUMN is
    # SQLite ≥3.25; running a second time fails with "no such column" since the
    # old name is gone — _apply_migrations swallows that.
    "ALTER TABLE requests RENAME COLUMN primary_used_percent_before"
    " TO five_hourly_used_percent_before",
    "ALTER TABLE requests RENAME COLUMN primary_used_percent_after"
    " TO five_hourly_used_percent_after",
    "ALTER TABLE requests RENAME COLUMN secondary_used_percent_before"
    " TO weekly_used_percent_before",
    "ALTER TABLE requests RENAME COLUMN secondary_used_percent_after TO weekly_used_percent_after",
    "ALTER TABLE requests RENAME COLUMN primary_reset_at TO five_hourly_reset_at",
    "ALTER TABLE requests RENAME COLUMN secondary_reset_at TO weekly_reset_at",
    "ALTER TABLE requests RENAME COLUMN primary_over_secondary_limit_percent"
    " TO five_hourly_over_weekly_limit_percent",
    # Quality labeling for ExploiterRouter training: user feedback and automated signals.
    "ALTER TABLE requests ADD COLUMN quality_score INTEGER",  # -1, 0, +1; NULL = unlabeled
    "ALTER TABLE requests ADD COLUMN quality_label_method TEXT",  # 'user', 'llm_judge_v1', etc
    # Prompt complexity classification for cost-per-complexity routing.
    "ALTER TABLE requests ADD COLUMN prompt_complexity_class INTEGER",  # 1, 2, 3; NULL = not classified
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
    # Router fields. requested_* is what the client sent before any virtual-model
    # rewriting; the existing model/reasoning_effort columns continue to mean what
    # was actually served upstream. routing_mode is one of:
    # 'pass-through' | 'auto-learning' | 'auto'.
    requested_model: str | None = None
    requested_reasoning_effort: str | None = None
    routing_mode: str | None = None
    # Complexity classification from embedded instruction in auto-learning requests.
    # 1, 2, or 3; NULL = not classified (non-auto-learning or marker not found).
    prompt_complexity_class: int | None = None


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
        IF NOT EXISTS for ADD COLUMN or RENAME COLUMN, hence the catch.
        """
        for stmt in _MIGRATIONS:
            try:
                self._conn.execute(stmt)
            except sqlite3.OperationalError as exc:
                msg = str(exc).lower()
                # Already-applied signals: ADD COLUMN re-run, or RENAME COLUMN
                # where the old name is gone (renamed) or the new name exists.
                if (
                    "duplicate column" in msg
                    or "no such column" in msg
                    or "there is already another column" in msg
                ):
                    continue
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
            qb.five_hourly_used_percent if qb is not None else None,
            qa.five_hourly_used_percent if qa is not None else None,
            qb.weekly_used_percent if qb is not None else None,
            qa.weekly_used_percent if qa is not None else None,
            qa.five_hourly_reset_at if qa is not None else None,
            qa.weekly_reset_at if qa is not None else None,
            qa.five_hourly_over_weekly_limit_percent if qa is not None else None,
            _pick(qa, qb, "credits_balance"),
            _bool_to_int(qa.credits_has_credits if qa is not None else None),
            _bool_to_int(qa.credits_unlimited if qa is not None else None),
            1 if crossover else 0,
            entry.requested_model,
            entry.requested_reasoning_effort,
            entry.routing_mode,
            entry.prompt_complexity_class,
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
                    five_hourly_used_percent_before, five_hourly_used_percent_after,
                    weekly_used_percent_before, weekly_used_percent_after,
                    five_hourly_reset_at, weekly_reset_at,
                    five_hourly_over_weekly_limit_percent,
                    credits_balance, credits_has_credits, credits_unlimited,
                    quota_reset_crossover,
                    requested_model, requested_reasoning_effort, routing_mode,
                    prompt_complexity_class
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?
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

    def last_session_prompt_tokens(self, session_id: str) -> int | None:
        """Query the most recent prompt_tokens for a session_id.

        Returns the prompt_tokens from the last request in this session, or None
        if the session is not found or has no prompt_tokens data. Used to infer
        the current context accumulation in a session for context-safe routing.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT prompt_tokens FROM requests"
                " WHERE session_id = ? AND prompt_tokens IS NOT NULL"
                " ORDER BY ts_start DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return row[0] if row is not None else None

    def record_quality(self, request_id: int, score: int, method: str) -> None:
        """Record a quality label for an existing request row.

        `request_id`: the rowid from the requests table
        `score`: -1 (bad), 0 (ok), or 1 (good)
        `method`: identifier for how the label was generated ('user', 'llm_judge_v1', etc)
        """
        if score not in (-1, 0, 1):
            raise ValueError(f"quality_score must be -1, 0, or 1; got {score}")
        with self._lock:
            self._conn.execute(
                "UPDATE requests SET quality_score = ?, quality_label_method = ?"
                " WHERE id = ?",
                (score, method, request_id),
            )

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
        before.five_hourly_used_percent is not None
        and after.five_hourly_used_percent is not None
        and after.five_hourly_used_percent < before.five_hourly_used_percent
    ) or (
        before.weekly_used_percent is not None
        and after.weekly_used_percent is not None
        and after.weekly_used_percent < before.weekly_used_percent
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
