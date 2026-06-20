from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from callosum.codex_quota import CodexQuotaSnapshot
from callosum.peer_quality import PeerQualityOpinion

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

-- One row per cell-level attempt within a single user request. The
-- `requests` row records the final outcome (last attempted cell, success
-- or failure); this table preserves the full attempt chain so future
-- training can learn from rerouted requests ("recommender picked X, X
-- failed with classification=Y, Y succeeded — when do we see this?").
CREATE TABLE IF NOT EXISTS request_routing_attempts (
    request_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    attempt_idx INTEGER NOT NULL,
    backend_id TEXT,
    model TEXT,
    reasoning_effort TEXT,
    status INTEGER,
    classification TEXT,
    latency_ms INTEGER,
    error_message TEXT,
    PRIMARY KEY (request_id, attempt_idx)
);
CREATE INDEX IF NOT EXISTS idx_request_routing_attempts_request_id
    ON request_routing_attempts(request_id);

CREATE TABLE IF NOT EXISTS peer_quality_opinions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL REFERENCES requests(id) ON DELETE CASCADE,
    session_id TEXT,
    judge_backend_id TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judge_reasoning_effort TEXT,
    subject_request_id INTEGER REFERENCES requests(id) ON DELETE CASCADE,
    subject_model TEXT NOT NULL,
    subject_reasoning_effort TEXT,
    score INTEGER NOT NULL CHECK (score IN (-1, 0, 1)),
    nonce TEXT NOT NULL,
    reason TEXT,
    raw_marker TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_peer_quality_opinions_request_id
    ON peer_quality_opinions(request_id);
CREATE INDEX IF NOT EXISTS idx_peer_quality_matrix
    ON peer_quality_opinions(judge_model, judge_reasoning_effort, subject_model, subject_reasoning_effort);

CREATE TABLE IF NOT EXISTS peer_quality_capture_metrics (
    request_id INTEGER PRIMARY KEY REFERENCES requests(id) ON DELETE CASCADE,
    session_id TEXT,
    judge_backend_id TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    judge_reasoning_effort TEXT,
    nonce TEXT NOT NULL,
    opinion_count INTEGER NOT NULL,
    echo_count INTEGER NOT NULL,
    malformed_count INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_peer_quality_capture_metrics_session
    ON peer_quality_capture_metrics(session_id, created_at);
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
    "UPDATE requests SET requested_reasoning_effort = reasoning_effort WHERE requested_reasoning_effort IS NULL",
    "UPDATE requests SET routing_mode = 'pass-through' WHERE routing_mode IS NULL",
    "CREATE INDEX IF NOT EXISTS idx_requests_routing_mode ON requests(routing_mode)",
    "CREATE INDEX IF NOT EXISTS idx_requests_served_cell ON requests(model, reasoning_effort)",
    # Quota-window column renames: upstream's x-codex-primary-* (5h window) and
    # x-codex-secondary-* (weekly window) are preserved as-is over the wire, but
    # we name our own columns by what they actually mean. RENAME COLUMN is
    # SQLite ≥3.25; running a second time fails with "no such column" since the
    # old name is gone — _apply_migrations swallows that.
    "ALTER TABLE requests RENAME COLUMN primary_used_percent_before TO five_hourly_used_percent_before",
    "ALTER TABLE requests RENAME COLUMN primary_used_percent_after TO five_hourly_used_percent_after",
    "ALTER TABLE requests RENAME COLUMN secondary_used_percent_before TO weekly_used_percent_before",
    "ALTER TABLE requests RENAME COLUMN secondary_used_percent_after TO weekly_used_percent_after",
    "ALTER TABLE requests RENAME COLUMN primary_reset_at TO five_hourly_reset_at",
    "ALTER TABLE requests RENAME COLUMN secondary_reset_at TO weekly_reset_at",
    "ALTER TABLE requests RENAME COLUMN primary_over_secondary_limit_percent TO five_hourly_over_weekly_limit_percent",
    # Quality labeling for cost-optimal router training: user feedback and automated signals.
    "ALTER TABLE requests ADD COLUMN quality_score INTEGER",  # -1, 0, +1; NULL = unlabeled
    "ALTER TABLE requests ADD COLUMN quality_label_method TEXT",  # 'user', 'llm_judge_v1', etc
    # Prompt complexity classification for cost-per-complexity routing.
    "ALTER TABLE requests ADD COLUMN prompt_complexity_class INTEGER",  # 1, 2, 3; NULL = not classified
    # Text extraction for label UI keyword search.
    "ALTER TABLE requests ADD COLUMN prompt_text TEXT",
    "ALTER TABLE requests ADD COLUMN response_text TEXT",
    # FTS5 virtual table for full-text search (content table references requests).
    """CREATE VIRTUAL TABLE IF NOT EXISTS requests_fts USING fts5(
    prompt_text,
    response_text,
    content='requests',
    content_rowid='id',
    tokenize='porter unicode61'
)""",
    # Triggers to keep FTS5 in sync with requests table.
    """CREATE TRIGGER IF NOT EXISTS requests_ai AFTER INSERT ON requests BEGIN
  INSERT INTO requests_fts(rowid, prompt_text, response_text)
  VALUES (new.id, new.prompt_text, new.response_text);
END""",
    """CREATE TRIGGER IF NOT EXISTS requests_au AFTER UPDATE ON requests BEGIN
  INSERT INTO requests_fts(requests_fts, rowid, prompt_text, response_text)
  VALUES ('delete', old.id, old.prompt_text, old.response_text);
  INSERT INTO requests_fts(rowid, prompt_text, response_text)
  VALUES (new.id, new.prompt_text, new.response_text);
END""",
    """CREATE TRIGGER IF NOT EXISTS requests_ad AFTER DELETE ON requests BEGIN
  INSERT INTO requests_fts(requests_fts, rowid, prompt_text, response_text)
  VALUES ('delete', old.id, old.prompt_text, old.response_text);
END""",
    # Recommender provenance — captures which classifier cell made the
    # routing decision, the literal text it returned, and whether the row
    # came from a live upstream call vs cache/fallback/alternative. NULL on
    # pass-through (recommender didn't fire) and on cache/fallback rows
    # (no classifier output to capture). The intended consumer is a future
    # local-classifier training pipeline: filter to
    # recommender_source IN ('upstream', 'alternative') and learn
    # input=prompt_text → label=(model, reasoning_effort).
    "ALTER TABLE requests ADD COLUMN recommender_classifier_cell TEXT",
    "ALTER TABLE requests ADD COLUMN recommender_raw_output TEXT",
    "ALTER TABLE requests ADD COLUMN recommender_source TEXT",
    "CREATE INDEX IF NOT EXISTS idx_requests_recommender_source ON requests(recommender_source)",
    # Phase 4 learning-router columns: raw float32 bytes of the prompt
    # embedding and (optionally) response embedding, populated when the
    # configured EmbeddingProvider is non-noop. Used by the kNN
    # predictor's reload() at startup and after each predictor-train
    # Dispatch job checkpoint. Existing production DBs may already have
    # these columns from an earlier label-UI plan migration; the
    # ALTER TABLE is wrapped in a try/except in the migration loop so
    # a "duplicate column" error is silently ignored.
    "ALTER TABLE requests ADD COLUMN prompt_embedding BLOB",
    "ALTER TABLE requests ADD COLUMN response_embedding BLOB",
    # Per-request effective routing mode. Distinct from `routing_mode`
    # which records recommender provenance (`pass-through` /
    # `auto-learning` / `auto`); `effective_routing_mode` captures
    # which routing-mode lens this individual request was processed
    # under — most importantly whether it was redirected to the
    # remote-only path as part of the canary baseline. Values:
    # `auto` (normal auto-mode), `canary_redirect` (operator was in
    # auto, this request was selected for the canary), `forced_remote`
    # / `forced_local` / `forced_offline` (operator explicitly chose
    # that mode), `pass-through` (recommender did not fire).
    # The dev loop and operator dashboards compare success rates
    # bucketed by this column to detect local-side regressions
    # against the remote-only baseline.
    "ALTER TABLE requests ADD COLUMN effective_routing_mode TEXT",
    "CREATE INDEX IF NOT EXISTS idx_requests_effective_routing_mode ON requests(effective_routing_mode)",
    "ALTER TABLE peer_quality_opinions ADD COLUMN subject_request_id INTEGER",
    "CREATE INDEX IF NOT EXISTS idx_peer_quality_opinions_subject_request_id "
    "ON peer_quality_opinions(subject_request_id)",
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
    # Per-request effective mode — see _MIGRATIONS column comment.
    # Populated by the request handler from the canary scheduler's
    # decision plus the operator state's mode. NULL on rows recorded
    # before this field landed; treat NULL as "auto" for analytical
    # purposes (most pre-existing traffic is auto-mode).
    effective_routing_mode: str | None = None
    # Complexity classification from embedded instruction in auto-learning requests.
    # 1, 2, or 3; NULL = not classified (non-auto-learning or marker not found).
    prompt_complexity_class: int | None = None
    # Original OpenAI-format client request dict (before any translation/routing).
    # Used to extract prompt text for label UI keyword search.
    client_request: dict[str, Any] | None = None
    # Recommender provenance — see _MIGRATIONS for column-level docs. Populated
    # only on rows where the cell recommender fired and got a live upstream
    # decision (upstream/alternative); NULL on pass-through, cache, and
    # fallback rows.
    recommender_classifier_cell: str | None = None
    recommender_raw_output: str | None = None
    recommender_source: str | None = None
    # Raw float32 bytes of the prompt embedding produced by the routing
    # EmbeddingProvider. NULL when the noop provider is active (cold-
    # start configuration) or when text extraction returned empty.
    # Persisted to the `prompt_embedding` BLOB column; the kNN
    # predictor reloads it via np.frombuffer at startup / after
    # training-job checkpoints.
    prompt_embedding: bytes | None = None


@dataclass(slots=True)
class RoutingAttempt:
    """One cell-level dispatch attempt within a single user request.

    The dispatch loop records one of these per cell it tries. attempt_idx
    starts at 0 (the recommender's primary pick) and increments for each
    reroute. A successful request produces one row with the success's
    backend_id + status=200; a failed-then-rerouted request produces
    multiple, with intermediate rows carrying the BackendError details.
    """

    attempt_idx: int
    backend_id: str | None
    model: str
    reasoning_effort: str
    status: int
    classification: str
    latency_ms: int
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class SessionAssistantTurn:
    request_id: int
    model: str
    reasoning_effort: str | None
    response_text: str


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
        self._new_request_callbacks: list[Callable[[int], None]] = []
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
                if "duplicate column" in msg or "no such column" in msg or "there is already another column" in msg:
                    continue
                raise

    @property
    def path(self) -> Path:
        return self._path

    @property
    def capture_bodies(self) -> bool:
        return self._capture_bodies

    def add_new_request_callback(self, cb: Callable[[int], None]) -> None:
        """Register a callback to be called with request_id when a new request is logged."""
        self._new_request_callbacks.append(cb)

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
        prompt_text = _extract_prompt_text(entry.client_request)
        response_text = _extract_response_text(entry.resp_payload)
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
            prompt_text,
            response_text,
            entry.recommender_classifier_cell,
            entry.recommender_raw_output,
            entry.recommender_source,
            entry.prompt_embedding,
            entry.effective_routing_mode,
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
                    prompt_complexity_class, prompt_text, response_text,
                    recommender_classifier_cell, recommender_raw_output,
                    recommender_source, prompt_embedding,
                    effective_routing_mode
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?
                )
                """,
                row,
            )
            request_id = cursor.lastrowid
            if request_id is None:
                raise RuntimeError("sqlite3 did not return a rowid for the inserted request")
            if self._capture_bodies and (
                entry.req_payload is not None or entry.resp_payload is not None or entry.upstream_headers is not None
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
                            json.dumps(entry.upstream_headers).encode() if entry.upstream_headers is not None else None
                        ),
                    ),
                )
            row_id = int(request_id)
        # Call callbacks outside lock to avoid deadlock if callback tries to query
        for cb in self._new_request_callbacks:
            with contextlib.suppress(Exception):
                cb(row_id)
        return row_id

    def record_routing_attempts(
        self,
        request_id: int,
        attempts: list[RoutingAttempt],
    ) -> None:
        """Bulk-insert per-attempt rows for a single request.

        Called by the dispatch layer at the end of a request after all
        cell-level retries have settled (success or final failure). A
        request that succeeded on the first attempt still gets one row
        here for symmetry — the routing_attempts table is then a complete
        picture of every cell every request actually touched.
        """
        if not attempts:
            return
        rows = [
            (
                request_id,
                a.attempt_idx,
                a.backend_id,
                a.model,
                a.reasoning_effort,
                a.status,
                a.classification,
                a.latency_ms,
                a.error_message,
            )
            for a in attempts
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO request_routing_attempts"
                " (request_id, attempt_idx, backend_id, model, reasoning_effort,"
                "  status, classification, latency_ms, error_message)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

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

    def recent_session_assistant_turns(self, session_id: str, *, limit: int = 8) -> list[SessionAssistantTurn]:
        """Return recent successful assistant outputs for hidden provenance tagging."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, model, reasoning_effort, response_text FROM requests"
                " WHERE session_id = ?"
                "   AND status = 200"
                "   AND model IS NOT NULL"
                "   AND response_text IS NOT NULL"
                " ORDER BY ts_start DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out: list[SessionAssistantTurn] = []
        for request_id, model, effort, text in rows:
            if isinstance(model, str) and isinstance(text, str) and text:
                out.append(
                    SessionAssistantTurn(
                        request_id=int(request_id),
                        model=model,
                        reasoning_effort=effort,
                        response_text=text,
                    )
                )
        return out

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
                "UPDATE requests SET quality_score = ?, quality_label_method = ? WHERE id = ?",
                (score, method, request_id),
            )

    def record_peer_quality_opinions(
        self,
        *,
        request_id: int,
        session_id: str | None,
        judge_backend_id: str,
        judge_model: str,
        judge_reasoning_effort: str | None,
        opinions: list[PeerQualityOpinion] | tuple[PeerQualityOpinion, ...],
        created_at: float,
    ) -> None:
        """Persist nonce-validated peer quality opinions for one request."""
        if not opinions:
            return
        rows = [
            (
                request_id,
                session_id,
                judge_backend_id,
                judge_model,
                judge_reasoning_effort,
                opinion.subject_request_id,
                opinion.subject_model,
                opinion.subject_reasoning_effort,
                opinion.score,
                opinion.nonce,
                opinion.reason,
                opinion.raw_marker,
                created_at,
            )
            for opinion in opinions
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT INTO peer_quality_opinions"
                " (request_id, session_id, judge_backend_id, judge_model, judge_reasoning_effort,"
                "  subject_request_id, subject_model, subject_reasoning_effort, score, nonce,"
                "  reason, raw_marker, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                rows,
            )

    def record_peer_quality_capture_metrics(
        self,
        *,
        request_id: int,
        session_id: str | None,
        judge_backend_id: str,
        judge_model: str,
        judge_reasoning_effort: str | None,
        nonce: str,
        opinion_count: int,
        echo_count: int,
        malformed_count: int,
        created_at: float,
    ) -> None:
        """Persist request-level qop capture counters for observability."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO peer_quality_capture_metrics"
                " (request_id, session_id, judge_backend_id, judge_model, judge_reasoning_effort,"
                "  nonce, opinion_count, echo_count, malformed_count, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    request_id,
                    session_id,
                    judge_backend_id,
                    judge_model,
                    judge_reasoning_effort,
                    nonce,
                    opinion_count,
                    echo_count,
                    malformed_count,
                    created_at,
                ),
            )

    def per_mode_stats_since(self, *, since_ts: float) -> dict[str, dict[str, int]]:
        """Aggregate per-effective_routing_mode counts since `since_ts`.

        Returns a dict keyed by mode → {total, success, failure},
        where success = status < 500 and failure = status >= 500.
        The result powers /status's rolling comparison between
        auto-mode and canary-redirect rows so operators (and the
        dev loop) can see at a glance whether local code is keeping
        up with the remote baseline.

        Modes encountered in the log but absent from this dict's
        keys had zero rows in the window. Callers should treat a
        missing key as zero, not as missing data."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT effective_routing_mode,"
                "       COUNT(*) AS total,"
                "       SUM(CASE WHEN status >= 500 THEN 1 ELSE 0 END) AS failures "
                "FROM requests "
                "WHERE ts_start >= ? AND effective_routing_mode IS NOT NULL "
                "GROUP BY effective_routing_mode",
                (since_ts,),
            ).fetchall()
        return {
            row[0]: {
                "total": int(row[1]),
                "failure": int(row[2] or 0),
                "success": int(row[1]) - int(row[2] or 0),
            }
            for row in rows
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _walk_text(node: Any) -> list[str]:
    """Concatenate every text-shaped string from a nested message body.

    Handles three shapes that callosum sees in the wild:
      * Chat completions `messages[].content` as str or as a list of
        parts with `text`.
      * Codex Responses API `input[].content[].text`.
      * Top-level `instructions` strings.
    """
    if node is None:
        return []
    if isinstance(node, str):
        return [node]
    if isinstance(node, list):
        out: list[str] = []
        for x in node:
            out.extend(_walk_text(x))
        return out
    if isinstance(node, dict):
        if isinstance(node.get("text"), str):
            return [node["text"]]
        if "content" in node:
            return _walk_text(node["content"])
        dict_out: list[str] = []
        for v in node.values():
            dict_out.extend(_walk_text(v))
        return dict_out
    return []


def _extract_prompt_text(client_request: dict[str, Any] | None) -> str | None:
    """Extract user-visible text from an OpenAI-format client request.

    Handles both Chat Completions (`messages`) and Codex Responses
    API (`input` + `instructions`). Earlier versions only read the
    Chat Completions shape, so every /v1/responses request — i.e.
    every Codex CLI session — silently logged NULL prompt_text.
    Caught by an embed-backfill audit showing 52,265 rows with
    req_payload populated but prompt_text NULL.
    """
    if client_request is None:
        return None
    try:
        parts: list[str] = []
        for key in ("input", "messages", "instructions"):
            if key in client_request:
                parts.extend(_walk_text(client_request[key]))
        text = "\n".join(p for p in parts if p)
        return text if text else None
    except Exception:
        return None


def _response_text_from_dict(resp: dict[str, Any]) -> str | None:
    """Pull assistant-visible text from a single response JSON object."""
    # Responses API shape.
    if "output" in resp:
        parts = _walk_text(resp["output"])
        text = "\n".join(p for p in parts if p)
        if text:
            return text
    # Chat Completions shape.
    choices = resp.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        msg = choices[0].get("message")
        if isinstance(msg, dict):
            c = msg.get("content")
            if isinstance(c, str) and c:
                return c
            if isinstance(c, list):
                parts = _walk_text(c)
                text = "\n".join(p for p in parts if p)
                if text:
                    return text
    return None


def _response_text_from_sse(blob: bytes) -> str | None:
    """Recover assistant text from a Codex Responses SSE stream blob.

    Streamed responses are stored as the raw `data: {...}` event blob, not a
    single JSON document, so `json.loads` over the whole thing fails. The
    terminal `response.completed` event carries the full response object; the
    per-output `response.output_text.done` events each carry the full
    accumulated text. Prefer the completed event, fall back to the done events.
    """
    completed_text: str | None = None
    done_parts: list[str] = []
    delta_parts: list[str] = []
    for line in blob.split(b"\n"):
        line = line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            ev = json.loads(data)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        et = ev.get("type")
        if et == "response.completed":
            # Codex Responses: terminal event carries the full response object.
            resp = ev.get("response")
            if isinstance(resp, dict):
                completed_text = _response_text_from_dict(resp) or completed_text
        elif et == "response.output_text.done":
            # Codex Responses: each output emits its full accumulated text.
            t = ev.get("text")
            if isinstance(t, str) and t:
                done_parts.append(t)
        else:
            # Chat Completions: accumulate streamed `choices[0].delta.content`.
            choices = ev.get("choices")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta")
                if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                    delta_parts.append(delta["content"])
    if completed_text:
        return completed_text
    if done_parts:
        return "\n".join(done_parts)
    if delta_parts:
        return "".join(delta_parts)
    return None


def _extract_response_text(resp_payload: bytes | None) -> str | None:
    """Extract assistant-visible text from a response payload.

    Handles three storage shapes: a single Chat Completions JSON
    (`choices[0].message.content`), a single Codex Responses JSON
    (`output[].content[].text`), and — for streamed responses — the raw SSE
    event blob, whose text lives in the terminal `response.completed` event
    rather than at the top level. The SSE shape is the common one: peer-quality
    capture only runs on streaming requests, and `recent_session_assistant_turns`
    can only surface a prior turn as a subject when its `response_text` is set,
    so failing to extract here silently starves capture of subjects.
    """
    if resp_payload is None:
        return None
    try:
        resp = json.loads(resp_payload)
        if isinstance(resp, dict):
            text = _response_text_from_dict(resp)
            if text:
                return text
    except Exception:
        pass
    # Streamed responses are SSE (`data: {...}` lines), not a single document.
    if b"data:" in resp_payload:
        return _response_text_from_sse(resp_payload)
    return None


def _compress(data: bytes | None) -> bytes | None:
    if data is None:
        return None
    return zlib.compress(data, level=6)


def decompress(blob: bytes | None) -> bytes | None:
    """Inverse of `_compress`. Public so analysis tools can read bodies back."""
    if blob is None:
        return None
    return zlib.decompress(blob)


def _is_reset_crossover(before: CodexQuotaSnapshot | None, after: CodexQuotaSnapshot | None) -> bool:
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
