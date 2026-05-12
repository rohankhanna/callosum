"""Quality labeling UI for router training."""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from codex_proxy.usage_log import UsageLog


class _SSEBroadcaster:
    """Manages EventSource subscriptions for new request notifications."""

    def __init__(self) -> None:
        self._queues: list[asyncio.Queue[int]] = []

    def subscribe(self) -> asyncio.Queue[int]:
        """Subscribe to new request notifications."""
        q: asyncio.Queue[int] = asyncio.Queue(maxsize=100)
        self._queues.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[int]) -> None:
        """Unsubscribe from notifications."""
        try:
            self._queues.remove(q)
        except ValueError:
            pass

    def notify(self, request_id: int) -> None:
        """Called from sync context when a new request is logged."""
        for q in list(self._queues):
            try:
                q.put_nowait(request_id)
            except asyncio.QueueFull:
                pass


_broadcaster: _SSEBroadcaster | None = None


def _get_db(usage_log: UsageLog) -> sqlite3.Connection:
    """Open a read connection to the usage log database."""
    conn = sqlite3.connect(str(usage_log.path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _do_search(
    conn: sqlite3.Connection,
    q: str = "",
    model: str = "",
    complexity: int | None = None,
    quality: str = "all",
    date_from: str = "",
    date_to: str = "",
    sort: str = "newest",
    limit: int = 20,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Execute search query against FTS5 index and filters."""
    where_clauses = ["r.prompt_text IS NOT NULL"]

    if model:
        where_clauses.append("r.model = ?")
    if complexity is not None:
        where_clauses.append("r.prompt_complexity_class = ?")
    if quality == "good":
        where_clauses.append("r.quality_score != -1 OR r.quality_score IS NULL")
    elif quality == "bad":
        where_clauses.append("r.quality_score = -1")
    elif quality == "neutral":
        where_clauses.append("r.quality_score = 0")
    elif quality == "labeled":
        where_clauses.append("r.quality_score IS NOT NULL")
    elif quality == "unlabeled":
        where_clauses.append("r.quality_score IS NULL")

    # Date filtering using ISO format comparison
    if date_from:
        where_clauses.append("r.ts_start >= ?")
    if date_to:
        where_clauses.append("r.ts_start <= ?")

    join_clause = ""
    params: list[Any] = []

    # FTS5 search
    if q:
        join_clause = "LEFT JOIN requests_fts ON requests_fts.rowid = r.id"
        where_clauses.append("(requests_fts.rowid IS NOT NULL AND requests_fts MATCH ?)")
        params.append(q)

    # Build parameter list for WHERE clause
    if model:
        params.append(model)
    if complexity is not None:
        params.append(complexity)
    if date_from:
        params.append(date_from)
    if date_to:
        params.append(date_to)

    where_sql = " AND ".join(where_clauses)

    # Sorting
    if sort == "oldest":
        order_sql = "r.ts_start ASC"
    elif sort == "most_tokens":
        order_sql = "r.prompt_tokens DESC NULLS LAST"
    elif sort == "least_tokens":
        order_sql = "r.prompt_tokens ASC NULLS LAST"
    else:  # "newest" is default
        order_sql = "r.ts_start DESC"

    sql = f"""
    SELECT
        r.id, r.ts_start, r.model, r.reasoning_effort, r.prompt_complexity_class,
        r.quality_score, r.prompt_tokens, r.response_bytes,
        SUBSTRING(r.prompt_text, 1, 500) as prompt_text,
        SUBSTRING(r.response_text, 1, 500) as response_text
    FROM requests r
    {join_clause}
    WHERE {where_sql}
    ORDER BY {order_sql}
    LIMIT ? OFFSET ?
    """

    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    return [dict(row) for row in rows]


def _do_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Get labeling statistics."""
    row = conn.execute(
        """
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN quality_score IS NOT NULL THEN 1 ELSE 0 END) as labeled,
            SUM(CASE WHEN quality_score = 1 THEN 1 ELSE 0 END) as good,
            SUM(CASE WHEN quality_score = 0 THEN 1 ELSE 0 END) as neutral,
            SUM(CASE WHEN quality_score = -1 THEN 1 ELSE 0 END) as bad
        FROM requests
        """
    ).fetchone()
    if row is None:
        return {"total": 0, "labeled": 0, "good": 0, "neutral": 0, "bad": 0}
    return dict(row)


def _do_models(conn: sqlite3.Connection) -> list[str]:
    """Get distinct model names."""
    rows = conn.execute(
        "SELECT DISTINCT model FROM requests WHERE model IS NOT NULL ORDER BY model"
    ).fetchall()
    return [row[0] for row in rows]


def _do_update_quality(
    conn: sqlite3.Connection, request_id: int, score: int, method: str = "user"
) -> None:
    """Update quality score for a request."""
    if score not in (-1, 0, 1):
        raise ValueError(f"quality_score must be -1, 0, or 1; got {score}")
    conn.execute(
        "UPDATE requests SET quality_score = ?, quality_label_method = ? WHERE id = ?",
        (score, method, request_id),
    )
    conn.commit()


def install_label_ui(app: Any, usage_log: UsageLog) -> None:
    """Install label UI router and register callbacks."""
    global _broadcaster
    _broadcaster = _SSEBroadcaster()

    router = APIRouter(prefix="/label", tags=["label"])

    @router.get("")
    async def label_page() -> FileResponse:
        """Serve label.html static page."""
        return FileResponse("~/Desktop/codex-proxy/src/codex_proxy/static/label.html")

    @router.get("/search")
    async def search(
        q: str = "",
        model: str = "",
        complexity: int | None = None,
        quality: str = "all",
        date_from: str = "",
        date_to: str = "",
        sort: str = "newest",
        limit: int = 20,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Search requests with filters and sorting."""
        conn = _get_db(usage_log)
        try:
            rows = _do_search(
                conn,
                q=q,
                model=model,
                complexity=complexity,
                quality=quality,
                date_from=date_from,
                date_to=date_to,
                sort=sort,
                limit=limit,
                offset=offset,
            )
            return {"results": rows, "offset": offset, "limit": limit}
        finally:
            conn.close()

    @router.get("/stats")
    async def stats() -> dict[str, Any]:
        """Get labeling statistics."""
        conn = _get_db(usage_log)
        try:
            return _do_stats(conn)
        finally:
            conn.close()

    @router.get("/models")
    async def models() -> dict[str, list[str]]:
        """Get distinct models from requests."""
        conn = _get_db(usage_log)
        try:
            model_list = _do_models(conn)
            return {"models": model_list}
        finally:
            conn.close()

    @router.post("/{request_id}/quality")
    async def set_quality(request_id: int, body: dict[str, Any]) -> dict[str, Any]:
        """Set quality score for a request."""
        score = body.get("score")
        if score is None:
            raise HTTPException(status_code=400, detail="score is required")
        if score not in (-1, 0, 1):
            raise HTTPException(status_code=400, detail="score must be -1, 0, or 1")

        conn = _get_db(usage_log)
        try:
            _do_update_quality(conn, request_id, int(score), "user")
            return {"success": True}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
        finally:
            conn.close()

    @router.get("/stream")
    async def stream() -> StreamingResponse:
        """SSE stream of new request IDs."""
        if _broadcaster is None:
            raise HTTPException(status_code=500, detail="broadcaster not initialized")

        q = _broadcaster.subscribe()

        async def generate():
            try:
                while True:
                    try:
                        request_id = await asyncio.wait_for(q.get(), timeout=30)
                        yield f"data: {request_id}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                _broadcaster.unsubscribe(q)

        return StreamingResponse(generate(), media_type="text/event-stream")

    # Register SSE callback
    usage_log.add_new_request_callback(lambda rid: _broadcaster.notify(rid) if _broadcaster else None)

    app.include_router(router)
