from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from callosum.usage_log import _walk_text, decompress


@dataclass(frozen=True, slots=True)
class SegmentSummary:
    kind: str
    chars: int
    est_prompt_tokens: int


@dataclass(frozen=True, slots=True)
class TurnSummary:
    request_id: int
    ts_start: float
    route: str
    requested_model: str | None
    served_model: str | None
    effective_routing_mode: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    segment_summaries: tuple[SegmentSummary, ...]


@dataclass(frozen=True, slots=True)
class ModeBucketSummary:
    effective_routing_mode: str
    turn_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class TimeBucketSummary:
    bucket_start: str
    turn_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    mode_summaries: tuple[ModeBucketSummary, ...]


def recent_turn_summaries(
    db_path: Path,
    *,
    limit: int = 10,
) -> list[TurnSummary]:
    if limit <= 0:
        return []
    if not db_path.exists():
        raise FileNotFoundError(f"usage log not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            """
            SELECT
                r.id,
                r.ts_start,
                r.route,
                r.requested_model,
                r.model,
                COALESCE(r.effective_routing_mode, 'auto'),
                r.prompt_tokens,
                r.completion_tokens,
                r.total_tokens,
                b.req_payload
            FROM requests AS r
            LEFT JOIN request_bodies AS b ON b.request_id = r.id
            ORDER BY r.id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        conn.close()
    return [
        TurnSummary(
            request_id=int(row[0]),
            ts_start=float(row[1]),
            route=str(row[2]),
            requested_model=row[3],
            served_model=row[4],
            effective_routing_mode=str(row[5]),
            prompt_tokens=row[6],
            completion_tokens=row[7],
            total_tokens=row[8],
            segment_summaries=_segment_summaries(row[9], row[6]),
        )
        for row in rows
    ]


def token_time_series(
    db_path: Path,
    *,
    bucket: str = "day",
    limit: int = 30,
) -> list[TimeBucketSummary]:
    if limit <= 0:
        return []
    if not db_path.exists():
        raise FileNotFoundError(f"usage log not found: {db_path}")
    bucket_expr = _bucket_expression(bucket)
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            f"""
            SELECT
                {bucket_expr} AS bucket_start,
                COALESCE(r.effective_routing_mode, 'auto') AS effective_routing_mode,
                COUNT(*) AS turn_count,
                COALESCE(SUM(r.prompt_tokens), 0) AS prompt_tokens,
                COALESCE(SUM(r.completion_tokens), 0) AS completion_tokens,
                COALESCE(SUM(r.total_tokens), 0) AS total_tokens
            FROM requests AS r
            GROUP BY bucket_start, effective_routing_mode
            ORDER BY bucket_start DESC, effective_routing_mode ASC
            """,
        ).fetchall()
    finally:
        conn.close()
    buckets: dict[str, list[ModeBucketSummary]] = {}
    order: list[str] = []
    for row in rows:
        bucket_start = str(row[0])
        if bucket_start not in buckets:
            buckets[bucket_start] = []
            order.append(bucket_start)
        buckets[bucket_start].append(
            ModeBucketSummary(
                effective_routing_mode=str(row[1]),
                turn_count=int(row[2]),
                prompt_tokens=int(row[3]),
                completion_tokens=int(row[4]),
                total_tokens=int(row[5]),
            )
        )
    return [
        _rollup_bucket(bucket_start, buckets[bucket_start])
        for bucket_start in reversed(order[:limit])
    ]


def render_recent_turns_json(
    db_path: Path,
    *,
    limit: int = 10,
) -> dict[str, Any]:
    turns = recent_turn_summaries(db_path, limit=limit)
    return {
        "db_path": str(db_path),
        "turn_count": len(turns),
        "turns": [
            {
                "request_id": turn.request_id,
                "ts_start": turn.ts_start,
                "route": turn.route,
                "requested_model": turn.requested_model,
                "served_model": turn.served_model,
                "effective_routing_mode": turn.effective_routing_mode,
                "prompt_tokens": turn.prompt_tokens,
                "completion_tokens": turn.completion_tokens,
                "total_tokens": turn.total_tokens,
                "segment_summaries": [
                    {
                        "kind": seg.kind,
                        "chars": seg.chars,
                        "est_prompt_tokens": seg.est_prompt_tokens,
                    }
                    for seg in turn.segment_summaries
                ],
            }
            for turn in turns
        ],
    }


def render_token_time_series_json(
    db_path: Path,
    *,
    bucket: str = "day",
    limit: int = 30,
) -> dict[str, Any]:
    series = token_time_series(db_path, bucket=bucket, limit=limit)
    return {
        "db_path": str(db_path),
        "bucket": bucket,
        "bucket_count": len(series),
        "series": [
            {
                "bucket_start": item.bucket_start,
                "turn_count": item.turn_count,
                "prompt_tokens": item.prompt_tokens,
                "completion_tokens": item.completion_tokens,
                "total_tokens": item.total_tokens,
                "mode_summaries": [
                    {
                        "effective_routing_mode": mode.effective_routing_mode,
                        "turn_count": mode.turn_count,
                        "prompt_tokens": mode.prompt_tokens,
                        "completion_tokens": mode.completion_tokens,
                        "total_tokens": mode.total_tokens,
                    }
                    for mode in item.mode_summaries
                ],
            }
            for item in series
        ],
    }


def _segment_summaries(
    req_payload_blob: bytes | None,
    prompt_tokens: int | None,
) -> tuple[SegmentSummary, ...]:
    payload = _decode_request_payload(req_payload_blob)
    if payload is None:
        return ()
    segments = _extract_segments(payload)
    if not segments:
        return ()
    total_chars = sum(len(text) for _, text in segments)
    summaries: list[SegmentSummary] = []
    for kind, text in segments:
        chars = len(text)
        est_tokens = _apportion_tokens(
            part_chars=chars,
            total_chars=total_chars,
            total_tokens=prompt_tokens,
        )
        summaries.append(
            SegmentSummary(kind=kind, chars=chars, est_prompt_tokens=est_tokens)
        )
    return tuple(summaries)


def _decode_request_payload(req_payload_blob: bytes | None) -> dict[str, Any] | None:
    raw = decompress(req_payload_blob)
    if raw is None:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _extract_segments(payload: dict[str, Any]) -> list[tuple[str, str]]:
    segments: list[tuple[str, str]] = []
    instructions = _collapse_text(payload.get("instructions"))
    if instructions:
        segments.append(("instructions", instructions))
    input_text = _collapse_text(payload.get("input"))
    if input_text:
        segments.append(("input", input_text))
    for message in payload.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        content = _collapse_text(message.get("content"))
        if content:
            segments.append((f"message:{role}", content))
    return segments


def _collapse_text(node: Any) -> str:
    parts = [part for part in _walk_text(node) if part]
    return "\n".join(parts)


def _apportion_tokens(
    *,
    part_chars: int,
    total_chars: int,
    total_tokens: int | None,
) -> int:
    if total_tokens is None or total_chars <= 0 or part_chars <= 0:
        return 0
    return int(round(total_tokens * (part_chars / total_chars)))


def _bucket_expression(bucket: str) -> str:
    if bucket == "day":
        return "strftime('%Y-%m-%dT00:00:00Z', r.ts_start, 'unixepoch')"
    if bucket == "hour":
        return "strftime('%Y-%m-%dT%H:00:00Z', r.ts_start, 'unixepoch')"
    raise ValueError(f"unsupported bucket: {bucket}")


def _rollup_bucket(
    bucket_start: str,
    modes: list[ModeBucketSummary],
) -> TimeBucketSummary:
    return TimeBucketSummary(
        bucket_start=bucket_start,
        turn_count=sum(item.turn_count for item in modes),
        prompt_tokens=sum(item.prompt_tokens for item in modes),
        completion_tokens=sum(item.completion_tokens for item in modes),
        total_tokens=sum(item.total_tokens for item in modes),
        mode_summaries=tuple(sorted(modes, key=lambda item: item.effective_routing_mode)),
    )
