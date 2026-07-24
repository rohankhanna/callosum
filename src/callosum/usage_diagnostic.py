from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from callosum.usage_log import _walk_text, decompress

# --- Peer-quality (in-band) request stream segmentation ---------------------
#
# The retired in-band peer-quality path (`_inject_peer_quality_prompt` in
# app.py) mutated the live upstream request in three additive ways before
# dispatch: (a) wrapped each eligible prior assistant text in a provenance tag
# `<model|effort|reqid>...original...</model|effort|reqid>`, (b) appended a
# trailing `developer` message carrying the audit instruction, and (c) placed
# one `<<qop ...>>` marker line per subject inside that instruction. The
# persisted `request_bodies.req_payload` is the POST-injection body, so these
# signatures are visible on rows captured while the in-band flag was on.
#
# The four segment kinds below decompose such a payload for the first-divergence
# audit (work tracker ): stable_front (the byte-identical prefix
# shared with the no-capture arm), provenance_mutated_history (the history block
# carrying provenance tags), live_tail (the current turn's user/tool input),
# and peer_opinion_suffix (the appended audit instruction). The sidecar-primary
# path () does not mutate the user request, so these kinds
# only appear on in-band rows; non-peer-quality turns keep the original
# `instructions` / `input` / `message:<role>` taxonomy via `_extract_segments`.

_AUDIT_SENTINEL = "Hidden Callosum quality audit"
_QOP_MARKER_PREFIX = "<<qop "
# A provenance tag wraps prior assistant text: `<label>text</label>` where label
# is `{model}|{request_id}` or `{model}|{effort}|{request_id}`. The backreference
# ties the closing tag to the opening label; DOTALL so content may span newlines.
_PROVENANCE_TAG_RE = re.compile(
    r"<([A-Za-z0-9_.\-]+(?:\|[\w.\-]+){1,2})>.*?</\1>",
    re.DOTALL,
)


def has_peer_quality_signature(payload: dict[str, Any]) -> bool:
    """True iff payload carries in-band peer-quality injection artifacts.

    Detects either the audit instruction (by its sentinel phrase or a <<qop)
    marker anywhere in the instructions or a developer message, or a provenance
    tag wrapping assistant text. Pure and regex-only; safe on any payload shape.
    """
    instructions = _collapse_text(payload.get("instructions"))
    if isinstance(instructions, str) and (
        _AUDIT_SENTINEL in instructions or _QOP_MARKER_PREFIX in instructions
    ):
        return True
    for role, text in _iter_message_items(payload):
        if not text:
            continue
        if role == "developer" and _AUDIT_SENTINEL in text:
            return True
        if _QOP_MARKER_PREFIX in text or _PROVENANCE_TAG_RE.search(text):
            return True
    return False


def extract_peer_quality_segments(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Decompose an in-band peer-quality payload into the four audit segments.

    Returns (kind, text) pairs in order: stable_front, then
    provenance_mutated_history, then live_tail, then peer_opinion_suffix
    (omitting any that are empty). Tag overhead is counted inside
    provenance_mutated_history because those are the bytes actually sent upstream.
    """
    segments: list[tuple[str, str]] = []
    instructions = _collapse_text(payload.get("instructions"))
    if isinstance(instructions, str) and instructions:
        # The audit is appended to `instructions` only in the fallback shape
        # (no input/messages); otherwise it is a trailing developer message.
        if _AUDIT_SENTINEL in instructions:
            idx = instructions.index(_AUDIT_SENTINEL)
            front = instructions[:idx].rstrip()
            if front:
                segments.append(("stable_front", front))
            segments.append(("peer_opinion_suffix", instructions[idx:]))
        else:
            segments.append(("stable_front", instructions))

    items = _iter_message_items(payload)
    tagged_idxs = [
        i
        for i, (role, text) in enumerate(items)
        if role == "assistant" and bool(_PROVENANCE_TAG_RE.search(text or ""))
    ]
    audit_idx = next(
        (i for i, (role, text) in enumerate(items) if role == "developer" and _AUDIT_SENTINEL in (text or "")),
        None,
    )

    if not tagged_idxs and audit_idx is None:
        # Signature came from the instructions fallback alone; all message items
        # are stable front.
        for _, text in items:
            if text:
                segments.append(("stable_front", text))
        return _coalesce_adjacent(segments)

    front_end = tagged_idxs[0] if tagged_idxs else (audit_idx if audit_idx is not None else len(items))
    for _, text in items[:front_end]:
        if text:
            segments.append(("stable_front", text))

    if tagged_idxs:
        for _, text in items[tagged_idxs[0] : tagged_idxs[-1] + 1]:
            if text:
                segments.append(("provenance_mutated_history", text))

    tail_start = tagged_idxs[-1] + 1 if tagged_idxs else front_end
    tail_end = audit_idx if audit_idx is not None else len(items)
    for _, text in items[tail_start:tail_end]:
        if text:
            segments.append(("live_tail", text))

    if audit_idx is not None:
        text = items[audit_idx][1]
        if text:
            segments.append(("peer_opinion_suffix", text))

    return _coalesce_adjacent(segments)


def first_divergence_position(a: bytes, b: bytes) -> int:
    """Index of the first differing byte between a and b.

    Returns -1 when the byte streams are equal. When one stream is a prefix of
    the other, returns the length of the shorter stream (the first extra byte).
    Used to localize where in-band injection first perturbs the upstream stream
    relative to the no-capture arm.
    """
    if a == b:
        return -1
    shortest = min(len(a), len(b))
    for i in range(shortest):
        if a[i] != b[i]:
            return i
    return shortest


def provenance_tag_overhead_chars(text: str) -> int:
    """Total chars of the <label> / </label> provenance wrappers in text.

    The bytes inside these wrappers are pure injection overhead on the upstream
    stream; the inner text is the original assistant content. Complements the
    injected_tokens column (which is the token-level additive cost).
    """
    overhead = 0
    for match in _PROVENANCE_TAG_RE.finditer(text or ""):
        label = match.group(1)
        overhead += 2 * len(label) + len("<></>")  # `<label>` + `</label>`
    return overhead


def _iter_message_items(payload: dict[str, Any]) -> list[tuple[str, str]]:
    """Normalize Responses input / Chat messages into (role, text)."""
    items = payload.get("input") if isinstance(payload.get("input"), list) else None
    if items is None:
        items = payload.get("messages") if isinstance(payload.get("messages"), list) else None
    if items is None:
        return []
    out: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "unknown")
        text = _collapse_text(item.get("content"))
        out.append((role, text))
    return out


def _coalesce_adjacent(segments: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge runs of adjacent same-kind segments into one (newline-joined)."""
    out: list[tuple[str, str]] = []
    for kind, text in segments:
        if not text:
            continue
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + "\n" + text)
        else:
            out.append((kind, text))
    return out


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
class TrafficKindBucketSummary:
    # Policy-purpose bucket. `traffic_kind` is the Callosum decision-purpose
    # axis (operator / canary_redirect / min_coverage_quota / peer_quality_capture).
    # Pre-F4 rows have NULL traffic_kind and coalesce to "legacy" so they are
    # visible without being mislabeled as real operator traffic.
    traffic_kind: str
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
    # None means the traffic_kind axis was not requested (group_by="mode").
    traffic_kind_summaries: tuple[TrafficKindBucketSummary, ...] | None = None


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
    group_by: str = "mode",
) -> list[TimeBucketSummary]:
    if limit <= 0:
        return []
    if group_by not in ("mode", "traffic_kind", "both"):
        raise ValueError(f"unsupported group_by: {group_by}")
    if not db_path.exists():
        raise FileNotFoundError(f"usage log not found: {db_path}")
    bucket_expr = _bucket_expression(bucket)
    want_mode = group_by in ("mode", "both")
    want_kind = group_by in ("traffic_kind", "both")
    conn = sqlite3.connect(str(db_path))
    try:
        mode_rows = _fetch_grouped(conn, bucket_expr, "mode") if want_mode else []
        kind_rows = _fetch_grouped(conn, bucket_expr, "traffic_kind") if want_kind else []
    finally:
        conn.close()
    mode_by_bucket = _assemble(mode_rows, ModeBucketSummary, "effective_routing_mode")
    kind_by_bucket = _assemble(kind_rows, TrafficKindBucketSummary, "traffic_kind")
    # Bucket totals are identical across axes (same rows grouped differently),
    # so derive them from whichever axis was queried. Prefer mode when present.
    totals_source = mode_rows if mode_rows else kind_rows
    bucket_order: list[str] = []
    seen: set[str] = set()
    for row in totals_source:
        bucket_start = str(row[0])
        if bucket_start not in seen:
            seen.add(bucket_start)
            bucket_order.append(bucket_start)
    # rows are newest-first (ORDER BY bucket_start DESC); take the newest `limit`
    # then reverse to oldest-first, matching the historical ordering.
    selected = bucket_order[:limit]
    selected.reverse()
    result: list[TimeBucketSummary] = []
    for bucket_start in selected:
        modes = tuple(mode_by_bucket.get(bucket_start, ()))
        kinds = kind_by_bucket.get(bucket_start)
        kind_tuple = tuple(kinds) if kinds is not None else None
        total_items: tuple[ModeBucketSummary | TrafficKindBucketSummary, ...] = modes if modes else (kind_tuple or ())
        result.append(
            TimeBucketSummary(
                bucket_start=bucket_start,
                turn_count=sum(item.turn_count for item in total_items),
                prompt_tokens=sum(item.prompt_tokens for item in total_items),
                completion_tokens=sum(item.completion_tokens for item in total_items),
                total_tokens=sum(item.total_tokens for item in total_items),
                mode_summaries=modes,
                traffic_kind_summaries=kind_tuple,
            )
        )
    return result


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
    group_by: str = "mode",
) -> dict[str, Any]:
    series = token_time_series(db_path, bucket=bucket, limit=limit, group_by=group_by)
    rendered: list[dict[str, Any]] = []
    for item in series:
        bucket_doc: dict[str, Any] = {
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
        if item.traffic_kind_summaries is not None:
            bucket_doc["traffic_kind_summaries"] = [
                {
                    "traffic_kind": kind.traffic_kind,
                    "turn_count": kind.turn_count,
                    "prompt_tokens": kind.prompt_tokens,
                    "completion_tokens": kind.completion_tokens,
                    "total_tokens": kind.total_tokens,
                }
                for kind in item.traffic_kind_summaries
            ]
        rendered.append(bucket_doc)
    return {
        "db_path": str(db_path),
        "bucket": bucket,
        "group_by": group_by,
        "bucket_count": len(series),
        "series": rendered,
    }


def _segment_summaries(
    req_payload_blob: bytes | None,
    prompt_tokens: int | None,
) -> tuple[SegmentSummary, ...]:
    payload = _decode_request_payload(req_payload_blob)
    if payload is None:
        return ()
    if has_peer_quality_signature(payload):
        segments = extract_peer_quality_segments(payload)
    else:
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
        summaries.append(SegmentSummary(kind=kind, chars=chars, est_prompt_tokens=est_tokens))
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


# (column, NULL-fallback) for each provenance axis. The fallback keeps legacy
# rows visible instead of dropping them: effective_routing_mode NULL -> 'auto'
# (pre-router / pass-through), traffic_kind NULL -> 'legacy' (pre-F4 rows).
_AXIS_CONFIG: dict[str, tuple[str, str]] = {
    "mode": ("effective_routing_mode", "auto"),
    "traffic_kind": ("traffic_kind", "legacy"),
}


def _fetch_grouped(
    conn: sqlite3.Connection,
    bucket_expr: str,
    axis: str,
) -> list[tuple[Any, ...]]:
    """Group token volume by time bucket and one provenance axis.

    axis="mode" groups by effective_routing_mode (NULL -> 'auto'); axis=
    "traffic_kind" groups by traffic_kind (NULL -> 'legacy' so pre-F4 rows are
    visible without being mislabeled as real operator traffic).
    """
    column, fallback = _AXIS_CONFIG[axis]
    return conn.execute(
        f"""
        SELECT
            {bucket_expr} AS bucket_start,
            COALESCE(r.{column}, ?) AS label,
            COUNT(*) AS turn_count,
            COALESCE(SUM(r.prompt_tokens), 0) AS prompt_tokens,
            COALESCE(SUM(r.completion_tokens), 0) AS completion_tokens,
            COALESCE(SUM(r.total_tokens), 0) AS total_tokens
        FROM requests AS r
        GROUP BY bucket_start, r.{column}
        ORDER BY bucket_start DESC, label ASC
        """,
        (fallback,),
    ).fetchall()


def _assemble(
    rows: list[tuple[Any, ...]],
    cls: type,
    label_field: str,
) -> dict[str, list[Any]]:
    by_bucket: dict[str, list[Any]] = {}
    for row in rows:
        bucket_start = str(row[0])
        by_bucket.setdefault(bucket_start, []).append(
            cls(
                **{
                    label_field: str(row[1]),
                    "turn_count": int(row[2]),
                    "prompt_tokens": int(row[3]),
                    "completion_tokens": int(row[4]),
                    "total_tokens": int(row[5]),
                }
            )
        )
    for items in by_bucket.values():
        items.sort(key=lambda item: getattr(item, label_field))
    return by_bucket
