from __future__ import annotations

import json
import math
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

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
# audit: stable_front (the byte-identical prefix
# shared with the no-capture arm), provenance_mutated_history (the history block
# carrying provenance tags), live_tail (the current turn's user/tool input),
# and peer_opinion_suffix (the appended audit instruction). The sidecar-primary
# path does not mutate the user request, so these kinds
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
    if isinstance(instructions, str) and (_AUDIT_SENTINEL in instructions or _QOP_MARKER_PREFIX in instructions):
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
        i for i, (role, text) in enumerate(items) if role == "assistant" and bool(_PROVENANCE_TAG_RE.search(text or ""))
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


# --- Per-session compounding input-token cost ----------------------
#
# In a multi-turn tool-using session every turn re-sends the entire growing
# transcript to the upstream model, so the cumulative input-token cost across
# a session grows as O(K^2) in the turn count even though the per-turn input
# only grows O(K). callosum sits on both legs of every turn (client->callosum
# and callosum->upstream) and tags tool turns from its own decoded request
# payloads — a vantage no external observability platform has — so the
# compounding-cost metric, per-turn marginal, and tool-turn attribution are
# irreducibly local to this routing layer (see logs/2026-08-04.md).
#
# Field names mirror the OpenTelemetry GenAI semantic conventions
# (gen_ai.usage.cache_read.input_tokens / new input / output_tokens, PR #3163)
# rather than inventing callosum-specific names: `cache_read_tokens` is the
# cache-hit prefix (the `cached_tokens` column), `new_tokens` is the uncached
# remainder (prompt_tokens - cache_read_tokens, matching the peer-quality
# sidecar cost report's `_uncached_input` formula), `output_tokens` is the
# completion. cache_creation is not separately tracked by the log today, so
# new_tokens folds it into the uncached remainder.


@dataclass(frozen=True, slots=True)
class CompoundingTurnSummary:
    request_id: int
    session_id: str
    # 0-based ordinal within the session, by ascending request id.
    turn_index: int
    ts_start: float
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    # OTel GenAI mirror: cache-hit prefix served from cache (= cached_tokens).
    cache_read_tokens: int | None
    # OTel GenAI mirror: uncached input = max(0, prompt_tokens - cache_read).
    new_tokens: int | None
    # OTel GenAI mirror: completion tokens (= completion_tokens).
    output_tokens: int | None
    # Running sum of prompt_tokens through this turn (the O(K^2) growth curve).
    cumulative_input: int | None
    # prompt_tokens[k] - prompt_tokens[k-1]; for the first turn, the input
    # itself (growth from an empty transcript). None when prompt_tokens is
    # None or the prior turn was None.
    marginal_input: int | None
    # True iff this turn's payload carries tool I/O (role "tool", a
    # function_call / function_call_output item, or an assistant tool_calls
    # block) — i.e. the transcript being re-sent includes tool results.
    is_tool_turn: bool


@dataclass(frozen=True, slots=True)
class SessionCompoundingSummary:
    session_id: str
    turn_count: int
    tool_turn_count: int
    # Sum of prompt_tokens across all turns — the total input re-sent over
    # the session, i.e. the quadratic re-send tax. The per-turn `cumulative_input`
    # on the last turn equals this value.
    cumulative_input_tokens: int
    total_output_tokens: int
    # prompt_tokens of the final turn (the session's terminal context size).
    final_turn_input_tokens: int | None
    # cumulative_input_tokens / final_turn_input_tokens — how many times the
    # terminal context was effectively re-sent. ~K/2 for linear per-turn growth.
    # None when the final turn has no/zero input.
    compounding_ratio: float | None
    turns: tuple[CompoundingTurnSummary, ...]


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


def is_tool_turn(payload: dict[str, Any]) -> bool:
    """True iff payload carries tool I/O being re-sent to the model.

    Detects tool-result content across both request shapes callosum proxies:
    Chat Completions (a role: "tool" message, or an assistant message
    carrying a non-empty tool_calls block) and the Responses API (an
    input item of type function_call / function_call_output or
    carrying a tool_call_id). A turn flagged here is one whose
    transcript includes tool results, so its (growing) input is part of the
    quadratic re-send cost the compounding metric measures. Pure and
    shape-safe; returns False on an empty or unrecognized payload.
    """
    items = payload.get("input") if isinstance(payload.get("input"), list) else None
    if items is None:
        items = payload.get("messages") if isinstance(payload.get("messages"), list) else None
    if not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        item_type = item.get("type")
        if role == "tool":
            return True
        if item_type in ("function_call", "function_call_output", "tool_call"):
            return True
        if "tool_call_id" in item:
            return True
        if role == "assistant" and isinstance(item.get("tool_calls"), list) and item["tool_calls"]:
            return True
    return False


def compounding_cost_summaries(
    db_path: Path,
    *,
    session_id: str | None = None,
    limit_sessions: int = 10,
    min_turns: int = 2,
) -> list[SessionCompoundingSummary]:
    """Group turns by session_id and measure per-session compounding cost.

    Turns are ordered by ascending request id within each session (id
    increases with ts_start, so this is chronological). Sessions with fewer
    than min_turns turns are dropped (a single turn has no compounding).
    When session_id is None all sessions are considered, ordered
    most-recently-active first (by max request id); otherwise only the named
    session is returned. Read-only; raises FileNotFoundError if db_path
    is absent. Rows with NULL session_id are ignored — they cannot be
    attributed to a session.
    """
    if limit_sessions <= 0:
        return []
    if not db_path.exists():
        raise FileNotFoundError(f"usage log not found: {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        if session_id is not None:
            rows = conn.execute(
                """
                SELECT r.id, r.session_id, r.ts_start,
                       r.prompt_tokens, r.completion_tokens, r.cached_tokens,
                       b.req_payload
                FROM requests AS r
                LEFT JOIN request_bodies AS b ON b.request_id = r.id
                WHERE r.session_id IS NOT NULL AND r.session_id = ?
                ORDER BY r.id ASC
                """,
                (session_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT r.id, r.session_id, r.ts_start,
                       r.prompt_tokens, r.completion_tokens, r.cached_tokens,
                       b.req_payload
                FROM requests AS r
                LEFT JOIN request_bodies AS b ON b.request_id = r.id
                WHERE r.session_id IS NOT NULL
                ORDER BY r.session_id, r.id ASC
                """,
            ).fetchall()
    finally:
        conn.close()
    # Group rows by session preserving the ascending-id order from the query.
    sessions: dict[str, list[tuple[Any, ...]]] = {}
    for row in rows:
        sid = str(row[1])
        sessions.setdefault(sid, []).append(row)
    # Most-recently-active session first: the session whose latest turn has
    # the largest request id comes first.
    ordered = sorted(sessions.values(), key=lambda rs: rs[-1][0], reverse=True)
    result: list[SessionCompoundingSummary] = []
    for rs in ordered:
        if len(rs) < min_turns:
            continue
        result.append(_build_session_summary(sid=str(rs[0][1]), rows=rs))
        if len(result) >= limit_sessions:
            break
    return result


def _build_session_summary(
    *,
    sid: str,
    rows: list[tuple[Any, ...]],
) -> SessionCompoundingSummary:
    turns: list[CompoundingTurnSummary] = []
    cumulative = 0
    prev_prompt: int | None = None
    tool_count = 0
    for index, row in enumerate(rows):
        prompt = row[3]
        completion = row[4]
        cached = row[5]
        cache_read = cached
        new = max(0, int(prompt) - (cached or 0)) if prompt is not None else None
        if prompt is not None:
            cumulative += int(prompt)
            cumulative_input: int | None = cumulative
            marginal = (int(prompt) - prev_prompt) if prev_prompt is not None else int(prompt)
            prev_prompt = int(prompt)
        else:
            cumulative_input = None
            marginal = None
        payload = _decode_request_payload(row[6])
        is_tool = is_tool_turn(payload) if payload is not None else False
        if is_tool:
            tool_count += 1
        turns.append(
            CompoundingTurnSummary(
                request_id=int(row[0]),
                session_id=sid,
                turn_index=index,
                ts_start=float(row[2]),
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_tokens=cached,
                cache_read_tokens=cache_read,
                new_tokens=new,
                output_tokens=completion,
                cumulative_input=cumulative_input,
                marginal_input=marginal,
                is_tool_turn=is_tool,
            )
        )
    final_input = turns[-1].prompt_tokens
    ratio = (cumulative / final_input) if final_input and final_input > 0 else None
    return SessionCompoundingSummary(
        session_id=sid,
        turn_count=len(turns),
        tool_turn_count=tool_count,
        cumulative_input_tokens=cumulative,
        total_output_tokens=sum(t.output_tokens or 0 for t in turns),
        final_turn_input_tokens=final_input,
        compounding_ratio=ratio,
        turns=tuple(turns),
    )


def render_compounding_cost_json(
    db_path: Path,
    *,
    session_id: str | None = None,
    limit_sessions: int = 10,
    min_turns: int = 2,
) -> dict[str, Any]:
    sessions = compounding_cost_summaries(
        db_path,
        session_id=session_id,
        limit_sessions=limit_sessions,
        min_turns=min_turns,
    )
    return {
        "db_path": str(db_path),
        "session_count": len(sessions),
        "sessions": [
            {
                "session_id": s.session_id,
                "turn_count": s.turn_count,
                "tool_turn_count": s.tool_turn_count,
                "cumulative_input_tokens": s.cumulative_input_tokens,
                "total_output_tokens": s.total_output_tokens,
                "final_turn_input_tokens": s.final_turn_input_tokens,
                "compounding_ratio": s.compounding_ratio,
                "turns": [
                    {
                        "request_id": t.request_id,
                        "session_id": t.session_id,
                        "turn_index": t.turn_index,
                        "ts_start": t.ts_start,
                        "prompt_tokens": t.prompt_tokens,
                        "completion_tokens": t.completion_tokens,
                        "cached_tokens": t.cached_tokens,
                        "cache_read_tokens": t.cache_read_tokens,
                        "new_tokens": t.new_tokens,
                        "output_tokens": t.output_tokens,
                        "cumulative_input": t.cumulative_input,
                        "marginal_input": t.marginal_input,
                        "is_tool_turn": t.is_tool_turn,
                    }
                    for t in s.turns
                ],
            }
            for s in sessions
        ],
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


def _format_percent(value: float) -> str:
    # Treat any non-finite value as unknown.
    if not math.isfinite(value):
        return "<unknown>"
    return f"{value:.1f}%"


def _format_ts(ts: float | None) -> str:
    if ts is None:
        return "<no reset time>"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --- F2 live terminal view -----------------------------------------
#
# Operator decision 2026-08-23: the F2 live-visualization
# path for the token-usage diagnostic is a terminal UI (TUI), not OTel/
# OpenLLMetry — self-contained, no external backend, fits the single-operator
# loopback CLI. The non-live time-series already ships as `callosum usage
# series`; this slice adds the live, auto-refreshing
# delivery mode over the same read-only SQLite log.
#
# The implementation is deliberately stdlib-only (ANSI escapes + a sleep loop):
# adding a TUI library (textual/rich) would be a new runtime dependency, which
# is an operator-judgment call (see the ruamel precedent on
# ), not an autonomous one. A refresh loop over the existing
# `render_token_time_series_json` + `render_recent_turns_json` payloads keeps
# the surface read-only, daemon-optional, and free of new dependencies. The
# rendering is a pure function over those payloads so it tests without a TTY.

# Hide/show the cursor and clear+home the screen. Kept as named constants so
# the run loop never embeds raw escapes inline.
_HIDE_CURSOR = "\033[?25l"
_SHOW_CURSOR = "\033[?25h"
_CLEAR_SCREEN = "\033[2J\033[H"


def render_usage_live_text(
    series_payload: dict[str, Any],
    recent_payload: dict[str, Any],
    *,
    width: int,
    fetched_at_iso: str,
    interval_s: int,
) -> str:
    """Render one live-view frame from the existing diagnostic payloads.

    Pure: no I/O, no clock. The caller builds series_payload via
    render_token_time_series_json and recent_payload via
    render_recent_turns_json, plus a timestamp string and terminal width,
    so this function is deterministic and unit-testable. Output is plain text
    (no ANSI escapes) — the run loop wraps it with clear/screen control.
    """
    width = max(width, 40)
    lines: list[str] = [
        "callosum usage — live",
        (
            f"db: {series_payload.get('db_path', '?')}  "
            f"bucket={series_payload.get('bucket', '?')}  "
            f"group_by={series_payload.get('group_by', '?')}  "
            f"interval={interval_s}s"
        ),
        f"fetched at: {fetched_at_iso}",
        "",
        "Token volume over time",
        _format_series_header(),
    ]
    for bucket in series_payload.get("series", []):
        lines.append(_format_series_row(bucket))
    lines.append("")
    recent_turns = recent_payload.get("turns", [])
    lines.append(f"Recent turns (limit={recent_payload.get('turn_count', len(recent_turns))})")
    lines.append(_format_recent_header(width))
    for turn in recent_turns:
        lines.append(_format_recent_row(turn, width))
    lines.append("")
    lines.append(f"refreshing every {interval_s}s — press Ctrl-C to exit")
    return "\n".join(lines)


def _format_series_header() -> str:
    return f"{'bucket':<20} {'turns':>7} {'prompt':>10} {'compl':>9} {'total':>10}"


def _format_series_row(bucket: dict[str, Any]) -> str:
    return (
        f"{str(bucket.get('bucket_start', '?')):<20.20} "
        f"{_format_count(bucket.get('turn_count')):>7} "
        f"{_format_count(bucket.get('prompt_tokens')):>10} "
        f"{_format_count(bucket.get('completion_tokens')):>9} "
        f"{_format_count(bucket.get('total_tokens')):>10}"
    )


def _format_recent_header(width: int) -> str:
    mode_w, served_w = _recent_string_widths(width)
    return f"{'id':>6} {'ts':<16} {'mode':<{mode_w}} {'served':<{served_w}} {'prompt':>8} {'compl':>7} {'total':>8}"


def _format_recent_row(turn: dict[str, Any], width: int) -> str:
    mode_w, served_w = _recent_string_widths(width)
    ts = _format_turn_ts(turn.get("ts_start"))
    return (
        f"{str(turn.get('request_id', '')):>6} "
        f"{ts:<16.16} "
        f"{_truncate(str(turn.get('effective_routing_mode', '?')), mode_w):<{mode_w}} "
        f"{_truncate(str(turn.get('served_model', '?')), served_w):<{served_w}} "
        f"{_format_count(turn.get('prompt_tokens')):>8} "
        f"{_format_count(turn.get('completion_tokens')):>7} "
        f"{_format_count(turn.get('total_tokens')):>8}"
    )


# Fixed numeric columns in the recent-turns table: id(6) + ts(16) + prompt(8)
# + compl(7) + total(8) + six single-space separators = 52 columns. The two
# string columns (mode, served) share whatever width remains.
_RECENT_FIXED_WIDTH = 52


def _recent_string_widths(width: int) -> tuple[int, int]:
    avail = max(width - _RECENT_FIXED_WIDTH, 18)
    mode_w = max(8, min(28, avail // 2))
    served_w = max(8, avail - mode_w)
    return mode_w, served_w


def _format_turn_ts(ts_start: Any) -> str:
    if ts_start is None:
        return "?"
    try:
        return datetime.fromtimestamp(float(ts_start), tz=UTC).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "?"


def _truncate(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


def _format_count(n: Any) -> str:
    if n is None:
        return "-"
    try:
        value = int(n)
    except (TypeError, ValueError):
        return "-"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.0f}k"
    return str(value)


def run_usage_live(
    db_path: Path,
    *,
    bucket: str = "day",
    group_by: str = "mode",
    series_limit: int = 14,
    recent_limit: int = 15,
    interval_s: int = 5,
    out_stream: TextIO,
) -> int:
    """Refresh a live terminal view of the usage log until interrupted.

    Reads the SQLite log directly via the existing read-only render helpers
    (daemon-optional). When out_stream is a TTY the view clears and
    redraws every interval_s seconds, hiding the cursor while running and
    restoring it on exit (including Ctrl-C). When it is not a TTY (piped or
    redirected) it prints a single text snapshot and exits 0, so the command
    never sprays ANSI escapes into a pipe. Returns 0 on clean exit, 1 if the
    usage log is absent.
    """
    if interval_s <= 0:
        raise ValueError(f"interval must be positive: {interval_s}")
    is_tty = out_stream.isatty()
    try:
        frame = _build_usage_live_frame(
            db_path,
            bucket=bucket,
            group_by=group_by,
            series_limit=series_limit,
            recent_limit=recent_limit,
            interval_s=interval_s,
        )
        if not is_tty:
            out_stream.write(frame + "\n")
            out_stream.flush()
            return 0
        out_stream.write(_HIDE_CURSOR)
        out_stream.flush()
        while True:
            frame = _build_usage_live_frame(
                db_path,
                bucket=bucket,
                group_by=group_by,
                series_limit=series_limit,
                recent_limit=recent_limit,
                interval_s=interval_s,
            )
            out_stream.write(_CLEAR_SCREEN + frame + "\n")
            out_stream.flush()
            time.sleep(interval_s)
    except KeyboardInterrupt:
        return 0
    finally:
        if is_tty:
            out_stream.write(_SHOW_CURSOR)
            out_stream.flush()
    return 0


def _build_usage_live_frame(
    db_path: Path,
    *,
    bucket: str,
    group_by: str,
    series_limit: int,
    recent_limit: int,
    interval_s: int,
) -> str:
    series_payload = render_token_time_series_json(
        db_path,
        bucket=bucket,
        limit=series_limit,
        group_by=group_by,
    )
    recent_payload = render_recent_turns_json(db_path, limit=recent_limit)
    width = shutil.get_terminal_size((80, 24)).columns
    fetched_at_iso = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return render_usage_live_text(
        series_payload,
        recent_payload,
        width=width,
        fetched_at_iso=fetched_at_iso,
        interval_s=interval_s,
    )
