from __future__ import annotations

import io
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from callosum.usage_diagnostic import (
    CompoundingTurnSummary,
    SegmentSummary,
    SessionCompoundingSummary,
    TrafficKindBucketSummary,
    _format_count,
    _format_turn_ts,
    _truncate,
    compounding_cost_summaries,
    extract_peer_quality_segments,
    first_divergence_position,
    has_peer_quality_signature,
    is_tool_turn,
    recent_turn_summaries,
    render_compounding_cost_json,
    render_recent_turns_json,
    render_token_time_series_json,
    render_usage_live_text,
    run_usage_live,
    token_time_series,
)
from callosum.usage_log import UsageLog, UsageLogEntry


def _entry(
    *,
    ts_start: float,
    prompt_tokens: int | None,
    req_payload: dict | None,
    effective_routing_mode: str | None = "selector_remote-only",
    requested_model: str | None = "callosum:remote-only",
    served_model: str | None = "model-a0e7",
    traffic_kind: str | None = None,
    session_id: str | None = "sess-A",
    cached_tokens: int | None = None,
    completion_tokens: int = 12,
) -> UsageLogEntry:
    return UsageLogEntry(
        ts_start=ts_start,
        ts_end=ts_start + 0.5,
        route="responses",
        stream=False,
        session_id=session_id,
        backend_id="primary",
        model=served_model,
        reasoning_effort="medium",
        status=200,
        classification=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=None,
        cached_tokens=cached_tokens,
        # `req_payload` is what the diagnostic json-loads + segment-walks; it
        # is stored zlib-compressed by UsageLog and round-trips via decompress().
        req_payload=json.dumps(req_payload).encode() if req_payload is not None else None,
        requested_model=requested_model,
        effective_routing_mode=effective_routing_mode,
        traffic_kind=traffic_kind,
    )


def test_recent_turns_apportion_prompt_tokens_by_char_share(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # instructions = "sys"*4 = 12 chars; input = "user"*16 = 64 chars; total 76.
    payload = {"instructions": "sys" * 4, "input": "user" * 16}
    log.record(_entry(ts_start=1.0, prompt_tokens=100, req_payload=payload))
    log.close()

    turns = recent_turn_summaries(db, limit=10)
    assert len(turns) == 1
    turn = turns[0]
    assert turn.requested_model == "callosum:remote-only"
    assert turn.served_model == "model-a0e7"
    assert turn.effective_routing_mode == "selector_remote-only"
    assert turn.prompt_tokens == 100

    kinds = {s.kind for s in turn.segment_summaries}
    assert kinds == {"instructions", "input"}
    by_kind = {s.kind: s for s in turn.segment_summaries}
    assert by_kind["instructions"].chars == 12
    assert by_kind["input"].chars == 64
    # 100 tokens split by char share: 12/76*100 ≈ 15.8 → 16, 64/76*100 ≈ 84.2 → 84.
    assert by_kind["instructions"].est_prompt_tokens == 16
    assert by_kind["input"].est_prompt_tokens == 84
    # Apportioned tokens are conserved (sum == prompt_tokens) for this clean case.
    assert sum(s.est_prompt_tokens for s in turn.segment_summaries) == 100


def test_recent_turns_ordered_newest_first_and_coalesces_null_mode(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1.0, prompt_tokens=50, req_payload={"input": "first"}, effective_routing_mode=None))
    log.record(
        _entry(ts_start=2.0, prompt_tokens=50, req_payload={"input": "second"}, effective_routing_mode="forced_local")
    )
    log.close()

    turns = recent_turn_summaries(db, limit=10)
    assert [t.request_id for t in turns] == [2, 1]
    # NULL effective_routing_mode must read back as "auto" (COALESCE), not None.
    assert turns[1].effective_routing_mode == "auto"
    assert turns[0].effective_routing_mode == "forced_local"


def test_recent_turns_handle_missing_payload_and_chat_messages(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # Chat-Completions shape: segments come from messages[*].content.
    chat = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello world"},
        ]
    }
    log.record(_entry(ts_start=1.0, prompt_tokens=60, req_payload=chat))
    # No payload blob at all (capture off for this row in a mixed log).
    log.record(_entry(ts_start=2.0, prompt_tokens=30, req_payload=None))
    log.close()

    turns = recent_turn_summaries(db, limit=10)
    chat_turn = next(t for t in turns if t.request_id == 1)
    assert {s.kind for s in chat_turn.segment_summaries} == {"message:system", "message:user"}
    # No payload => empty segments, but tokens still pass through.
    bare = next(t for t in turns if t.request_id == 2)
    assert bare.segment_summaries == ()
    assert bare.prompt_tokens == 30


def test_limit_bounds_and_missing_db(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    for i in range(5):
        log.record(_entry(ts_start=float(i), prompt_tokens=10, req_payload={"input": "x"}))
    log.close()

    assert recent_turn_summaries(db, limit=0) == []
    assert len(recent_turn_summaries(db, limit=3)) == 3

    with pytest.raises(FileNotFoundError):
        recent_turn_summaries(tmp_path / "nope.sqlite", limit=5)


def test_render_recent_turns_json_shape(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1.0, prompt_tokens=40, req_payload={"input": "abc"}))
    log.close()

    doc = render_recent_turns_json(db, limit=5)
    assert doc["db_path"] == str(db)
    assert doc["turn_count"] == 1
    turn = doc["turns"][0]
    assert set(turn.keys()) >= {
        "request_id",
        "ts_start",
        "route",
        "requested_model",
        "served_model",
        "effective_routing_mode",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "segment_summaries",
    }
    assert turn["segment_summaries"][0]["kind"] == "input"
    assert isinstance(turn["segment_summaries"][0]["est_prompt_tokens"], int)


def test_token_time_series_groups_by_day_and_mode(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "abc"}))
    log.record(
        _entry(
            ts_start=1_200.0,
            prompt_tokens=60,
            req_payload={"input": "defg"},
            effective_routing_mode="forced_local",
        )
    )
    log.record(
        _entry(
            ts_start=90_000.0,
            prompt_tokens=30,
            req_payload={"input": "xyz"},
            effective_routing_mode=None,
        )
    )
    log.close()

    series = token_time_series(db, limit=2)
    assert len(series) == 2
    assert series[0].bucket_start == "1970-01-01T00:00:00Z"
    assert series[0].turn_count == 2
    assert series[0].prompt_tokens == 100
    assert [m.effective_routing_mode for m in series[0].mode_summaries] == [
        "forced_local",
        "selector_remote-only",
    ]
    assert series[1].bucket_start == "1970-01-02T00:00:00Z"
    assert series[1].turn_count == 1
    assert series[1].mode_summaries[0].effective_routing_mode == "auto"


def test_render_token_time_series_json_shape(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "abc"}))
    log.close()

    doc = render_token_time_series_json(db, bucket="day", limit=5)
    assert doc["db_path"] == str(db)
    assert doc["bucket"] == "day"
    assert doc["bucket_count"] == 1
    bucket = doc["series"][0]
    assert set(bucket.keys()) >= {
        "bucket_start",
        "turn_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "mode_summaries",
    }
    assert bucket["mode_summaries"][0]["effective_routing_mode"] == "selector_remote-only"


def test_token_time_series_groups_by_traffic_kind(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "a"}, traffic_kind="operator"))
    log.record(
        _entry(ts_start=1_100.0, prompt_tokens=60, req_payload={"input": "b"}, traffic_kind="min_coverage_quota")
    )
    log.record(
        _entry(ts_start=1_200.0, prompt_tokens=20, req_payload={"input": "c"}, traffic_kind="peer_quality_capture")
    )
    log.close()

    series = token_time_series(db, limit=2, group_by="traffic_kind")
    assert len(series) == 1
    bucket = series[0]
    assert bucket.traffic_kind_summaries is not None
    # operator + min_coverage_quota + peer_quality_capture, sorted by label
    assert [k.traffic_kind for k in bucket.traffic_kind_summaries] == [
        "min_coverage_quota",
        "operator",
        "peer_quality_capture",
    ]
    # mode axis was not requested -> empty, totals derived from traffic_kind rows
    assert bucket.mode_summaries == ()
    assert bucket.turn_count == 3
    assert bucket.prompt_tokens == 120


def test_token_time_series_legacy_null_traffic_kind_coalesces(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # Pre-F4 rows: traffic_kind left NULL -> must show as 'legacy', not 'operator'.
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "a"}))
    log.close()

    series = token_time_series(db, limit=2, group_by="traffic_kind")
    assert series[0].traffic_kind_summaries is not None
    assert [k.traffic_kind for k in series[0].traffic_kind_summaries] == ["legacy"]


def test_token_time_series_both_axes(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "a"}, traffic_kind="operator"))
    log.record(
        _entry(
            ts_start=1_100.0,
            prompt_tokens=50,
            req_payload={"input": "b"},
            traffic_kind="canary_redirect",
            effective_routing_mode="canary_redirect",
        )
    )
    log.close()

    series = token_time_series(db, limit=2, group_by="both")
    bucket = series[0]
    assert bucket.mode_summaries  # populated
    assert bucket.traffic_kind_summaries is not None  # populated
    # Totals are identical regardless of axis.
    mode_total = sum(m.total_tokens for m in bucket.mode_summaries)
    kind_total = sum(k.total_tokens for k in bucket.traffic_kind_summaries)
    assert mode_total == kind_total == bucket.total_tokens
    assert {k.traffic_kind for k in bucket.traffic_kind_summaries} == {"operator", "canary_redirect"}


def test_token_time_series_default_group_by_leaves_traffic_kind_none(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "a"}, traffic_kind="operator"))
    log.close()

    series = token_time_series(db, limit=2)  # default group_by="mode"
    assert series[0].traffic_kind_summaries is None
    assert series[0].mode_summaries  # back-compat: mode axis populated as before


def test_token_time_series_rejects_bad_group_by(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "a"}))
    log.close()
    with pytest.raises(ValueError):
        token_time_series(db, group_by="nonsense")


def test_render_token_time_series_json_includes_traffic_kind(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000.0, prompt_tokens=40, req_payload={"input": "a"}, traffic_kind="operator"))
    log.close()

    doc = render_token_time_series_json(db, group_by="traffic_kind")
    assert doc["group_by"] == "traffic_kind"
    bucket = doc["series"][0]
    assert "traffic_kind_summaries" in bucket
    assert bucket["traffic_kind_summaries"][0]["traffic_kind"] == "operator"
    # mode_summaries still present (empty) for shape stability.
    assert bucket["mode_summaries"] == []


def test_traffic_kind_bucket_summary_is_frozen() -> None:
    k = TrafficKindBucketSummary(
        traffic_kind="operator",
        turn_count=1,
        prompt_tokens=10,
        completion_tokens=2,
        total_tokens=12,
    )
    with pytest.raises(FrozenInstanceError):
        k.turn_count = 2  # type: ignore[misc]


def test_segment_summary_is_frozen() -> None:
    seg = SegmentSummary(kind="input", chars=4, est_prompt_tokens=2)
    with pytest.raises(FrozenInstanceError):
        seg.kind = "instructions"  # type: ignore[misc]


# --- Peer-quality (in-band) stream segmentation + first-divergence ----------
# These cover the retired in-band path's payload shape (work tracker ):
# provenance tags `<model|effort|reqid>...</model|effort|reqid>` around prior
# assistant text + a trailing developer audit instruction carrying `<<qop ...>>`
# markers. Non-peer-quality turns must keep the original taxonomy unchanged.

_AUDIT_INSTRUCTION = (
    "Hidden Callosum quality audit (mandatory, applies even when you call a tool). "
    "Emit the marker below with score and reason replaced by your honest judgement.\n"
    "<<qop nonce=abc subject=model-a0e7|medium subject_request_id=42 score=+1 reason=short>>"
)


def _on_arm_responses_payload() -> dict:
    return {
        "instructions": "sys",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do the thing"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "input_text", "text": "<model-a0e7|medium|42>did the thing</model-a0e7|medium|42>"}
                ],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now do this"}]},
            {
                "type": "message",
                "role": "developer",
                "content": [{"type": "input_text", "text": _AUDIT_INSTRUCTION}],
            },
        ],
    }


def _on_arm_chat_payload() -> dict:
    return {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "assistant", "content": "<model-a0e7|42>did it</model-a0e7|42>"},
            {"role": "user", "content": "go"},
            {"role": "developer", "content": _AUDIT_INSTRUCTION},
        ]
    }


def test_off_arm_responses_uses_original_taxonomy(tmp_path: Path) -> None:
    # No provenance tags, no audit sentinel -> existing _extract_segments path.
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    payload = {
        "instructions": "sys",
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
    }
    assert has_peer_quality_signature(payload) is False
    log.record(_entry(ts_start=1.0, prompt_tokens=50, req_payload=payload))
    log.close()

    turns = recent_turn_summaries(db, limit=5)
    assert len(turns) == 1
    assert {s.kind for s in turns[0].segment_summaries} == {"instructions", "input"}


def test_on_arm_responses_segments_into_four_kinds(tmp_path: Path) -> None:
    payload = _on_arm_responses_payload()
    assert has_peer_quality_signature(payload) is True

    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1.0, prompt_tokens=300, req_payload=payload))
    log.close()

    turns = recent_turn_summaries(db, limit=5)
    segs = {s.kind: s for s in turns[0].segment_summaries}
    assert set(segs) == {"stable_front", "provenance_mutated_history", "live_tail", "peer_opinion_suffix"}
    # Tag overhead is counted inside the mutated-history segment (bytes sent).
    assert segs["provenance_mutated_history"].chars == len("<model-a0e7|medium|42>did the thing</model-a0e7|medium|42>")
    assert segs["peer_opinion_suffix"].chars == len(_AUDIT_INSTRUCTION)
    assert segs["live_tail"].chars == len("now do this")
    # Char-share apportionment is conserved within rounding drift (4 segments).
    assert abs(sum(s.est_prompt_tokens for s in turns[0].segment_summaries) - 300) <= 2


def test_on_arm_chat_segments_into_four_kinds(tmp_path: Path) -> None:
    payload = _on_arm_chat_payload()
    assert has_peer_quality_signature(payload) is True

    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1.0, prompt_tokens=200, req_payload=payload))
    log.close()

    turns = recent_turn_summaries(db, limit=5)
    segs = {s.kind: s for s in turns[0].segment_summaries}
    assert set(segs) == {"stable_front", "provenance_mutated_history", "live_tail", "peer_opinion_suffix"}
    assert segs["stable_front"].chars == len("sys")
    assert segs["provenance_mutated_history"].chars == len("<model-a0e7|42>did it</model-a0e7|42>")
    assert segs["live_tail"].chars == len("go")
    assert abs(sum(s.est_prompt_tokens for s in turns[0].segment_summaries) - 200) <= 2


def test_instructions_fallback_splits_out_suffix() -> None:
    # No input/messages -> audit appended to instructions (app.py fallback shape).
    payload = {"instructions": "sys\n\n" + _AUDIT_INSTRUCTION}
    assert has_peer_quality_signature(payload) is True
    segs = {k: v for k, v in extract_peer_quality_segments(payload)}
    assert set(segs) == {"stable_front", "peer_opinion_suffix"}
    assert segs["stable_front"] == "sys"
    assert segs["peer_opinion_suffix"] == _AUDIT_INSTRUCTION


def test_multi_subject_history_coalesces_and_conserves(tmp_path: Path) -> None:
    payload = {
        "instructions": "sys",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ask one"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "<model-a0e7|medium|41>ans one</model-a0e7|medium|41>"}],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "ask two"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "<model-a0e7|medium|42>ans two</model-a0e7|medium|42>"}],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now this"}]},
            {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": _AUDIT_INSTRUCTION}]},
        ],
    }
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1.0, prompt_tokens=1000, req_payload=payload))
    log.close()

    turns = recent_turn_summaries(db, limit=5)
    segs = {s.kind: s for s in turns[0].segment_summaries}
    # Two tagged assistant items + the interleaved user item coalesce into one
    # provenance_mutated_history segment; live_tail is the final user turn.
    assert set(segs) == {"stable_front", "provenance_mutated_history", "live_tail", "peer_opinion_suffix"}
    assert segs["live_tail"].chars == len("now this")
    assert sum(s.est_prompt_tokens for s in turns[0].segment_summaries) == 1000


def test_first_divergence_position() -> None:
    assert first_divergence_position(b"abc", b"abc") == -1
    assert first_divergence_position(b"abc", b"abd") == 2
    assert first_divergence_position(b"abc", b"abcdef") == 3
    assert first_divergence_position(b"abcdef", b"abc") == 3
    # The off-vs-on arm divergence lands at the first provenance tag opening:
    # the off-arm assistant text is the bare "did the thing"; the on-arm opens
    # with "<model-a0e7|medium|42>". The shared stable front (instructions + first
    # user turn) is byte-identical, so divergence is inside the assistant item.
    on_payload = _on_arm_responses_payload()
    off_arm = {
        "instructions": "sys",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "do the thing"}]},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "input_text", "text": "did the thing"}],
            },
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now do this"}]},
        ],
    }
    serialized_on = json.dumps(on_payload).encode()
    serialized_off = json.dumps(off_arm).encode()
    pos = first_divergence_position(serialized_off, serialized_on)
    assert pos >= 0
    # Divergence is exactly at the assistant content: bare text off-arm vs tag on-arm.
    assert serialized_off[pos:].startswith(b"did the thing")
    assert serialized_on[pos:].startswith(b"<model-a0e7|medium|42>")


def test_has_peer_quality_signature_truth_table() -> None:
    assert has_peer_quality_signature({}) is False
    assert has_peer_quality_signature({"messages": [{"role": "user", "content": "hi"}]}) is False
    assert has_peer_quality_signature({"input": [{"role": "assistant", "content": "plain text"}]}) is False
    assert (
        has_peer_quality_signature({"messages": [{"role": "assistant", "content": "<model-a0e7|42>x</model-a0e7|42>"}]})
        is True
    )
    assert has_peer_quality_signature({"messages": [{"role": "developer", "content": _AUDIT_INSTRUCTION}]}) is True
    assert has_peer_quality_signature({"instructions": _AUDIT_INSTRUCTION}) is True


# --- Per-session compounding input-token cost () ------------
# Each turn of a multi-turn session re-sends the growing transcript, so the
# cumulative input across a session grows O(K^2) while per-turn input grows
# O(K). These tests cover the decomposition (cache_read/new/output mirroring
# the OTel GenAI convention), the running cumulative + per-turn marginal, the
# compounding_ratio, tool-turn attribution, and session filtering/ordering.


def test_is_tool_turn_truth_table() -> None:
    # No payload shape -> False.
    assert is_tool_turn({}) is False
    # Plain user turn -> False.
    assert is_tool_turn({"messages": [{"role": "user", "content": "hi"}]}) is False
    assert is_tool_turn({"input": [{"type": "message", "role": "user", "content": "hi"}]}) is False
    # Chat Completions: a tool-result message -> True.
    assert is_tool_turn({"messages": [{"role": "tool", "content": "42"}]}) is True
    # Chat Completions: an assistant turn carrying tool_calls -> True.
    assert is_tool_turn({"messages": [{"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]}]}) is True
    # Responses API: function_call_output -> True.
    assert is_tool_turn({"input": [{"type": "function_call_output", "output": "42"}]}) is True
    # Responses API: function_call -> True.
    assert is_tool_turn({"input": [{"type": "function_call", "name": "run"}]}) is True
    # Any item carrying tool_call_id -> True (covers unnamed tool-result shapes).
    assert is_tool_turn({"input": [{"tool_call_id": "call_1", "output": "42"}]}) is True


def test_compounding_cost_basic_session_growth(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # Three-turn session: per-turn input grows 100 -> 150 -> 200; cache hit
    # grows 0 -> 50 -> 100, so the uncached `new_tokens` is a constant 100.
    for i, (prompt, cached) in enumerate([(100, 0), (150, 50), (200, 100)]):
        log.record(
            _entry(
                ts_start=float(i),
                prompt_tokens=prompt,
                cached_tokens=cached,
                req_payload={"input": f"turn {i}"},
                session_id="sess-A",
            )
        )
    log.close()

    sessions = compounding_cost_summaries(db, limit_sessions=10)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.session_id == "sess-A"
    assert s.turn_count == 3
    assert s.tool_turn_count == 0  # plain input turns only
    turns = s.turns
    assert [t.turn_index for t in turns] == [0, 1, 2]
    # OTel-mirrored decomposition: cache_read = cached_tokens, new = prompt - cached.
    assert [t.cache_read_tokens for t in turns] == [0, 50, 100]
    assert [t.new_tokens for t in turns] == [100, 100, 100]
    assert [t.output_tokens for t in turns] == [12, 12, 12]
    # Running cumulative input: 100, 100+150=250, 250+200=450.
    assert [t.cumulative_input for t in turns] == [100, 250, 450]
    # Marginal vs prior turn: first turn = its own input; later = delta.
    assert [t.marginal_input for t in turns] == [100, 50, 50]
    # Session-level rollup: cumulative = 450, final = 200, ratio = 2.25.
    assert s.cumulative_input_tokens == 450
    assert s.final_turn_input_tokens == 200
    assert s.compounding_ratio == 2.25
    assert s.total_output_tokens == 36


def test_compounding_cost_tool_turn_attribution(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # Turn 1 re-sends a tool result back to the model.
    tool_payload = {"messages": [{"role": "user", "content": "run it"}, {"role": "tool", "content": "result=42"}]}
    log.record(_entry(ts_start=0.0, prompt_tokens=100, req_payload={"input": "ask"}, session_id="sess-T"))
    log.record(_entry(ts_start=1.0, prompt_tokens=180, req_payload=tool_payload, session_id="sess-T"))
    log.close()

    sessions = compounding_cost_summaries(db, limit_sessions=10)
    assert len(sessions) == 1
    s = sessions[0]
    assert s.tool_turn_count == 1
    assert [t.is_tool_turn for t in s.turns] == [False, True]


def test_compounding_cost_min_turns_filters_single_turn_sessions(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=0.0, prompt_tokens=100, req_payload={"input": "only"}, session_id="solo"))
    log.record(_entry(ts_start=1.0, prompt_tokens=120, req_payload={"input": "a"}, session_id="pair"))
    log.record(_entry(ts_start=2.0, prompt_tokens=140, req_payload={"input": "b"}, session_id="pair"))
    log.close()

    # Default min_turns=2 drops the single-turn session.
    sessions = compounding_cost_summaries(db, limit_sessions=10)
    assert [s.session_id for s in sessions] == ["pair"]
    # min_turns=1 includes it (solo has no compounding, but is not hidden).
    sessions_all = compounding_cost_summaries(db, limit_sessions=10, min_turns=1)
    assert {s.session_id for s in sessions_all} == {"solo", "pair"}
    # The solo session has ratio == 1.0 (cumulative == final == 100).
    solo = next(s for s in sessions_all if s.session_id == "solo")
    assert solo.compounding_ratio == 1.0


def test_compounding_cost_session_filter_and_limit(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    for i in range(3):
        log.record(_entry(ts_start=float(i), prompt_tokens=100 + i, req_payload={"input": "x"}, session_id="alpha"))
    for i in range(3):
        log.record(_entry(ts_start=float(i), prompt_tokens=200 + i, req_payload={"input": "y"}, session_id="beta"))
    log.close()

    # --session-id restricts to one session.
    alpha = compounding_cost_summaries(db, session_id="alpha", limit_sessions=10)
    assert len(alpha) == 1
    assert alpha[0].session_id == "alpha"
    assert alpha[0].turn_count == 3
    # --limit-sessions=1 picks the most-recently-active session (beta has the
    # larger max request id).
    limited = compounding_cost_summaries(db, limit_sessions=1)
    assert len(limited) == 1
    assert limited[0].session_id == "beta"


def test_compounding_cost_handles_null_prompt_tokens(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=0.0, prompt_tokens=100, req_payload={"input": "a"}, session_id="sess-N"))
    # A legacy/errored row with NULL prompt_tokens must not break the metric.
    log.record(_entry(ts_start=1.0, prompt_tokens=None, req_payload={"input": "b"}, session_id="sess-N"))
    log.record(_entry(ts_start=2.0, prompt_tokens=300, req_payload={"input": "c"}, session_id="sess-N"))
    log.close()

    sessions = compounding_cost_summaries(db, limit_sessions=10)
    assert len(sessions) == 1
    s = sessions[0]
    turns = s.turns
    # The middle turn has None prompt -> None new/cumulative/marginal; the
    # running cumulative carries forward (treats the gap as 0).
    assert turns[1].prompt_tokens is None
    assert turns[1].new_tokens is None
    assert turns[1].cumulative_input is None
    assert turns[1].marginal_input is None
    # Cumulative resumes: 100 (turn 0) + 0 (turn 1) + 300 (turn 2) = 400.
    assert turns[2].cumulative_input == 400
    # final_turn_input is the last non-None? No — it is the last turn's
    # prompt_tokens literally (300 here), so ratio = 400/300.
    assert s.final_turn_input_tokens == 300
    assert s.compounding_ratio == 400 / 300


def test_compounding_cost_orders_sessions_by_most_recent_activity(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # alpha starts first but has the smaller max id; beta is more recent.
    for i in range(2):
        log.record(_entry(ts_start=float(i), prompt_tokens=100, req_payload={"input": "x"}, session_id="alpha"))
    for i in range(2):
        log.record(_entry(ts_start=float(i), prompt_tokens=100, req_payload={"input": "y"}, session_id="beta"))
    log.close()

    sessions = compounding_cost_summaries(db, limit_sessions=10)
    assert [s.session_id for s in sessions] == ["beta", "alpha"]


def test_compounding_cost_bounds_and_missing_db(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=0.0, prompt_tokens=100, req_payload={"input": "x"}, session_id="a"))
    log.record(_entry(ts_start=1.0, prompt_tokens=120, req_payload={"input": "y"}, session_id="a"))
    log.close()

    assert compounding_cost_summaries(db, limit_sessions=0) == []
    with pytest.raises(FileNotFoundError):
        compounding_cost_summaries(tmp_path / "nope.sqlite", limit_sessions=5)


def test_compounding_cost_ignores_null_session_id(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    # Rows with no session_id cannot be attributed and must be skipped.
    log.record(_entry(ts_start=0.0, prompt_tokens=100, req_payload={"input": "x"}, session_id=None))
    log.record(_entry(ts_start=1.0, prompt_tokens=120, req_payload={"input": "y"}, session_id="a"))
    log.record(_entry(ts_start=2.0, prompt_tokens=140, req_payload={"input": "z"}, session_id="a"))
    log.close()

    sessions = compounding_cost_summaries(db, limit_sessions=10)
    assert [s.session_id for s in sessions] == ["a"]


def test_render_compounding_cost_json_shape(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=0.0, prompt_tokens=100, cached_tokens=0, req_payload={"input": "a"}, session_id="s"))
    log.record(_entry(ts_start=1.0, prompt_tokens=150, cached_tokens=50, req_payload={"input": "b"}, session_id="s"))
    log.close()

    doc = render_compounding_cost_json(db, limit_sessions=5)
    assert doc["db_path"] == str(db)
    assert doc["session_count"] == 1
    session = doc["sessions"][0]
    assert set(session.keys()) >= {
        "session_id",
        "turn_count",
        "tool_turn_count",
        "cumulative_input_tokens",
        "total_output_tokens",
        "final_turn_input_tokens",
        "compounding_ratio",
        "turns",
    }
    turn = session["turns"][0]
    assert set(turn.keys()) >= {
        "request_id",
        "session_id",
        "turn_index",
        "ts_start",
        "prompt_tokens",
        "completion_tokens",
        "cached_tokens",
        "cache_read_tokens",
        "new_tokens",
        "output_tokens",
        "cumulative_input",
        "marginal_input",
        "is_tool_turn",
    }
    assert turn["cache_read_tokens"] == 0
    assert turn["new_tokens"] == 100
    assert session["compounding_ratio"] == pytest.approx(250 / 150)  # (100+150)/150


def test_compounding_turn_summary_is_frozen() -> None:
    t = CompoundingTurnSummary(
        request_id=1,
        session_id="s",
        turn_index=0,
        ts_start=0.0,
        prompt_tokens=10,
        completion_tokens=2,
        cached_tokens=0,
        cache_read_tokens=0,
        new_tokens=10,
        output_tokens=2,
        cumulative_input=10,
        marginal_input=10,
        is_tool_turn=False,
    )
    with pytest.raises(FrozenInstanceError):
        t.is_tool_turn = True  # type: ignore[misc]


def test_session_compounding_summary_is_frozen() -> None:
    s = SessionCompoundingSummary(
        session_id="s",
        turn_count=1,
        tool_turn_count=0,
        cumulative_input_tokens=10,
        total_output_tokens=2,
        final_turn_input_tokens=10,
        compounding_ratio=1.0,
        turns=(),
    )
    with pytest.raises(FrozenInstanceError):
        s.turn_count = 2  # type: ignore[misc]


# --- F2 live terminal view () --------------------------------


def test_format_count_humanizes_and_handles_none() -> None:
    assert _format_count(None) == "-"
    assert _format_count(0) == "0"
    assert _format_count(999) == "999"
    assert _format_count(1_000) == "1k"
    assert _format_count(2_400_000) == "2.4M"
    assert _format_count("not-a-number") == "-"


def test_truncate() -> None:
    assert _truncate("abc", 5) == "abc"
    assert _truncate("abcdef", 5) == "abcd…"
    assert _truncate("ab", 2) == "ab"
    assert _truncate("abc", 0) == ""
    assert _truncate("abc", 1) == "a"


def test_format_turn_ts() -> None:
    assert _format_turn_ts(None) == "?"
    assert _format_turn_ts(0) == "1970-01-01 00:00"
    assert _format_turn_ts("not-a-number") == "?"


def test_render_usage_live_text_shape(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000_000_000.0, prompt_tokens=1_500, req_payload={"input": "x"}))
    log.record(
        _entry(
            ts_start=1_000_010_000.0,
            prompt_tokens=2_400_000,
            req_payload={"input": "y"},
            served_model="model-a0e8",
            effective_routing_mode="forced_local",
        )
    )
    log.close()

    series = render_token_time_series_json(db, limit=5)
    recent = render_recent_turns_json(db, limit=5)
    frame = render_usage_live_text(
        series,
        recent,
        width=120,
        fetched_at_iso="2026-08-23T14:02:00Z",
        interval_s=5,
    )
    lines = frame.splitlines()

    assert lines[0] == "callosum usage — live"
    assert "interval=5s" in lines[1]
    assert lines[2] == "fetched at: 2026-08-23T14:02:00Z"
    assert "Token volume over time" in frame
    assert "Recent turns (limit=2)" in frame
    assert lines[-1] == "refreshing every 5s — press Ctrl-C to exit"
    # Humanized token counts appear in the rendered frame.
    assert "2.4M" in frame
    # The recent-turns panel surfaces the served model and routing mode.
    assert "model-a0e8" in frame
    assert "forced_local" in frame
    # No ANSI escapes in the pure render (the run loop adds them).
    assert "\033[" not in frame


def test_render_usage_live_text_clamps_narrow_width(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(
        _entry(
            ts_start=1_000_000_000.0,
            prompt_tokens=100,
            req_payload={"input": "x"},
            served_model="a-very-long-served-model-name",
            effective_routing_mode="a-very-long-routing-mode-name",
        )
    )
    log.close()
    series = render_token_time_series_json(db, limit=5)
    recent = render_recent_turns_json(db, limit=5)
    frame = render_usage_live_text(
        series,
        recent,
        width=40,
        fetched_at_iso="2026-08-23T14:02:00Z",
        interval_s=3,
    )
    # The narrow terminal truncates the recent-turns string columns rather
    # than overflowing: the full long names must not survive into the frame,
    # but the ellipsis marker from truncation does.
    assert "a-very-long-served-model-name" not in frame
    assert "a-very-long-routing-mode-name" not in frame
    assert "…" in frame


def test_run_usage_live_non_tty_single_snapshot(tmp_path: Path) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000_000_000.0, prompt_tokens=500, req_payload={"input": "x"}))
    log.close()

    out = io.StringIO()
    assert out.isatty() is False
    rc = run_usage_live(db, interval_s=1, out_stream=out)

    assert rc == 0
    text = out.getvalue()
    assert "\033[" not in text  # no ANSI escapes when not a TTY
    assert "callosum usage — live" in text
    # A single snapshot, not a refresh loop.
    assert text.count("callosum usage — live") == 1


def test_run_usage_live_tty_loop_exits_on_interrupt_and_restores_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = tmp_path / "u.sqlite"
    log = UsageLog(db, capture_bodies=True)
    log.record(_entry(ts_start=1_000_000_000.0, prompt_tokens=500, req_payload={"input": "x"}))
    log.close()

    class TTYStream(io.StringIO):
        def isatty(self) -> bool:
            return True

    def raise_keyboard(_s: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr("callosum.usage_diagnostic.time.sleep", raise_keyboard)

    out = TTYStream()
    rc = run_usage_live(db, interval_s=1, out_stream=out)

    assert rc == 0
    text = out.getvalue()
    assert "\033[?25l" in text  # cursor hidden while running
    assert "\033[?25h" in text  # cursor restored on exit
    assert "\033[2J\033[H" in text  # screen cleared each frame
    assert "callosum usage — live" in text


def test_run_usage_live_rejects_nonpositive_interval(tmp_path: Path) -> None:
    out = io.StringIO()
    with pytest.raises(ValueError):
        run_usage_live(tmp_path / "u.sqlite", interval_s=0, out_stream=out)


def test_run_usage_live_missing_db_propagates(tmp_path: Path) -> None:
    out = io.StringIO()
    with pytest.raises(FileNotFoundError):
        run_usage_live(tmp_path / "nope.sqlite", interval_s=1, out_stream=out)
