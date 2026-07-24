from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from callosum.usage_diagnostic import (
    SegmentSummary,
    TrafficKindBucketSummary,
    extract_peer_quality_segments,
    first_divergence_position,
    has_peer_quality_signature,
    recent_turn_summaries,
    render_recent_turns_json,
    render_token_time_series_json,
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
) -> UsageLogEntry:
    return UsageLogEntry(
        ts_start=ts_start,
        ts_end=ts_start + 0.5,
        route="responses",
        stream=False,
        session_id="sess-A",
        backend_id="primary",
        model=served_model,
        reasoning_effort="medium",
        status=200,
        classification=None,
        prompt_tokens=prompt_tokens,
        completion_tokens=12,
        total_tokens=None,
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
    log.record(_entry(ts_start=1_100.0, prompt_tokens=60, req_payload={"input": "b"}, traffic_kind="min_coverage_quota"))
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
                "content": [{"type": "input_text", "text": "<model-a0e7|medium|42>did the thing</model-a0e7|medium|42>"}],
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
    assert segs["provenance_mutated_history"].chars == len(
        "<model-a0e7|medium|42>did the thing</model-a0e7|medium|42>"
    )
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
    assert has_peer_quality_signature({"messages": [{"role": "assistant", "content": "<model-a0e7|42>x</model-a0e7|42>"}]}) is True
    assert has_peer_quality_signature({"messages": [{"role": "developer", "content": _AUDIT_INSTRUCTION}]}) is True
    assert has_peer_quality_signature({"instructions": _AUDIT_INSTRUCTION}) is True
