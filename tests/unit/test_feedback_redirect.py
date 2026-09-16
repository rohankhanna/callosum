"""Tests for the surface-only feedback redirect ().

Covers: the conservative snippet scrubber, the redirect-message formatter,
the reserved auto-send switch (OFF / not wired), and the UsageLog
feedback-suggestion audit methods (derive-on-read pending list, get one,
acknowledge/dismiss decision, status, count).
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from callosum.feedback_redirect import (
    FeedbackAutoSendNotWiredError,
    build_redirect_message,
    feedback_auto_send_enabled,
    format_feedback_redirect,
    scrub_snippet,
)
from callosum.usage_log import UsageLog, UsageLogEntry

# ---------------------------------------------------------------------------
# scrub_snippet
# ---------------------------------------------------------------------------


def test_scrub_redacts_openai_api_key() -> None:
    out = scrub_snippet("the key is sk-abcdefghijklmnopqrstuvwxyz0123 and done")
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in out
    assert "[REDACTED:api-key]" in out
    assert "done" in out


def test_scrub_redacts_bearer_token_case_insensitive() -> None:
    out = scrub_snippet("Authorization: Bearer abc123def456ghi789jkl012mno345")
    assert "abc123def456ghi789jkl012mno345" not in out
    assert "[REDACTED:bearer]" in out
    # The literal Bearer word is also consumed by the replacement.
    assert "Bearer abc123def456ghi789jkl012mno345" not in out


def test_scrub_redacts_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpM"
    out = scrub_snippet(f"token={jwt}")
    assert jwt not in out
    assert "[REDACTED:jwt]" in out


def test_scrub_redacts_account_id_value_not_key() -> None:
    out = scrub_snippet('chatgpt_user_id=acct_abc123def456 and account_id="x_y_z_99"')
    # The key names survive; only the values are redacted.
    assert "chatgpt_user_id" in out
    assert "account_id" in out
    assert "acct_abc123def456" not in out
    assert "x_y_z_99" not in out
    assert out.count("[REDACTED:account]") == 2


def test_scrub_redacts_long_opaque_run() -> None:
    blob = "a" * 64  # 64 hex/base64-alphabet chars
    out = scrub_snippet(f"prefix {blob} suffix")
    assert blob not in out
    assert "[REDACTED:opaque]" in out


def test_scrub_leaves_short_normal_text_intact() -> None:
    text = "def foo(x: int) -> str: return str(x)  # a normal comment"
    # No secret-shaped substrings => text passes through unchanged.
    assert scrub_snippet(text) == text


def test_scrub_empty_returns_empty() -> None:
    assert scrub_snippet("") == ""


# ---------------------------------------------------------------------------
# format_feedback_redirect / build_redirect_message
# ---------------------------------------------------------------------------


def test_format_feedback_redirect_scrubs_and_formats() -> None:
    msg = format_feedback_redirect(
        request_id=42,
        thread_id="thread-xyz",
        model="gpt-x",
        reasoning_effort="high",
        detector="user",
        prompt_text="fix this sk-abcdefghijklmnopqrstuvwxyz0123 please",
        response_text="here is a 64char secret: " + "a" * 64,
    )
    assert "request 42" in msg
    assert "thread thread-xyz" in msg
    assert "gpt-x/high" in msg
    assert "detector: user" in msg
    # Scrubbed — no raw secrets in the rendered message.
    assert "sk-abcdefghijklmnopqrstuvwxyz0123" not in msg
    assert "a" * 64 not in msg
    assert "[REDACTED:api-key]" in msg
    assert "[REDACTED:opaque]" in msg


def test_format_feedback_redirect_handles_missing_thread_and_effort() -> None:
    msg = format_feedback_redirect(
        request_id=7,
        thread_id=None,
        model="model-a0g2",
        reasoning_effort=None,
        detector=None,
        prompt_text="hi",
        response_text="lo",
    )
    assert "thread (none)" in msg
    assert "detector: unknown" in msg
    assert "model-a0g2" in msg
    # No trailing slash when effort is absent.
    assert "model-a0g2/" not in msg


def test_build_redirect_message_uses_prescrubbed_input() -> None:
    # build_redirect_message does NOT scrub; the caller must scrub first.
    raw = "sk-rawsecret0123456789abcdefghij"
    from callosum.feedback_redirect import _RedirectContext  # noqa: WPS437

    msg = build_redirect_message(
        _RedirectContext(
            request_id=1,
            thread_id="t",
            model="m",
            reasoning_effort=None,
            detector="d",
            scrubbed_prompt=raw,
            scrubbed_response=raw,
        )
    )
    # Passed through verbatim — confirms the formatter itself is scrub-free.
    assert raw in msg


# ---------------------------------------------------------------------------
# reserved auto-send switch (future automation stage, OFF, not wired)
# ---------------------------------------------------------------------------


def test_auto_send_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("CALLOSUM_FEEDBACK_AUTO_SEND_ENABLED", raising=False)
    assert feedback_auto_send_enabled() is False


def test_auto_send_enabled_only_on_exact_one(monkeypatch) -> None:
    monkeypatch.setenv("CALLOSUM_FEEDBACK_AUTO_SEND_ENABLED", "1")
    assert feedback_auto_send_enabled() is True
    monkeypatch.setenv("CALLOSUM_FEEDBACK_AUTO_SEND_ENABLED", "true")
    assert feedback_auto_send_enabled() is False
    monkeypatch.setenv("CALLOSUM_FEEDBACK_AUTO_SEND_ENABLED", "0")
    assert feedback_auto_send_enabled() is False


def test_auto_send_not_wired_error_is_raised_for_future_guard() -> None:
    # The reserved guard exists so a future call site that observes
    # feedback_auto_send_enabled() True without an injection path raises
    # rather than silently no-ops. We only assert the class is raisable.
    try:
        raise FeedbackAutoSendNotWiredError("not wired")
    except FeedbackAutoSendNotWiredError as exc:
        assert "not wired" in str(exc)


# ---------------------------------------------------------------------------
# UsageLog feedback-suggestion audit methods
# ---------------------------------------------------------------------------


def _entry(**overrides: object) -> UsageLogEntry:
    base: dict[str, object] = dict(
        ts_start=1000.0,
        ts_end=1000.25,
        route="responses",
        stream=True,
        session_id="sess-1",
        backend_id="primary",
        model="model-a0e7",
        reasoning_effort="medium",
        status=200,
        classification="ok",
        request_bytes=512,
        response_bytes=2048,
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        cached_tokens=20,
        reasoning_tokens=5,
    )
    base.update(overrides)
    return UsageLogEntry(**base)  # type: ignore[arg-type]


def _flag(
    log: UsageLog,
    *,
    prompt: str,
    response: str,
    detector: str = "user",
    session_id: str = "sess-1",
    model: str = "gpt-x",
    effort: str = "medium",
    ts_start: float = 1000.0,
) -> int:
    """Insert a request row, then flag it quality_score=-1 with snippet text."""
    rid = log.record(
        _entry(
            session_id=session_id,
            model=model,
            reasoning_effort=effort,
            ts_start=ts_start,
            ts_end=ts_start + 0.25,
        )
    )
    conn = sqlite3.connect(log.path)
    conn.execute(
        "UPDATE requests SET prompt_text = ?, response_text = ?, "
        "quality_score = -1, quality_label_method = ? WHERE id = ?",
        (prompt, response, detector, rid),
    )
    conn.commit()
    conn.close()
    return rid


def _good(log: UsageLog, *, prompt: str = "ok prompt", response: str = "ok response", ts_start: float = 1000.0) -> int:
    """Insert a request row with a +1 quality score (not a bad output)."""
    rid = log.record(_entry(ts_start=ts_start, ts_end=ts_start + 0.25))
    conn = sqlite3.connect(log.path)
    conn.execute(
        "UPDATE requests SET prompt_text = ?, response_text = ?, quality_score = 1, "
        "quality_label_method = 'user' WHERE id = ?",
        (prompt, response, rid),
    )
    conn.commit()
    conn.close()
    return rid


def test_list_pending_returns_only_flagged_undecided(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    bad1 = _flag(log, prompt="p1", response="r1", detector="user", ts_start=2000.0)
    bad2 = _flag(log, prompt="p2", response="r2", detector="implicit_failure_v1", ts_start=3000.0)
    _good(log, ts_start=4000.0)  # not flagged — must not appear
    pending = log.list_pending_feedback_suggestions()
    assert [s.request_id for s in pending] == [bad2, bad1]  # newest first
    assert pending[0].quality_label_method == "implicit_failure_v1"
    assert pending[0].prompt_text == "p2"
    assert pending[0].model == "gpt-x"
    assert pending[0].session_id == "sess-1"
    log.close()


def test_list_pending_skips_rows_without_snippet_text(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rid = log.record(_entry(ts_start=1000.0))
    conn = sqlite3.connect(log.path)
    # Flagged but no snippet text → not surfaceable.
    conn.execute(
        "UPDATE requests SET quality_score = -1, quality_label_method = 'user' WHERE id = ?",
        (rid,),
    )
    conn.commit()
    conn.close()
    assert log.list_pending_feedback_suggestions() == []
    assert log.pending_feedback_suggestion_count() == 0
    log.close()


def test_get_feedback_suggestion_returns_none_for_unflagged(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    good = _good(log)
    assert log.get_feedback_suggestion(good) is None
    bad = _flag(log, prompt="p", response="r")
    assert log.get_feedback_suggestion(bad) is not None
    log.close()


def test_decide_acknowledge_removes_from_pending(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rid = _flag(log, prompt="p", response="r")
    assert log.pending_feedback_suggestion_count() == 1
    assert log.feedback_suggestion_status(rid) is None
    ok = log.decide_feedback_suggestion(rid, "acknowledged", decided_at=5000.0)
    assert ok is True
    assert log.feedback_suggestion_status(rid) == "acknowledged"
    assert log.pending_feedback_suggestion_count() == 0
    assert log.list_pending_feedback_suggestions() == []
    log.close()


def test_decide_dismiss_removes_from_pending(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rid = _flag(log, prompt="p", response="r")
    ok = log.decide_feedback_suggestion(rid, "dismissed", decided_at=5000.0)
    assert ok is True
    assert log.feedback_suggestion_status(rid) == "dismissed"
    assert log.pending_feedback_suggestion_count() == 0
    log.close()


def test_decide_is_upsert_reverses_prior_decision(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rid = _flag(log, prompt="p", response="r")
    log.decide_feedback_suggestion(rid, "acknowledged", decided_at=5000.0)
    # Change of mind: now dismissed.
    log.decide_feedback_suggestion(rid, "dismissed", decided_at=6000.0)
    assert log.feedback_suggestion_status(rid) == "dismissed"
    log.close()


def test_decide_rejects_unflagged_request(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    good = _good(log)
    ok = log.decide_feedback_suggestion(good, "acknowledged", decided_at=5000.0)
    assert ok is False  # no quality_score=-1 to decide on
    log.close()


def test_decide_rejects_unknown_request(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    ok = log.decide_feedback_suggestion(999999, "acknowledged", decided_at=5000.0)
    assert ok is False
    log.close()


def test_decide_rejects_invalid_status(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rid = _flag(log, prompt="p", response="r")
    try:
        log.decide_feedback_suggestion(rid, "maybe", decided_at=5000.0)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid status")
    log.close()


def test_get_feedback_suggestion_works_after_decision(tmp_path: Path) -> None:
    # Unlike list_pending, get returns the snippet even after a decision so
    # `callosum feedback show` can review a just-decided item.
    log = UsageLog(tmp_path / "u.sqlite")
    rid = _flag(log, prompt="p", response="r")
    log.decide_feedback_suggestion(rid, "acknowledged", decided_at=5000.0)
    s = log.get_feedback_suggestion(rid)
    assert s is not None
    assert s.prompt_text == "p"
    log.close()


def test_feedback_suggestions_table_created_on_fresh_db(tmp_path: Path) -> None:
    # The CREATE TABLE IF NOT EXISTS in _SCHEMA must materialize the table
    # on a fresh DB so the methods work without a separate migration.
    log = UsageLog(tmp_path / "u.sqlite")
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_suggestions'").fetchone()
    conn.close()
    assert row is not None
    # And the count method works (returns 0, not a "no such table" error).
    assert log.pending_feedback_suggestion_count() == 0
    log.close()


def test_pending_count_excludes_decided(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    a = _flag(log, prompt="a", response="a", ts_start=1000.0)
    _flag(log, prompt="b", response="b", ts_start=2000.0)
    assert log.pending_feedback_suggestion_count() == 2
    log.decide_feedback_suggestion(a, "dismissed", decided_at=time.time())
    assert log.pending_feedback_suggestion_count() == 1
    log.close()
