from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from codex_proxy.codex_quota import CodexQuotaSnapshot
from codex_proxy.usage_log import UsageLog, UsageLogEntry, decompress


def _snap(*, primary: int, secondary: int) -> CodexQuotaSnapshot:
    return CodexQuotaSnapshot(
        plan_type="plus",
        active_limit="premium",
        primary_used_percent=primary,
        secondary_used_percent=secondary,
        primary_window_minutes=300,
        secondary_window_minutes=10080,
        primary_reset_at=1777000000,
        secondary_reset_at=1777500000,
        primary_reset_after_seconds=6000,
        secondary_reset_after_seconds=100000,
        primary_over_secondary_limit_percent=0,
        credits_balance=None,
        credits_has_credits=False,
        credits_unlimited=False,
        observed_at=time.time(),
    )


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


def test_record_inserts_row_with_computed_latency(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(_entry())
    assert rowid >= 1
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    (latency, route, stream, session_id) = conn.execute(
        "SELECT latency_ms, route, stream, session_id FROM requests WHERE id = ?",
        (rowid,),
    ).fetchone()
    assert latency == 250
    assert route == "responses"
    assert stream == 1
    assert session_id == "sess-1"
    log.close()


def test_record_persists_quota_before_and_after(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(
        _entry(
            quota_before=_snap(primary=1, secondary=53),
            quota_after=_snap(primary=2, secondary=54),
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        """
        SELECT primary_used_percent_before, primary_used_percent_after,
               secondary_used_percent_before, secondary_used_percent_after,
               plan_type, quota_reset_crossover
        FROM requests WHERE id = ?
        """,
        (rowid,),
    ).fetchone()
    assert row == (1, 2, 53, 54, "plus", 0)
    log.close()


def test_record_flags_window_reset_crossover(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(
        _entry(
            quota_before=_snap(primary=98, secondary=53),
            # primary window reset mid-request: after < before.
            quota_after=_snap(primary=1, secondary=54),
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    (crossover,) = conn.execute(
        "SELECT quota_reset_crossover FROM requests WHERE id = ?", (rowid,)
    ).fetchone()
    assert crossover == 1
    log.close()


def test_record_writes_compressed_bodies_round_trip(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite", capture_bodies=True)
    req = b'{"model":"model-a0e7","input":[]}'
    resp = b'{"id":"resp-1","object":"response"}'
    headers = {"x-codex-plan-type": "plus", "x-codex-primary-used-percent": "2"}
    rowid = log.record(_entry(req_payload=req, resp_payload=resp, upstream_headers=headers))
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT req_payload, resp_payload, upstream_headers"
        " FROM request_bodies WHERE request_id = ?",
        (rowid,),
    ).fetchone()
    assert row is not None
    assert decompress(row[0]) == req
    assert decompress(row[1]) == resp
    roundtripped_headers = decompress(row[2])
    assert roundtripped_headers is not None
    assert json.loads(roundtripped_headers) == headers
    log.close()


def test_capture_bodies_off_drops_blobs(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite", capture_bodies=False)
    rowid = log.record(
        _entry(
            req_payload=b"...",
            resp_payload=b"...",
            upstream_headers={"x-codex-plan-type": "plus"},
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT COUNT(*) FROM request_bodies WHERE request_id = ?", (rowid,)
    ).fetchone()
    assert row[0] == 0
    log.close()


def test_null_quota_yields_null_columns_but_not_crossover(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(_entry(quota_before=None, quota_after=None))
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        """
        SELECT primary_used_percent_before, primary_used_percent_after,
               plan_type, quota_reset_crossover
        FROM requests WHERE id = ?
        """,
        (rowid,),
    ).fetchone()
    assert row == (None, None, None, 0)
    log.close()
