from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from callosum.codex_quota import CodexQuotaSnapshot
from callosum.usage_log import UsageLog, UsageLogEntry, decompress


def _snap(*, five_hourly: int, weekly: int) -> CodexQuotaSnapshot:
    return CodexQuotaSnapshot(
        plan_type="plus",
        active_limit="premium",
        five_hourly_used_percent=five_hourly,
        weekly_used_percent=weekly,
        five_hourly_window_minutes=300,
        weekly_window_minutes=10080,
        five_hourly_reset_at=1777000000,
        weekly_reset_at=1777500000,
        five_hourly_reset_after_seconds=6000,
        weekly_reset_after_seconds=100000,
        five_hourly_over_weekly_limit_percent=0,
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
            quota_before=_snap(five_hourly=1, weekly=53),
            quota_after=_snap(five_hourly=2, weekly=54),
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        """
        SELECT five_hourly_used_percent_before, five_hourly_used_percent_after,
               weekly_used_percent_before, weekly_used_percent_after,
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
            quota_before=_snap(five_hourly=98, weekly=53),
            # 5-hour window reset mid-request: after < before.
            quota_after=_snap(five_hourly=1, weekly=54),
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    (crossover,) = conn.execute("SELECT quota_reset_crossover FROM requests WHERE id = ?", (rowid,)).fetchone()
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
        "SELECT req_payload, resp_payload, upstream_headers FROM request_bodies WHERE request_id = ?",
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
    row = conn.execute("SELECT COUNT(*) FROM request_bodies WHERE request_id = ?", (rowid,)).fetchone()
    assert row[0] == 0
    log.close()


def test_null_quota_yields_null_columns_but_not_crossover(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(_entry(quota_before=None, quota_after=None))
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        """
        SELECT five_hourly_used_percent_before, five_hourly_used_percent_after,
               plan_type, quota_reset_crossover
        FROM requests WHERE id = ?
        """,
        (rowid,),
    ).fetchone()
    assert row == (None, None, None, 0)
    log.close()


def test_router_fields_persist(tmp_path: Path) -> None:
    """Variation/routing fields round-trip cleanly. requested_* captures the
    client's intent before the explorer router rewrote the body; model and
    reasoning_effort columns continue to mean what was actually served.
    """
    log = UsageLog(tmp_path / "u.sqlite")
    # Auto-learning request: client asked for 'auto-learning', proxy served
    # model-a0c3 at low reasoning to fill that cell.
    rowid = log.record(
        _entry(
            model="model-a0c3",
            reasoning_effort="low",
            requested_model="auto-learning",
            requested_reasoning_effort=None,
            routing_mode="auto-learning",
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        """
        SELECT model, reasoning_effort, requested_model,
               requested_reasoning_effort, routing_mode
        FROM requests WHERE id = ?
        """,
        (rowid,),
    ).fetchone()
    assert row == ("model-a0c3", "low", "auto-learning", None, "auto-learning")
    log.close()


def test_traffic_kind_persists(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(_entry(traffic_kind="quota_explore"))
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    (traffic_kind,) = conn.execute(
        "SELECT traffic_kind FROM requests WHERE id = ?",
        (rowid,),
    ).fetchone()
    assert traffic_kind == "quota_explore"
    log.close()


def test_traffic_kind_index_exists(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    log.record(_entry())
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    indices = conn.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='requests'").fetchall()
    names = {n for (n,) in indices}
    assert "idx_requests_traffic_kind" in names
    log.close()


def test_pass_through_request_records_requested_equals_served(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    rowid = log.record(
        _entry(
            model="model-a0e7",
            reasoning_effort="xhigh",
            requested_model="model-a0e7",
            requested_reasoning_effort="xhigh",
            routing_mode="pass-through",
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT model, requested_model, routing_mode FROM requests WHERE id = ?",
        (rowid,),
    ).fetchone()
    assert row == ("model-a0e7", "model-a0e7", "pass-through")
    log.close()


def test_migration_backfills_existing_rows(tmp_path: Path) -> None:
    """Open a v1-style DB (no router columns), insert a row directly via
    raw SQL, then re-open via UsageLog so the migration runs. Backfill should
    populate requested_* from served_* and routing_mode='pass-through'.
    """
    db_path = tmp_path / "u.sqlite"
    # Create a minimal v1-shape table by hand, simulating a pre-router DB.
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_start REAL NOT NULL, ts_end REAL NOT NULL, latency_ms INTEGER NOT NULL,
            route TEXT NOT NULL, stream INTEGER NOT NULL, session_id TEXT,
            backend_id TEXT NOT NULL, model TEXT, reasoning_effort TEXT,
            status INTEGER NOT NULL, classification TEXT,
            request_bytes INTEGER, response_bytes INTEGER,
            prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER,
            cached_tokens INTEGER, reasoning_tokens INTEGER,
            plan_type TEXT, active_limit TEXT,
            primary_used_percent_before INTEGER, primary_used_percent_after INTEGER,
            secondary_used_percent_before INTEGER, secondary_used_percent_after INTEGER,
            primary_reset_at INTEGER, secondary_reset_at INTEGER,
            primary_over_secondary_limit_percent INTEGER,
            credits_balance TEXT, credits_has_credits INTEGER, credits_unlimited INTEGER,
            quota_reset_crossover INTEGER NOT NULL DEFAULT 0
        )
        """
    )
    conn.execute(
        "INSERT INTO requests (ts_start, ts_end, latency_ms, route, stream,"
        " backend_id, model, reasoning_effort, status, classification)"
        " VALUES (1000.0, 1000.5, 500, 'responses', 1, 'primary', 'model-a0e7', 'xhigh', 200, 'ok')"
    )
    conn.commit()
    conn.close()
    # Now reopen via UsageLog — migration runs.
    log = UsageLog(db_path)
    log.close()
    conn = sqlite3.connect(db_path)
    row = conn.execute("SELECT requested_model, requested_reasoning_effort, routing_mode FROM requests").fetchone()
    assert row == ("model-a0e7", "xhigh", "pass-through")
