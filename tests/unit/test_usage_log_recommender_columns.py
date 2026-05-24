"""Tests for the recommender_* columns on the requests table.

These columns are the data plumbing for (B) — training a local classifier
that takes over routing decisions later. The columns must:

- Land in the schema via _MIGRATIONS on existing databases.
- Be populated from UsageLogEntry fields when the recommender fires.
- Stay NULL on pass-through rows so a training query filtering on
  `recommender_source = 'upstream'` doesn't pick up rows the recommender
  never saw.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from callosum.usage_log import UsageLog, UsageLogEntry


def _entry(**overrides: object) -> UsageLogEntry:
    base = dict(
        ts_start=1.0,
        ts_end=2.0,
        route="/v1/responses",
        stream=False,
        session_id=None,
        backend_id="primary",
        model="model-a0e7",
        reasoning_effort="high",
        status=200,
        classification="ok",
    )
    base.update(overrides)
    return UsageLogEntry(**base)  # type: ignore[arg-type]


def test_pass_through_row_leaves_recommender_columns_null(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    log.record(_entry())
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT recommender_classifier_cell, recommender_raw_output, recommender_source"
        " FROM requests"
    ).fetchone()
    assert row == (None, None, None)


def test_recommender_columns_persist_when_set(tmp_path: Path) -> None:
    log = UsageLog(tmp_path / "u.sqlite")
    log.record(
        _entry(
            routing_mode="auto",
            recommender_classifier_cell="model-a0c3 low",
            recommender_raw_output="model-a0e7 high",
            recommender_source="upstream",
        )
    )
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    row = conn.execute(
        "SELECT routing_mode, recommender_classifier_cell,"
        " recommender_raw_output, recommender_source FROM requests"
    ).fetchone()
    assert row == ("auto", "model-a0c3 low", "model-a0e7 high", "upstream")


def test_index_on_recommender_source_exists(tmp_path: Path) -> None:
    """Distillation training queries filter on recommender_source; the
    index keeps that scan cheap as the requests table grows."""
    log = UsageLog(tmp_path / "u.sqlite")
    log.record(_entry())  # trigger migrations
    conn = sqlite3.connect(tmp_path / "u.sqlite")
    indices = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='requests'"
    ).fetchall()
    names = {n for (n,) in indices}
    assert "idx_requests_recommender_source" in names
