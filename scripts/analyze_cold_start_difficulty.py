#!/usr/bin/env python3
"""Tune the cold-start difficulty tiers against the request log.

The cold-start router (`routing/router.py::_task_difficulty`) assigns a
deterministic difficulty tier when the predictor has no differentiated
opinion. That tier drives how much reasoning-effort / capability the
selector demands. This script checks the central tuning assumption —
that prompt size predicts reasoning difficulty — against real traffic,
and reports the numbers needed to re-tune the token thresholds.

It is READ-ONLY. It opens the request log immutably and never writes.

Key cuts:
  * prompt_tokens percentiles + density buckets, so token thresholds can
    be placed in low-density valleys rather than across busy regions.
  * mean reasoning / completion tokens per size tier, controlling for
    the served reasoning-effort, which is the cleanest available proxy
    for "how much thinking the task actually took". If this curve does
    NOT rise with size, raw size is a poor difficulty signal and should
    not escalate the difficulty tier (it is a context-window concern,
    handled separately by _window_fit_factor).

Usage:
    python scripts/analyze_cold_start_difficulty.py [--db PATH]
                                                    [--include-synthetic]
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

DEFAULT_DB = Path.home() / ".local" / "state" / "callosum" / "requests.sqlite"

# Mirror the live thresholds in routing/router.py so the report describes
# the tiers the router actually assigns. Keep in sync when re-tuning.
TIER_EDGES = (2_000, 16_000, 64_000)
PERCENTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def _connect(db: Path) -> sqlite3.Connection:
    if not db.exists():
        raise SystemExit(f"request log not found: {db}")
    # Immutable open: read-only, no locking, no accidental writes.
    return sqlite3.connect(f"file:{db}?immutable=1", uri=True)


def _tier_expr(col: str = "prompt_tokens") -> str:
    lo, mid, hi = TIER_EDGES
    return f"CASE WHEN {col}>={hi} THEN 4 WHEN {col}>={mid} THEN 3 WHEN {col}>={lo} THEN 2 ELSE 1 END"


def _where(include_synthetic: bool) -> str:
    clause = "prompt_tokens IS NOT NULL AND status=200"
    if not include_synthetic:
        clause += " AND routing_mode NOT LIKE '%synthetic%'"
    return clause


def _percentiles(conn: sqlite3.Connection, where: str) -> None:
    total = conn.execute(f"SELECT count(*) FROM requests WHERE {where}").fetchone()[0]
    print(f"\nprompt_tokens distribution (n={total})")
    if not total:
        return
    for p in PERCENTILES:
        offset = int(p * total)
        row = conn.execute(
            f"SELECT prompt_tokens FROM requests WHERE {where} ORDER BY prompt_tokens LIMIT 1 OFFSET {offset}"
        ).fetchone()
        print(f"  p{int(p * 100):02d} = {row[0]:>8}")


def _tier_buckets(conn: sqlite3.Connection, where: str) -> None:
    print("\nsize-tier buckets (all served effort)")
    rows = conn.execute(
        f"SELECT {_tier_expr()} tier, count(*), "
        f"CAST(avg(completion_tokens) AS INT), CAST(avg(reasoning_tokens) AS INT) "
        f"FROM requests WHERE {where} GROUP BY tier ORDER BY tier"
    ).fetchall()
    print(f"  {'tier':<5}{'n':>8}{'avg_compl':>11}{'avg_reason':>12}")
    for tier, n, compl, reason in rows:
        print(f"  {tier:<5}{n:>8}{compl or 0:>11}{reason or 0:>12}")


def _reasoning_curve(conn: sqlite3.Connection, where: str) -> None:
    print("\nreasoning curve, controlling for served high/xhigh effort")
    print("  (if avg_reason does not rise with tier, size is a weak difficulty signal)")
    rows = conn.execute(
        f"SELECT {_tier_expr()} tier, count(*), "
        f"CAST(avg(reasoning_tokens) AS INT), CAST(avg(completion_tokens) AS INT) "
        f"FROM requests WHERE {where} "
        f"AND reasoning_effort IN ('high','xhigh') AND reasoning_tokens IS NOT NULL "
        f"GROUP BY tier ORDER BY tier"
    ).fetchall()
    print(f"  {'tier':<5}{'n':>8}{'avg_reason':>12}{'avg_compl':>11}")
    by_tier: dict[int, int] = {}
    for tier, n, reason, compl in rows:
        print(f"  {tier:<5}{n:>8}{reason or 0:>12}{compl or 0:>11}")
        by_tier[tier] = reason or 0
    if by_tier:
        peak = max(by_tier, key=lambda t: by_tier[t])
        top = max(by_tier)
        print(f"  => reasoning peaks at tier {peak} (top size tier is {top})")
        if peak < top:
            print(
                f"  => beyond tier {peak}, larger prompts do NOT need more reasoning; "
                "raw size should not escalate difficulty past that point"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="request log path")
    ap.add_argument(
        "--include-synthetic",
        action="store_true",
        help="include synthetic exploration traffic (default: real traffic only)",
    )
    args = ap.parse_args()

    conn = _connect(args.db)
    try:
        where = _where(args.include_synthetic)
        print(f"request log: {args.db}")
        print(f"filter: {where}")
        print(f"tier edges (tokens): {TIER_EDGES}")
        _percentiles(conn, where)
        _tier_buckets(conn, where)
        _reasoning_curve(conn, where)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
