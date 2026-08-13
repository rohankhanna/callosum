#!/usr/bin/env python3
"""Operator entry point for the tiered merge/promotion gate.

Thin wrapper around callosum gate for cron / CI use. Exits 0 when Tier 1
is green (merge not blocked), 1 when merge-blocked. Run from the repo root:

    python scripts/run_gate.py                # all three tiers, human-readable
    python scripts/run_gate.py --tier 1       # Tier 1 only (the merge gate)
    python scripts/run_gate.py --json         # machine-readable report
"""

from __future__ import annotations

import sys

from callosum.cli import main

if __name__ == "__main__":
    # Prepend "gate" so callers invoke this script as `python scripts/run_gate.py
    # [--tier N ...]` without typing the subcommand themselves.
    sys.argv = [sys.argv[0], "gate", *sys.argv[1:]]
    sys.exit(main())