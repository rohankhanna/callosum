#!/usr/bin/env python3
"""Thin CLI wrapper for the model fit probe job.

Forwards to callosum.jobs.model_probe (see callosum.jobs.model_probe).
Run as python scripts/probe_model_fit.py --db-path <usage.sqlite> [--max-models N].
"""

from __future__ import annotations

import sys

from callosum.jobs.model_probe import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))