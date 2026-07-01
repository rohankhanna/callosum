#!/usr/bin/env python3
"""Machine-checkable P3 promotion gate (CLI wrapper).

This is the command work tracker release conditions invoke via --check (see
work tracker ). The gate logic lives in
callosum.jobs.p3_gate; this script is a thin CLI wrapper.

Exit code: 0 if all P3 release conditions are met, 1 otherwise. JSON on stdout.

Usage:

  # Read-only gate check against the live usage log:
  scripts/check_p3_gate.py --db-path ~/.local/state/callosum/requests.sqlite

  # Full daily cycle: convert new opinions to labels, then check the gate:
  scripts/check_p3_gate.py --db-path ~/.local/state/callosum/requests.sqlite \\
      --apply-labels --checkpoint-path ~/.local/state/callosum/peer_quality_labels.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from callosum.jobs.p3_gate import (
    DEFAULT_MAX_CLASS_FRACTION,
    DEFAULT_MIN_CELLS,
    DEFAULT_MIN_LABELS,
    DEFAULT_MIN_LIFT,
    DEFAULT_MIN_PER_CELL,
    DEFAULT_SAMPLE_LIMIT,
    run,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--checkpoint-path", type=Path, default=None,
                        help="labeler checkpoint; required if --apply-labels")
    parser.add_argument("--apply-labels", action="store_true",
                        help="run the shadow labeler before evaluating the gate")
    parser.add_argument("--min-labels", type=int, default=DEFAULT_MIN_LABELS)
    parser.add_argument("--min-cells", type=int, default=DEFAULT_MIN_CELLS)
    parser.add_argument("--min-per-cell", type=int, default=DEFAULT_MIN_PER_CELL)
    parser.add_argument("--min-lift", type=float, default=DEFAULT_MIN_LIFT)
    parser.add_argument("--max-class-fraction", type=float, default=DEFAULT_MAX_CLASS_FRACTION)
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    parser.add_argument("--label-batch-size", type=int, default=200)
    args = parser.parse_args(argv)

    try:
        result, rc = run(
            db_path=args.db_path,
            checkpoint_path=args.checkpoint_path,
            apply_labels=args.apply_labels,
            min_labels=args.min_labels,
            min_cells=args.min_cells,
            min_per_cell=args.min_per_cell,
            min_lift=args.min_lift,
            max_class_fraction=args.max_class_fraction,
            sample_limit=args.sample_limit,
            label_batch_size=args.label_batch_size,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())