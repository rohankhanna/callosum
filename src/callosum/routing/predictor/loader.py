"""Extension point: labeled-row loader for predictor reload.

The trained loader (reads quality-scored rows from the request-log SQLite) is
private.  This stub returns an empty stream so the default predictor stays at
its cold-start uniform prior.

To implement a custom loader:

    from callosum.routing.protocols import LabeledRow

    def labeled_rows_from_request_log(db_path, *, limit=None):
        # Yield LabeledRow records from your data source.
        ...
        yield LabeledRow(request_id=1, cell_used="model-a1b2 high", outcome=1.0)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from callosum.routing.protocols import LabeledRow


def labeled_rows_from_request_log(
    db_path: Path,
    *,
    limit: int | None = None,
) -> Iterable[LabeledRow]:
    """Stub: yields no rows.  Implement to load labeled data."""
    return
    yield  # pragma: no cover  — make it a generator
