"""Extension point: labeled-row loader for predictor reload.

The default loader returns no rows, so the default predictor stays at its
cold-start uniform prior.  Replace this loader to read quality-scored rows
from the request-log SQLite database or another data source.

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
    """Yield no rows. Replace this loader to provide labeled data."""
    return
    yield  # pragma: no cover  — make it a generator
