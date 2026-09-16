"""Extension point: per-model cost-rank provider.

The default provider falls back to priority ordering from the backend catalog.
Replace it to derive ``cost_rank`` from measured quota burn or another signal.

To implement a custom cost model:

    from pathlib import Path
    from collections.abc import Callable

    class MyCostRankProvider:
        def __init__(self, usage_log_path, *, catalog_priorities, **kwargs):
            ...
        def rank_for(self, model, default):
            # Return a cost rank (0 = cheapest).
            return default
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


class CostRankProvider:
    """Default provider that returns the caller-supplied rank.

    Override ``rank_for`` to implement a custom measured-cost model.
    """

    def __init__(
        self,
        usage_log_path: Path,
        *,
        catalog_priorities: Callable[[], dict[str, int]],
        overrides: dict[str, int] | None = None,
        enabled: bool = True,
        min_nonzero_samples: int = 10,
        window_seconds: int = 30 * 24 * 3600,
        base_rank: int = 10,
        refresh_seconds: int = 3600,
        clock: Callable[[], float] | None = None,
    ) -> None:
        del usage_log_path, catalog_priorities, overrides, enabled
        del min_nonzero_samples, window_seconds, base_rank, refresh_seconds, clock

    def rank_for(self, model: str, default: int) -> int:
        return default


def measured_cost_ranks(*args: object, **kwargs: object) -> dict[str, int]:
    """Return no measured ranks by default."""
    return {}
