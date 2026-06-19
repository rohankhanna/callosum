"""Extension point: cost-label data structures and quantization helpers.

The trained cost-label extraction (quantized meter-tick windows from the
request log) is private.  This stub preserves the data structures
(``QuotaMeterSpec``, ``FIVE_HOURLY_METER``, ``WEEKLY_METER``) used by public
modules and provides no-op implementations of the analysis functions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuotaMeterSpec:
    name: str
    unit: str
    before_column: str
    after_column: str
    reset_at_column: str


FIVE_HOURLY_METER = QuotaMeterSpec(
    name="five_hourly",
    unit="five_hourly_used_percent",
    before_column="five_hourly_used_percent_before",
    after_column="five_hourly_used_percent_after",
    reset_at_column="five_hourly_reset_at",
)
WEEKLY_METER = QuotaMeterSpec(
    name="weekly",
    unit="weekly_used_percent",
    before_column="weekly_used_percent_before",
    after_column="weekly_used_percent_after",
    reset_at_column="weekly_reset_at",
)


@dataclass(frozen=True, slots=True)
class CostLabelQualitySummary:
    total: int = 0
    serialized: int = 0

    @property
    def excluded(self) -> int:
        return self.total - self.serialized

    def to_metadata(self) -> dict[str, object]:
        return {}

    def to_meter_metadata(self, meter: QuotaMeterSpec) -> dict[str, object]:
        metadata = self.to_metadata()
        metadata["meter"] = meter.name
        metadata["unit"] = meter.unit
        return metadata


@dataclass(frozen=True, slots=True)
class QuantizedCostWindow:
    model: str = ""
    reasoning_effort: str = ""
    uncached_input: float = 0.0
    cached_input: float = 0.0
    output: float = 0.0
    reasoning: float = 0.0
    total_tokens: float = 0.0
    delta: float = 0.0
    row_count: int = 0
    tick_row_id: int = 0
    ts_start: float = 0.0
    ts_end: float = 0.0


def cost_label_quality_summary(*args: object, **kwargs: object) -> CostLabelQualitySummary:
    """Stub: returns an empty summary."""
    return CostLabelQualitySummary()


def cutoff_for_window(window_seconds: int, *, now: float | None = None) -> float:
    """Stub: returns a cutoff timestamp for the given window."""
    return (now if now is not None else time.time()) - window_seconds


def quantized_cost_windows(*args: object, **kwargs: object) -> list[QuantizedCostWindow]:
    """Stub: returns no cost windows."""
    return []
