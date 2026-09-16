"""Extension point: forward cost estimator and composite cost model.

The default implementation preserves the interfaces used by the router and
selector while returning neutral zero-cost estimates.  Replace it to predict
per-request meter burn from measured data.

To implement a custom cost estimator, subclass ``CostUsageEstimator`` and
override ``estimate`` / ``finalize``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from callosum.cell_grid import Cell
from callosum.routing.cost_labels import FIVE_HOURLY_METER, WEEKLY_METER
from callosum.routing.usage_estimate import (
    Estimate,
    EstimateInput,
    FinalizedObservation,
)


class CompositeCostEstimate:
    """Pair of per-meter cost estimates."""

    five_hourly: Estimate
    weekly: Estimate

    def __init__(self) -> None:
        zero = Estimate(
            point=0.0,
            low=0.0,
            high=0.0,
            unit="weekly_used_percent",
            source="stub",
            verifiable=False,
        )
        self.five_hourly = zero
        self.weekly = zero

    def for_meter(self, meter: str) -> Estimate:
        if meter == FIVE_HOURLY_METER.name:
            return self.five_hourly
        if meter == WEEKLY_METER.name:
            return self.weekly
        raise KeyError(meter)


class CostModelProvider:
    """Single-meter cost model provider."""

    def __init__(self, usage_log_path: Path, *, meter: object | None = None, **kwargs: object) -> None:
        del usage_log_path, meter, kwargs

    def rate_for(self, cell: Cell) -> float:
        return 0.0


class CompositeCostModelProvider:
    """Pair of meter-specific providers."""

    def __init__(self, usage_log_path: Path, **kwargs: object) -> None:
        self.five_hourly = CostModelProvider(usage_log_path, meter=FIVE_HOURLY_METER, **kwargs)
        self.weekly = CostModelProvider(usage_log_path, meter=WEEKLY_METER, **kwargs)


class CostUsageEstimator:
    """Returns neutral zero-cost estimates for all cells."""

    def __init__(self, provider: CostModelProvider, *, is_remote: Callable[[Cell], bool] | None = None) -> None:
        del provider, is_remote

    @property
    def id(self) -> str:
        return "cost-v1"

    def estimate(self, inp: EstimateInput) -> Estimate:
        return Estimate(
            point=0.0,
            low=0.0,
            high=0.0,
            unit=WEEKLY_METER.unit,
            source="stub",
            verifiable=False,
        )

    def finalize(
        self,
        request_id: int,
        cell: Cell,
        observed_output_tokens: int | None,
        observed_value: float | None,
    ) -> FinalizedObservation:
        return FinalizedObservation(
            request_id=request_id,
            cell=cell,
            observed_output_tokens=observed_output_tokens,
            observed_value=observed_value,
            unit=WEEKLY_METER.unit,
            verifiable=False,
            source="stub",
        )


class CompositeCostUsageEstimator:
    """Composite estimator over both meters."""

    def __init__(self, provider: CompositeCostModelProvider) -> None:
        self.five_hourly = CostUsageEstimator(provider.five_hourly)
        self.weekly = CostUsageEstimator(provider.weekly)

    def estimate(self, inp: EstimateInput) -> CompositeCostEstimate:
        return CompositeCostEstimate()
