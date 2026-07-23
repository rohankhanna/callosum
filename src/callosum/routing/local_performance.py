"""Parametric local-model performance model.

This module keeps the local-LLM latency story separate from the generic
request-time estimator. It models the local runtime in three regimes:

* underutilized: the request fits cleanly in the unified pool
* pool-edge: KV cache pushes against the pool headroom
* overhang: the request exceeds the pool and spills to swap

The formulas are intentionally simple and conservative. They are meant to be
fed by metadata from local LLM gateway plus later calibration from observed runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class LocalPerformanceRegime(StrEnum):
    UNDERUTILIZED = "underutilized"
    POOL_EDGE = "pool-edge"
    OVERHANG = "overhang"


@dataclass(frozen=True, slots=True)
class LocalPerformanceModel:
    """Per-model parametric performance surface.

    Units:
    - bytes for memory quantities
    - tokens for lengths
    - milliseconds for latency
    - tokens/s for decode rate
    """

    model_id: str
    quantization: str | None
    pool_bytes: int
    free_bytes: int
    weight_bytes: int | None
    kv_bytes_per_token: int | None
    activation_bytes: int | None
    fit_limit_override_tokens: int | None
    prefill_ms_per_token: float
    decode_tokens_per_second: float
    decode_bandwidth_kappa: float
    swap_penalty_ms: float
    edge_slack_tokens: int
    expected_output_tokens: int = 512

    def fit_limit_tokens(self) -> int | None:
        if self.fit_limit_override_tokens is not None:
            return max(0, self.fit_limit_override_tokens)
        if self.weight_bytes is None or self.kv_bytes_per_token is None or self.activation_bytes is None:
            return None
        remaining = self.free_bytes - self.weight_bytes - self.activation_bytes
        if remaining <= 0:
            return 0
        return max(0, remaining // self.kv_bytes_per_token)

    def regime_for(self, input_tokens: int) -> LocalPerformanceRegime:
        fit_limit = self.fit_limit_tokens()
        if fit_limit is None:
            return LocalPerformanceRegime.UNDERUTILIZED
        if input_tokens <= fit_limit:
            if fit_limit - input_tokens <= self.edge_slack_tokens:
                return LocalPerformanceRegime.POOL_EDGE
            return LocalPerformanceRegime.UNDERUTILIZED
        return LocalPerformanceRegime.OVERHANG

    def ttft_ms(self, input_tokens: int) -> float:
        base = self.prefill_ms_per_token * input_tokens
        regime = self.regime_for(input_tokens)
        if regime == LocalPerformanceRegime.OVERHANG:
            return base + self.swap_penalty_ms
        if regime == LocalPerformanceRegime.POOL_EDGE:
            # Soft cliff: a modest re-prefill tax when the pool edge is close.
            return base * 1.35
        return base

    def decode_rate_tps(self, input_tokens: int) -> float:
        rate = self.decode_tokens_per_second
        if input_tokens <= 0:
            return max(0.0, rate)
        if self.regime_for(input_tokens) == LocalPerformanceRegime.POOL_EDGE:
            return max(0.0, rate / max(1.0, 1.0 + self.decode_bandwidth_kappa))
        if self.regime_for(input_tokens) == LocalPerformanceRegime.OVERHANG:
            return max(0.0, rate / max(1.0, self.decode_bandwidth_kappa * 4.0))
        return max(0.0, rate)

    def turn_latency_ms(self, *, input_tokens: int, output_tokens: int, tool_overhead_ms: float = 0.0) -> float:
        decode_rate = self.decode_rate_tps(input_tokens)
        decode_ms = (output_tokens / decode_rate * 1000.0) if decode_rate > 0 else self.swap_penalty_ms
        return self.ttft_ms(input_tokens) + decode_ms + tool_overhead_ms

    def thread_latency_ms(
        self,
        *,
        initial_input_tokens: int,
        turns: int,
        output_tokens: int,
        growth_rate: float = 0.0,
        tool_overhead_ms: float = 0.0,
    ) -> float:
        total = 0.0
        current = float(initial_input_tokens)
        for _ in range(max(0, turns)):
            total += self.turn_latency_ms(
                input_tokens=int(current),
                output_tokens=output_tokens,
                tool_overhead_ms=tool_overhead_ms,
            )
            current *= 1.0 + max(0.0, growth_rate)
        return total


def build_local_performance_model(
    *,
    model_id: str,
    quantization: str | None,
    pool_bytes: int,
    free_bytes: int,
    weight_bytes: int | None,
    kv_bytes_per_token: int | None,
    activation_bytes: int | None,
    estimated_tokens_per_second: float | None,
    fit_limit_tokens: int | None = None,
    prefill_ms_per_token: float = 2.0,
    decode_bandwidth_kappa: float = 1.0,
    swap_penalty_ms: float = 60_000.0,
    edge_slack_tokens: int = 8_192,
) -> LocalPerformanceModel:
    decode_tps = (
        estimated_tokens_per_second
        if estimated_tokens_per_second and estimated_tokens_per_second > 0
        else 10.0
    )
    effective_edge_slack_tokens = edge_slack_tokens
    if fit_limit_tokens is not None and fit_limit_tokens > 0:
        effective_edge_slack_tokens = min(edge_slack_tokens, max(1, fit_limit_tokens // 8))
    return LocalPerformanceModel(
        model_id=model_id,
        quantization=quantization,
        pool_bytes=pool_bytes,
        free_bytes=free_bytes,
        weight_bytes=weight_bytes,
        kv_bytes_per_token=kv_bytes_per_token,
        activation_bytes=activation_bytes,
        fit_limit_override_tokens=fit_limit_tokens,
        prefill_ms_per_token=prefill_ms_per_token,
        decode_tokens_per_second=decode_tps,
        decode_bandwidth_kappa=decode_bandwidth_kappa,
        swap_penalty_ms=swap_penalty_ms,
        edge_slack_tokens=effective_edge_slack_tokens,
    )
