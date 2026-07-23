from __future__ import annotations

from callosum.routing.local_performance import (
    LocalPerformanceRegime,
    build_local_performance_model,
)


def test_underutilized_fit_and_latency() -> None:
    model = build_local_performance_model(
        model_id="local-a",
        quantization="bf16",
        pool_bytes=121 * 1024**3,
        free_bytes=94 * 1024**3,
        weight_bytes=40 * 1024**3,
        kv_bytes_per_token=2 * 1024**2,
        activation_bytes=2 * 1024**3,
        estimated_tokens_per_second=20.0,
        prefill_ms_per_token=1.5,
        decode_bandwidth_kappa=1.0,
    )
    assert model.regime_for(1_000) == LocalPerformanceRegime.UNDERUTILIZED
    assert model.ttft_ms(1_000) == 1_500.0
    assert model.decode_rate_tps(1_000) == 20.0
    assert model.turn_latency_ms(input_tokens=1_000, output_tokens=500) > model.ttft_ms(1_000)


def test_pool_edge_soft_cliff_reduces_decode_rate() -> None:
    model = build_local_performance_model(
        model_id="local-b",
        quantization="bf16",
        pool_bytes=121 * 1024**3,
        free_bytes=94 * 1024**3,
        weight_bytes=20 * 1024**3,
        kv_bytes_per_token=1024**3,
        activation_bytes=8 * 1024**3,
        estimated_tokens_per_second=30.0,
        prefill_ms_per_token=2.0,
        decode_bandwidth_kappa=2.0,
        edge_slack_tokens=100,
    )
    fit_limit = model.fit_limit_tokens()
    assert fit_limit is not None
    assert model.regime_for(fit_limit - 50) == LocalPerformanceRegime.POOL_EDGE
    assert model.decode_rate_tps(fit_limit - 50) < 30.0


def test_overhang_adds_swap_penalty_and_threads_grow() -> None:
    model = build_local_performance_model(
        model_id="local-c",
        quantization="bf16",
        pool_bytes=121 * 1024**3,
        free_bytes=10 * 1024**3,
        weight_bytes=9 * 1024**3,
        kv_bytes_per_token=512 * 1024**2,
        activation_bytes=2 * 1024**3,
        estimated_tokens_per_second=12.0,
        prefill_ms_per_token=1.0,
        swap_penalty_ms=50_000.0,
    )
    assert model.regime_for(10_000) == LocalPerformanceRegime.OVERHANG
    assert model.ttft_ms(10_000) >= 50_000.0
    assert model.thread_latency_ms(initial_input_tokens=10_000, turns=3, output_tokens=100, growth_rate=0.25) > 0.0
