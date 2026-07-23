"""Tests for the curated local-model catalog."""

from __future__ import annotations

from callosum.local import ModelEntry
from callosum.local_model_catalog import curate_local_models, curated_local_model_ids
from callosum.usage_log import ModelFitProbe


def _model(
    model_id: str,
    *,
    quantization: str | None = "bf16",
    runnable_on_host: bool | None = True,
    status: str | None = "working",
    throughput: float | None = 12.0,
    surfaces: tuple[str, ...] = ("responses",),
    family: str = "test",
) -> ModelEntry:
    return ModelEntry(
        id=model_id,
        endpoint="http://127.0.0.1:11434",
        runtime="ollama",
        runtime_model=model_id,
        family=family,
        context_window=128_000,
        api_surfaces=surfaces,
        enabled=True,
        local_quantization=quantization,
        local_runnable_on_host=runnable_on_host,
        local_status=status,
        estimated_tokens_per_second=throughput,
    )


def _probe(
    model_id: str,
    *,
    fits: bool = True,
    achievable: int = 87_376,
    advertised: int = 131_072,
) -> ModelFitProbe:
    return ModelFitProbe(
        model_id=model_id,
        content_hash="abc",
        advertised_context_tokens=advertised,
        achievable_context_tokens=achievable,
        max_concurrency=1.3 if fits else 0.3,
        fits_full_context=fits,
        probed_at=0.0,
    )


def test_curated_catalog_admits_training_precision_response_model() -> None:
    curated = curate_local_models([_model("small-good")], min_tokens_per_second=10.0)
    assert curated[0].admitted is True
    assert curated[0].reasons == ()


def test_curated_catalog_excludes_post_training_quantized_models() -> None:
    curated = curate_local_models([_model("small-q4", quantization="q4_k_m")])
    assert curated[0].admitted is False
    assert "non-training-precision:q4_k_m" in curated[0].reasons


def test_curated_catalog_excludes_chat_only_and_not_runnable_models() -> None:
    curated = curate_local_models(
        [
            _model("chat-only", surfaces=("chat",)),
            _model("not-runnable", runnable_on_host=False),
        ]
    )
    assert curated[0].admitted is False
    assert "no-responses-surface" in curated[0].reasons
    assert curated[1].admitted is False
    assert "not-runnable-on-host" in curated[1].reasons


def test_curated_catalog_respects_throughput_floor() -> None:
    curated = curate_local_models(
        [
            _model("slow", throughput=4.0),
            _model("fast", throughput=18.0),
        ],
        min_tokens_per_second=10.0,
    )
    assert curated[0].admitted is False
    assert "throughput-below-floor:4.000" in curated[0].reasons
    assert curated[1].admitted is True


def test_curated_local_model_ids_returns_admitted_subset() -> None:
    ids = curated_local_model_ids(
        [
            _model("good"),
            _model("bad", quantization="q4_k_m"),
        ]
    )
    assert ids == ["good"]


def test_native_quant_gpt_oss_mxfp4_is_training_precision() -> None:
    """model-a0d2 was post-trained at mxfp4, so mxfp4 is its training precision."""
    curated = curate_local_models([_model("model-a0d2", quantization="mxfp4", family="model-a0d2")])
    assert curated[0].admitted is True
    assert curated[0].reasons == ()


def test_native_quant_mxfp4_rejected_for_non_gpt_oss_family() -> None:
    curated = curate_local_models([_model("other", quantization="mxfp4", family="something-else")])
    assert curated[0].admitted is False
    assert "non-training-precision:mxfp4" in curated[0].reasons


def test_overflows_pool_at_full_context_rejects() -> None:
    curated = curate_local_models(
        [_model("big")],
        probe_results={"big": _probe("big", fits=False, achievable=2048, advertised=131_072)},
    )
    assert curated[0].admitted is False
    assert "overflows-pool-at-full-context:2048/131072" in curated[0].reasons


def test_fits_full_context_admits() -> None:
    curated = curate_local_models(
        [_model("ok")], probe_results={"ok": _probe("ok", fits=True)}
    )
    assert curated[0].admitted is True


def test_unprobed_model_admitted_optimistically() -> None:
    curated = curate_local_models([_model("unprobed")], probe_results={})
    assert curated[0].admitted is True
    assert curated[0].reasons == ()
