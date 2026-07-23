"""Curated view over local models.

This keeps the "small local-GPU-runnable fleet" as a distinct selection
surface from generic discovery. The catalog is intentionally conservative:
it only admits local models that are runnable on-host, serve responses,
expose a usable throughput signal when present, are at training precision
(no post-hoc down/up casts — a native released quant such as model-a0d2 mxfp4
counts as training precision), and — once a full-context fit probe has been
run — can serve their full advertised context window on this host. Models
that do not fit at full context are streamed from remote sources rather than
kept local.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from callosum.local import ModelEntry
from callosum.usage_log import ModelFitProbe

_TRAINING_PRECISION = frozenset({"bf16", "fp16", "f16", "bfloat16"})

# Models released (post-trained) at a quantization that IS their training
# precision — not a post-hoc downcast. model-a0d2 was post-trained with mxfp4
# quantization of its MoE weights, so mxfp4 is model-a0d2's training precision
# and is admitted as such. Matched by family prefix because families carry a
# size suffix (e.g. "model-a0d2").
_NATIVE_TRAINING_QUANT_PREFIX: dict[str, str] = {"model-a0d2": "mxfp4"}


def _is_training_precision(quant: str | None, family: str) -> bool:
    """True if quant is the model's training precision (or unknown).

    Unknown quantization (None) is not itself a rejection reason — it defers to
    the other admission checks. Native released quants declared above count as
    training precision.
    """
    if quant is None:
        return True
    q = quant.lower()
    if q in _TRAINING_PRECISION:
        return True
    fam = family.lower()
    return any(
        fam.startswith(prefix) and q == native.lower()
        for prefix, native in _NATIVE_TRAINING_QUANT_PREFIX.items()
    )


@dataclass(frozen=True, slots=True)
class CuratedLocalModel:
    model: ModelEntry
    admitted: bool
    reasons: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.model.id


def curate_local_models(
    models: list[ModelEntry],
    *,
    min_tokens_per_second: float | None = None,
    probe_results: Mapping[str, ModelFitProbe] | None = None,
) -> list[CuratedLocalModel]:
    """Return a conservative catalog of small local models.

    The caller can set min_tokens_per_second to require a throughput
    floor; when unset, throughput is only used if the source provides it.
    probe_results carries the latest full-context fit probe per model id;
    a model whose probe says it cannot serve its full advertised context
    window on this host is rejected with overflows-pool-at-full-context and
    left for remote streaming. A model with no probe result is admitted
    optimistically until a probe re-measures it.
    """
    out: list[CuratedLocalModel] = []
    for model in models:
        reasons: list[str] = []

        if not model.enabled:
            reasons.append("disabled")
        if "responses" not in model.api_surfaces:
            reasons.append("no-responses-surface")
        if model.local_runnable_on_host is False:
            reasons.append("not-runnable-on-host")
        if model.local_status and model.local_status not in {"working", "unmeasured", "unknown"}:
            reasons.append(f"status:{model.local_status}")
        if model.local_quantization is not None and not _is_training_precision(
            model.local_quantization, model.family
        ):
            reasons.append(f"non-training-precision:{model.local_quantization}")
        if probe_results is not None:
            probe = probe_results.get(model.id)
            if probe is not None and not probe.fits_full_context:
                reasons.append(
                    f"overflows-pool-at-full-context:{probe.achievable_context_tokens}"
                    f"/{probe.advertised_context_tokens}"
                )
        if (
            min_tokens_per_second is not None
            and model.estimated_tokens_per_second is not None
            and model.estimated_tokens_per_second < min_tokens_per_second
        ):
            reasons.append(f"throughput-below-floor:{model.estimated_tokens_per_second:.3f}")

        admitted = not reasons
        out.append(CuratedLocalModel(model=model, admitted=admitted, reasons=tuple(reasons)))
    return out


def curated_local_model_ids(
    models: list[ModelEntry],
    *,
    min_tokens_per_second: float | None = None,
    probe_results: Mapping[str, ModelFitProbe] | None = None,
) -> list[str]:
    """Convenience wrapper returning the admitted model ids only."""
    return [
        item.id
        for item in curate_local_models(
            models,
            min_tokens_per_second=min_tokens_per_second,
            probe_results=probe_results,
        )
        if item.admitted
    ]