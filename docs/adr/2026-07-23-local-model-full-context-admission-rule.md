# ADR: Local-model full-context memory-fit admission rule

Date: 2026-07-23

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-07-23 (introducing commit `e6d6f86`, merged to `main` as `c276005`).
> It records a past decision; the implementation landed prior to this
> document. The date above is the decision date.

## Context

Callosum routes across a small curated local-GPU fleet and remote LLM
backends. The local fleet is a distinct selection surface from generic
hub discovery: the router must never be handed a model it cannot actually
serve on this host. On the GB10 unified-memory machine (~121 GiB shared
CPU/GPU pool), a model whose weights plus KV cache do not fit at its full
advertised context window is not "local-usable" — it forces a context cap
or an OOM/swap mid-stream. If such a model were admitted and the router
then routed it a request whose context window the model cannot fulfill,
the request would stall or die at request time, after the router had
already committed the turn to it.

The failure mode is specifically a **request-time** surprise: the router
picks the cell, hands it a prompt sized within the model's advertised
window, and only at serve time does the host discover it cannot back that
window. The cost is a mid-stream stall/OOM on a turn the router believed
was sound. Recovering at request time means either a context cap (silent
degradation — the model serves less than it advertises) or a fallback
re-dispatch (a wasted turn). The cleaner shape is to gate this **at
curation time**, before the router ever sees the model: a model that
cannot serve its full advertised context on this host is dropped from the
admitted local fleet and streamed from a remote source instead.

This makes the judgment **mechanical and self-enforcing** rather than a
recurring manual cull, so future model pulls are auto-evaluated. It
realizes the training-precision + full-context-fit admission policy
(work tracker ).

## Decision

Adopt a **self-enforcing routing gate** in
`local_model_catalog.curate_local_models`
(`src/callosum/local_model_catalog.py`) that applies Callosum-owned
admission rules over raw hub discovery. A local model is admitted to the
curated local fleet **only if** it passes all of:

1. **Runnable-on-host** — `model.local_runnable_on_host` is not `False`
   (the hub confirms the model can load on this host at all).
2. **Responses-capable** — the model exposes a `responses` API surface
   (`"responses" in model.api_surfaces`); a model with no responses
   surface is not a callosum-routable local cell.
3. **Training precision (no post-hoc casts)** — the model's weights are
   at the precision the model was trained at (typically `bf16`/`fp16`).
   No post-hoc downcasts (`q4_k_m`, `q8_0`, `fp8`, `nvfp4`, `w4a16`, gguf
   conversions) and no upcasts. A **native released quant the model was
   post-trained at** counts as training precision — currently `model-a0d2`
   × `mxfp4` (model-a0d2 was post-trained with mxfp4 quantization of its
   MoE weights; mxfp4 is its training precision, not a post-hoc
   downcast), matched by family prefix because families carry a size
   suffix (e.g. `model-a0d2`). Unknown quantization (`None`) is not
   itself a rejection reason; it defers to the other checks.
4. **Full-context memory-fit (the model-fit probe)** — when a probe
   result exists for the model's current artifact, the model fits this
   host at its full advertised context window. The probe reads vLLM's
   own startup log lines (`max_seq_len`, `GPU KV cache size`,
   `Maximum concurrency`) and records `fits_full_context =
   max_concurrency >= 1.0` (the MVP bar is concurrency 1 — one request
   at the full window). Failure → rejection reason
   `overflows-pool-at-full-context:<achievable>/<advertised>`, and the
   model is left for remote streaming.
5. **Optional throughput floor** — when the caller sets
   `min_tokens_per_second`, a model whose `estimated_tokens_per_second`
   falls below the floor is rejected with
   `throughput-below-floor:<tps>`. When unset, throughput is only used
   if the source provides it.

A model with **no probe result** is admitted **optimistically** (no
reason) so the fleet is not bricked before the first probe runs; the
probe later flips non-fitting models to rejected. The admitted set plus
per-model rejection reasons are exposed as a **stable consumer surface**
(`CuratedLocalModel.admitted` + `reasons` tuple; backend-facing seam
`backends/local_direct.LocalModelRegistryBackend.curated_local_models` /
`admitted_local_model_ids` / `local_model_admission_reasons`), so
downstream consumers inspect both admitted ids and rejection reasons
without re-encoding the catalog policy.

The admitted set is consumed by the local backend in the live prod path:
`backends/local_direct.LocalModelRegistryBackend.curated_local_models` (line
~237) calls `curate_local_models` with `probe_results` threaded from a
60 s TTL cache of `usage_log.all_model_fit_probes()`. Rejected local
models are dropped from routing (`router._admitted_local_cells`,
`cost_weighted` selector) and picked up by remote backends automatically.

The probe itself runs as a **short-lived subprocess when the operator is
idle and memory headroom is sufficient**: `app._PeriodicModelProbeSpawner`
fires every `CALLOSUM_MODEL_PROBE_INTERVAL_S` (default 300 s), only when
no request has completed in the last `CALLOSUM_MODEL_PROBE_QUIET_S`
(default 60 s) **and** `MemAvailable >= CALLOSUM_MODEL_PROBE_MEM_FLOOR_GIB`
(default 8 GiB), spawning `python -m callosum.jobs.model_probe
--max-models K` (default 1) serially (VRAM-contention) with an exclusive
GPU lock and a preflight `MemAvailable >= weights + margin` admission
check before each load. Kill switch: `CALLOSUM_MODEL_PROBE_ENABLED=0`.

## Consequences

**Positive:**

- The local fleet is **honest about what it can serve**: an admitted
  local model can actually back its full advertised context window on
  this host, so the router never commits a turn to a model that will
  stall or OOM mid-stream on a within-window prompt.
- The rule is **upstream of routing**: the router never sees an
  un-runnable model. Rejected models are dropped from
  `router._admitted_local_cells` and picked up by remote backends
  automatically — no request-time fallback path is needed for a
  context-overflow.
- Rejected models have **recorded reasons** (the `reasons` tuple on
  `CuratedLocalModel`, surfaced through `local_model_admission_reasons`),
  so the operator can inspect *why* a model is not in the local fleet
  without re-deriving the admission logic.
- The probe is **self-enforcing**: future model pulls are auto-evaluated
  by the idle-gated spawner, so the fleet stays honest without a
  recurring manual cull.
- A model with no probe result is admitted optimistically, so the fleet
  is not bricked before the first probe runs; the probe later flips
  non-fitting models to rejected.

**Negative / accepted:**

- The probe runs as a short-lived heavyweight subprocess (it loads the
  model under vLLM to read vLLM's own KV-cache measurement). This is
  gated to idle windows + sufficient memory headroom + an exclusive GPU
  lock + serial execution, so it does not contend with live traffic —
  but it means a freshly-pulled model is admitted optimistically until
  the next idle window probes it.
- The MVP bar is **concurrency 1** at the full window (one request can
  fill the window). Stricter bars (≥N concurrent at full window; a
  headroom margin) are deferred — both reuse the already-recorded
  `max_concurrency`, so they are ~one-line admission changes once tuned
  (work tracker , ).
- The MVP probes `runtime == "vllm"` models only; `gpt_oss` runtime,
  model-a0e0, and ollama entries are not probed and fall back to the
  hub's existing `local_fit_limit_tokens` prior + the
  training-precision check.
- The probe trusts vLLM's resolved `max_seq_len` rather than the
  model's true `config.json` `max_position_embeddings` (overriding
  `--max-model-len` to test the true value is deferred).

**Reversibility:** normal git-history change. The admission rule is
env-tunable / config: the probe spawner has a kill switch
(`CALLOSUM_MODEL_PROBE_ENABLED=0`, default on) and tunables for interval,
quiet window, memory floor, and per-run model cap
(`CALLOSUM_MODEL_PROBE_INTERVAL_S`, `CALLOSUM_MODEL_PROBE_QUIET_S`,
`CALLOSUM_MODEL_PROBE_MEM_FLOOR_GIB`, `CALLOSUM_MODEL_PROBE_MAX_MODELS`);
the throughput floor is caller-supplied and optional. Disabling the
probe falls back to optimistic admission (no `overflows-pool-at-full-context`
rejection), and the training-precision + runnable-on-host + responses-capable
checks remain. This ADR records the full-context memory-fit admission
shape as the agreed one.