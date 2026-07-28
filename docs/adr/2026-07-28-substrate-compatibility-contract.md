# ADR: Substrate-owned compatibility contract for local model cells

Date: 2026-07-28
work tracker:  (P2) under shim-reduction program 
Grounded by: P1 inventory (`docs/investigations/2026-07-28-shim-reduction-p1-inventory.md`), local LLM gateway substrate survey, LiteLLM 1.82.6 survey.

## Status

Proposed — design only. No behavior change ships with this ADR. It defines the
boundary the P3 auto-dev rewrite, P4 upstream handoff, and P5 shim retirement
execute against. Implementation is gated on operator approval of P3/P4/P5
per-node; this ADR is the contract those nodes satisfy.

## Context

P1 inventoried 14 callosum-owned protocol-translation surfaces and found
**three parallel chat↔responses translation stacks with no shared module**
(`litellm_gateway.py`, `codex_auth_vault.py`, `local_direct.py`). Most of this
translation duplicates capability the substrates already own. Two substrate
surveys ground the contract in reality rather than aspiration:

### What local LLM gateway already owns (responses-proxy)
- Serves `/v1/responses` natively (`responses_proxy/http_app.py:75`).
- Strips in-band reasoning tags on **both** stream + non-stream paths
  (`translation.py:119` `InbandReasoningSplitter`; `sse.py:222,693`) — the
  original of callosum's ported copy, applied unconditionally.
- Normalizes all three reasoning aliases `thinking`/`reasoning`/`reasoning_content`
  (`translation.py:24-47`).
- Emits usage in `input_tokens`/`output_tokens` shape (`sse.py:754`).
- Translates tool-call shape both directions (`translation.py:182-338`).
- Advertises `api_surfaces` per model (`cli.py:2256-2278,3103`).
- **Gaps:** no `tools`/`modalities`/`throughput` in `capabilities --json`
  (only `context_window`, `quantization`, `supported_reasoning_levels`); no
  `drop_params` knob (implicit whitelist on responses, verbatim on chat); no
  "verified chat translation" flag for responses lanes (hard-coded
  responses-only even though `/v1/chat/completions` exists at `http_app.py:55`);
  no first-class cancellation contract; no durable `previous_response_id`/`store`
  (in-process only).

### What LiteLLM 1.82.6 already owns (gateway fallback)
- Native `/v1/responses` endpoint (since v1.63.8) with internal chat↔responses
  bridge. Callosum today POSTs to `/v1/chat/completions` and hand-translates —
  **the decisive lever** (see Decision §1).
- `drop_params: true` **is already enabled** in `gateway/litellm.yaml:58`
  (callosum's comment at `litellm_gateway.py:977-981` claiming `drop_params:false`
  is **stale**). `drop_params` only drops unsupported *OpenAI* params
  ([Issue #10791](https://github.com/BerriAI/litellm/issues/10791)); non-OpenAI
  params like the Codex `reasoning` routing hint may still pass through.
- Reasoning-alias normalization to canonical `reasoning_content` (v1.63.0+);
  `thinking` is Anthropic-only (`thinking_blocks`), **not** a generic alias.
- Dual usage shapes: `input_tokens`/`output_tokens` on `/responses`,
  `prompt_tokens`/`completion_tokens` on `/chat`.
- Bidirectional tool-call shape translation; `tool_choice` flattening
  ([PR #27622](https://github.com/BerriAI/litellm/pull/27622)) is recent —
  verify it landed in 1.82.6 before relying on it.
- **Version gap:** the explicit `use_chat_completions_api: true` opt-in for
  `openai/`-prefixed models with custom `api_base` (the exact local LLM gateway
  topology) shipped in **v1.83.14** ([PR #25346](https://github.com/BerriAI/litellm/pull/25346)),
  **newer than the deployed 1.82.6**. On 1.82.6 the bridge relies on model
  `mode`/`supported_endpoints` metadata which local LLM gateway's `openai/<model>`
  entries may not populate. Upgrading the LiteLLM image to ≥1.83.14 lets the
  contract explicitly opt in chat-only local models.

### Why callosum still translates today
Callosum's `litellm_gateway.py` POSTs to `/v1/chat/completions` and hand-translates
to/from Responses because (a) it predates LiteLLM's native `/v1/responses`, and
(b) local LLM gateway's responses-proxy only fronts a subset of cells. The
hand-translation is therefore compensating for **a path choice, not a missing
substrate capability** on the cells the proxy fronts.

## Decision

### 1. Consume the advertised native surface; do not translate. (Primary lever)

A cell's `api_surfaces` advertisement is the contract. When a cell advertises
the surface the client requested (`responses` for Codex `/v1/responses` traffic;
`chat` for `/v1/chat/completions` traffic), callosum **POSTs to that native
surface and passes the body through**. Callosum ceases to translate
chat↔responses for contract-conformant cells.

- For `responses`-advertising cells: POST to the substrate's `/v1/responses`
  (local LLM gateway responses-proxy, or LiteLLM's native `/v1/responses`) and pass
  the Responses body through. The substrate owns event shape, reasoning
  normalization, usage shape, tool shape, in-band tag stripping.
- For `chat`-advertising cells hit by `/v1/chat/completions` traffic: POST to
  `/v1/chat/completions` and pass through. No response translation.
- Cross-surface mismatches (chat client hitting a responses-only cell, or vice
  versa) are **routed around**: the capability filter excludes the cell for that
  request rather than callosum translating. This retires the
  `_responses_sse_to_chat_sse` inverse translator (`local_direct.py:604-770`)
  and the chat↔responses round-trips in `local_direct.py:381-540`.

**Residual callosum-owned translation** survives only for cells where **no
substrate owns the requested surface** — e.g. a bare ollama/vllm/model-a0e0
chat-only cell not fronted by any responses-proxy, hit by Codex
`/v1/responses` traffic. That residual path is the `inband_reasoning` transform's
defended domain (KEEP-with-release-condition from P1). The contract marks it
**TEMPORARY-DEBT with a close condition**: "a substrate fronts this cell's
responses surface," at which point the residual translator deletes.

### 2. The seven substrate conformance invariants (P1 §A4)

A substrate's native `/v1/responses` surface is **contract-conformant** iff it
guarantees, on every response:

1. `usage.input_tokens` is present and non-null (Codex CLI hard-fails
   "missing field 'input_tokens'" otherwise). local LLM gateway: ✅ `sse.py:754`.
   LiteLLM `/responses`: ✅ native.
2. `output[]` is never empty (emit an empty message item if there is no
   content). local LLM gateway: ✅. LiteLLM: ✅.
3. Ordering: `reasoning` items first, then `function_call` items, then
   `message` (o1/o3 convention). local LLM gateway: ✅. LiteLLM: ✅.
4. Consecutive `function_call` Responses items collapse into one assistant
   chat message with parallel `tool_calls` on the chat bridge.
   local LLM gateway: ✅ `translation.py:182-271`. LiteLLM: ✅ (older than 1.82.6).
5. `finish_reason`/`status` mapping for `incomplete`/`cancelled`/`content_filter`
   does not mask budget truncations as natural stops.
   local LLM gateway: partial. LiteLLM: ✅.
6. Reasoning-alias unification: `thinking`/`reasoning_content`/`reasoning` →
   one reasoning item. local LLM gateway: ✅ all three (`translation.py:24-47`).
   LiteLLM: ✅ `reasoning`+`reasoning_content`; `thinking` is Anthropic-only —
   **callosum must not rely on LiteLLM to unify `thinking` for non-Anthropic
   cells.**
7. `tool_call_at_scale` capability is probed **through the substrate** (the
   harness probes the fronted cell, not the raw runtime), so a substrate fix
   flips the finding to `pass` and the routing at-scale gate
   (`routing/capability.py:77-91`) stops dropping the cell. A callosum-side
   transform does **not** flip this finding. **GAP on both substrates today.**

### 3. Capability advertisement contract (substrate-side gaps → P4 handoff)

The substrate must expose, per model in `local-llm capabilities --json`
(`local.capability_matrix.v1`):

| Field | local LLM gateway today | Contract requires |
|---|---|---|
| `context_window` | ✅ | ✅ |
| `quantization` | ✅ | ✅ |
| `supported_reasoning_levels` | ✅ (model-a0d2 only honest) | ✅ — must be honest per runtime, not inferred |
| `supports_tools` | ❌ | ✅ — so callosum stops probing |
| `modalities` (text/vision/audio) | ❌ | ✅ |
| `throughput` (tokens/s) | ❌ (only `graph_metrics.ceiling_search_completion_tokens_per_second`) | ✅ — first-class field |
| `api_surfaces` | ✅ | ✅ — plus a `verified_chat_translation: bool` flag for responses lanes |
| `verified_chat_translation` | ❌ (hard-coded responses-only) | ✅ — so callosum can route chat traffic to responses lanes when verified |

Closing these gaps is substrate work → **P4 upstream handoff** to local LLM gateway
(handoff prompt only; no external edits from callosum). Until closed, callosum
keeps its capability probes for tools/modalities and its `cell_capabilities`
defaults (`local_direct.py:238-271`) as TEMPORARY-DEBT.

### 4. `drop_params` and request-field hygiene

- The substrate (LiteLLM) honors `drop_params: true` for unsupported *OpenAI*
  params — **already enabled**. Callosum's `_strip_codex_only_fields`
  (`litellm_gateway.py:976-1005`) becomes **redundant for OpenAI-standard
  params** (`parallel_tool_calls`).
- **KEEP a thin strip only for non-OpenAI params** the substrate may pass
  through (Issue #10791): the Codex `reasoning` routing hint (distinct from
  `reasoning_effort`). This is belt-and-suspenders until the contract verifies
  LiteLLM classifies `reasoning` as droppable.
- **Fix the stale comment** at `litellm_gateway.py:977-981` (claims
  `drop_params:false`; actually `true`). Small drive-by, fold into P5.

### 5. Downgrade / quarantine behavior when the contract is missing

When a cell's probed behavior **violates** the contract (a conformance invariant
fails, or an advertised surface is missing), callosum does **not** author a
transform (that is the P3 rewrite). Instead:

- `quarantine_cell`: drop the cell from the routable pool for the violated
  surface, record the contract violation as the reason, and surface it in
  `/status` + the capability profile.
- `file_upstream_gap`: emit a handoff artifact (P4) describing the violated
  invariant + probe commands + acceptance criteria, targeted at the owning
  substrate.
- `route_native`: if a conformant alternative surface exists for the same cell,
  route there instead.
- `verify_fix`: re-probe after an upstream fix lands; flip quarantine off when
  the invariant passes.

This replaces the current `adapter_hint` → "author a transform" path
(`pre_filter.py:30-34`, `perception.py:337-353`) with `adapter_hint` →
"quarantine / file upstream / route-native / verify." The `"no feasible adapter"`
sentinel (`pre_filter.py:99-106`) generalizes: **every** `adapter_hint` becomes a
gap report, not a transform-authoring instruction. (P3 implements this.)

### 6. Callosum's permanent owned surface (unchanged by this contract)

Callosum keeps: routing, cell-grid composition, capability verification
(observational probing/classification), policy attribution, logging/usage,
cooldown/quota health, the marker scrub, the per-CLI endpoint architecture
(`transforms/<cli>/`), and the transforms *framework* (`protocol.py`,
`registry.py`). The framework hosts any residual per-cell transform until its
release condition fires; `build_default_registry` shrinks back to empty as
shims retire.

## Acceptance tests (contract classification)

A new test module `tests/unit/test_substrate_contract.py` encodes the contract
as a classifier over a cell's advertised surfaces + probed conformance:

- `contract_conformant(cell, surface)` → True iff the surface is advertised and
  all 7 invariants pass for that surface.
- `contract_gap(cell, surface)` → surface advertised but ≥1 invariant fails or
  a capability field (tools/modalities/throughput) is missing →
  `file_upstream_gap` action.
- `contract_missing(cell, surface)` → surface not advertised → `route_native`
  to an alternative surface or `quarantine_cell`.
- `residual_translation_required(cell, surface)` → no substrate owns the
  requested surface for this cell → TEMPORARY-DEBT with close condition.

The MOVE-UPSTREAM test bodies from P1 §C (`test_litellm_gateway.py` translation
tests, `test_local_direct_backend.py:247-445`, `test_inband_reasoning.py`) are
restated in the handoff (P4) as **substrate conformance acceptance tests** for
the owning substrate repo — they move out of callosum when the residual
translator deletes.

## Consequences

- **Retires ~10 MOVE-UPSTREAM translation surfaces** (P1 §A2) once cells are
  fronted by a contract-conformant substrate surface — the bulk of
  `litellm_gateway.py`'s 1302 lines and `local_direct.py`'s inverse translator.
- **Does not require a new caching framework or dependency** — reuses
  `api_surfaces` (already shipped) and the existing capability probe path.
- **Substrate-side work required** (P4 handoff): local LLM gateway capability-matrix
  gaps (tools/modalities/throughput/verified_chat_translation) and a LiteLLM
  image bump to ≥1.83.14 for the explicit `use_chat_completions_api` opt-in.
  Until those land, callosum keeps its probes + residual translator
  (TEMPORARY-DEBT, no regression).
- **P3 rewrite target unchanged** but now has a concrete action set
  (route_native / file_upstream_gap / quarantine_cell / verify_fix / remove_shim)
  and a contract to quarantine against.
- **Rollback:** feature-branch `--no-ff` merges; the residual translator is
  never deleted until the substrate fronting is verified, so any regression is
  a single revert. The contract is additive — `contract_conformant` returning
  False everywhere falls back to today's behavior.

## Non-goals

- Defining which substrate *owns* a given cell (local LLM gateway vs LiteLLM) —
  that is a P4 per-cell routing/handoff decision, not a contract property. The
  contract is substrate-agnostic.
- Mid-stream failover or cross-backend stream stitching (already a non-goal per
  `The Project Documentation`).
- Moving the per-CLI codex endpoint transform architecture — that is in-scope
  KEEP per the per-CLI-endpoint architecture and is not a shim.