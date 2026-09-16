# ADR: Three-tier, per-field local-model capability precedence

Date: 2026-08-05

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-08-05 (introducing commit `a36e110`, merged to `main` as
> `8554d0b`). It records a past decision; the implementation landed
> prior to this document. The date above is the decision date.

## Context

A routing cell's `modalities` and `supports_tools` feed the hard
`CapabilityFilter` — a request whose prompt modality or tool-need a cell
cannot serve is excluded before any cost/quality scoring. Getting those
fields wrong is expensive in both directions: trust a runtime's
self-report blindly and a wrong `tools` claim sends tool-calling traffic
to a model that mangles it; default pessimistically and a non-ollama
runtime that handles tools fine is starved of tool traffic.

Before this decision, local capabilities came from a single source:
`litellm_gateway` inlined an ollama `/api/show` parse. Two things made
that insufficient:

- The sibling local model registry was about to ship **canonical** capability
  fields (a human-curated, authoritative per-model record) that should
  override whatever a runtime self-reports.
- The `local_direct` backend also needed capabilities, but did not share
  the gateway's parse — duplicating it would drift.

A whole-cell "trust source X" rule is too coarse: the hub may be
canonical for `supports_tools` (curated) while the runtime's
`/api/show` is the better source for `context_window` (measured). The
precedence must be **per field**, not per cell.

## Decision

Resolve each capability field of `LocalModelRegistryBackend.cell_capabilities`
through a three-tier, **per-field** precedence
(`src/callosum/backends/_ollama_capabilities.py`,
`local_direct.py`, `litellm_gateway.py`):

1. **Hub-canonical.** When the hub emits a field, it is canonical truth
   (defensively parsed off `CapabilityRow`). This tier is inert until
   the sibling hub ships the fields — the parse is in place but reads
   nothing until then.
2. **Direct-ask stopgap.** Otherwise, ask the runtime via ollama
   `/api/show` (`fetch_ollama_capabilities`). Gated by
   `CALLOSUM_LOCAL_CAPABILITIES_STOPGAP` = `off` | `modalities` (default)
   | `all`. Modalities are **strictly additive** (a runtime can add
   `image`/`audio` but never revoke the always-present `text`), so the
   modality stopgap ships on. **Tool accuracy is explicitly enabled**: the
   runtime tool-probe is one-directional — it can *revoke* a wrong
   `tools` claim but cannot *grant* one — so the tool stopgap is off by
   default.
3. **Conservative defaults.** Where neither source speaks:
   `text`-only modalities, `supports_tools=True` (optimistic, so
   non-ollama runtimes with no direct-ask source stay tool-routable,
   probe-revocable).

The ollama `/api/show` parse was extracted into the shared
`_ollama_capabilities` module so `litellm_gateway` and `local_direct`
share one parser (and to break a circular import: `litellm_gateway`
imports `DEFAULT_HEALTH_TIMEOUT_S` back from it). The parser returns
primitives (`OllamaShowCapabilities`), not `CellCapabilities`, so each
backend merges with its own surrounding fields. It never raises — all
transport/HTTP/parse failures return `None`, matching the original
"must NEVER bubble up" contract.

The hard `CapabilityFilter` itself is **unchanged** — this decision is
upstream of the filter, in capability *population*. A wrong capability
is now a population bug, not a filter bug.

## Consequences

**Positive:**

- The hub becomes canonical where it speaks, without having to cover
  every field or every model before shipping — fields it does not emit
  fall through to the direct-ask or default tiers.
- Per-field precedence means a curated `supports_tools` and a measured
  `context_window` can coexist on the same cell from different sources.
- `local_direct` and `litellm_gateway` share one parse, so capability
  semantics cannot drift between backends.
- Non-ollama runtimes stay tool-routable by default (optimistic
  `supports_tools=True`) rather than being silently excluded; a wrong
  claim is correctable by the probe, not fatal.

**Negative / accepted:**

- The hub-canonical tier is inert until the sibling hub ships the
  fields; until then the effective behavior is tiers 2–3.
- Tool accuracy depends on a one-directional probe (revoke-only), so a
  runtime that *under*-claims tools cannot be corrected by the stopgap
  — the explicitly enabled tool stopgap is the deliberate trade for not
  trusting blind self-reports.
- `supports_tools=True` as the default means a non-ollama runtime that
  silently mishandles tools will receive tool traffic until the probe
  revokes it; the optimistic default trades a bounded amount of bad
  tool traffic for not starving tool-capable runtimes.

**Reversibility:** normal git-history change. The direct-ask tier is
env-tunable (`CALLOSUM_LOCAL_CAPABILITIES_STOPGAP`) without a code
change; the hub-canonical tier activates automatically when the hub
ships the fields. This ADR records the per-field, hub-canonical →
direct-ask → conservative-default shape as the agreed one.
