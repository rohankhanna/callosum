# ADR: Reasoning-effort cap — top-N tiers by rank (temporary guardrail, ollama-cloud immune)

Date: 2026-08-04

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision enacted
> 2026-08-04 (introducing commit `396cde8`, merged to `main` as
> `21198d3`). It records a past decision; the implementation landed
> prior to this document. The date above is the decision date.

## Context

While the peer-quality and cost models are still stabilizing, automatic
routing could spend the most expensive reasoning efforts (`high`,
`xhigh`) too freely on non-ollama backends — burning quota and budget on
tiers whose marginal quality benefit over cheaper efforts is not yet
justified by learned evidence. There was no guardrail keeping the
expensive tiers rare during this stabilization period.

A level-by-name cap ("cap `xhigh`") would not generalize if the
canonical `low < medium < high < xhigh` ladder grows, and would be
fragile to rename. A flat cap on every high effort would also wrongly
regulate the cloud lane (ollama-cloud), whose reasoning levels are not
the standard regulated ladder and whose quota is a separate custody
path.

## Decision

Adopt a **temporary operating cap on the top-N canonical reasoning
efforts by severity rank** (`src/callosum/app.py` lines 2544-2577,
config in `src/callosum/config.py` lines 224-228), applied before the
router and the coverage quota see the candidate pool:

- The cap targets the **top-N canonical reasoning efforts by severity
  rank** — the last N rungs of the `low < medium < high < xhigh` ladder
  (`FALLBACK_REASONING_LEVELS[-effort_cap_top_n:]`), so `{high, xhigh}`
  for the default `effort_cap_top_n = 2`. Capping by rank, not by name,
  generalizes if the ladder grows and never regulates the cheap tiers.
- If a capped effort's recent successful share is at or above
  `effort_cap_pct` (default 1%) over `effort_cap_window_seconds` (default
  seven days, aligned with the upstream weekly quota window), that
  effort's cells are removed from automatic routing while cheaper
  alternatives are available.
- **ollama-cloud is immune outright**: models served by exempt backend
  kinds (`_EFFORT_CAP_IMMUNE_BACKEND_KINDS`) are never dropped and are
  excluded from the share denominator, so the cloud lane is unaffected.
- Efforts outside the canonical ladder (e.g. an ollama-cloud model's
  `default`) are never capped.
- Explicit `callosum:<source>/<model>:<effort>` pins bypass this filter
  entirely (a pinned effort is an operator choice, not automatic
  routing). If no uncapped alternative exists, the pool is preserved
  rather than fabricating an outage.
- The cap is exposed in `/status` as `router.effort_cap` with the
  enabled flag, cap percentage, top-N, cap window, and the exempt
  (`immune`) backend kinds.

This is intentionally a **temporary operating cap, not a learned quality
claim**. Remove it once the learned router has enough reliable cost and
quality evidence to spend high reasoning effort deliberately.

## Consequences

**Positive:**

- High and xhigh efforts are kept rare for non-ollama backends during
  model stabilization, bounding quota/budget burn on the expensive
  tiers while the learned router matures.
- The cloud lane (ollama-cloud) is unaffected: its cells are never
  dropped and excluded from the share denominator, so cloud quota is
  not regulated by a cap designed for the standard ladder.
- The cap is observable via `router.effort_cap` in `/status`, and is
  env/config-tunable (`effort_cap_enabled`, `effort_cap_pct`,
  `effort_cap_top_n`, `effort_cap_window_seconds` in `[auto_router]`).

**Negative / accepted:**

- Temporary by design: the cap must be removed (or `effort_cap_enabled`
  set off) once the learned router can spend high effort deliberately;
  leaving it on past that point would suppress legitimate high-effort
  routing.
- Capping by rank means adding a new top rung (e.g. a future `xxhigh`)
  would automatically fall under the cap; that is the intended
  generalization, but it means the regulated set is implicit, not
  enumerated.
- If every candidate is a capped effort at/over the share threshold and
  no uncapped alternative exists, the pool is preserved (no fabricated
  outage) — the cap is best-effort, not absolute.

**Reversibility:** normal git-history change. The cap is a temporary
guardrail with a kill switch (`effort_cap_enabled`); this ADR records
the top-N-by-rank + ollama-cloud-immune shape as the agreed one, to be
retired once the learned router matures.
