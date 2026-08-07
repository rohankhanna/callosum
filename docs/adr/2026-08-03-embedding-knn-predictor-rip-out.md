# ADR: Rip out the BGE embedding + KNN quality-predictor subsystem

Date: 2026-08-03

## Status

Accepted

> Note: this ADR was authored 2026-08-13 to record a decision that was
> enacted 2026-08-03 (merged to `main` as `d2f3a4e`, deployed live and
> durable the same day). It records a past decision rather than proposing
> a new one; the implementation, deploy, and follow-on doc/code-sync
> work all landed prior to this document. The date above is the decision
> date.

## Context

Callosum's router historically carried a prompt-embedding + KNN
quality-predictor subsystem: a BGE (`bge-large-en-v1.5`) prompt-embedding
path that accrued per-request embeddings, and a KNN quality-predictor
that used those embeddings to lift a "P3 flip" confidence metric toward a
future contextual bandit. This was turned on for accrual on 2026-06-26
with the intent of eventually feeding a contextual (prompt-conditioned)
routing bandit.

The router's quality/relevance direction instead converged on a
**non-contextual** bandit: a per-cell peer-quality matrix populated by
out-of-band LLM peer judging (the peer-quality sidecar + `PeerJudgeLabeler`),
with `CellMajorityPriorPredictor` (`id=cell_majority_prior`) as its cold
form. Because the bandit is non-contextual by operator decision — "we
don't need embeddings as long as we use the LLMs to do peer-quality
rating" — prompt embeddings have no future role, and the KNN-lift P3
flip metric is forfeit by design.

Keeping the embedding subsystem imposed real, ongoing costs:

- A heavy ML dependency stack (`torch`, `sentence-transformers`,
  `transformers`, `tokenizers`, `scipy`, `scikit-learn`, `sympy`,
  `triton`, `safetensors`, …) that dominated the build's failure-prone
  step (a large `torch` re-download under the build's isolated cache)
  and inflated the runtime artifact.
- A `prompt_embedding` accrual column and a deferred-for-embedding prune
  policy that held old unlabeled rows indefinitely waiting for an
  embedding that would never come — contributing to `requests.sqlite`
  bloat.
- Stale doc/code surfaces describing embeddings/KNN as live or
  aspirational, diverging from the actual non-contextual routing model.

## Decision

Remove the BGE prompt-embedding + KNN quality-predictor subsystem
entirely. The bandit stays **non-contextual** (per-cell peer-quality
matrix, no prompt-conditioning). Embeddings are not to be re-introduced.

Concretely (enacted on `feat/rip-embedding-knn`, commits `39e3077` +
fallout `2641c23`, merged `--no-ff` to `main` as `d2f3a4e`):

- Ripped the embedding provider + KNN predictor code paths.
- Removed `embedding_provider="bge-large-en-v1.5"` from the live config
  (`~/.config/callosum/config.toml`) — `RoutingConfig(extra="forbid")`
  makes a leftover field startup-fatal, so this removal had to land
  before the deploy. The `[auto_router.routing]` block now selects
  `quality_predictor="uniform"` + `cell_selector="cost-weighted"`.
- Made the column-prune `guard_column` (was `prompt_embedding`)
  optional and dropped it from both prune policies, so old unlabeled
  rows are pruned on age + row-filter rather than deferred forever.
- Shed the heavy ML dependency stack: removed the `[embeddings]` pip
  extra, the `callosum-embed-backfill` console-script entry, the
  `[dependency-groups] embeddings` mirror, and the build-script
  `[embeddings]` hardcode (`pyproject.toml` + `install_runtime_venv.sh`);
  regenerated `uv.lock` to drop the whole ML stack (later commit
  `f12d3dd`).
- Synced the curated docs and in-code comments to the rip-out so no
  surface still describes embeddings/KNN as live or aspirational.

**Survived the rip:** `CellMajorityPriorPredictor` (the cold form of the
non-contextual bandit), the peer-quality sidecar/capture path +
`PeerJudgeLabeler`, and the slimmed `peer_quality_shadow_report`. The
predictor loader now learns from all labeled rows, not just
embedding-bearing ones — a small training-signal gain.

## Consequences

**Positive:**

- The build's failure-prone step (large `torch` re-download) is gone;
  the runtime artifact and dependency surface are dramatically smaller
  and faster to install.
- `requests.sqlite` bloat hygiene is restored: rows are no longer held
  indefinitely for an embedding that never comes.
- Docs and code comments now match the actual non-contextual routing
  model, eliminating a class of stale-surface confusion.

**Negative / accepted:**

- The P3 flip / KNN-lift confidence metric is forfeit by design. Any
  future contextual (prompt-conditioned) routing would require
  re-introducing an embedding/feature path — explicitly out of scope
  under the non-contextual bandit decision; do not re-introduce
  embeddings to serve it.
- The dormant `prompt_embedding` / `response_embedding` columns remain
  in the live `requests.sqlite`. They are harmless and dormant, and a
  `DROP COLUMN` rewrites the whole table (a destructive op on a ~24GB
  live DB). The DROP is deliberately held for an explicit maintenance
  window with a prior backup; it is not autonomously performed.

**Reversibility:** the rip is a normal git-history change. Re-introducing
embeddings would be a new feature decision, not a revert, and is
discouraged by this ADR's non-contextual-bandit rationale.