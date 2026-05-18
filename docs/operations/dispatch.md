# Dispatch Integration

**Dispatch status:** `N/A: no long-running automation`

codex-proxy no longer runs any workload that needs Dispatch's cron-like
scheduling, restart-safe job tracking, or resource-aware admission.

Background tasks the proxy does run — synthetic auto-learning ticker,
periodic backend smoke tester, periodic cooldown prober — are reactive
to live in-memory backend state (quota snapshots, cooldown timestamps,
request traffic). They are not job-shaped: they would gain nothing from
Dispatch's queue / DAG / restart-safety guarantees and would lose the
tight coupling to the proxy's in-process state. They stay in-proxy.

## History

A previous iteration had a "model-based router" that periodically
re-fit a cost model from logged request data; that refit was the only
plausible Dispatch candidate (small, periodic, restart-safe). The
model-based router was abandoned because the corpus turned out to be
too sparse to train against (~300 unique prompts after dedup) and was
replaced by an upstream-classifier-driven routing strategy (see
`src/codex_proxy/cell_recommender.py`). The recommender's classifier
runs inline per request and is amortized by an in-memory cache — no
training, no refit, no Dispatch.

If a future workload appears that genuinely fits Dispatch's shape
(e.g. a real ML training run, an embedding backfill, an off-hour
analytics batch), revisit this document and submit the job through
the Dispatch CLI per the install + safety-gate guidance in the
polestar The Baselines Document.

## Polestar reference

The canonical policy is `~/Desktop/polestar/The Baselines Document`
("Dispatch Local Orchestration") and `The Compliance Checklist`
("Dispatch Orchestration"). Both permit `N/A` declarations when a repo
has no long-running automation; this document is the explanation.
