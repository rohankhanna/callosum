# Dispatch Integration

**Dispatch status:** Yes

callosum runs one Dispatch-shaped workload as of Phase 5 of the
learning-router refactor:

- **`callosum.jobs.embed_backfill`** — backfill prompt embeddings for
  request-log rows that have `prompt_text` populated but
  `prompt_embedding IS NULL`. Seeds the kNN predictor's training
  corpus from historical request data after switching the
  EmbeddingProvider config from `noop` to `bge-large-en-v1.5` (or
  any other real provider).

## Install

```
python -m pip install --user -e /home/<user>/Desktop/scheduler-orchestration
```

The CLI lands at `~/.local/bin/dispatch`.

## Operator commands

```
dispatch --help
dispatch server start
dispatch server status
dispatch server logs
dispatch server stop
dispatch bootstrap --help
```

## Runtime directory

`SCHED_ORCH_RUNTIME_DIR` defaults to
`~/.local/share/scheduler-orchestration/runtime` per Dispatch's standard
layout. Callosum does not require a project-local override.

## Safety defaults

Dispatch execution is disabled by default. Set
`SCHED_ORCH_ENABLE_SCHEDULER_EXEC=1` and pass `DISPATCH_API_KEY` when
submitting jobs from callosum (a future commit may add a "submit
backfill" admin endpoint to /status; today, submit manually).

## Submit embedding backfill manually

```
EXAMPLE_ONLY python -m callosum.jobs.embed_backfill \
    --db-path /home/<user>/.local/state/callosum/requests.sqlite \
    --batch-size 64
```

The job:

- Reads rows where `prompt_text IS NOT NULL AND prompt_embedding IS NULL`
  in ascending id order.
- Computes embeddings in batches, writes them back in single transactions.
- Persists the cursor to a checkpoint file after each batch (atomic
  write — survives SIGKILL mid-write).
- Honors SIGTERM: finishes the in-flight batch, saves the checkpoint,
  exits 0. Dispatch can preempt freely.
- On restart, resumes from the checkpoint — no work re-done.

Resource hint when submitting through Dispatch: `device_preference=gpu`,
`priority=low`. The job should run only when GPU is idle so it doesn't
contend with live routing-path embedding requests.

## What's NOT Dispatch-shaped

- The proxy's reactive workloads (periodic backend smoke tester,
  cooldown prober, capability discovery) stay in-process. They are
  tightly coupled to in-memory backend state and gain nothing from
  Dispatch's queue / DAG / restart-safety guarantees.
- Per-request embedding generation runs inline in the routing pipeline
  (via the EmbeddingProvider Protocol). Only the *backfill* of
  historical rows is a Dispatch job.

## History

Earlier versions of this doc declared `N/A: no long-running automation`
because the model-based router was abandoned for sparse-data reasons
and the upstream-classifier-driven recommender that replaced it ran
inline per request. Phase 5 of the learning-router refactor (2026)
puts the embedding-backfill workload back into Dispatch territory.

## control plane reference

The canonical policy is `~/Desktop/control plane/The Baselines Document`
("Dispatch Local Orchestration") and `The Compliance Checklist`
("Dispatch Orchestration").
