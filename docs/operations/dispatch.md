# Dispatch Integration

**Dispatch status:** Yes

callosum runs two Dispatch-shaped workloads as of the peer-quality
learning-router work:

- **`callosum.jobs.embed_backfill`** — backfill prompt embeddings for
  request-log rows that have `prompt_text` populated but
  `prompt_embedding IS NULL`. Seeds the kNN predictor's training
  corpus from historical request data after switching the
  EmbeddingProvider config from `noop` to `bge-large-en-v1.5` (or
  any other real provider).
- **`callosum.jobs.apply_peer_quality_labels`** — derive conservative
  request-level `quality_score` candidates from captured
  `peer_quality_opinions`. This is shadow-mode only: it writes label
  provenance for future KNN training but does not change live routing.

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

## Managed service boundary

Dispatch jobs are orthogonal to the Callosum service's own deployment
boundary. The managed Callosum service itself should run from the
installed runtime CLI at:

```bash
~/.local/share/callosum/runtime/venv/bin/callosum
```

not from `uv run` against the repo checkout. See
`docs/operations/runtime_deploy.md`.

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

## Submit peer-quality labels manually

```
EXAMPLE_ONLY python -m callosum.jobs.apply_peer_quality_labels \
    --db-path /home/<user>/.local/state/callosum/requests.sqlite \
    --checkpoint-path /home/<user>/.local/state/callosum/apply_peer_quality_labels.ckpt \
    --batch-size 200 \
    --min-opinions 1 \
    --dry-run
```

The job:

- Reads captured peer-quality opinions and resolves each to a subject
  request. New opinion rows prefer exact `subject_request_id`; legacy
  rows fall back to the latest prior same-session subject-cell request.
- Requires peer-quality capture to have produced source rows first.
  Check `/status` at `router.peer_quality_capture`; the configured
  `CALLOSUM_PEER_QUALITY_CAPTURE_RATE` must be above `0` for new
  capture attempts.
- With `--dry-run`, resolves and reports candidate progress through the
  same path but does not update request rows or checkpoint state.
- Writes only rows where `quality_score IS NULL`, using
  `quality_label_method='peer_quality_v1'`.
- Persists the cursor checkpoint after each batch and honors SIGTERM
  between batches.
- Leaves live routing untouched. Inspect `/status` at
  `router.peer_quality_shadow` for counts, embedded-label readiness,
  and held-out KNN shadow-eval metrics over `peer_quality_v1` rows only.
  `knn_shadow_readiness.missing` is the operator-facing list of evidence
  still needed before a P3 canary can be considered.

When the service is stopped, inspect the same report directly from the
usage log:

```
EXAMPLE_ONLY python -m callosum.jobs.peer_quality_shadow_report \
    --db-path /home/<user>/.local/state/callosum/requests.sqlite
```

## Run a bounded peer-quality capture window

For a managed user service, use a temporary systemd user-manager
environment rather than editing the unit file:

```
systemctl --user set-environment CALLOSUM_PEER_QUALITY_CAPTURE_RATE=1
callosum service restart
curl -s http://127.0.0.1:8765/status | jq '.router.peer_quality_capture, .router.peer_quality_shadow'
```

Then send normal multi-turn traffic through the proxy. Peer opinions need
same-session prior assistant turns from a different cell; single-turn
traffic and tool-only turns do not produce useful source rows.

When the window is over, disable capture and restart:

```
systemctl --user unset-environment CALLOSUM_PEER_QUALITY_CAPTURE_RATE
callosum service restart
```

After the window, run the offline report and label preview:

```
python -m callosum.jobs.peer_quality_shadow_report \
    --db-path /home/<user>/.local/state/callosum/requests.sqlite

python -m callosum.jobs.apply_peer_quality_labels \
    --db-path /home/<user>/.local/state/callosum/requests.sqlite \
    --checkpoint-path /home/<user>/.local/state/callosum/apply_peer_quality_labels.ckpt \
    --dry-run
```

## What's NOT Dispatch-shaped

- The proxy's reactive workloads (periodic backend smoke tester,
  cooldown prober, capability discovery) stay in-process. They are
  tightly coupled to in-memory backend state and gain nothing from
  Dispatch's queue / DAG / restart-safety guarantees.
- Per-request embedding generation runs inline in the routing pipeline
  (via the EmbeddingProvider Protocol). Historical embedding backfill
  and peer-quality label application are Dispatch-shaped jobs.

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
