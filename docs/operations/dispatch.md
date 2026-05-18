# Dispatch Integration

**Dispatch status:** `Yes`

codex-proxy uses [Dispatch](file://~/Desktop/scheduler-orchestration)
(local orchestration substrate, loopback-only, restart-safe job graph
runner) for its only long-running automation that needs cron-like
cadence and operator-observable job state: the periodic refit of the
cost-router's lookup tables.

In-process background tasks (synthetic auto-learning ticker, smoke
tester, periodic backend cooldown prober) remain in-proxy because they
are reactive to live traffic and tightly coupled to in-memory backend
state; they are not job-shaped.

## Install (canonical, per polestar)

```
python -m pip install --user -e ~/Desktop/scheduler-orchestration
```

This makes the `dispatch` CLI available from any directory through the
user's PATH.

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
`~/.local/share/scheduler-orchestration/runtime` (per the Dispatch
canonical service definition for this machine). Repo-local runtime
directories are not used.

## API key — per-application

Dispatch issues API keys on a per-application basis: codex-proxy uses
its own key, distinct from any other consumer's. The canonical issuance
flow is the Dispatch CLI's `bootstrap` subcommand, which creates a user
(if needed), logs in, creates the project, and mints a key in one pass:

```
EXAMPLE_ONLY dispatch bootstrap \
    --username <your-username> \
    --password <your-password> \
    --project-name codex-proxy \
    --key-label refit-router \
    --print-summary
```

Run `dispatch bootstrap --help` for the authoritative argument list.
The minted key is what `DISPATCH_API_KEY` should be set to when
submitting the refit job below.

## Safety gates

Per polestar `The Baselines Document` and `The Compliance Checklist`, Dispatch
execution is disabled by default. To actually run the refit job:

- `SCHED_ORCH_ENABLE_SCHEDULER_EXEC=1`
- `DISPATCH_API_KEY=<codex-proxy-key-from-bootstrap-above>`

The Dispatch server must remain loopback-only by default
(`127.0.0.1:8780`); LAN or public binding requires an explicit
repo-local security decision.

## Refit-router job

The refit job's payload is a single HTTP POST against the proxy's
`/control/refit-router` admin endpoint. The endpoint is synchronous:
when it returns, the new v1 + v2 lookup tables are live in-process.

Job spec lives at `dispatch/refit-router.job.json`. Schedule: every
3600 seconds (one hour).

### Submit (manual, one-time setup)

```
EXAMPLE_ONLY dispatch jobs submit dispatch/refit-router.job.json
```

(replace `EXAMPLE_ONLY` with nothing when running for real)

### Verify it ran

```
EXAMPLE_ONLY curl -fsS http://127.0.0.1:8765/status | jq .router
```

`router.last_fit_age_s` should be less than the schedule interval
(3600s). `router.is_ready` indicates whether the v1 floor is met;
`router.v2_buckets_with_data` lists which `(complexity, model, effort)`
buckets have enough labeled rows that v2 routing engages for them.

### Manual fire (smoke test the endpoint independently of Dispatch)

```
EXAMPLE_ONLY curl -fsS -X POST http://127.0.0.1:8765/control/refit-router
```

Returns `{refit_at_ts, is_ready, v1_cell_count, v2_bucket_count}`.

## What changed when this moved out of in-process

Before: a refit branch lived inside `synthetic.py:_tick`, firing every
`optimal_refit_interval_seconds`. If synthetics were disabled (no
backends or no usage log), the refit never ran.

After: `lifespan` still primes once at startup. The hourly refresh is
now Dispatch's responsibility. The synthetic ticker no longer touches
the cost router.

## Other workloads considered, kept in-proxy

| Workload | Why it stayed in-process |
|---|---|
| Synthetic auto-learning request floor | Tightly coupled to live backend quota snapshots; pacing is reactive |
| Periodic smoke tester | Same — reads live backend health |
| Periodic backend cooldown prober | Same — reactive to in-memory cooldown state |
| Request dispatch / SSE streaming | Hot path, not job-shaped |

## Future Dispatch candidates (deferred)

- Usage-log retention trim (~90 days) — see in-repo memory note
  `project_usage_log_retention_pending`, target review by
  2026-08-14.
- v2-of-v2 router: a real learned classifier (small ML model that
  predicts complexity from prompt text) — would belong in Dispatch
  because training is non-trivial, GPU-eligible, and can be queued
  rather than blocking.
