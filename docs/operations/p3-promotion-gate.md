# P3 Promotion Gate — Probe Commands, Thresholds & Capture Schedule

**Status:** Specification (callosum-side response to work tracker handoff
). Supplies the machine-checkable `--check` commands,
thresholds, capture data dir, and cadence that work tracker release conditions
(, scheduler ) consume to gate the P3
flip (: uniform → knn quality predictor).

**Date:** 2026-07-01

## 1. What P3 is and what the gate protects

P3 flips `auto_router.routing.quality_predictor` from `"uniform"` to
`"knn"` in `~/.config/callosum/config.toml`. Embeddings (`bge-large-en-v1.5`)
are already live; the flip makes the KNN predictor authoritative for cell
selection.

The router also exposes `quality_predictor = "cell_majority_prior"` as an
explicit, non-default candidate. It uses the leave-one-out per-cell majority
signal identified by the shadow diagnostics; it does not change this gate,
which remains bound to the KNN metric until the operator explicitly chooses a
different rollout policy.

The gate exists because **a near-constant label target is unlearnable**: if
peer-quality labels are near single-class, the KNN predictor collapses to the
majority class and is *worse than* (or at best equal to) the trivial
always-guess-majority baseline. Accruing more near-single-class labels does
not fix this. The gate therefore treats **judge discrimination as the binding
constraint, not volume.**

## 2. Corrected model (what changed since the 2026-06-21 handoff)

The original handoff assumed P3 was volume-blocked. The 2026-06-26 regime
reset and the 2026-07-01 measurement revise that:

| | Absolute regime (n=42, pre-reset) | Honor-code regime (n=520, 2026-07-01) |
|---|---|---|
| label distribution | {+1:38, 0:4, −1:0} | {+1:1, 0:39, −1:10} |
| majority class | +1 (0.905) | 0 (0.78) |
| KNN exact rate | 0.667 | 0.78 |
| baseline exact rate | 0.905 | 0.78 |
| **KNN lift over majority** | **−0.238** | **0.0** |
| beats majority baseline | false | **false** |

The honor-code judging prompt (deployed 2026-06-26) **cured the +1 leniency**
but **overcorrected into a 0-dominated neutral class**. KNN now ties the
baseline exactly (lift 0.0). Coverage is ample (13 cells, 520 embedded labels,
50 eval samples; `knn_shadow_ready = true` structurally), but the **metric
gate fails.** This empirically confirms : P3 is blocked on
judge discrimination, not volume.

## 3. Release conditions (the gate)

Three conditions. Gates 1–2 are necessary floors; **gate 3 is binding.**

| # | Gate | Threshold (default) | Binding? |
|---|---|---|---|
| 1 | **VOLUME** — embedded peer-quality labels | `MIN_LABELS = 100` | necessary |
| 2 | **COVERAGE** — cells with ≥ `MIN_PER_CELL` labels | `MIN_CELLS = 2`, `MIN_PER_CELL = 10` | necessary |
| 3 | **METRIC** — KNN leave-one-out exact agreement beats majority-class baseline by a margin | `MIN_LIFT = 0.05` (strict `>`) | **binding** |

A **DISCRIMINATION diagnostic** (max single-class fraction of the label
distribution ≤ `DEFAULT_MAX_CLASS_FRACTION = 0.70`) is reported but NOT
blocking — it is the cheap proxy for why gate 3 fails, and gate 3 subsumes it
(single-class labels ⇒ lift 0 ⇒ gate 3 fails).

Thresholds are operator-tunable; defaults are grounded in the 2026-06-26
operator decision ("re-decide the flip at n≥100 once exact agreement provably
beats the majority-class baseline by a margin") and the code minimums
(`MIN_SHADOW_LABELS=20`, `MIN_SHADOW_CELLS=2`, `MIN_SHADOW_EVALS=10` in
`src/callosum/routing/labeler/peer_quality.py`).

## 4. Machine-checkable probe command

The gate is implemented in `src/callosum/jobs/p3_gate.py` and exposed as a CLI
script. Exit code `0` ⇔ all release conditions met; `1` ⇔ not met (JSON on
stdout for diagnostics). This is the command work tracker `--check` invokes.

### 4.1 Read-only gate check

```bash
scripts/check_p3_gate.py --db-path ~/.local/state/callosum/requests.sqlite
```

### 4.2 Full daily cycle (label then check)

```bash
scripts/check_p3_gate.py \
  --db-path ~/.local/state/callosum/requests.sqlite \
  --apply-labels \
  --checkpoint-path ~/.local/state/callosum/peer_quality_labels.ckpt
```

`--apply-labels` runs the shadow labeler
(`callosum.jobs.apply_peer_quality_labels`) first to convert newly accrued
opinions into embedded `peer_quality_v1` labels, then evaluates the gate. The
labeler is shadow-mode only and never switches live routing.

### 4.3 Programmatic / from work tracker `--check`

```bash
# work tracker release condition ( / ) --check command:
~/Desktop/callosum/.venv/bin/python \
  ~/Desktop/callosum/scripts/check_p3_gate.py \
  --db-path ~/.local/state/callosum/requests.sqlite \
  --apply-labels \
  --checkpoint-path ~/.local/state/callosum/peer_quality_labels.ckpt
# exit 0 -> condition met (flip eligible); exit 1 -> not met
```

> **Note — work tracker `--check` runs in a bare shell.** work tracker executes the
> `--check` command directly (system interpreter via the script shebang, from
> an unspecified working directory). It does **not** inherit this project's
> venv, so `scripts/check_p3_gate.py …` alone fails with `ModuleNotFoundError:
> No module named callosum`. Always prefix the command with the absolute
> `.venv/bin/python` and use absolute paths — exactly as above. The same
> caveat applies to `scripts/reset_peer_quality_regime.py` if it is ever wired
> as a probe.

### 4.4 Output schema (JSON)

```json
{
  "all_met": false,
  "blocking_on": ["metric"],
  "knn_shadow_ready": true,
  "gates": {
    "volume":   { "met": true, "observed": 520, "threshold": 100 },
    "coverage": { "met": true, "observed_cells_with_min_per_cell": 10,
                  "threshold_cells": 2, "threshold_per_cell": 10 },
    "metric":   { "met": false, "available": true, "observed_lift": 0.0,
                  "threshold_lift": 0.05, "beats_majority_baseline": false,
                  "reason": "knn does not beat majority-class baseline" },
    "discrimination_diagnostic": { "met": false, "observed_max_class_fraction": 0.78,
                  "threshold_max_class_fraction": 0.7,
                  "label_distribution": {"-1":10,"0":39,"1":1} }
  },
  "observed": { "captured_opinions": 2737, "embedded_labels": 520, "cells": 13,
                "knn_exact_rate": 0.78, "baseline_majority_exact_rate": 0.78,
                "knn_lift_over_majority": 0.0 }
}
```

### 4.5 Equivalent read paths (no daemon required)

- Standalone shadow report: `uv run python -m callosum.jobs.peer_quality_shadow_report --db-path <path>`
- Live service: `callosum status` → `router.peer_quality_shadow` block carries the same fields.

The same shadow-report surface also carries the sidecar measurement: the
`sidecar_breakdown` reports completed sidecar request token totals,
`tokens_per_opinion`, and `tokens_per_labeled_subject` so usable-label yield can
be read from one JSON document.

Peer-quality judging is now **synchronous and automatic** — there is no queue
to drain and no command to run. When a turn completes, if it was sampled at
`CALLOSUM_PEER_QUALITY_SIDECAR_ENQUEUE_RATE` (default `0.1`) and the serving
backend is not near quota exhaustion, the live server fires one background
judge request in-process and records the opinion directly into
`peer_quality_opinions` (with `nonce="sidecar"`). Labels accrue on their own as
traffic flows; just run the shadow report periodically to watch the counts
grow. The earlier batch-runner / backfill / experiment jobs and the systemd
timer were removed — they were over-engineering for this single-operator
loopback service.

## 5. Capture schedule

### 5.1 Capture-start date

**2026-06-26** — the regime reset. The 300 absolute-regime opinions + 42
derived labels were hard-deleted to avoid mixing regimes; the honor-code
judging prompt was deployed (daemon live). v2-only accrual runs from this
date. `peer_quality_capture_metrics` were intentionally retained as
diagnostics, so `capture_breakdown` is stale relative to the reset opinions
table — do not treat `capture_breakdown` counts as v2-only.

### 5.2 Capture rate

`CALLOSUM_PEER_QUALITY_CAPTURE_RATE` — Bernoulli sample gate, currently
**1.0** (live, every request sampled). 0 = off (default). Set via env on the
daemon. There is **no separate daily cap** on opinion accrual.

### 5.3 Per-probe cadence

| Probe | Cadence | Trigger |
|---|---|---|
| `apply_peer_quality_labels` | daily, before gate check | converts accrued opinions → embedded labels |
| `peer_quality_shadow_report` | daily (inside gate check; P3 default evaluates the full peer-label pool) | produces the metric the gate asserts |
| `check_p3_gate.py` (the `--check`) | daily | the work tracker release condition |

The work tracker due-probe scheduler () already runs
`work tracker check --due` daily via a systemd user timer (`work tracker-check.timer`,
`Persistent=true`, lingers). The daily cycle is therefore: timer fires →
`work tracker check --due` → invokes the release condition's `--check` command
(§4.3, with `--apply-labels`) → exit code computes `met`.

### 5.4 `--watch` (file-triggered) — the deferred path unit

work tracker supports `--watch` path triggers (). The
`work tracker-check.path` unit is currently deferred *until a real callosum capture
dir + `--watch` condition exist* (). **This spec unblocks
that.** Recommended watch target:

- **Capture data dir:** `~/.local/state/callosum/`
- **Hot file:** `requests.sqlite` (the single usage log; `peer_quality_opinions`,
  `peer_quality_capture_metrics`, and the `requests.quality_score` /
  `prompt_embedding` / `traffic_kind` columns all live here)
- **Checkpoint file:** `peer_quality_labels.ckpt` (mtime advances when the
  labeler runs)

`PathModified=` on the dir fires on any write; for a tighter trigger, watch
the checkpoint file (advances only when labels are freshly applied). Pair
with the daily timer as a fallback — `--watch` gives promptness, the timer
gives a guaranteed floor.

## 6. Codex-quota constraint

In-band organic accrual is **structurally starved** of cross-cell prose pairs
in real Codex traffic (): the eligible-subject condition
(prose from a *different* cell present in context) essentially never co-occurs.
At capture rate 1.0 the daemon still accrues ~550 opinions/day, but they are
near-single-class and do not advance the metric gate.

The documented fallback is **out-of-band shadow eval** (`admin_cell_call` asks
a different cell to judge a logged prose turn), which decouples from organic
diversity but **consumes Codex quota**. The constraints:

- **Per-account quota windows:** `x-codex-*` headers expose a 5-hourly and a
  weekly window (`codex_quota.py`). Out-of-band judging spend counts against
  these.
- **Per-turn overhead budget:** `_PEER_QUALITY_OVERHEAD_TOKEN_BUDGET = 1200`
  tokens per injected judgment (`app.py:135`), amortized defer-not-skip
  (, meter = case B: audit tokens subtracted from
  Codex-facing usage).
- **Dedup gate:** `judged_subject_request_ids` prevents re-judging the same
  subject.

**Implication for the schedule:** the *capture* (in-band) side is effectively
free of quota cost but does not advance the metric gate. The *discrimination*
fix (comparative/pairwise judging, stricter rubric, or the continuous-spectrum
direction in ) is what must advance the metric gate; any
out-of-band judging used to generate discriminating labels must be amortized
against the 5-hourly/weekly windows. Until a discrimination fix is shipped, the
daily `--check` will keep returning `met=false` on the metric gate regardless
of how long capture runs.

## 7. Current verdict (2026-07-01)

```
$ scripts/check_p3_gate.py --db-path ~/.local/state/callosum/requests.sqlite
exit=1   all_met=false   blocking_on=["metric"]
```

- Volume: **met** (520 ≥ 100)
- Coverage: **met** (10 cells with ≥10 labels ≥ 2)
- Metric: **NOT met** (lift 0.0, does not beat majority baseline)
- Discrimination diagnostic: 0.78 max-class fraction (0-dominated)

**P3 is not eligible to flip.** The blocker is judge discrimination (labels
collapse to neutral 0 under the honor-code prompt), not data volume. The next
lever is on the judging side, not the capture side:
 (continuous/spectrum score + judge-strength-aware
aggregation) or comparative/pairwise judging ( candidate
fixes), sequenced after the discrete honor-code version is proven to beat the
majority baseline.

## 8. Files

| Path | Purpose |
|---|---|
| `src/callosum/jobs/p3_gate.py` | Gate logic: `evaluate_gate`, `run`, thresholds |
| `scripts/check_p3_gate.py` | CLI wrapper (work tracker `--check` target) |
| `tests/unit/jobs/test_p3_gate.py` | Unit tests for the gate logic |
| `src/callosum/routing/labeler/peer_quality.py` | Shadow report + `MIN_SHADOW_*` constants |
| `src/callosum/jobs/apply_peer_quality_labels.py` | Shadow labeler (opinions → labels) |
| `src/callosum/jobs/peer_quality_shadow_report.py` | Standalone shadow-report CLI |
