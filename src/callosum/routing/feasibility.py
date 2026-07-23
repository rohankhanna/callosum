"""Coverage feasibility: can a cell FINISH a request of this size before
the local stall-guard fires?

Sibling to routing/capability.py (hard capability constraints) and to
_window_fit_factor in router.py (soft context-window fit). Capability
asks "can this cell serve this request at all?"; window-fit asks "does the
prompt fit the advertised context window?"; feasibility asks "can it finish
within the stall-guard budget?" — the question the coverage doom loop
exposed as missing (work tracker ````). A cell that fits the
window but whose predicted p95 completion exceeds
CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S will trip the first-byte guard on a
large real turn, time out with status=0, record no completed sample, and
— under the even-split minimum-coverage quota — get re-targeted indefinitely
because it stays under its coverage floor forever.

This module promotes the forward time estimator (````) from a
soft scheduling tie-break to a HARD feasibility constraint on the FORCED-COVERAGE
path (cold-start random selection in router.py + quota forcing in
app.py). It stays a soft tie-break on the normal selection path (the
router's cost/quality pick), so a learned-optimal cell is never hard-excluded
for being slow — only forced coverage avoids it.

Cold-cell grace — the deliberate tradeoff named in ````: a
cell whose latency prediction rests on a model-pooled / global-prior / flat
fallback prior RATHER THAN its own measured fit is NOT hard-excluded on that
prior. The prior is uncertain, and hard-excluding on it would starve coverage
of exactly the cold cells we most need to sample (the ones with zero measured
rows). Such a cell stays eligible for one forced attempt; if it then times
out, the post-timeout cooldown in routing/quota.py skips it for the next
cycle so the doom loop still breaks. The cost is at most one wasted forced
turn per cold cell — the price of not starving coverage. The cell re-arms
only after it records a real completed sample.

The budget is the stall-guard FIRST-BYTE timeout (the deadline a large-context
prefill trips), not a total-completion cap: the stall guard has no total
deadline, only a first-byte and an inter-byte idle deadline. For the large real
turns that trigger the doom loop, prefill dominates total latency, so the
predicted p95 total is a conservative proxy for "will it make the first-byte
deadline" — a cell whose p95 total sits under the first-byte budget will
comfortably start streaming in time; one over it almost certainly will not.
Comparing the p95 (Estimate.high), not the p50, keeps the constraint on the
comfortable side the handoff asks for.
"""

from __future__ import annotations

from callosum.cell_grid import Cell
from callosum.routing.usage_estimate import (
    EstimateInput,
    OutputTokenForecaster,
    UsageEstimator,
)

# A latency prediction built on the cell's OWN measured fit (or an operator
# override) is trusted as a hard feasibility constraint. Anything else —
# model-pooled, global-prior, flat fallback — is a prior, not a measurement:
# cold-cell grace keeps such cells eligible. Matches the source stamps
# emitted by TimeModelProvider._resolve / _override_model.
_TRUSTED_SOURCES = ("cell-measured", "override")


def feasibility_eligible(
    cell: Cell,
    input_tokens: int,
    *,
    time_estimator: UsageEstimator | None,
    output_forecaster: OutputTokenForecaster | None,
    budget_s: float,
) -> bool:
    """True iff cell may be forced onto a request of input_tokens for
    coverage without a predicted stall-guard timeout.

    Returns True (eligible) when:

      * there is no estimator available — the router cold-starts without one,
        so feasibility must not gate (grace);
      * the cell's latency prediction is TRUSTED (cell-measured / override) AND
        its predicted p95 (Estimate.high) sits within budget_s — the
        cell is predicted to finish comfortably under the first-byte guard;
      * the prediction is UNTRUSTED (model-pooled / global-prior / fallback) —
        cold-cell grace: do not hard-exclude on an unmeasured prior.

    Returns False only for a TRUSTED measured fit whose p95 exceeds the
    budget — the one case where we confidently know the forced turn will time
    out, so forcing it would just burn a turn and record no sample.
    """
    if time_estimator is None or output_forecaster is None:
        return True
    forecast = output_forecaster.forecast(cell, input_tokens)
    estimate = time_estimator.estimate(EstimateInput(cell=cell, input_tokens=input_tokens, output=forecast))
    if not estimate.source.startswith(_TRUSTED_SOURCES):
        return True  # cold-cell grace — do not hard-exclude on a prior
    return estimate.high <= budget_s * 1000.0
