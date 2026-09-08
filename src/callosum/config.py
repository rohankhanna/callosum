from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from callosum.routing.factory import RoutingConfig

BackendType = Literal["codex_auth_vault", "codex_gateway"]

# Default upstream base URL for Codex auth vault backends.
CODEX_AUTH_VAULT_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"

# Behavioral stall guard for LOCAL streaming backends. Replaces the old
# size-based pre-flight cap (MAX_LOCAL_TOOL_REQUEST_BYTES): a hardcoded byte
# count is the wrong unit — whether a request fits is a function of the chosen
# model's context window, not a constant, and the codebase already routes on
# that (soft window-fit in router.py + the empirical `tool_call_at_scale`
# capability gate in capability.py). The byte cap existed only to fail fast on
# the OTHER failure mode: some local runtimes accept a large tool request, then
# emit no bytes until the full transport timeout, so local-only "looks like a
# loop". We catch that behaviorally instead — give the model a fair budget to
# start (cold load + prefill), then bound the gap between tokens. A genuine
# hang fails fast and `transient` (routing falls back to another cell); a large
# request the model CAN serve just streams. Both env-tunable.
#
# FIRST_BYTE: max wait for the first streamed byte (covers cold weight-load +
# prefill of a big prompt). IDLE: max gap between subsequent bytes once the
# model is producing (a working model emits tokens ms apart; a long gap means a
# stall). Tune up for slow hardware / very large prompts.
LOCAL_STREAM_FIRST_BYTE_TIMEOUT_S = float(os.getenv("CALLOSUM_LOCAL_FIRST_BYTE_TIMEOUT_S", "180"))
LOCAL_STREAM_IDLE_TIMEOUT_S = float(os.getenv("CALLOSUM_LOCAL_STREAM_IDLE_TIMEOUT_S", "45"))


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8765
    client_auth_token_env: str | None = None
    control_auth_token_env: str | None = None
    # On startup, probe each non-cooldown backend with one minimal upstream
    # call so the operator sees auth health (or weekly-exhaustion, or any
    # other backend issue) immediately in the launch log instead of finding
    # out via failed user requests later. Each probe burns a few hundred
    # tokens of quota — disable if you restart often.
    startup_smoke_test: bool = True
    # In addition to the startup pass, re-run the smoke test on this interval
    # so operators see live backend state without restarting (e.g. when a
    # weekly window resets or auth gets refreshed externally). Set to 0 to
    # disable the periodic re-run; the startup pass still happens. Default
    # 3600 (1h) — matches the model-list refresh cadence and keeps quota cost
    # bounded.
    smoke_test_interval_seconds: int = 3600


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path | None = None


class UsageLogConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    # Default-on during the consumption-modeling phase: capture request and
    # response payloads plus upstream headers so we can correlate tokens +
    # reasoning effort with Δquota. Flip to false once the model is trained.
    capture_bodies: bool = True


class AuthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # When set, /auth/* registration + login endpoints become available, and
    # /v1/* requests must carry `Authorization: Bearer <api-key>` issued via
    # the auth flow. When unset, auth is disabled (single-operator mode).
    db: Path | None = None
    # How long a session token (returned by /auth/login) stays valid.
    session_ttl_seconds: int = 1800


class CodexCatalogConfig(BaseModel):
    """Project the live Callosum catalog into a codex `/model` picker file.

    codex's interactive `/model` picker is driven by a model catalog override
    declared in `~/.codex/config.toml` as `model_catalog_json = "<path>"`, not
    by the provider's `/v1/models`. When enabled, Callosum re-emits that file
    from the same catalog `/v1/models` serves, so the picker lists and switches
    between Callosum lanes from a single config file. See
    callosum.codex_catalog and.
    """

    model_config = ConfigDict(extra="forbid")

    # Off by default: emitting the file is only useful when a codex config on
    # this host points `model_catalog_json` at `output_path`. Operators opt in.
    enabled: bool = False
    # Where the codex-shaped catalog file is written. The codex config's
    # `model_catalog_json` must point at this same absolute path.
    output_path: Path = Path.home() / ".codex" / "callosum-catalog.json"
    # codex binary used to source a rich ModelInfo template (run under a
    # disposable CODEX_HOME so it yields codex's bundled default catalog).
    codex_bin: str = "codex"
    # Safety re-emit cadence (the reconciler also re-emits on catalog-hash
    # change and on startup). Seconds.
    refresh_interval_seconds: int = 1800
    # Operator-declared lanes that should appear in the picker even when no
    # backend serves them yet (e.g. "callosum:remote/model-a0e8:high"). Selecting
    # one surfaces a clean "not available yet" message at dispatch.
    declared_lanes: list[str] = Field(default_factory=list)


class AutoRouterConfig(BaseModel):
    """Auto-router configuration: the learning-router pipeline plus live
    operational knobs (context-safety margin, cooldown prober, cost/time
    estimators).

    The former synthetic background-topper worker (auto-learning-synthetic)
    has been removed (); its only surviving piece is the
    coverage machinery (CellCoverage / coverage_order), now reused by
    the per-cell minimum-coverage quota. See
    docs/architecture/min_coverage_quota.md.
    """

    model_config = ConfigDict(extra="forbid")

    # Request-scoped hard cap for dispatch retry/failover.
    dispatch_retry_budget_seconds: float = 360.0
    # Shared cap on backend attempts across all cells for one user request.
    dispatch_retry_max_backend_attempts: int = 4

    # Context-safe routing: min headroom (tokens) to leave above current session size
    # when picking a model. Prevents routing to models with insufficient context.
    router_context_safety_margin: int = 8192
    # How often the cooldown prober re-probes any backend whose persisted
    # cooldown_until_ts is in the future. The probe bypasses the cooldown-
    # skip guard and clears the cooldown if the probe succeeds — the only
    # way the proxy can self-heal from a stale-snapshot lockout (e.g.
    # upstream's reported weekly_reset_at was wrong, account was topped up
    # out of band, original 429 was transient). 0 to disable.
    cooldown_probe_interval_seconds: int = 3600

    # Learning router. Pluggable pipeline:
    #   features → capability filter → quality predict → cost-weighted select
    # Each stage is a Protocol with swappable implementations. Cold-start
    # defaults (uniform predictor + cost-weighted selector) produce
    # local-first cost-ordered routing without any ML deps. The learned
    # predictor is `cell_majority_prior` (a per-cell majority-baseline
    # prior that ignores prompt embeddings); an embedding/KNN predictor
    # was evaluated and removed.
    routing: RoutingConfig = Field(default_factory=lambda: RoutingConfig())

    # Per-cell minimum-coverage quota (): a deterministic
    # routing rule, not a self-tuning system. Every compatible
    # (model, reasoning_effort) cell, within its lane, must receive at least
    # its even split of a fixed budget over a rolling window, so quality
    # labels keep accruing across the whole grid instead of decaying on the
    # cells the normal cost/quality pick would starve. When a cell's recent
    # usage is below its floor the router steers one turn to the least-used
    # eligible cell; otherwise routing is untouched. See
    # docs/architecture/min_coverage_quota.md.
    min_coverage_quota_enabled: bool = False
    # Total coverage budget: the fraction of recent traffic reserved for the
    # per-cell floor, split EVENLY across the lane's candidate cells, so each
    # cell's effective floor is `min_coverage_budget_pct / n_candidates`. This
    # keeps total forced coverage bounded by the budget no matter how many
    # cells the grid grows to — a flat per-cell floor is only feasible while
    # n <= 1/floor and silently crowds out the normal pick as the grid grows.
    # At n=10 a 0.10 budget reproduces the old flat 1% per cell.
    min_coverage_budget_pct: float = 0.10
    # Optional absolute minimum per-cell floor; 0 disables (pure even split).
    # Safety net so the even-split floor never decays to ~0 on very large grids.
    # When set above budget/n for some cell it overrides the even split there
    # (and total forced coverage can then exceed the budget — opt-in).
    min_coverage_floor_pct: float = 0.0
    # Window over which a cell's share is measured. 30 days.
    min_coverage_window_seconds: int = 2_592_000

    # Coverage feasibility (): a cell is forced onto a
    # coverage turn only when the forward time estimator predicts it can
    # FINISH within the stall-guard first-byte budget — otherwise the forced
    # turn times out (status 0), records no completed sample, and the cell
    # stays under its floor forever, so the even-split quota re-targets it
    # indefinitely (the doom loop). Promotes the time estimator from a soft
    # scheduling tie-break to a hard constraint on the FORCED-COVERAGE path
    # only; the normal (cost/quality) selection path keeps it as a soft
    # tie-break. Cold cells with no measured latency fit stay eligible (grace)
    # so coverage is not starved of the cells it most needs to sample. Disable
    # to fall back to the pre-fix window-fit-only filter (escape hatch). See
    # routing/feasibility.py.
    min_coverage_feasibility_enabled: bool = True
    # Post-timeout cooldown (the second half of): a cell that
    # just timed out on a forced-coverage turn is skipped by the quota for
    # this window so it is not immediately re-targeted. Re-arms only after the
    # cell records a real completed sample. The backstop to feasibility —
    # bounds a cold cell's wasted forced turns to one. See
    # cell_grid.recent_quota_cooldown_cells.
    min_coverage_cooldown_enabled: bool = True
    # How far back a failed forced-coverage turn keeps a cell cooled. Short
    # by design — just enough to skip the next selection cycle; the cell
    # re-arms as soon as it completes any sample. 10 min.
    min_coverage_cooldown_window_seconds: int = 600

    # Guardrail for reasoning-cost blowups: cap the highest-severity reasoning
    # tiers so automatic routing keeps the expensive levels rare. The cap
    # targets the top-N canonical reasoning efforts BY SEVERITY RANK (the last
    # N rungs of the low<medium<high<xhigh ladder), not by name — so it
    # generalizes if the ladder grows and never regulates the cheap tiers.
    # Efforts outside the canonical ladder (e.g. an ollama-cloud model's
    # default) are never capped, and models served by exempt backend kinds
    # (ollama-cloud) are immune outright: their cells are never dropped and
    # excluded from the share denominator. Explicit
    # callosum:<source>/<model>:<effort> pins bypass this filter entirely.
    effort_cap_enabled: bool = True
    effort_cap_pct: float = 0.01
    effort_cap_top_n: int = 2
    # Seven days keeps the cap aligned with the upstream weekly quota window.
    effort_cap_window_seconds: int = 604_800

    # Dynamic per-model cost_rank derived from MEASURED weekly-quota burn
    # (weekly_used_percent deltas in the request log), replacing the flat
    # remote constant. Catalog priority is the cold-start prior; the override
    # map wins outright. See callosum.routing.cost_model.
    cost_rank_dynamic_enabled: bool = True
    # Minimum non-zero-delta observations before a model's measured mean is
    # trusted (integer-% resolution means most rows log a 0 delta).
    cost_rank_min_nonzero_samples: int = 10
    # Only request-log rows newer than this feed the rank (bounds the scan and
    # tracks the current quota regime). 30 days.
    cost_rank_window_seconds: int = 2_592_000
    # Cheapest measured remote model starts here; local cells stay at 0.
    cost_rank_base: int = 10
    # How often the rank map is recomputed (kept off the hot path).
    cost_rank_refresh_seconds: int = 3600
    # Operator overrides: {model_slug: cost_rank}. Wins over measured + prior.
    cost_rank_overrides: dict[str, int] = Field(default_factory=dict)

    # Forward cost estimator (): per-request prediction of
    # ChatGPT weekly-quota burn in weekly_used_percent points, from the same
    # measured deltas the cost_rank reads. Feeds /status pre-flight ranges and
    # the routing reward cost term. See callosum.routing.cost_estimator.
    cost_estimate_enabled: bool = True
    # Minimum non-zero-delta rows before a cell's/model's measured per-token
    # burn rate is trusted (integer-% resolution → most rows log a 0 delta).
    cost_estimate_min_nonzero_samples: int = 10
    # Only rows newer than this feed the rate (bounds the scan, tracks the
    # current quota regime). 30 days.
    cost_estimate_window_seconds: int = 2_592_000
    # Flat per-token burn used only when the log has no usable measured signal
    # at all (weekly_used_percent points per token).
    cost_estimate_fallback_rate: float = 5e-6
    # How often the per-cell rate map is recomputed (kept off the hot path).
    cost_estimate_refresh_seconds: int = 3600
    # Operator overrides: {model_slug: pct_per_token}. Wins over measured + prior.
    cost_estimate_overrides: dict[str, float] = Field(default_factory=dict)
    # Forward time estimator (): per-request prediction of
    # wall-clock latency in ms for BOTH remote and local cells (local is NOT
    # zero — it is often the slow path), fit t ≈ a·input + b·output + c against
    # the same request log's latency_ms. Feeds a future pre-flight ETA range
    # and data-driven per-cell timeout tuning. See
    # callosum.routing.time_estimator.
    time_estimate_enabled: bool = True
    # Minimum timing rows before a cell's/model's measured fit is trusted
    # (latency has no integer-resolution problem, so these are total samples,
    # not non-zero deltas — contrast cost_estimate_min_nonzero_samples).
    time_estimate_min_samples: int = 10
    # Only rows newer than this feed the fit (bounds the scan, tracks the
    # current latency regime). 30 days.
    time_estimate_window_seconds: int = 2_592_000
    # Flat per-token slope (ms/token) and fixed overhead (ms) used only when
    # the log has no usable measured signal at all.
    time_estimate_fallback_ms_per_token: float = 12.0
    time_estimate_fallback_base_ms: float = 500.0
    # Cold-start slowdown applied to a remote-dominated global prior when it is
    # reused for a LOCAL cell ("local is the slow path"); yields to the cell's
    # own measured fit once it has data. The time analogue of the cost
    # estimator's catalog-priority tilt.
    time_estimate_local_slowdown: float = 4.0
    # How often the per-cell fit is recomputed (kept off the hot path).
    time_estimate_refresh_seconds: int = 3600
    # Operator overrides: {model_slug: [ms_per_token, base_ms]}. Wins outright.
    time_estimate_overrides: dict[str, list[float]] = Field(default_factory=dict)
    # Shared output-token forecaster (): cold-start global
    # output:input ratio yields to per-cell measured ratios after this many
    # observed rows. Consumed by BOTH the cost and time estimators.
    output_forecast_min_obs: int = 20
    # Flat output:input ratio used when the log has no measured rows at all.
    output_forecast_fallback_ratio: float = 1.0


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: BackendType = "codex_auth_vault"
    vault_path: Path | None = None
    models: list[str] = Field(default_factory=list)
    codex_base_url: str = CODEX_AUTH_VAULT_DEFAULT_BASE_URL
    api_key: str = ""
    base_url: str = CODEX_AUTH_VAULT_DEFAULT_BASE_URL

    @model_validator(mode="after")
    def _validate_credential_source(self) -> BackendConfig:
        """Enforce credential fields are set per backend type."""
        if self.type == "codex_auth_vault" and not self.vault_path:
            raise ValueError("codex_auth_vault requires vault_path")
        if self.type == "codex_gateway" and not self.api_key:
            raise ValueError("codex_gateway requires api_key")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    usage_log: UsageLogConfig = Field(default_factory=UsageLogConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    auto_router: AutoRouterConfig = Field(default_factory=AutoRouterConfig)
    codex_catalog: CodexCatalogConfig = Field(default_factory=CodexCatalogConfig)
    backends: list[BackendConfig] = Field(default_factory=list)


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        data = tomllib.load(f)

    return Config.model_validate(data)
