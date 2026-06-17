from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from callosum.auth_vault import AuthVault
from callosum.backend import Backend
from callosum.backends.codex_auth_vault import (
    DEFAULT_BASE_URL as CODEX_AUTH_VAULT_DEFAULT_BASE_URL,
)
from callosum.backends.codex_auth_vault import CodexAuthVaultBackend
from callosum.routing.factory import RoutingConfig
from callosum.state import StateStore

BackendType = Literal["codex_auth_vault", "credential_proxy"]

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


class AutoRouterConfig(BaseModel):
    """Background synthetic-request worker for the auto-learning explorer.

    The PRIMARY controller is per-account weekly-exhaustion: every tick,
    for each account, fire synthetics at the rate that lands weekly at
    100% by reset, given projected human burn (last N hours of organic
    rate × safety margin). This honors the invariant "weekly capacity
    is never wasted" since paid-monthly weekly slots can't be regained.

    The FALLBACK controller is the original floor + pct + ceiling logic.
    It runs only when no quota snapshot is available yet (cold start —
    backend hasn't served a request yet, so we don't know weekly state),
    or as an absolute safety cap.

    All synthetic_* defaults are 0 (worker disabled). Set non-zero values
    to enable — either the floor (cold-start mode) or just plug it in and
    let the weekly-exhaustion controller do its thing once quota snapshots
    arrive.
    """

    model_config = ConfigDict(extra="forbid")

    # Cold-start fallback bounds (used when no quota_snapshot is available).
    # All synthetic_* defaults are 0 (worker disabled). Set non-zero values to enable.
    synthetic_floor_per_day: int = 0
    synthetic_pct_of_organic: float = 0.0
    synthetic_hard_ceiling_per_day: int = 0
    # How often the worker wakes to check whether to fire another synthetic.
    synthetic_check_interval_seconds: int = 300

    # Weekly-exhaustion controller knobs. Sensible defaults baked in.
    # `pct_per_synthetic_estimate`: cost of one synthetic request, in weekly%
    # points. Hand-set v1; learned by the cost model in v2.
    pct_per_synthetic_estimate: float = 0.1
    # `prediction_window_hours`: how far back to look when estimating organic
    # burn rate per account. 168h = 7d.
    prediction_window_hours: int = 168
    # `prediction_safety_margin`: multiplier on projected human burn so the
    # controller errs toward leaving the human room (1.20 = +20%).
    prediction_safety_margin: float = 1.20
    # Per-tick cap on synthetics fired (one tick interval). Spreads connection
    # load even if the math says fire many.
    max_synthetics_per_tick: int = 10
    # Stop firing on an account once weekly_used_percent crosses this — close
    # enough to 100 that we don't risk a 429 on a real human request.
    weekly_target_pct: float = 95.0
    # Pause firing on an account when 5h is near-exhausted (otherwise we'd
    # 429-loop until the 5h window rolls).
    five_hourly_pause_pct: float = 95.0
    # Aggressive-exhaustion mode: when weekly_used_percent meets this threshold,
    # bypass normal pacing and fire continuously until confirmed 429 from upstream.
    aggressive_exhaustion_pct: float = 98.0
    # How many consecutive 429 responses confirm real exhaustion (vs transient).
    aggressive_exhaustion_consecutive_429s: int = 3
    # Max requests per aggressive burst (safety cap per backend per tick).
    aggressive_exhaustion_max_per_burst: int = 50
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
    # defaults (noop embedding + uniform predictor + cost-weighted selector)
    # produce local-first cost-ordered routing without any ML deps. Phases
    # 4+ swap in BGE embeddings, a k-NN predictor, and the labeler.
    routing: RoutingConfig = Field(default_factory=lambda: RoutingConfig())

    # Arm-level exploration. When enabled, synthetic auto-learning traffic
    # (requested_model == "auto-learning-synthetic") targets the LEAST-sampled
    # compatible cell instead of the cost-cheapest one, so coverage
    # accumulates across the whole (model, reasoning_effort) grid. Organic
    # traffic is never affected. See callosum.routing.exploration.
    exploration_enabled: bool = True

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
    proxy_url: str | None = None
    upstream_url: str | None = None
    custody_account: str | None = None
    models: list[str] = Field(default_factory=list)
    codex_base_url: str = CODEX_AUTH_VAULT_DEFAULT_BASE_URL

    @model_validator(mode="after")
    def _validate_credential_source(self) -> BackendConfig:
        """Enforce exactly one of vault_path or proxy_url is set."""
        has_vault = self.vault_path is not None
        has_proxy = self.proxy_url is not None
        if has_vault and has_proxy:
            raise ValueError("cannot specify both vault_path and proxy_url")
        if self.type == "codex_auth_vault" and not has_vault:
            raise ValueError("codex_auth_vault requires vault_path")
        if self.type == "credential_proxy":
            if not has_proxy:
                raise ValueError("credential_proxy requires proxy_url")
            if not self.upstream_url:
                raise ValueError("credential_proxy requires upstream_url")
        return self


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    usage_log: UsageLogConfig = Field(default_factory=UsageLogConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    auto_router: AutoRouterConfig = Field(default_factory=AutoRouterConfig)
    backends: list[BackendConfig] = Field(default_factory=list)


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)


def build_backend(
    cfg: BackendConfig,
    *,
    state_store: StateStore | None = None,
) -> Backend:
    # `models` is an OPTIONAL cold-start hint, not a contract. Both
    # backend kinds discover their model catalog dynamically via
    # `refresh_advertised_models` at startup. Operator-provided lists
    # in config are useful only as a fallback when discovery hasn't
    # run yet or has failed — and even then, the smoke test will report
    # "no advertised_models" cleanly rather than crash. Hard-coding
    # current model names into config would defeat dynamic routing
    # and create maintenance churn each time OpenAI ships a new model.
    if cfg.type == "credential_proxy":
        from callosum.backends.credential_proxy import CredentialProxyBackend

        if cfg.proxy_url is None or cfg.upstream_url is None:
            raise ValueError(f"credential_proxy backend {cfg.id!r}: proxy_url and upstream_url are required")
        return CredentialProxyBackend(
            id=cfg.id,
            proxy_url=cfg.proxy_url,
            upstream_url=cfg.upstream_url,
            advertised_models=frozenset(cfg.models),
            custody_account=cfg.custody_account,
            state_store=state_store,
        )
    else:  # codex_auth_vault
        if cfg.vault_path is None:
            raise ValueError(f"codex_auth_vault backend {cfg.id!r}: vault_path is required")
        vault = AuthVault(path=cfg.vault_path)
        return CodexAuthVaultBackend(
            id=cfg.id,
            vault=vault,
            advertised_models=frozenset(cfg.models),
            base_url=cfg.codex_base_url,
            state_store=state_store,
        )


def build_backends(cfg: Config, *, env: Mapping[str, str] | None = None) -> list[Backend]:
    # `env` is accepted for signature compatibility with callers that used to
    # resolve API-key env vars here. Codex auth vaults are file-backed and do
    # not read the environment.
    del env
    state_store = StateStore(cfg.state.dir) if cfg.state.dir is not None else None
    return [build_backend(bc, state_store=state_store) for bc in cfg.backends]
