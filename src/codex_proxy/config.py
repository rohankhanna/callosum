from __future__ import annotations

import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codex_proxy.auth_vault import AuthVault
from codex_proxy.backend import Backend
from codex_proxy.backends.codex_auth_vault import (
    DEFAULT_BASE_URL as CODEX_AUTH_VAULT_DEFAULT_BASE_URL,
)
from codex_proxy.backends.codex_auth_vault import CodexAuthVaultBackend
from codex_proxy.state import StateStore

BackendType = Literal["codex_auth_vault"]


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


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: BackendType = "codex_auth_vault"
    vault_path: Path
    models: list[str] = Field(default_factory=list)
    codex_base_url: str = CODEX_AUTH_VAULT_DEFAULT_BASE_URL


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
    if not cfg.models:
        raise ValueError(f"backend {cfg.id!r}: models is required")
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
