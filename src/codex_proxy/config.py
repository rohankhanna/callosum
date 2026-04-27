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
    """Background synthetic-request topper for the auto-learning explorer.

    Synthetic requests SUPPLEMENT organic auto-learning traffic; they never
    replace it. Two bounds, both active simultaneously:

    - `synthetic_floor_per_day`: minimum synthetics fired per UTC day, regardless
      of organic volume. Guarantees corpus velocity on quiet days.
    - `synthetic_pct_of_organic`: synthetics may grow up to this fraction of
      today's organic volume. Keeps the corpus from being dominated by
      synthetic prompts on busy days.
    - `synthetic_hard_ceiling_per_day`: absolute upper bound — quota safety net.

    Effective target per day ≈ min(hard_ceiling, max(floor, ceil(pct * organic))).

    All defaults are 0 (worker disabled). Set non-zero values to enable.
    """

    model_config = ConfigDict(extra="forbid")

    synthetic_floor_per_day: int = 0
    synthetic_pct_of_organic: float = 0.0
    synthetic_hard_ceiling_per_day: int = 0
    # How often the worker wakes to check whether to fire another synthetic.
    synthetic_check_interval_seconds: int = 300


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
