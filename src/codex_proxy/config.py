from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codex_proxy.auth_vault import AuthVault
from codex_proxy.backend import Backend
from codex_proxy.backends.azure_openai import AzureOpenAIBackend
from codex_proxy.backends.codex_auth_vault import (
    DEFAULT_BASE_URL as CODEX_AUTH_VAULT_DEFAULT_BASE_URL,
)
from codex_proxy.backends.codex_auth_vault import CodexAuthVaultBackend
from codex_proxy.backends.openai_api_key import OpenAIApiKeyBackend
from codex_proxy.state import StateStore

BackendType = Literal["openai_api_key", "azure_openai", "codex_auth_vault"]
SessionMode = Literal["stateless", "sticky", "sticky_replay"]


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8765
    client_auth_token_env: str | None = None
    control_auth_token_env: str | None = None


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    default_mode: SessionMode = "stateless"
    allow_consumer_auth_backends: bool = False


class StateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: Path | None = None


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: BackendType
    api_key_env: str | None = None
    base_url: str = "https://api.openai.com/v1"
    models: list[str] = Field(default_factory=list)
    endpoint: str | None = None
    api_version: str | None = None
    deployments: dict[str, str] = Field(default_factory=dict)
    vault_path: Path | None = None
    codex_base_url: str = CODEX_AUTH_VAULT_DEFAULT_BASE_URL


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    state: StateConfig = Field(default_factory=StateConfig)
    backends: list[BackendConfig] = Field(default_factory=list)


def load_config(path: Path) -> Config:
    with path.open("rb") as f:
        data = tomllib.load(f)
    return Config.model_validate(data)


def build_backend(
    cfg: BackendConfig,
    *,
    policy: PolicyConfig,
    env: Mapping[str, str] | None = None,
    state_store: StateStore | None = None,
) -> Backend:
    resolved_env = env if env is not None else os.environ
    if cfg.type == "codex_auth_vault" and not policy.allow_consumer_auth_backends:
        raise ValueError(
            f"backend {cfg.id!r}: consumer-auth backends are disabled; "
            "set policy.allow_consumer_auth_backends = true to enable."
        )
    if cfg.type == "openai_api_key":
        if cfg.api_key_env is None:
            raise ValueError(f"backend {cfg.id!r}: api_key_env is required for openai_api_key")
        api_key = resolved_env.get(cfg.api_key_env)
        if not api_key:
            raise ValueError(f"backend {cfg.id!r}: env var {cfg.api_key_env!r} is not set")
        return OpenAIApiKeyBackend(
            id=cfg.id,
            api_key=api_key,
            advertised_models=frozenset(cfg.models),
            base_url=cfg.base_url,
            state_store=state_store,
        )
    if cfg.type == "codex_auth_vault":
        if cfg.vault_path is None:
            raise ValueError(f"backend {cfg.id!r}: vault_path is required for codex_auth_vault")
        if not cfg.models:
            raise ValueError(f"backend {cfg.id!r}: models is required for codex_auth_vault")
        vault = AuthVault(path=cfg.vault_path)
        return CodexAuthVaultBackend(
            id=cfg.id,
            vault=vault,
            advertised_models=frozenset(cfg.models),
            base_url=cfg.codex_base_url,
            state_store=state_store,
        )
    if cfg.type == "azure_openai":
        if cfg.endpoint is None:
            raise ValueError(f"backend {cfg.id!r}: endpoint is required for azure_openai")
        if cfg.api_version is None:
            raise ValueError(f"backend {cfg.id!r}: api_version is required for azure_openai")
        if not cfg.deployments:
            raise ValueError(f"backend {cfg.id!r}: deployments is required for azure_openai")
        if cfg.api_key_env is None:
            raise ValueError(f"backend {cfg.id!r}: api_key_env is required for azure_openai")
        api_key = resolved_env.get(cfg.api_key_env)
        if not api_key:
            raise ValueError(f"backend {cfg.id!r}: env var {cfg.api_key_env!r} is not set")
        return AzureOpenAIBackend(
            id=cfg.id,
            endpoint=cfg.endpoint,
            api_key=api_key,
            api_version=cfg.api_version,
            deployments=cfg.deployments,
            state_store=state_store,
        )
    raise ValueError(f"backend {cfg.id!r}: unsupported type {cfg.type!r}")


def build_backends(cfg: Config, *, env: Mapping[str, str] | None = None) -> list[Backend]:
    state_store = StateStore(cfg.state.dir) if cfg.state.dir is not None else None
    return [
        build_backend(bc, policy=cfg.policy, env=env, state_store=state_store)
        for bc in cfg.backends
    ]
