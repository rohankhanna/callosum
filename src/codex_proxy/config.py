from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from codex_proxy.backend import Backend
from codex_proxy.backends.openai_api_key import OpenAIApiKeyBackend

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


class BackendConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: BackendType
    api_key_env: str | None = None
    base_url: str = "https://api.openai.com/v1"
    models: list[str] = Field(default_factory=list)


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
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
        )
    raise NotImplementedError(f"backend type {cfg.type!r} is not yet supported")


def build_backends(cfg: Config, *, env: Mapping[str, str] | None = None) -> list[Backend]:
    return [build_backend(bc, policy=cfg.policy, env=env) for bc in cfg.backends]
