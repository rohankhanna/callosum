from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from codex_proxy.config import (
    BackendConfig,
    Config,
    PolicyConfig,
    build_backend,
    build_backends,
    load_config,
)


def test_load_minimal_config(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[[backends]]
id = "primary"
type = "openai_api_key"
api_key_env = "OPENAI_API_KEY"
models = ["model-a0f5-mini"]
"""
    )
    cfg = load_config(path)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8765
    assert cfg.policy.default_mode == "stateless"
    assert cfg.policy.allow_consumer_auth_backends is False
    assert cfg.state.dir is None
    assert len(cfg.backends) == 1
    bc = cfg.backends[0]
    assert bc.id == "primary"
    assert bc.type == "openai_api_key"
    assert bc.api_key_env == "OPENAI_API_KEY"
    assert bc.models == ["model-a0f5-mini"]


def test_load_config_with_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[state]
dir = "{state_dir}"

[[backends]]
id = "primary"
type = "openai_api_key"
api_key_env = "OPENAI_API_KEY"
models = ["model-a0f5-mini"]
"""
    )
    cfg = load_config(path)
    assert cfg.state.dir == state_dir


def test_load_rejects_unknown_backend_type(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
[[backends]]
id = "x"
type = "totally_made_up"
"""
    )
    with pytest.raises(ValidationError):
        load_config(path)


async def test_build_backend_reads_api_key_from_env() -> None:
    bc = BackendConfig(id="p", type="openai_api_key", api_key_env="MY_KEY", models=["m"])
    backend = build_backend(bc, policy=PolicyConfig(), env={"MY_KEY": "sk-abc"})
    try:
        assert backend.id == "p"
        assert "m" in backend.advertised_models
    finally:
        await backend.aclose()


def test_build_backend_errors_when_env_missing() -> None:
    bc = BackendConfig(id="p", type="openai_api_key", api_key_env="MISSING", models=["m"])
    with pytest.raises(ValueError, match="MISSING"):
        build_backend(bc, policy=PolicyConfig(), env={})


def test_build_backend_requires_api_key_env_for_openai() -> None:
    bc = BackendConfig(id="p", type="openai_api_key", models=["m"])
    with pytest.raises(ValueError, match="api_key_env"):
        build_backend(bc, policy=PolicyConfig(), env={})


async def test_build_backend_constructs_azure_backend() -> None:
    bc = BackendConfig(
        id="azure-main",
        type="azure_openai",
        api_key_env="AZURE_KEY",
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
    )
    backend = build_backend(bc, policy=PolicyConfig(), env={"AZURE_KEY": "az-secret"})
    try:
        assert backend.id == "azure-main"
        assert backend.kind == "azure_openai"
        assert "model-a0f5-mini" in backend.advertised_models
    finally:
        await backend.aclose()


def test_build_azure_backend_requires_endpoint() -> None:
    bc = BackendConfig(
        id="azure",
        type="azure_openai",
        api_key_env="K",
        api_version="2024-10-01-preview",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
    )
    with pytest.raises(ValueError, match="endpoint"):
        build_backend(bc, policy=PolicyConfig(), env={"K": "az"})


def test_build_azure_backend_requires_api_version() -> None:
    bc = BackendConfig(
        id="azure",
        type="azure_openai",
        api_key_env="K",
        endpoint="https://example.openai.azure.com",
        deployments={"model-a0f5-mini": "model-a0f5-mini-prod"},
    )
    with pytest.raises(ValueError, match="api_version"):
        build_backend(bc, policy=PolicyConfig(), env={"K": "az"})


def test_build_azure_backend_requires_deployments() -> None:
    bc = BackendConfig(
        id="azure",
        type="azure_openai",
        api_key_env="K",
        endpoint="https://example.openai.azure.com",
        api_version="2024-10-01-preview",
    )
    with pytest.raises(ValueError, match="deployments"):
        build_backend(bc, policy=PolicyConfig(), env={"K": "az"})


def test_build_backend_rejects_codex_auth_vault_without_gate() -> None:
    bc = BackendConfig(id="p", type="codex_auth_vault", models=["m"])
    with pytest.raises(ValueError, match="consumer-auth"):
        build_backend(bc, policy=PolicyConfig(allow_consumer_auth_backends=False), env={})


def test_build_backend_errors_on_codex_auth_vault_even_when_gated() -> None:
    bc = BackendConfig(id="p", type="codex_auth_vault", models=["m"])
    with pytest.raises(NotImplementedError):
        build_backend(bc, policy=PolicyConfig(allow_consumer_auth_backends=True), env={})


async def test_build_backends_iterates() -> None:
    cfg = Config(
        backends=[
            BackendConfig(id="a", type="openai_api_key", api_key_env="K1", models=["m"]),
            BackendConfig(id="b", type="openai_api_key", api_key_env="K2", models=["m"]),
        ]
    )
    backends = build_backends(cfg, env={"K1": "sk-1", "K2": "sk-2"})
    try:
        assert [b.id for b in backends] == ["a", "b"]
    finally:
        for b in backends:
            await b.aclose()
