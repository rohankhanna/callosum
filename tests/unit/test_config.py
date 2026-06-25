from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from callosum.config import (
    BackendConfig,
    Config,
    build_backend,
    build_backends,
    load_config,
)


def _write_vault(tmp_path: Path, *, name: str = "auth.json") -> Path:
    vault = tmp_path / name
    vault.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "a",
                    "refresh_token": "r",
                    "account_id": "acct",
                }
            }
        )
    )
    return vault


def test_load_minimal_config(tmp_path: Path) -> None:
    vault = _write_vault(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[[backends]]
id = "primary"
vault_path = "{vault}"
models = ["model-a0d0"]
"""
    )
    cfg = load_config(path)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8765
    assert cfg.state.dir is None
    assert len(cfg.backends) == 1
    bc = cfg.backends[0]
    assert bc.id == "primary"
    assert bc.type == "codex_auth_vault"
    assert bc.vault_path == vault
    assert bc.models == ["model-a0d0"]
    # Live auto-router defaults survive (synthetic-worker knobs removed —
    # ).
    assert cfg.auto_router.router_context_safety_margin == 8192
    assert cfg.auto_router.cost_rank_dynamic_enabled is True


def test_load_config_with_state_dir(tmp_path: Path) -> None:
    vault = _write_vault(tmp_path)
    state_dir = tmp_path / "state"
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[state]
dir = "{state_dir}"

[[backends]]
id = "primary"
vault_path = "{vault}"
models = ["model-a0d0"]
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
vault_path = "/tmp/nope.json"
"""
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_load_rejects_legacy_policy_section(tmp_path: Path) -> None:
    # The [policy] block used to gate consumer-auth backends and set a default
    # session mode. It was removed; configs that still declare it should fail
    # validation rather than silently load.
    vault = _write_vault(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text(
        f"""
[policy]
default_mode = "sticky"

[[backends]]
id = "primary"
vault_path = "{vault}"
models = ["model-a0d0"]
"""
    )
    with pytest.raises(ValidationError):
        load_config(path)


def test_backend_config_requires_vault_path() -> None:
    with pytest.raises(ValidationError):
        BackendConfig(id="p", models=["model-a0d0"])  # type: ignore[call-arg]


def test_build_backend_accepts_missing_models(tmp_path: Path) -> None:
    """`models` in TOML is an OPTIONAL cold-start hint. Both backend
    kinds discover their catalog dynamically via
    refresh_advertised_models. Hard-coding a static list in config
    defeated dynamic routing and forced operator churn each time
    OpenAI shipped a model. build_backend used to reject empty
    `models`; this test pins the relaxation."""
    vault = _write_vault(tmp_path)
    bc = BackendConfig(id="p", vault_path=vault)
    backend = build_backend(bc)
    assert backend.advertised_models == frozenset()


async def test_build_backend_constructs_codex_auth_vault(tmp_path: Path) -> None:
    vault = _write_vault(tmp_path)
    bc = BackendConfig(id="vault", vault_path=vault, models=["model-a0d0"])
    backend = build_backend(bc)
    try:
        assert backend.id == "vault"
        assert backend.kind == "codex_auth_vault"
        assert "model-a0d0" in backend.advertised_models
    finally:
        await backend.aclose()


async def test_build_backends_iterates(tmp_path: Path) -> None:
    vault_a = _write_vault(tmp_path, name="a.json")
    vault_b = _write_vault(tmp_path, name="b.json")
    cfg = Config(
        backends=[
            BackendConfig(id="a", vault_path=vault_a, models=["model-a0d0"]),
            BackendConfig(id="b", vault_path=vault_b, models=["model-a0d0"]),
        ]
    )
    backends = build_backends(cfg)
    try:
        assert [b.id for b in backends] == ["a", "b"]
    finally:
        for b in backends:
            await b.aclose()
