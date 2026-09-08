from __future__ import annotations

from collections.abc import Mapping

from callosum.auth_vault import AuthVault
from callosum.backend import Backend
from callosum.config import BackendConfig, Config
from callosum.state import StateStore


def build_backend(
    cfg: BackendConfig,
    *,
    state_store: StateStore | None = None,
) -> Backend:
    """Instantiate a backend from its configuration entry.

    The models field is an optional cold-start hint, not a contract.
    The backend discovers its model catalog dynamically via
    refresh_advertised_models at startup. Operator-provided lists in
    config are useful only as a fallback when discovery has not run yet
    or has failed.
    """
    from callosum.backends.codex_auth_vault import CodexAuthVaultBackend

    if cfg.type == "codex_gateway":
        from callosum.backends.codex_gateway import CodexGatewayBackend

        return CodexGatewayBackend(
            id=cfg.id,
            api_key=cfg.api_key,
            advertised_models=frozenset(cfg.models),
            base_url=cfg.base_url,
            state_store=state_store,
        )

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
    """Build all backends declared in the configuration.

    env is accepted for signature compatibility with callers that used
    to resolve API-key env vars here. Codex auth vaults are file-backed and
    do not read the environment.
    """
    del env
    state_store = StateStore(cfg.state.dir) if cfg.state.dir is not None else None
    return [build_backend(bc, state_store=state_store) for bc in cfg.backends]
