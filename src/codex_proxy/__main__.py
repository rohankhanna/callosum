from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from codex_proxy.app import create_app
from codex_proxy.auth import AuthService
from codex_proxy.auth_db import AuthDB
from codex_proxy.backends.openrouter_free import OpenRouterFreeBackend
from codex_proxy.config import build_backends, load_config
from codex_proxy.usage_log import UsageLog


def _default_config_path() -> Path:
    return Path("~/.config/codex-proxy/config.toml").expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(prog="codex-proxy")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"config not found: {args.config}")

    cfg = load_config(args.config)
    backends = build_backends(cfg)
    # Auto-register OpenRouter free-tier as a fallback backend if the user
    # has set OPENROUTER_API_KEY. No TOML edit required — the backend
    # auto-discovers models from OpenRouter's catalog and picks per request.
    # Shadow-advertise whatever the other (Codex) backends currently advertise,
    # re-evaluated on every access so dynamic model discovery propagates.
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    if openrouter_key:
        codex_backends = list(backends)  # snapshot of non-OpenRouter backends

        def _shadow_models_now() -> frozenset[str]:
            return frozenset(m for b in codex_backends for m in b.advertised_models)

        backends.append(
            OpenRouterFreeBackend(
                id="openrouter-free",
                api_key=openrouter_key,
                shadow_models=_shadow_models_now,
            )
        )
    usage_log = (
        UsageLog(cfg.usage_log.path, capture_bodies=cfg.usage_log.capture_bodies)
        if cfg.usage_log.path is not None
        else None
    )
    auth_service: AuthService | None = None
    if cfg.auth.db is not None:
        auth_service = AuthService(
            AuthDB(cfg.auth.db),
            session_ttl_seconds=cfg.auth.session_ttl_seconds,
        )
    host = args.host if args.host is not None else cfg.server.host
    port = args.port if args.port is not None else cfg.server.port
    uvicorn.run(
        create_app(
            backends=backends,
            usage_log=usage_log,
            auth_service=auth_service,
            auto_router_config=cfg.auto_router,
            startup_smoke_test=cfg.server.startup_smoke_test,
        ),
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
