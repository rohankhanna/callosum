from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import uvicorn

from callosum.app import create_app
from callosum.auth import AuthService
from callosum.auth_db import AuthDB
from callosum.backends.litellm_gateway import (
    DEFAULT_BASE_URL as LITELLM_GATEWAY_DEFAULT_BASE_URL,
)
from callosum.backends.litellm_gateway import LiteLLMGatewayBackend
from callosum.backends.openrouter_free import OpenRouterFreeBackend
from callosum.config import build_backends, load_config
from callosum.usage_log import UsageLog


def _default_config_path() -> Path:
    return Path("~/.config/callosum/config.toml").expanduser()


def main() -> None:
    # Ensure our `callosum.startup` logger emits INFO + WARNING to stdout.
    # Without this, the root logger defaults to WARNING and the smoke-test
    # OK/SKIPPED lines get silently dropped — operator only sees FAILED.
    logging.getLogger("callosum.startup").setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        # Use UTC timestamps in ISO 8601 format per polestar compliance
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        # Convert to UTC
        logging.Formatter.converter = lambda *args: datetime.now(timezone.utc).timetuple()

    parser = argparse.ArgumentParser(prog="callosum")
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
    # Auto-register a LiteLLM gateway backend when the operator points us at
    # one. Gateway is provided by `local LLM gateway` and exposes local models
    # (ollama / vllm / model-a0e0 / etc.) behind one OpenAI-compatible URL.
    # Optional and additive: when CALLOSUM_LITELLM_GATEWAY_URL is unset (or
    # the gateway is unreachable), Codex backends remain authoritative.
    litellm_url = os.environ.get(
        "CALLOSUM_LITELLM_GATEWAY_URL", LITELLM_GATEWAY_DEFAULT_BASE_URL
    )
    if os.environ.get("CALLOSUM_LITELLM_GATEWAY_ENABLED") == "1":
        backends.append(
            LiteLLMGatewayBackend(
                id="local LLM gateway",
                base_url=litellm_url,
                master_key=os.environ.get("CALLOSUM_LITELLM_MASTER_KEY"),
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

    # Use uvicorn's signal handling configuration and explicitly set to allow shutdown
    uvicorn.run(
        create_app(
            backends=backends,
            usage_log=usage_log,
            auth_service=auth_service,
            auto_router_config=cfg.auto_router,
            startup_smoke_test=cfg.server.startup_smoke_test,
            smoke_test_interval_seconds=cfg.server.smoke_test_interval_seconds,
        ),
        host=host,
        port=port,
        # Ensure signals are properly handled
        access_log=False,
        # Use the default signal handlers (don't override them)
        # This lets uvicorn handle SIGINT/SIGTERM properly
    )


if __name__ == "__main__":
    main()
