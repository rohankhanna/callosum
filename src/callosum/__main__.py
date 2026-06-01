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
from callosum.backends.local_direct import LocalModelRegistryBackend
from callosum.config import build_backends, load_config
from callosum.local import LocalModelRegistrySource
from callosum.operator_state import OperatorState
from callosum.usage_log import UsageLog


def _default_config_path() -> Path:
    return Path("~/.config/callosum/config.toml").expanduser()


def main() -> None:
    # Ensure our `callosum.startup` logger emits INFO + WARNING to stdout.
    # Without this, the root logger defaults to WARNING and the smoke-test
    # OK/SKIPPED lines get silently dropped — operator only sees FAILED.
    logging.getLogger("callosum.startup").setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        # Use UTC timestamps in ISO 8601 format per control plane compliance
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
    # Operator-state DB: persistent runtime knobs the CLI manages
    # (inference-param overrides today; denylist + mode + priority in
    # later steps). Lives alongside auth.sqlite under state.dir.
    operator_state = OperatorState(cfg.state.dir / "operator_state.sqlite")
    backends = build_backends(cfg)
    # Auto-register a LiteLLM gateway backend when the operator points us at
    # one. Gateway is provided by `local LLM gateway` and exposes local models
    # (ollama / vllm / model-a0e0 / etc.) behind one OpenAI-compatible URL.
    # Optional and additive: when CALLOSUM_LITELLM_GATEWAY_URL is unset (or
    # the gateway is unreachable), Codex backends remain authoritative.
    # Prefer local LLM gateway as the source of truth when its CLI is on
    # PATH. Falls back to LiteLLMGatewayBackend (litellm.yaml-driven)
    # when the operator doesn't have local LLM gateway installed.
    if (
        os.environ.get("CALLOSUM_LOCAL_DISABLED") != "1"
        and LocalModelRegistrySource.is_available()
    ):
        source = LocalModelRegistrySource()
        backends.append(
            LocalModelRegistryBackend(
                id="local LLM gateway",
                source=source,
                operator_state=operator_state,
            )
        )
    litellm_url = os.environ.get(
        "CALLOSUM_LITELLM_GATEWAY_URL", LITELLM_GATEWAY_DEFAULT_BASE_URL
    )
    if os.environ.get("CALLOSUM_LITELLM_GATEWAY_ENABLED") == "1":
        # CALLOSUM_LITELLM_TIMEOUT_S overrides the per-call timeout for the
        # local backend. Default is generous (300s) to absorb cold-load
        # latency on large local models; lower it on fast hardware or when
        # models are kept warm. Malformed values fall back to the default
        # rather than crashing the proxy at startup.
        litellm_timeout_raw = os.environ.get("CALLOSUM_LITELLM_TIMEOUT_S")
        try:
            litellm_timeout = (
                float(litellm_timeout_raw)
                if litellm_timeout_raw is not None
                else None
            )
        except ValueError:
            litellm_timeout = None
        backend_kwargs: dict[str, object] = {
            "id": "local LLM gateway",
            "base_url": litellm_url,
            "master_key": os.environ.get("CALLOSUM_LITELLM_MASTER_KEY"),
            "operator_state": operator_state,
        }
        if litellm_timeout is not None:
            backend_kwargs["timeout_s"] = litellm_timeout
        backends.append(LiteLLMGatewayBackend(**backend_kwargs))  # type: ignore[arg-type]
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
            operator_state=operator_state,
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
