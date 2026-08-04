from __future__ import annotations

import argparse
import logging
import os
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from callosum.app import create_app
from callosum.auth import AuthService
from callosum.auth_db import AuthDB
from callosum.backend import Backend
from callosum.backends.litellm_gateway import (
    DEFAULT_BASE_URL as LITELLM_GATEWAY_DEFAULT_BASE_URL,
)
from callosum.backends.litellm_gateway import LiteLLMGatewayBackend
from callosum.backends.local_direct import LocalModelRegistryBackend
from callosum.backends.ollama_cloud import (
    DEFAULT_MODEL_SUFFIX as OLLAMA_CLOUD_DEFAULT_MODEL_SUFFIX,
)
from callosum.backends.ollama_cloud import DEFAULT_OLLAMA_URL as OLLAMA_CLOUD_DEFAULT_URL
from callosum.backends.ollama_cloud import OllamaCloudBackend
from callosum.config import Config, build_backends, load_config
from callosum.local import LocalModelRegistrySource
from callosum.operator_state import OperatorState
from callosum.usage_log import UsageLog


def _default_config_path() -> Path:
    return Path("~/.config/callosum/config.toml").expanduser()


def _configure_logging() -> None:
    """Wire the startup logger and UTC-timestamped root logger. Called
    by both the legacy `main()` entry and the new `serve_with_args()`
    helper so log formatting stays consistent across both code paths."""
    logging.getLogger("callosum.startup").setLevel(logging.INFO)
    if not logging.getLogger().handlers:
        # Use UTC timestamps in ISO 8601 format per control plane compliance
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s:  %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        # Convert to UTC
        logging.Formatter.converter = lambda *args: datetime.now(UTC).timetuple()


def build_runtime_backends(cfg: Config, *, operator_state: OperatorState) -> list[Backend]:
    """Build the same backend set the live server uses.

    Shared by the daemon and resumable batch jobs that need to execute real
    backend calls against the configured fleet.
    """
    backends = build_backends(cfg)
    local_added = False
    if os.environ.get("CALLOSUM_LOCAL_DISABLED") != "1" and LocalModelRegistrySource.is_available():
        source = LocalModelRegistrySource()
        backends.append(
            LocalModelRegistryBackend(
                id="local LLM gateway",
                source=source,
                operator_state=operator_state,
                usage_log_path=cfg.usage_log.path,
            )
        )
        local_added = True
    litellm_url = os.environ.get("CALLOSUM_LITELLM_GATEWAY_URL", LITELLM_GATEWAY_DEFAULT_BASE_URL)
    if os.environ.get("CALLOSUM_LITELLM_GATEWAY_ENABLED") == "1" and not local_added:
        litellm_timeout_raw = os.environ.get("CALLOSUM_LITELLM_TIMEOUT_S")
        try:
            litellm_timeout = float(litellm_timeout_raw) if litellm_timeout_raw is not None else None
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
        logging.getLogger("callosum.startup").warning(
            "LiteLLMGatewayBackend registered as local-llm fallback. "
            "Known limitation: chat-completions hangs for any model whose "
            "upstream runtime is a responses-only proxy (-responses-proxy "
            "entries). Prefer LocalModelRegistryBackend (auto-registered when the "
            "`local-llm` CLI is on PATH) for production use."
        )
    # Ollama Cloud: cloud models served by the local ollama daemon under
    # `ollama signin`. The daemon holds the Ollama Cloud auth; callosum holds
    # NO credential. This is an INDEPENDENT backend (own id, own fleet), not a
    # local fallback — cloud requests burn real Ollama Cloud quota, so it must
    # route as remote (BackendKind="ollama_cloud"), not as a free local cell.
    # Env-gated OFF by default so enabling is a no-op for live routing until the
    # operator turns it on. See backends/ollama_cloud.py + work tracker
    #  / .
    if os.environ.get("CALLOSUM_OLLAMA_CLOUD_ENABLED") == "1":
        ollama_cloud_url = os.environ.get(
            "CALLOSUM_OLLAMA_CLOUD_URL", OLLAMA_CLOUD_DEFAULT_URL
        )
        ollama_cloud_suffix = os.environ.get(
            "CALLOSUM_OLLAMA_CLOUD_MODEL_SUFFIX", OLLAMA_CLOUD_DEFAULT_MODEL_SUFFIX
        )
        backends.append(
            OllamaCloudBackend(
                id="ollama-cloud",
                ollama_url=ollama_cloud_url,
                model_suffix=ollama_cloud_suffix,
            )
        )
    return backends


def serve_with_args(args: argparse.Namespace) -> None:
    """Start the proxy daemon using pre-parsed CLI args.

    This is the public entry the unified `callosum serve` subcommand
    in `callosum.cli` calls into. The arg shape expected:
        args.config: Path | None  (None → use _default_config_path())
        args.host:   str  | None
        args.port:   int  | None

    Kept distinct from main() so the cli.py subcommand doesn't have to
    duplicate argparse setup — it parses its own args and hands the
    Namespace to this function.
    """
    _configure_logging()
    config_path = args.config if args.config is not None else _default_config_path()
    if not config_path.exists():
        # Mirror argparse.error's behavior — write to stderr and exit non-zero.
        import sys as _sys

        _sys.stderr.write(f"error: config not found: {config_path}\n")
        _sys.exit(2)
    _run_server(config_path, args.host, args.port)


def main() -> None:
    """Legacy entry: bare `callosum` once started the daemon directly.
    Preserved so `python -m callosum` continues to work as a server
    invocation. The unified `callosum` CLI entry point now lives in
    `callosum.cli:main` — see that module's docstring.
    """
    _configure_logging()

    parser = argparse.ArgumentParser(prog="callosum")
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    if not args.config.exists():
        parser.error(f"config not found: {args.config}")
    _run_server(args.config, args.host, args.port)


def _run_server(config_path: Path, host: str | None, port: int | None) -> None:
    """Core server startup. Extracted from main() so both the legacy
    entry and the new `serve_with_args()` helper share one
    implementation."""

    cfg = load_config(config_path)
    # Operator-state DB: persistent runtime knobs the CLI manages
    # (inference-param overrides today; denylist + mode + priority in
    # later steps). Lives alongside auth.sqlite under state.dir.
    if cfg.state.dir is None:
        import sys as _sys

        _sys.stderr.write("error: config.state.dir must be set\n")
        _sys.exit(2)
    operator_state = OperatorState(cfg.state.dir / "operator_state.sqlite")
    # Earned-autonomy ladder (Tier A). First-run seeds at L1_MANUAL;
    # subsequent runs reuse the persisted level. Lives alongside
    # operator_state.sqlite under state.dir.
    from callosum.autonomy import AutonomyStore

    autonomy_store = AutonomyStore(cfg.state.dir / "autonomy.sqlite")
    # Retention runner (Tier G) — owns the v1 policy list bound to the
    # actual db / repo / filesystem locations on this host. Wired to
    # the same usage_log / autonomy db paths the proxy uses, so
    # archives + deletes operate on the same data.
    from callosum.retention import DEFAULT_ARCHIVE_DIR, RetentionRunner

    repo_root_for_retention: Path | None = None
    callosum_pkg = Path(__file__).resolve().parent
    candidate_root = callosum_pkg.parents[1]
    if (candidate_root / ".git").exists():
        repo_root_for_retention = candidate_root
    research_runner_runs_dir = Path.home() / ".local" / "state" / "research_runner" / "model-research" / "runs"
    retention_runner = RetentionRunner(
        usage_log_path=cfg.usage_log.path,
        autonomy_db_path=cfg.state.dir / "autonomy.sqlite",
        repo_root=repo_root_for_retention,
        research_runner_runs_dir=(research_runner_runs_dir if research_runner_runs_dir.exists() else None),
        archive_dir=DEFAULT_ARCHIVE_DIR,
    )
    # Self-assessment runner (Tier C) — weekly metacognition. Reuses
    # the same autonomy store + usage log so its reads are coherent
    # with what the operator sees in /admin/autonomy/audit. Feedback
    # root is the project repo so artifacts land under feedback/
    # decisions/ where future agents can read them.
    from callosum.self_assessment import SelfAssessmentRunner

    self_assessment_runner = SelfAssessmentRunner(
        autonomy_store=autonomy_store,
        usage_log_path=cfg.usage_log.path,
        feedback_root=(repo_root_for_retention if repo_root_for_retention is not None else None),
    )
    backends = build_runtime_backends(cfg, operator_state=operator_state)
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
    host = host if host is not None else cfg.server.host
    port = port if port is not None else cfg.server.port

    # Use uvicorn's signal handling configuration and explicitly set to allow shutdown
    uvicorn.run(
        create_app(
            backends=backends,
            usage_log=usage_log,
            auth_service=auth_service,
            auto_router_config=cfg.auto_router,
            codex_catalog_config=cfg.codex_catalog,
            startup_smoke_test=cfg.server.startup_smoke_test,
            smoke_test_interval_seconds=cfg.server.smoke_test_interval_seconds,
            operator_state=operator_state,
            autonomy_store=autonomy_store,
            retention_runner=retention_runner,
            self_assessment_runner=self_assessment_runner,
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
