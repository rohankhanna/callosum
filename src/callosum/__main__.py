from __future__ import annotations

import argparse
import logging
import os
import shlex
from datetime import UTC, datetime
from pathlib import Path

import uvicorn

from callosum.app import create_app
from callosum.auth import AuthService
from callosum.auth_db import AuthDB
from callosum.backend import Backend
from callosum.backends.codex_gateway import DEFAULT_BASE_URL as CODEX_GATEWAY_DEFAULT_BASE_URL
from callosum.backends.codex_gateway import CodexGatewayBackend
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
from callosum.backends.openrouter import DEFAULT_BASE_URL as OPENROUTER_DEFAULT_BASE_URL
from callosum.backends.openrouter import (
    DEFAULT_BLOCKED_COUNTRIES as OPENROUTER_DEFAULT_BLOCKED_COUNTRIES,
)
from callosum.backends.openrouter import (
    DEFAULT_EXCLUDE_FAMILIES as OPENROUTER_DEFAULT_EXCLUDE_FAMILIES,
)
from callosum.backends.openrouter import OpenRouterBackend
from callosum.config import Config, load_config
from callosum.local import LocalModelRegistrySource
from callosum.operator_state import OperatorState
from callosum.usage_log import UsageLog
from callosum.wiring import build_backends


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


def _local_llm_cli_command() -> list[str] | None:
    """Resolve the local-llm catalog CLI argv from the env override.

    CALLOSUM_LOCAL_LLM_CLI, if set and non-blank, is shell-split into an
    argv list so the operator can point callosum at a known-good binary —
    e.g. CALLOSUM_LOCAL_LLM_CLI="uv run --directory
    /home/.../local LLM gateway python -m local.cli" — instead of
    relying on the local-llm console script on PATH, which silently dies
    when its pipx venv loses the package (orphaned shebang →
    ModuleNotFoundError, exit 1). Returns None to use the default
    ["local-llm"] PATH lookup.
    """
    raw = os.environ.get("CALLOSUM_LOCAL_LLM_CLI")
    if not raw or not raw.strip():
        return None
    return shlex.split(raw)


def build_runtime_backends(cfg: Config, *, operator_state: OperatorState) -> list[Backend]:
    """Build the same backend set the live server uses.

    Shared by the daemon and resumable batch jobs that need to execute real
    backend calls against the configured fleet.
    """
    backends = build_backends(cfg)
    local_added = False
    local_cli = _local_llm_cli_command()
    if os.environ.get("CALLOSUM_LOCAL_DISABLED") != "1":
        # probe_availability classifies the failure (missing / broken / timeout)
        # rather than collapsing it to a bare bool, so the operator sees the
        # real cause in the startup log instead of only the downstream
        # LiteLLM-fallback's opaque reason="network" in /status.
        available, reason = LocalModelRegistrySource.probe_availability(local_cli)
        if available:
            source = LocalModelRegistrySource(cli_command=local_cli)
            backends.append(
                LocalModelRegistryBackend(
                    id="local LLM gateway",
                    source=source,
                    operator_state=operator_state,
                    usage_log_path=cfg.usage_log.path,
                )
            )
            local_added = True
        else:
            logging.getLogger("callosum.startup").warning(
                "LocalModelRegistryBackend not registered: local-llm catalog CLI %s "
                "(probe ran %r). Point CALLOSUM_LOCAL_LLM_CLI at a known-good "
                "binary to override the PATH lookup.",
                reason,
                local_cli if local_cli is not None else ["local-llm", "--help"],
            )
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
    # Ollama Cloud is an independent remote backend. It talks directly to
    # ollama.com with the configured API key and is disabled by default.
    if os.environ.get("CALLOSUM_OLLAMA_CLOUD_ENABLED") == "1":
        ollama_cloud_url = os.environ.get("CALLOSUM_OLLAMA_CLOUD_URL", OLLAMA_CLOUD_DEFAULT_URL)
        ollama_cloud_suffix = os.environ.get("CALLOSUM_OLLAMA_CLOUD_MODEL_SUFFIX", OLLAMA_CLOUD_DEFAULT_MODEL_SUFFIX)
        ollama_cloud_api_key = os.environ.get("CALLOSUM_OLLAMA_CLOUD_API_KEY", "")

        backends.append(
            OllamaCloudBackend(
                id="ollama-cloud",
                api_key=ollama_cloud_api_key,
                ollama_url=ollama_cloud_url,
                model_suffix=ollama_cloud_suffix,
            )
        )
    # OpenRouter: the hosted OpenAI-compatible aggregator. Callosum sends
    # requests directly to OpenRouter using the configured API key. This is a
    # conservative-overflow remote band member (priority offset 2000,
    # behind Codex and ollama_cloud's 1000, before local's 10_000) so the
    # operator spends paid Codex quota and the already-paid ollama_cloud
    # first. Full auto-discovery by default; a family-exclude drops models
    # ollama_cloud already serves so OpenRouter doesn't get paid for them.
    # Data-residency is enforced per-request via OpenRouter `provider.ignore`
    # denying providers with datacenters/headquarters in authoritarian
    # regimes (default CN/RU/KP), auto-derived from the /providers endpoint.
    # Env-gated OFF by default. The API key is read from the environment so
    # it is never stored in Callosum's configuration file.
    if os.environ.get("CALLOSUM_OPENROUTER_ENABLED") == "1":
        openrouter_url = os.environ.get("CALLOSUM_OPENROUTER_BASE_URL", OPENROUTER_DEFAULT_BASE_URL)
        or_api_key = os.environ.get("CALLOSUM_OPENROUTER_API_KEY", "")
        or_model_filter = os.environ.get("CALLOSUM_OPENROUTER_MODEL_FILTER", "all")
        or_allowlist = frozenset(s for s in (os.environ.get("CALLOSUM_OPENROUTER_MODELS", "") or "").split(",") if s)
        or_model_prefix = os.environ.get("CALLOSUM_OPENROUTER_MODEL_PREFIX") or None
        or_blocked_countries = (
            frozenset(s for s in (os.environ.get("CALLOSUM_OPENROUTER_BLOCKED_COUNTRIES", "") or "").split(",") if s)
            or OPENROUTER_DEFAULT_BLOCKED_COUNTRIES
        )
        or_blocked_providers = frozenset(
            s for s in (os.environ.get("CALLOSUM_OPENROUTER_BLOCKED_PROVIDERS", "") or "").split(",") if s
        )
        or_allowed_providers = frozenset(
            s for s in (os.environ.get("CALLOSUM_OPENROUTER_ALLOWED_PROVIDERS", "") or "").split(",") if s
        )
        or_exclude_families = (
            frozenset(s for s in (os.environ.get("CALLOSUM_OPENROUTER_EXCLUDE_FAMILIES", "") or "").split(",") if s)
            or OPENROUTER_DEFAULT_EXCLUDE_FAMILIES
        )
        or_catalog_refresh_raw = os.environ.get("CALLOSUM_OPENROUTER_CATALOG_REFRESH_S")
        or_catalog_refresh: float | None = None
        if or_catalog_refresh_raw is not None:
            try:
                or_catalog_refresh = float(or_catalog_refresh_raw)
            except ValueError:
                logging.getLogger("callosum.startup").warning(
                    "CALLOSUM_OPENROUTER_CATALOG_REFRESH_S not a float; using default"
                )
        or_kwargs: dict[str, object] = {
            "id": "openrouter",
            "base_url": openrouter_url,
            "model_filter": or_model_filter,
            "allowlist": or_allowlist,
            "model_prefix": or_model_prefix,
            "blocked_countries": or_blocked_countries,
            "blocked_providers": or_blocked_providers,
            "allowed_providers": or_allowed_providers,
            "exclude_families": or_exclude_families,
            "api_key": or_api_key,
        }
        if or_catalog_refresh is not None:
            or_kwargs["catalog_refresh_s"] = or_catalog_refresh
        backends.append(OpenRouterBackend(**or_kwargs))  # type: ignore[arg-type]
    # Codex gateway: a generic Codex-compatible endpoint that takes a plain
    # API key. The operator points base_url at a Codex-compatible endpoint
    # and provides a long-lived Bearer token. No OAuth, no token refresh,
    # no proxy concepts. Env-gated OFF by default.
    if os.environ.get("CALLOSUM_CODEX_GATEWAY_ENABLED") == "1":
        codex_gw_url = os.environ.get("CALLOSUM_CODEX_GATEWAY_BASE_URL", CODEX_GATEWAY_DEFAULT_BASE_URL)
        codex_gw_api_key = os.environ.get("CALLOSUM_CODEX_GATEWAY_API_KEY", "")
        backends.append(
            CodexGatewayBackend(
                id="codex-gateway",
                api_key=codex_gw_api_key,
                base_url=codex_gw_url,
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
