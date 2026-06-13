"""Wrapper around the local LLM gateway CLI.

local LLM gateway manages a garage of locally-served models across multiple
runtimes (ollama, vllm, responses_proxy, etc.). Its CLI is the
single source of truth for what's available and what each model can do.
callosum shells out to it instead of hardcoding model lists.

Two CLI surfaces we consume:

  local-llm models local --json
    → { "entries": [ { "model": {...}, "artifacts": {...} }, ... ] }
    Per-model: id, endpoint, runtime, runtime_model, api_surfaces,
    context_window, family, enabled.

  local-llm capabilities --json
    → { "rows": [ { "model_id": ..., "host_fit": {...}, ... }, ... ] }
    Per-model: deeper capability info (host fit, quantization,
    deployment profile). Phase 5 only consumes the basics (context
    window). Future phases can use the rest.

Subprocess calls are cached with a TTL — the user changes their model
garage on the order of minutes/hours, not seconds, so a 60s refresh
window is plenty. Failures are logged and treated as "no models" —
callosum continues operating with whatever Codex backends are
configured.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


DEFAULT_REFRESH_S = 60.0
DEFAULT_CLI_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model from local LLM gateway's registry, normalized for callosum.

    `endpoint` is the http://host:port the runtime listens on.
    `runtime_model` is the model name the runtime expects (e.g.
    "model-a0d7" for ollama). `api_surfaces` enumerates which API
    callable shapes the runtime supports — "chat" for OpenAI chat
    completions, "responses" for the Codex Responses API.
    """

    id: str
    endpoint: str
    runtime: str
    runtime_model: str
    family: str
    context_window: int | None
    api_surfaces: tuple[str, ...]
    enabled: bool

    @classmethod
    def from_cli_entry(cls, entry: dict[str, Any]) -> ModelEntry | None:
        """Parse one entry from `local-llm models local --json`. Returns
        None if required fields are missing (defensive — local LLM gateway
        output shape evolves; never crash callosum)."""
        if not isinstance(entry, dict):
            return None
        m = entry.get("model")
        if not isinstance(m, dict):
            return None
        model_id = m.get("id")
        endpoint = m.get("endpoint")
        runtime = m.get("runtime")
        if not isinstance(model_id, str) or not model_id:
            return None
        if not isinstance(endpoint, str) or not endpoint:
            return None
        if not isinstance(runtime, str) or not runtime:
            return None
        surfaces = m.get("api_surfaces")
        if isinstance(surfaces, list):
            api_surfaces = tuple(s for s in surfaces if isinstance(s, str))
        elif isinstance(m.get("api"), str):
            api_surfaces = (m["api"],)
        else:
            api_surfaces = ()
        ctx = m.get("context_window")
        return cls(
            id=model_id,
            endpoint=endpoint,
            runtime=runtime,
            runtime_model=m.get("runtime_model", model_id) or model_id,
            family=m.get("family", "") or "",
            context_window=int(ctx) if isinstance(ctx, int) and ctx > 0 else None,
            api_surfaces=api_surfaces,
            enabled=bool(m.get("enabled", True)),
        )


@dataclass(slots=True)
class _CacheState:
    fetched_at: float = 0.0
    models: list[ModelEntry] = field(default_factory=list)
    healthy: bool = False


class LocalModelRegistrySource:
    """Thin client over the local LLM gateway CLI.

    Default behavior: subprocess-call `local-llm models local --json`
    on a 60s TTL. Callers (the LocalModelRegistryBackend) consume the parsed
    ModelEntry list for advertised_models and capability lookup.

    The CLI path is configurable for tests; default uses the
    `local-llm` executable on PATH.
    """

    def __init__(
        self,
        *,
        cli_command: list[str] | None = None,
        refresh_s: float = DEFAULT_REFRESH_S,
        timeout_s: float = DEFAULT_CLI_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> None:
        # Default to the user-installed CLI from local LLM gateway. Users
        # can override by passing an explicit command (typically
        # `["uv", "run", "python", "-m", "local.cli"]` for
        # development installs).
        self._cli = cli_command if cli_command is not None else ["local-llm"]
        self._refresh_s = refresh_s
        self._timeout_s = timeout_s
        self._env = dict(env) if env is not None else None
        self._lock = threading.Lock()
        self._cache = _CacheState()

    @staticmethod
    def is_available(cli_command: list[str] | None = None) -> bool:
        """Quick check: does the CLI command resolve to an executable?
        Used by __main__.py to decide whether to instantiate this
        source at all."""
        cmd = cli_command if cli_command is not None else ["local-llm"]
        try:
            proc = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0

    def models(self, *, force: bool = False) -> list[ModelEntry]:
        """Return the parsed model list. Cached for `refresh_s`; pass
        force=True to bypass the cache (used by tests + by the periodic
        refresh hook)."""
        with self._lock:
            now = time.time()
            if not force and (now - self._cache.fetched_at) < self._refresh_s and self._cache.healthy:
                return list(self._cache.models)
            self._cache = self._fetch_locked()
            return list(self._cache.models)

    def _fetch_locked(self) -> _CacheState:
        """Invoke the CLI and parse. Any failure → empty result with
        healthy=False so the next call re-tries on the next refresh tick
        instead of caching the failure long-term."""
        cmd = self._cli + ["models", "local", "--json"]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                env=self._merged_env(),
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning(
                "local LLM gateway: CLI invocation failed (%s); models list is empty this cycle",
                type(exc).__name__,
            )
            return _CacheState(fetched_at=time.time(), models=[], healthy=False)
        if proc.returncode != 0:
            logger.warning(
                "local LLM gateway: CLI exited %d; stderr=%r",
                proc.returncode,
                proc.stderr[:300] if proc.stderr else "",
            )
            return _CacheState(fetched_at=time.time(), models=[], healthy=False)
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            logger.warning("local LLM gateway: CLI emitted non-JSON output (%s)", exc)
            return _CacheState(fetched_at=time.time(), models=[], healthy=False)
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            return _CacheState(fetched_at=time.time(), models=[], healthy=False)
        models: list[ModelEntry] = []
        for entry in entries:
            parsed = ModelEntry.from_cli_entry(entry)
            if parsed is None:
                continue
            if not parsed.enabled:
                continue
            models.append(parsed)
        return _CacheState(fetched_at=time.time(), models=models, healthy=True)

    def _merged_env(self) -> dict[str, str] | None:
        """Some local LLM gateway installs need specific env (PATH for uv,
        SCHED_ORCH_RUNTIME_DIR for Dispatch integration). Caller can
        provide overrides; we layer them on os.environ."""
        if self._env is None:
            return None
        merged = dict(os.environ)
        merged.update(self._env)
        return merged
