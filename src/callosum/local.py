"""Wrapper around the the local LLM gateway CLI.

the local LLM gateway manages a garage of locally-served models across multiple
runtimes (ollama, vllm, responses_proxy, etc.). Its CLI is the
single source of truth for what's available and what each model can do.
callosum shells out to it instead of hardcoding model lists.

Two CLI surfaces we consume:

  the local LLM gateway models local --json
    → { "entries": [ { "model": {...}, "artifacts": {...} }, ... ] }
    Per-model: id, endpoint, runtime, runtime_model, api_surfaces,
    context_window, family, enabled.

  the local LLM gateway capabilities --json
    → { "rows": [ { "model_id": ..., "host_fit": {...}, ... }, ... ] }
    Per-model: deeper capability info (host fit, quantization,
    deployment profile). Phase 5 only consumes the basics (context
    window). Future phases can use the rest.

Subprocess calls are cached **load-once**: the garage changes on the
order of minutes/hours (only when the operator adds/removes a model —
a restart event, not steady state), so a healthy snapshot is served
indefinitely after the first fetch and re-fetched only on an explicit
`force=True` (startup pass, boot resync, operator reload). A *failed*
snapshot (the local LLM gateway unreachable) still retries on read, throttled to
once per `refresh_s`, so a cold boot recovers without a restart while a
down the local LLM gateway isn't hammered on every lookup. Restart always reloads.
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


#: Max age before a *failed* (unhealthy) cache retries. A healthy snapshot
#: never auto-expires — it is re-fetched only via `force=True`. Kept as a
#: throttle so a down the local LLM gateway isn't hammered on every read while a cold
#: boot still recovers on the next refresh tick.
DEFAULT_REFRESH_S = 60.0
DEFAULT_CLI_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One model from the local LLM gateway's registry, normalized for callosum.

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
    #: Ordered reasoning-effort levels the model exposes, per the local LLM gateway's
    #: registry. Always begins with "default". model-a0d2 cells advertise
    #: ("default", "low", "medium", "high"); reason-by-default models (model-a0g2,
    #: nemotron reasoning, phi-4-reasoning-plus) advertise only ("default",).
    #: Defaults to ("default",) when the field is absent (older hub builds).
    supported_reasoning_levels: tuple[str, ...] = ("default",)
    local_quantization: str | None = None
    local_runnable_on_host: bool | None = None
    local_status: str | None = None
    estimated_tokens_per_second: float | None = None
    local_pool_bytes: int | None = None
    local_fit_limit_tokens: int | None = None
    local_prefill_ms_per_token: float | None = None
    local_decode_bandwidth_kappa: float | None = None

    @classmethod
    def from_cli_entry(cls, entry: dict[str, Any]) -> ModelEntry | None:
        """Parse one entry from `the local LLM gateway models local --json`. Returns
        None if required fields are missing (defensive — the local LLM gateway
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
        levels_raw = m.get("supported_reasoning_levels")
        levels = tuple(s for s in levels_raw if isinstance(s, str) and s) if isinstance(levels_raw, list) else ()
        # Contract guarantees "default" leads the list; default to it when the
        # field is absent (older hub builds) or parsed empty.
        if not levels:
            levels = ("default",)
        return cls(
            id=model_id,
            endpoint=endpoint,
            runtime=runtime,
            runtime_model=m.get("runtime_model", model_id) or model_id,
            family=m.get("family", "") or "",
            context_window=int(ctx) if isinstance(ctx, int) and ctx > 0 else None,
            api_surfaces=api_surfaces,
            enabled=bool(m.get("enabled", True)),
            supported_reasoning_levels=levels,
        )


@dataclass(frozen=True, slots=True)
class CapabilityRow:
    """One row from `the local LLM gateway capabilities --json`, normalized for callosum.

    The hub's capability matrix is the authoritative structured source for
    local runtime facts that are not part of the simpler `models local` roster:
    host-fit, quantization, lifecycle status, and any future graph metrics such
    as measured throughput. Callosum can consume the matrix opportunistically
    without requiring every field to be present today.
    """

    model_id: str
    quantization_label: str | None
    runnable_on_host: bool | None
    status: str | None
    estimated_tokens_per_second: float | None
    pool_bytes: int | None
    fit_limit_tokens: int | None
    prefill_ms_per_token: float | None
    decode_bandwidth_kappa: float | None
    modalities: frozenset[str] | None = None
    supports_tools: bool | None = None

    @classmethod
    def from_cli_row(cls, row: dict[str, Any]) -> CapabilityRow | None:
        if not isinstance(row, dict):
            return None
        model_id = row.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            return None
        quant = row.get("quantization")
        quantization_label = None
        if isinstance(quant, dict):
            label = quant.get("label")
            if isinstance(label, str) and label:
                quantization_label = label
        host_fit = row.get("host_fit")
        runnable = None
        if isinstance(host_fit, dict):
            rh = host_fit.get("runnable_on_host")
            if isinstance(rh, bool):
                runnable = rh
        status = row.get("status") if isinstance(row.get("status"), str) else None
        metrics = row.get("graph_metrics")
        estimated_tps = None
        fit_limit_tokens = None
        prefill_ms_per_token = None
        decode_bandwidth_kappa = None
        if isinstance(metrics, dict):
            for key in (
                "estimated_clean_total_tokens_per_second",
                "estimated_clean_output_tokens_per_second",
                "ceiling_search_completion_tokens_per_second",
            ):
                value = metrics.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    estimated_tps = float(value)
                    break
            for key in (
                "ceiling_search_prompt_tokens",
                "measured_clean_prompt_tokens",
                "estimated_clean_prompt_tokens",
            ):
                value = metrics.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    fit_limit_tokens = int(value)
                    break
            total_tokens = metrics.get("measured_clean_total_tokens")
            duration_seconds = metrics.get("local_first_avg_duration_seconds")
            if (
                isinstance(total_tokens, (int, float))
                and total_tokens > 0
                and isinstance(duration_seconds, (int, float))
                and duration_seconds > 0
            ):
                prefill_ms_per_token = max(0.1, float(duration_seconds) * 1000.0 / float(total_tokens))
            if estimated_tps is not None:
                ceiling_tps = metrics.get("ceiling_search_completion_tokens_per_second")
                if isinstance(ceiling_tps, (int, float)) and ceiling_tps > 0 and estimated_tps > 0:
                    decode_bandwidth_kappa = max(1.0, estimated_tps / float(ceiling_tps))
        host = row.get("host")
        pool_bytes = None
        if isinstance(host, dict):
            total_vram_gb = host.get("total_vram_gb")
            total_ram_gb = host.get("total_ram_gb")
            unified = bool(host.get("unified_memory"))
            chosen_gb = total_ram_gb if unified else total_vram_gb
            if isinstance(chosen_gb, (int, float)) and chosen_gb > 0:
                pool_bytes = int(float(chosen_gb) * 1024**3)
        modalities: frozenset[str] | None = None
        raw_modalities = row.get("modalities")
        if isinstance(raw_modalities, list) and all(isinstance(m, str) for m in raw_modalities):
            normalized = {m.lower() for m in raw_modalities}
            if normalized:
                normalized.add("text")
                modalities = frozenset(normalized)
        supports_tools: bool | None = None
        raw_supports_tools = row.get("supports_tools")
        if isinstance(raw_supports_tools, bool):
            supports_tools = raw_supports_tools
        return cls(
            model_id=model_id,
            quantization_label=quantization_label,
            runnable_on_host=runnable,
            status=status,
            estimated_tokens_per_second=estimated_tps,
            pool_bytes=pool_bytes,
            fit_limit_tokens=fit_limit_tokens,
            prefill_ms_per_token=prefill_ms_per_token,
            decode_bandwidth_kappa=decode_bandwidth_kappa,
            modalities=modalities,
            supports_tools=supports_tools,
        )


@dataclass(slots=True)
class _CacheState:
    fetched_at: float = 0.0
    models: list[ModelEntry] = field(default_factory=list)
    capabilities: dict[str, CapabilityRow] = field(default_factory=dict)
    healthy: bool = False
    # Why the last fetch landed where it did: "ok" / "missing" / "timeout" /
    # "broken" / "unknown"(never fetched). Surfaced through health() so /status
    # names the real cause (broken sibling CLI vs. empty garage) instead of an
    # opaque "unknown".
    last_fetch_reason: str = "unknown"


class LocalModelRegistrySource:
    """Thin client over the the local LLM gateway CLI.

    Load-once cache: the first read shells out to `the local LLM gateway models local
    --json` (and `the local LLM gateway capabilities --json`); subsequent reads return
    the same healthy snapshot indefinitely. The garage changes on the
    order of minutes/hours — a restart event, not steady state — so no
    TTL-driven auto-refresh. `force=True` re-fetches (startup pass,
    `_catalog_boot_resync`, operator reload); a *failed* fetch retries on
    read at most once per `refresh_s` so a cold boot recovers without a
    restart while a down the local LLM gateway isn't hammered.

    The CLI path is configurable for tests; default uses the
    `the local LLM gateway` executable on PATH.
    """

    def __init__(
        self,
        *,
        cli_command: list[str] | None = None,
        refresh_s: float = DEFAULT_REFRESH_S,
        timeout_s: float = DEFAULT_CLI_TIMEOUT_S,
        env: dict[str, str] | None = None,
    ) -> None:
        # Default to the user-installed CLI from the local LLM gateway. Users
        # can override by passing an explicit command (typically
        # `["uv", "run", "python", "-m", "local.cli"]` for
        # development installs).
        self._cli = cli_command if cli_command is not None else ["the local LLM gateway"]
        # Throttle for retrying a *failed* (unhealthy) cache; a healthy
        # snapshot never auto-expires.
        self._refresh_s = refresh_s
        self._timeout_s = timeout_s
        self._env = dict(env) if env is not None else None
        self._lock = threading.Lock()
        self._cache = _CacheState()
        # Transient: the most recent _run_json outcome. _fetch_locked captures
        # this into the cache snapshot it returns; last_fetch_reason reads
        # from the cache so it reflects the last *completed* fetch, not an
        # in-flight capabilities call that may clobber this.
        self._last_fetch_reason: str = "unknown"

    @staticmethod
    def probe_availability(
        cli_command: list[str] | None = None,
    ) -> tuple[bool, str]:
        """Classify whether the catalog CLI is usable. Returns
        (available, reason) where reason is one of:

        - "ok"      — CLI present and exits 0.
        - "missing" — binary not on PATH (FileNotFoundError). Install or
          PATH issue; the sibling repo's console script isn't resolvable.
        - "broken"  — binary present but exits non-zero. The classic
          symptom of an orphaned pipx venv that lost its package: the shebang
          points at a venv whose site-packages no longer contain
          local, so the interpreter raises ModuleNotFoundError and
          the CLI exits 1.
        - "timeout" — CLI hung past the 5s probe window.

        Distinct reasons let the operator diagnose the real failure instead
        of the downstream fallback backend's opaque reason="network" in
        /status. Used by __main__.py at registration time.
        """
        cmd = cli_command if cli_command is not None else ["the local LLM gateway"]
        try:
            proc = subprocess.run(
                cmd + ["--help"],
                capture_output=True,
                text=True,
                timeout=5.0,
            )
        except FileNotFoundError:
            return False, "missing"
        except subprocess.TimeoutExpired:
            return False, "timeout"
        if proc.returncode != 0:
            return False, "broken"
        return True, "ok"

    @staticmethod
    def is_available(cli_command: list[str] | None = None) -> bool:
        """Quick boolean check: does the CLI resolve and exit 0?

        Thin wrapper over :meth:`probe_availability` retained for callers that
        only need the go/no-go (tests, the model-probe spawner). Callers that
        can act on the *reason* (e.g. __main__.py registration logging) should
        call probe_availability directly.
        """
        return LocalModelRegistrySource.probe_availability(cli_command)[0]

    @property
    def last_fetch_reason(self) -> str:
        """Why the most recent completed fetch landed where it did.

        Mirrors _CacheState.last_fetch_reason: "ok" / "missing" /
        "timeout" / "broken" / "unknown" (never fetched). Read by
        LocalModelRegistryBackend.health() to surface a distinct health reason.
        """
        return self._cache.last_fetch_reason

    def models(self, *, force: bool = False) -> list[ModelEntry]:
        """Return the parsed model list.

        Load-once: once a healthy snapshot exists it is returned indefinitely
        (no TTL auto-refresh — the garage changes on restart, not steady
        state). `force=True` bypasses the cache and re-fetches (startup pass,
        boot resync, operator reload). A *failed* (unhealthy) snapshot still
        retries on read, throttled to once per `refresh_s`, so a cold boot
        recovers without a restart while a down the local LLM gateway isn't hammered.
        """
        with self._lock:
            now = time.time()
            cache = self._cache
            if not force:
                if cache.healthy:
                    return list(cache.models)
                # Unhealthy: retry at most once per refresh_s.
                if (now - cache.fetched_at) < self._refresh_s:
                    return list(cache.models)
            self._cache = self._fetch_locked()
            return list(self._cache.models)

    def capabilities(self, *, force: bool = False) -> dict[str, CapabilityRow]:
        """Return the parsed capability-matrix rows keyed by model id.

        Same load-once semantics as `models()`: the two CLI surfaces describe
        the same local fleet and share one cache cycle, so a healthy snapshot
        is served indefinitely and only `force=True` or a failed-cache retry
        re-fetches.
        """
        with self._lock:
            now = time.time()
            cache = self._cache
            if not force:
                if cache.healthy:
                    return dict(cache.capabilities)
                if (now - cache.fetched_at) < self._refresh_s:
                    return dict(cache.capabilities)
            self._cache = self._fetch_locked()
            return dict(self._cache.capabilities)

    def _fetch_locked(self) -> _CacheState:
        """Invoke the CLI and parse. Any failure → empty result with
        healthy=False so the next call re-tries on the next refresh tick
        instead of caching the failure long-term. The captured
        last_fetch_reason records *why* the models fetch landed where it
        did (read before the capabilities sub-fetch, which may clobber the
        transient self._last_fetch_reason)."""
        payload = self._run_json(["models", "local", "--json"])
        if payload is None:
            return _CacheState(
                fetched_at=time.time(),
                models=[],
                capabilities={},
                healthy=False,
                last_fetch_reason=self._last_fetch_reason,
            )
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            # CLI ran and emitted a dict, but not the expected shape — treat
            # as a broken catalog CLI (version skew / wrong subcommand output).
            return _CacheState(
                fetched_at=time.time(),
                models=[],
                capabilities={},
                healthy=False,
                last_fetch_reason="broken",
            )
        models: list[ModelEntry] = []
        capabilities = self._fetch_capabilities()
        for entry in entries:
            parsed = ModelEntry.from_cli_entry(entry)
            if parsed is None:
                continue
            if not parsed.enabled:
                continue
            cap = capabilities.get(parsed.id)
            if cap is not None:
                parsed = ModelEntry(
                    id=parsed.id,
                    endpoint=parsed.endpoint,
                    runtime=parsed.runtime,
                    runtime_model=parsed.runtime_model,
                    family=parsed.family,
                    context_window=parsed.context_window,
                    api_surfaces=parsed.api_surfaces,
                    enabled=parsed.enabled,
                    supported_reasoning_levels=parsed.supported_reasoning_levels,
                    local_quantization=cap.quantization_label,
                    local_runnable_on_host=cap.runnable_on_host,
                    local_status=cap.status,
                    estimated_tokens_per_second=cap.estimated_tokens_per_second,
                    local_pool_bytes=cap.pool_bytes,
                    local_fit_limit_tokens=cap.fit_limit_tokens,
                    local_prefill_ms_per_token=cap.prefill_ms_per_token,
                    local_decode_bandwidth_kappa=cap.decode_bandwidth_kappa,
                )
            models.append(parsed)
        return _CacheState(
            fetched_at=time.time(),
            models=models,
            capabilities=capabilities,
            healthy=True,
            last_fetch_reason="ok",
        )

    def _fetch_capabilities(self) -> dict[str, CapabilityRow]:
        payload = self._run_json(["capabilities", "--json"], warn_label="capability matrix")
        if payload is None:
            return {}
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            return {}
        host = payload.get("host") if isinstance(payload.get("host"), dict) else None
        out: dict[str, CapabilityRow] = {}
        for row in rows:
            if host is not None and isinstance(row, dict) and not isinstance(row.get("host"), dict):
                row = {**row, "host": host}
            parsed = CapabilityRow.from_cli_row(row)
            if parsed is None:
                continue
            out[parsed.model_id] = parsed
        return out

    def _run_json(self, args: list[str], *, warn_label: str = "models list") -> dict[str, Any] | None:
        cmd = self._cli + args
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                env=self._merged_env(),
            )
        except FileNotFoundError as exc:
            self._last_fetch_reason = "missing"
            logger.warning(
                "the local LLM gateway: CLI not found (%s); %s unavailable this cycle",
                exc,
                warn_label,
            )
            return None
        except subprocess.TimeoutExpired:
            self._last_fetch_reason = "timeout"
            logger.warning(
                "the local LLM gateway: CLI timed out after %ss reading %s",
                self._timeout_s,
                warn_label,
            )
            return None
        if proc.returncode != 0:
            self._last_fetch_reason = "broken"
            logger.warning(
                "the local LLM gateway: CLI exited %d while reading %s; stderr=%r",
                proc.returncode,
                warn_label,
                proc.stderr[:300] if proc.stderr else "",
            )
            return None
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            self._last_fetch_reason = "broken"
            logger.warning("the local LLM gateway: CLI emitted non-JSON %s (%s)", warn_label, exc)
            return None
        if not isinstance(payload, dict):
            self._last_fetch_reason = "broken"
            logger.warning("the local LLM gateway: CLI emitted non-object %s", warn_label)
            return None
        self._last_fetch_reason = "ok"
        return payload

    def _merged_env(self) -> dict[str, str] | None:
        """Some the local LLM gateway installs need specific env (PATH for uv,
        SCHED_ORCH_RUNTIME_DIR for Dispatch integration). Caller can
        provide overrides; we layer them on os.environ."""
        if self._env is None:
            return None
        merged = dict(os.environ)
        merged.update(self._env)
        return merged
