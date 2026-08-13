"""One-shot full-context memory-fit probe for a local model.

Loads a model via the local LLM gateway CLI just long enough to read vLLM's
startup KV-cache allocation, decides whether the model can serve its full
advertised context window on this host, persists the result, and stops the
model. Each model is probed at most once per artifact (content hash), so
re-probing happens only after a re-pull or version change.

vLLM logs the answer we need at startup, into the serve log file that
local-llm serve --model <id> --json reports back to us:

    core.py:97] ... max_seq_len=<N> ...                      # resolved context
    kv_cache_utils.py:1307] GPU KV cache size: <K> tokens    # achievable KV
    kv_cache_utils.py:1312] Maximum concurrency for <N> tokens per request: <M>x

M >= 1.0 at N means at least one request can fill that window — the
model fits at full context on this host. We do not roll our own KV math;
vLLM measures it.

Host safety follows the unified-memory overlapping-loads anti-pattern
(learnings ):

- an exclusive GPU lock is taken for the load
  (LOCAL_LLM_EXCLUSIVE_GPU_LOCK=1), reusing local LLM gateway's
  locked_exec path so the probe never overlaps another heavyweight load
  that honors the same lock;
- a preflight admission check blocks the load when host headroom is too tight;
- the runner never loads two models at once (serial probes).

MVP scope: runtime == "vllm" models only. The custom gpt_oss runtime,
model-a0e0, and ollama entries are admitted via the catalog's existing
training-precision check + the hub's local_fit_limit_tokens prior; probing
them is a deferred refinement (see work tracker).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from callosum.local import ModelEntry
from callosum.usage_log import ModelFitProbe, UsageLog

logger = logging.getLogger(__name__)

DEFAULT_CLI_COMMAND: list[str] = ["local-llm"]
DEFAULT_SERVE_TIMEOUT_S = 600.0  # cold weight load on unified memory can be slow
DEFAULT_STOP_TIMEOUT_S = 90.0
DEFAULT_MODELS_TIMEOUT_S = 30.0
# vLLM safe mode (auto on Blackwell 12.1) reserves only 4 GiB for KV cache by
# default, which starves large-context probing. We override it to a realistic
# budget derived from the host pool minus the model's weights. Keep utilization
# conservative so the load does not collide with anything else resident.
DEFAULT_UTILIZATION_FRACTION = 0.80
DEFAULT_ACTIVATION_MARGIN_BYTES = 2 * 1024**3  # 2 GiB for activations/overhead
DEFAULT_KV_FLOOR_BYTES = 1 * 1024**3  # never request less than 1 GiB KV
# Two-tier failure backoff for the periodic model-fit probe. A (model, weights,
# lane) combo that keeps failing is retried on FAILURE_BACKOFF_S for up to
# FAILURE_CONFIRMATIONS consecutive attempts (to rule out transient causes),
# then marked confirmed-broken and skipped permanently until the model artifact
# OR the lane (serving definition + vLLM/transformers stack) changes -- which
# resets the combo and re-probes. This matches the design's "re-probe only on
# artifact change" philosophy while still giving a broken combo enough chances.
# Env-tunable: CALLOSUM_MODEL_PROBE_FAILURE_BACKOFF_S,
# CALLOSUM_MODEL_PROBE_FAILURE_CONFIRMATIONS.
DEFAULT_FAILURE_BACKOFF_S = 86400.0  # 24h between confirmation attempts
DEFAULT_FAILURE_CONFIRMATIONS = 3  # consecutive failures -> confirmed broken

_RE_MAX_SEQ_LEN = re.compile(r"max_seq_len=(\d+)")
_RE_KV_CACHE_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_RE_MAX_CONCURRENCY = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
_RE_TARGET = re.compile(r"target=(\S+)")


@dataclass(frozen=True, slots=True)
class VllmFitLog:
    """The fit-relevant lines parsed from a vLLM serve log."""

    max_seq_len: int | None
    kv_cache_tokens: int | None
    max_concurrency: float | None


def _run_cli(
    cli_command: list[str],
    args: list[str],
    *,
    timeout_s: float,
    env: dict[str, str] | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    cmd = [*cli_command, *args]
    run_env = None
    if env is not None:
        run_env = dict(os.environ)
        run_env.update(env)
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        timeout=timeout_s,
        env=run_env,
    )


def _read_memavailable_bytes() -> int | None:
    """Return MemAvailable from /proc/meminfo in bytes, or None."""
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except OSError:
        return None
    return None


def _artifact_dir_for(cli_command: list[str], model_id: str) -> str | None:
    """Resolve a model's on-disk artifact dir from local-llm models local --json.

    The hub's per-entry artifacts.detail carries a target=<path> segment
    pointing at the model's download directory. This is fragile by design of the
    hub's text output; the durable fix is a hub-owned probe-fit subcommand
    (external handoff). Returns None if it can't be resolved.
    """
    try:
        proc = _run_cli(
            cli_command,
            ["models", "local", "--json"],
            timeout_s=DEFAULT_MODELS_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    for entry in payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        m = entry.get("model")
        if isinstance(m, dict) and m.get("id") == model_id:
            art = entry.get("artifacts")
            if isinstance(art, dict):
                detail = art.get("detail")
                if isinstance(detail, str):
                    match = _RE_TARGET.search(detail)
                    if match:
                        return match.group(1)
    return None


def _weight_files(artifact_dir: str) -> list[Path]:
    """Return weight files (.safetensors, then .gguf) in the artifact dir."""
    base = Path(artifact_dir)
    files = sorted(base.glob("*.safetensors"))
    if files:
        return files
    return sorted(base.glob("*.gguf"))


def _content_hash(artifact_dir: str) -> str | None:
    """Hash the model's weight file set + sizes — changes on re-pull/version change."""
    files = _weight_files(artifact_dir)
    if not files:
        return None
    hasher = hashlib.sha256()
    for f in files:
        try:
            size = f.stat().st_size
        except OSError:
            continue
        hasher.update(f.name.encode("utf-8"))
        hasher.update(b":")
        hasher.update(str(size).encode("ascii"))
        hasher.update(b"|")
    return hasher.hexdigest()


def _weight_bytes(artifact_dir: str) -> int | None:
    """Total on-disk weight bytes (≈ resident weight footprint)."""
    files = _weight_files(artifact_dir)
    if not files:
        return None
    total = 0
    found = False
    for f in files:
        try:
            total += f.stat().st_size
            found = True
        except OSError:
            continue
    return total if found else None


def preflight_admits(
    *,
    pool_bytes: int,
    weight_bytes: int | None,
    utilization_fraction: float = DEFAULT_UTILIZATION_FRACTION,
    activation_margin_bytes: int = DEFAULT_ACTIVATION_MARGIN_BYTES,
) -> tuple[bool, str]:
    """Decide whether to attempt the load at all (host headroom gate).

    Blocks the load before it starts when the model's weights alone would
    exceed the usable pool budget. Returns (admitted, reason).
    """
    usable = int(pool_bytes * utilization_fraction)
    if weight_bytes is not None and weight_bytes >= usable - activation_margin_bytes:
        return False, f"weights {weight_bytes} leave no headroom in usable pool {usable}"
    free = _read_memavailable_bytes()
    if free is not None and weight_bytes is not None and free < weight_bytes + activation_margin_bytes:
        return False, f"MemAvailable {free} < weights {weight_bytes} + margin {activation_margin_bytes}"
    if free is not None and free < activation_margin_bytes:
        return False, f"MemAvailable {free} below margin {activation_margin_bytes}"
    return True, ""


def _kv_budget_bytes(
    *,
    pool_bytes: int,
    weight_bytes: int | None,
    utilization_fraction: float = DEFAULT_UTILIZATION_FRACTION,
    activation_margin_bytes: int = DEFAULT_ACTIVATION_MARGIN_BYTES,
) -> int:
    """KV cache budget = usable pool − weights − activation margin."""
    usable = int(pool_bytes * utilization_fraction)
    if weight_bytes is not None:
        remaining = usable - weight_bytes - activation_margin_bytes
    else:
        remaining = usable - activation_margin_bytes
    return max(DEFAULT_KV_FLOOR_BYTES, remaining)


def _parse_vllm_fit_log(log_text: str) -> VllmFitLog:
    max_seq = _RE_MAX_SEQ_LEN.search(log_text)
    kv = _RE_KV_CACHE_TOKENS.search(log_text)
    conc = _RE_MAX_CONCURRENCY.search(log_text)
    return VllmFitLog(
        max_seq_len=int(max_seq.group(1)) if max_seq else None,
        kv_cache_tokens=int(kv.group(1).replace(",", "")) if kv else None,
        max_concurrency=float(conc.group(2)) if conc else None,
    )


def _stop_model(cli_command: list[str], model_id: str) -> None:
    """Best-effort teardown so probed models do not stay resident."""
    try:
        _run_cli(cli_command, ["stop", "--model", model_id], timeout_s=DEFAULT_STOP_TIMEOUT_S)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("model_probe: stop failed for %s", model_id)


def _backoff_state_path(usage_log: UsageLog) -> Path | None:
    """Where per-model failure-backoff state lives: next to the usage_log db.

    Co-located with requests.sqlite so it follows the same state dir and needs
    no extra config. Returns None if the usage_log exposes no path (backoff
    then silently disabled — back-compat for tests/mocks).
    """
    p = getattr(usage_log, "path", None)
    return Path(p).parent / "model_probe_backoff.json" if p else None


def _read_backoff_state(path: Path) -> dict[str, dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_backoff_state(path: Path, state: dict[str, dict[str, Any]]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh)
        os.replace(tmp, path)
    except OSError:
        logger.debug("model_probe: could not persist backoff state to %s", path)


def _record_backoff_failure(
    path: Path | None,
    model_id: str,
    content_hash: str,
    lane_hash: str,
    reason: str,
    ts: float,
    confirmations: int,
) -> None:
    """Record a probe failure for the (model, weights, lane) combo.

    Failures are NOT written to the usage_log (its model_fit_probes table is the
    success-profile store read by catalog admission), so a separate JSON file
    tracks per-combo failure state. The spawner runs one probe job at a time, so
    there is no concurrent writer to guard here.

    The combo key is (content_hash, lane_hash): if either the model artifact or
    the lane (serving definition + stack) changed since the last failure, the
    consecutive-failure counter resets to 1 -- a changed combo is a fresh
    evaluation, not a continuation of a known-broken one. Once
    consecutive_failures reaches confirmations the combo is marked
    confirmed_broken and skipped permanently until the combo changes again.
    """
    if path is None or not content_hash:
        return
    state = _read_backoff_state(path)
    entry = state.get(model_id, {})
    same_combo = entry.get("content_hash") == content_hash and entry.get("lane_hash") == lane_hash
    failures = (int(entry.get("consecutive_failures", 0)) + 1) if same_combo else 1
    state[model_id] = {
        "content_hash": content_hash,
        "lane_hash": lane_hash,
        "consecutive_failures": failures,
        "last_failed_at": ts,
        "last_reason": reason[:300],
        "confirmed_broken": failures >= confirmations,
    }
    _write_backoff_state(path, state)


def _clear_backoff(path: Path | None, model_id: str) -> None:
    """Drop a model's failure-backoff entry once it probes successfully."""
    if path is None:
        return
    state = _read_backoff_state(path)
    if model_id in state:
        state.pop(model_id, None)
        _write_backoff_state(path, state)


_STACK_VERSIONS_CACHE: dict[tuple[str, ...], str] = {}


def _stack_versions(cli_command: list[str]) -> str:
    """Best-effort 'vllm <v> transformers <v>' from the local-llm serving venv.

    The serving stack sits between the model weights and the local-llm CLI,
    so its versions are part of the lane identity. The local-llm CLI's own
    shebang venv (pipx) does not contain vllm; the model is actually served from
    a separate venv, resolved in priority order: the
    CALLOSUM_LOCAL_LLM_VENV_PYTHON override, then the repo venv at
    ~/Desktop/local LLM gateway/.venv/bin/python, then the CLI's shebang python.
    Returns "" if none can be imported (e.g. in CI or unit tests) -- the lane
    hash then falls back to config files only. Cached per cli_command.
    """
    key = tuple(cli_command)
    if key in _STACK_VERSIONS_CACHE:
        return _STACK_VERSIONS_CACHE[key]
    candidates: list[str] = []
    env_py = os.environ.get("CALLOSUM_LOCAL_LLM_VENV_PYTHON")
    if env_py:
        candidates.append(env_py)
    candidates.append(os.path.expanduser("~/Desktop/local LLM gateway/.venv/bin/python"))
    try:
        with open(cli_command[0], "rb") as fh:
            m = re.match(rb"#!\s*(\S+)", fh.readline())
        if m:
            candidates.append(m.group(1).decode("utf-8", "replace"))
    except OSError:
        pass
    for py in candidates:
        if not (os.path.isfile(py) and os.access(py, os.X_OK)):
            continue
        try:
            proc = subprocess.run(
                [py, "-c", "import vllm,transformers;print(vllm.__version__,transformers.__version__)"],
                capture_output=True,
                text=True,
                timeout=8,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            _STACK_VERSIONS_CACHE[key] = proc.stdout.strip()
            return _STACK_VERSIONS_CACHE[key]
    _STACK_VERSIONS_CACHE[key] = ""
    return ""


def _lane_hash(model_id: str, cli_command: list[str]) -> str:
    """Fingerprint of the lane: the local LLM gateway serving definition for this
    model plus the vLLM/transformers stack version.

    "Lane" = everything between the model weights and the local-llm CLI:
    the registry serving config (launch/manifest/sources/requirements) and the
    serving stack (vllm/transformers). Editing the serving config OR upgrading
    the stack changes this hash, which resets the failure counter and re-probes
    the combo -- so a vLLM upgrade that might fix a broken lane is detected.
    """
    base = os.environ.get(
        "CALLOSUM_LOCAL_LLM_MODELS_DIR",
        os.path.expanduser("~/.local/share/local-llm/live/models"),
    )
    d = Path(base) / model_id
    parts: list[str] = []
    for fname in ("launch.yaml", "manifest.yaml", "sources.yaml", "requirements.yaml"):
        try:
            parts.append((d / fname).read_text(encoding="utf-8", errors="replace"))
        except OSError:
            parts.append("")
    parts.append(_stack_versions(cli_command))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def execute_model_probe(
    model: ModelEntry,
    *,
    usage_log: UsageLog,
    cli_command: list[str] | None = None,
    serve_timeout_s: float = DEFAULT_SERVE_TIMEOUT_S,
    pool_bytes: int | None = None,
    utilization_fraction: float = DEFAULT_UTILIZATION_FRACTION,
    activation_margin_bytes: int = DEFAULT_ACTIVATION_MARGIN_BYTES,
    now: float | None = None,
    failure_backoff_s: float | None = None,
    failure_confirmations: int | None = None,
    backoff_state_path: Path | None = None,
) -> ModelFitProbe | None:
    """Probe one model's full-context fit; persist + return the result.

    Returns None when the model is skipped (already probed for this artifact,
    failed preflight, or serve did not reach readiness) so the caller can move
    on. A None is NOT persisted as a result — skipped models remain eligible
    for a later probe run.
    """
    cli = cli_command if cli_command is not None else DEFAULT_CLI_COMMAND
    ts = float(now if now is not None else time.time())

    artifact_dir = _artifact_dir_for(cli, model.id)
    content_hash = _content_hash(artifact_dir) if artifact_dir else None

    # Dedup: skip if we already have a result for this exact artifact.
    if (
        content_hash is not None
        and usage_log.get_model_fit_probe_for_hash(model_id=model.id, content_hash=content_hash) is not None
    ):
        logger.info("model_probe: %s already probed for content_hash=%s; skipping", model.id, content_hash[:12])
        return None

    # Two-tier failure backoff (lane-aware). Compute the lane identity and the
    # resolved knobs once; they are reused at every failure-return below.
    bo_path = backoff_state_path if backoff_state_path is not None else _backoff_state_path(usage_log)
    backoff_s = (
        failure_backoff_s
        if failure_backoff_s is not None
        else float(os.environ.get("CALLOSUM_MODEL_PROBE_FAILURE_BACKOFF_S", str(DEFAULT_FAILURE_BACKOFF_S)))
    )
    confirmations = (
        failure_confirmations
        if failure_confirmations is not None
        else int(os.environ.get("CALLOSUM_MODEL_PROBE_FAILURE_CONFIRMATIONS", str(DEFAULT_FAILURE_CONFIRMATIONS)))
    )
    lane_hash = _lane_hash(model.id, cli) if content_hash is not None else ""
    if content_hash is not None and bo_path is not None:
        bo = _read_backoff_state(bo_path).get(model.id)
        if bo and bo.get("content_hash") == content_hash and bo.get("lane_hash") == lane_hash:
            # Same (model, weights, lane) combo as the last failure.
            if bo.get("confirmed_broken"):
                logger.info(
                    "model_probe: %s confirmed broken (lane=%s artifact=%s); "
                    "skipping until the lane or model artifact changes",
                    model.id,
                    lane_hash[:8],
                    content_hash[:8],
                )
                return None
            age = ts - float(bo.get("last_failed_at", 0.0))
            if age < backoff_s:
                logger.info(
                    "model_probe: %s last failed %.0fs ago (< confirm-backoff %.0fs, attempt %d/%d); skipping",
                    model.id,
                    age,
                    backoff_s,
                    int(bo.get("consecutive_failures", 0)),
                    confirmations,
                )
                return None
            # Backoff expired -> fall through for another confirmation attempt.
        # Combo changed (weights and/or lane) -> the old entry is stale; fall
        # through and re-probe. The failure recorder resets the counter for the
        # new combo.

    pool = pool_bytes if pool_bytes is not None else (model.local_pool_bytes or _read_memavailable_bytes())
    if pool is None:
        logger.warning("model_probe: %s has no pool size; skipping", model.id)
        return None
    weight = _weight_bytes(artifact_dir) if artifact_dir else None

    admitted, reason = preflight_admits(
        pool_bytes=pool,
        weight_bytes=weight,
        utilization_fraction=utilization_fraction,
        activation_margin_bytes=activation_margin_bytes,
    )
    if not admitted:
        logger.info("model_probe: %s deferred by preflight (%s)", model.id, reason)
        _record_backoff_failure(
            bo_path,
            model.id,
            content_hash or "",
            lane_hash,
            f"preflight: {reason}",
            ts,
            confirmations,
        )
        return None

    kv_budget = _kv_budget_bytes(
        pool_bytes=pool,
        weight_bytes=weight,
        utilization_fraction=utilization_fraction,
        activation_margin_bytes=activation_margin_bytes,
    )
    # Defeat vLLM safe mode's 4 GiB KV cap; take the exclusive GPU lock.
    serve_env = {
        "VLLM_KV_CACHE_MEMORY_BYTES": str(kv_budget),
        "LOCAL_LLM_EXCLUSIVE_GPU_LOCK": "1",
    }
    try:
        proc = _run_cli(
            cli,
            ["serve", "--model", model.id, "--json", "--timeout", str(int(serve_timeout_s))],
            timeout_s=serve_timeout_s + 60.0,
            env=serve_env,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("model_probe: serve invocation failed for %s (%s)", model.id, exc)
        _record_backoff_failure(
            bo_path,
            model.id,
            content_hash or "",
            lane_hash,
            f"serve invocation failed: {exc}",
            ts,
            confirmations,
        )
        return None
    if proc.returncode != 0:
        logger.warning("model_probe: serve returned rc=%s for %s: %s", proc.returncode, model.id, proc.stderr[:300])
        _record_backoff_failure(
            bo_path,
            model.id,
            content_hash or "",
            lane_hash,
            f"serve rc={proc.returncode}: {proc.stderr[:200]}",
            ts,
            confirmations,
        )
        return None
    try:
        serve_payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        logger.warning("model_probe: serve emitted non-JSON for %s", model.id)
        _record_backoff_failure(
            bo_path,
            model.id,
            content_hash or "",
            lane_hash,
            "serve emitted non-JSON",
            ts,
            confirmations,
        )
        return None

    log_file = serve_payload.get("log_file")
    ready = bool(serve_payload.get("ready"))
    try:
        if not ready or not log_file:
            tail = serve_payload.get("log_tail") or proc.stderr
            logger.warning("model_probe: %s did not reach readiness: %s", model.id, str(tail)[:300])
            _record_backoff_failure(
                bo_path,
                model.id,
                content_hash or "",
                lane_hash,
                f"not ready: {str(tail)[:200]}",
                ts,
                confirmations,
            )
            return None
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            log_text = fh.read()
    except OSError as exc:
        logger.warning("model_probe: could not read log_file %s: %s", log_file, exc)
        _record_backoff_failure(
            bo_path,
            model.id,
            content_hash or "",
            lane_hash,
            f"log read failed: {exc}",
            ts,
            confirmations,
        )
        return None
    finally:
        _stop_model(cli, model.id)

    fit = _parse_vllm_fit_log(log_text)
    if fit.max_seq_len is None or fit.max_concurrency is None:
        logger.warning(
            "model_probe: %s log missing fit lines (max_seq_len=%s concurrency=%s)",
            model.id,
            fit.max_seq_len,
            fit.max_concurrency,
        )
        _record_backoff_failure(
            bo_path,
            model.id,
            content_hash or "",
            lane_hash,
            f"log missing fit lines (max_seq_len={fit.max_seq_len} concurrency={fit.max_concurrency})",
            ts,
            confirmations,
        )
        return None

    fits_full_context = fit.max_concurrency >= 1.0
    advertised = fit.max_seq_len  # the window vLLM actually resolved + tested
    achievable = fit.kv_cache_tokens if fit.kv_cache_tokens is not None else fit.max_seq_len
    probe = ModelFitProbe(
        model_id=model.id,
        content_hash=content_hash or "",
        advertised_context_tokens=int(advertised),
        achievable_context_tokens=int(achievable),
        max_concurrency=float(fit.max_concurrency),
        fits_full_context=bool(fits_full_context),
        probed_at=ts,
    )
    usage_log.upsert_model_fit_probe(
        model_id=probe.model_id,
        content_hash=probe.content_hash,
        advertised_context_tokens=probe.advertised_context_tokens,
        achievable_context_tokens=probe.achievable_context_tokens,
        max_concurrency=probe.max_concurrency,
        fits_full_context=probe.fits_full_context,
        probed_at=probe.probed_at,
    )
    _clear_backoff(bo_path, model.id)
    logger.info(
        "model_probe: %s fits_full_context=%s advertised=%s achievable=%s concurrency=%.3fx",
        model.id,
        probe.fits_full_context,
        probe.advertised_context_tokens,
        probe.achievable_context_tokens,
        probe.max_concurrency,
    )
    return probe
