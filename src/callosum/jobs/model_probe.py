"""One-shot per-model full-context fit probe runner.

A resumable Dispatch-style batch job: enumerate local vLLM models, probe each
that has not been measured for its current artifact (content hash), serially,
and persist results to the usage_log. Probes are one-time per artifact — once a
model's content hash has a recorded result it is skipped until the artifact
changes (re-pull / version bump).

SIGTERM is cooperative: the job finishes the model it is currently probing,
then exits before starting another. The probe subprocess cannot be interrupted
mid-load without risking a half-started vLLM instance, so we let the in-flight
probe complete and stop cleanly.

Unlike the peer-quality sidecar, this job is synchronous (the probe shells out
to local-llm rather than driving async callosum backends) and does not need
a durable candidate queue — the fleet is small and stable, so candidates are
self-discovered each run and dedup is by the persisted result row.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
from pathlib import Path

from callosum.local import LocalModelRegistrySource, ModelEntry
from callosum.model_probe import (
    DEFAULT_CLI_COMMAND,
    DEFAULT_SERVE_TIMEOUT_S,
    execute_model_probe,
)
from callosum.usage_log import UsageLog

logger = logging.getLogger(__name__)

_should_exit = False


def _on_sigterm(signum: int, frame: object) -> None:
    del signum, frame
    global _should_exit
    _should_exit = True


def _probe_candidates(source: LocalModelRegistrySource) -> list[ModelEntry]:
    """Local vLLM models worth probing (enabled, responses-capable).

    Probing a precision-rejected model wastes one load, but the dedup makes it
    one-time and harmless; filtering to precision-admitted candidates only is a
    deferred refinement (see work tracker).
    """
    return [
        m
        for m in source.models()
        if m.enabled and m.runtime == "vllm" and "responses" in m.api_surfaces
    ]


def run(
    *,
    db_path: Path,
    max_models: int | None = None,
    cli_command: list[str] | None = None,
    serve_timeout_s: float = DEFAULT_SERVE_TIMEOUT_S,
) -> tuple[dict[str, int], int]:
    if not db_path.exists():
        print(f"db not found: {db_path}", file=sys.stderr)
        return {"probed": 0, "skipped": 0, "failed": 0}, 1
    cli = cli_command if cli_command is not None else DEFAULT_CLI_COMMAND
    usage_log = UsageLog(db_path)
    source = LocalModelRegistrySource(cli_command=cli)
    candidates = _probe_candidates(source)
    logger.info("model_probe: %d candidate(s)", len(candidates))

    probed = 0
    skipped = 0
    failed = 0
    for model in candidates:
        if _should_exit:
            logger.info("model_probe: SIGTERM; exiting before next model")
            break
        if max_models is not None and probed + failed >= max_models:
            logger.info("model_probe: reached --max-models")
            break
        try:
            probe = execute_model_probe(
                model,
                usage_log=usage_log,
                cli_command=cli,
                serve_timeout_s=serve_timeout_s,
                pool_bytes=model.local_pool_bytes,
            )
        except Exception as exc:  # noqa: BLE001 — a probe failure must not abort the batch
            failed += 1
            logger.warning("model_probe: %s raised %s: %s", model.id, type(exc).__name__, exc)
            continue
        if probe is None:
            skipped += 1
        else:
            probed += 1
    return {"probed": probed, "skipped": skipped, "failed": failed, "candidates": len(candidates)}, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, type=Path)
    parser.add_argument("--max-models", type=int, help="Cap of probes to attempt this run.")
    parser.add_argument(
        "--serve-timeout",
        type=float,
        default=DEFAULT_SERVE_TIMEOUT_S,
        help="Per-model serve readiness timeout in seconds.",
    )
    parser.add_argument(
        "--cli",
        nargs="+",
        default=list(DEFAULT_CLI_COMMAND),
        help="local-llm CLI command (default: local-llm).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    signal.signal(signal.SIGTERM, _on_sigterm)
    summary, rc = run(
        db_path=args.db_path,
        max_models=args.max_models,
        cli_command=args.cli,
        serve_timeout_s=args.serve_timeout,
    )
    json.dump(summary, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())