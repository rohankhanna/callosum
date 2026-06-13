#!/usr/bin/env python3
"""Bounded Codex CLI repro harness for callosum local-only routing.

This script is meant to be run by the operator after manually switching
callosum to local-only. It does not change routing mode. It captures:

- callosum status before and after the run
- callosum health
- local-llm model registry summary
- the exact Codex command's stdout, stderr, exit code, and duration

Artifacts are written under tmp/codex-local-only-repro/<timestamp>/.

Examples:

    scripts/codex_local_only_repro.py --case smoke --timeout-s 180
    scripts/codex_local_only_repro.py --case repo_inspect --timeout-s 240
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error, request

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "tmp" / "codex-local-only-repro"
DEFAULT_TIMEOUT_S = 180.0
DEFAULT_PROMPT = 'Reply with exactly this JSON object and nothing else: {"callosum_local_only_probe":"ok"}'
CASES: dict[str, str] = {
    "smoke": DEFAULT_PROMPT,
    "repo_inspect": (
        "Do not modify files. Inspect README.md and The Project Documentation, then answer "
        "with exactly two bullets: one naming the repo purpose, one naming one non-goal."
    ),
    "tool_roundtrip": (
        "Do not modify files. Use the shell to run pwd and list the top-level Python "
        "source files under src/callosum. Then answer with a concise summary."
    ),
    "loop_risk": (
        "Do not modify files. Think briefly and answer in at most 120 words: explain "
        "why a local model router might appear to loop even when the router itself is "
        "returning HTTP 200. Stop after the answer."
    ),
}


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _run(
    argv: list[str],
    *,
    timeout_s: float,
    cwd: Path = REPO_ROOT,
) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        timed_out = False
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = _decode_timeout_payload(exc.stdout)
        stderr = _decode_timeout_payload(exc.stderr)
        returncode = None
    ended = time.time()
    return {
        "argv": argv,
        "returncode": returncode,
        "timed_out": timed_out,
        "started_at": started,
        "ended_at": ended,
        "duration_s": ended - started,
        "stdout": stdout,
        "stderr": stderr,
    }


def _decode_timeout_payload(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _http_get(url: str, *, timeout_s: float = 5.0) -> dict[str, Any]:
    started = time.time()
    try:
        with request.urlopen(url, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return {
                "ok": True,
                "status": resp.status,
                "duration_s": time.time() - started,
                "body": body,
            }
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {
            "ok": False,
            "status": exc.code,
            "duration_s": time.time() - started,
            "body": body,
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": None,
            "duration_s": time.time() - started,
            "error": repr(exc),
        }


def _load_status(raw: dict[str, Any]) -> dict[str, Any] | None:
    if raw.get("returncode") != 0:
        return None
    try:
        parsed = json.loads(str(raw.get("stdout") or ""))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _summarize_models(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("returncode") != 0:
        return {"ok": False, "error": "local-llm command failed"}
    try:
        payload = json.loads(str(raw.get("stdout") or ""))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"non-json local-llm output: {exc}"}
    entries = payload.get("entries") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        return {"ok": False, "error": "local-llm output missing entries[]"}
    enabled_responses: list[str] = []
    enabled_chat_only: list[str] = []
    disabled: list[str] = []
    for entry in entries:
        model = entry.get("model") if isinstance(entry, dict) else None
        if not isinstance(model, dict):
            continue
        model_id = model.get("id")
        if not isinstance(model_id, str):
            continue
        if not bool(model.get("enabled", True)):
            disabled.append(model_id)
            continue
        surfaces = model.get("api_surfaces")
        surfaces_set = set(surfaces) if isinstance(surfaces, list) else set()
        if "responses" in surfaces_set:
            enabled_responses.append(model_id)
        elif "chat" in surfaces_set:
            enabled_chat_only.append(model_id)
    return {
        "ok": True,
        "enabled_responses": sorted(enabled_responses),
        "enabled_chat_only": sorted(enabled_chat_only),
        "disabled_count": len(disabled),
    }


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=DEFAULT_TIMEOUT_S,
        help=f"Wall-clock timeout for the Codex command. Default: {DEFAULT_TIMEOUT_S:g}.",
    )
    parser.add_argument(
        "--allow-non-local-only",
        action="store_true",
        help="Run even if callosum status does not report routing=local-only.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Artifact directory. Default: tmp/codex-local-only-repro/<UTC timestamp>.",
    )
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="smoke",
        help="Predefined Codex prompt to run when no explicit command is supplied.",
    )
    parser.add_argument(
        "codex_command",
        nargs=argparse.REMAINDER,
        help=(
            "Command to run after '--'. Default: codex exec <small JSON prompt>. "
            "Example: -- codex exec --skip-git-repo-check 'say ok'"
        ),
    )
    args = parser.parse_args(argv)

    out_dir = args.output_dir or (OUTPUT_ROOT / _utc_stamp())
    out_dir.mkdir(parents=True, exist_ok=False)

    status_before = _run(["callosum", "status"], timeout_s=15.0)
    parsed_status = _load_status(status_before)
    routing = parsed_status.get("routing") if parsed_status is not None else None
    if routing != "local-only" and not args.allow_non_local_only:
        _write_json(out_dir / "status_before.json", status_before)
        status_error = str(status_before.get("stderr") or "").strip()
        print(
            f"Refusing to run Codex because callosum routing is {routing!r}, not 'local-only'.",
            file=sys.stderr,
        )
        if status_error:
            print(f"Status command stderr: {status_error}", file=sys.stderr)
        print(f"Wrote partial artifacts to {out_dir}", file=sys.stderr)
        return 2

    health = _http_get(os.environ.get("CALLOSUM_BASE_URL", "http://127.0.0.1:8765") + "/health")
    local_models_raw = _run(["local-llm", "models", "local", "--json"], timeout_s=20.0)
    local_models_summary = _summarize_models(local_models_raw)

    command = list(args.codex_command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        command = ["codex", "exec", CASES[args.case]]

    codex_result = _run(command, timeout_s=args.timeout_s)
    status_after = _run(["callosum", "status"], timeout_s=15.0)

    _write_json(out_dir / "status_before.json", status_before)
    _write_json(out_dir / "status_after.json", status_after)
    _write_json(out_dir / "health.json", health)
    _write_json(out_dir / "local_models_summary.json", local_models_summary)
    _write_json(out_dir / "codex_result.json", _without_large_text(codex_result))
    _write_text(out_dir / "codex.stdout.txt", str(codex_result.get("stdout") or ""))
    _write_text(out_dir / "codex.stderr.txt", str(codex_result.get("stderr") or ""))
    _write_json(
        out_dir / "summary.json",
        {
            "routing_before": routing,
            "routing_after": (_load_status(status_after) or {}).get("routing"),
            "codex_returncode": codex_result.get("returncode"),
            "codex_timed_out": codex_result.get("timed_out"),
            "codex_duration_s": codex_result.get("duration_s"),
            "case": args.case,
            "output_dir": str(out_dir),
        },
    )

    print(f"Artifacts: {out_dir}")
    print(
        "Codex: "
        f"returncode={codex_result.get('returncode')} "
        f"timed_out={codex_result.get('timed_out')} "
        f"duration_s={codex_result.get('duration_s'):.3f}"
    )
    print("After this run, restore the router yourself: callosum routing set remote-only")
    return 124 if codex_result.get("timed_out") else int(codex_result.get("returncode") or 0)


def _without_large_text(result: dict[str, Any]) -> dict[str, Any]:
    slim = dict(result)
    slim["stdout_path"] = "codex.stdout.txt"
    slim["stderr_path"] = "codex.stderr.txt"
    slim.pop("stdout", None)
    slim.pop("stderr", None)
    return slim


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
