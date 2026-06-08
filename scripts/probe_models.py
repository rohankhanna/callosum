#!/usr/bin/env python3
"""Black-box model probing: run a battery of scenarios against every local
model the callosum-fronted gateway advertises, capture verbatim wire data,
and dump a per-(model, scenario) directory of artifacts that the next phase
analyzes and uses to design transforms.

Usage:
    CALLOSUM_LITELLM_MASTER_KEY=... python scripts/probe_models.py [--model M] [--scenario S]

Outputs to tests/integration/fixtures/probed/<model-slug>/<scenario-id>/:
    request.json        the exact body posted to the gateway
    response.json       the parsed response (for non-stream calls)
    response.raw        the raw response body bytes
    metadata.json       latency, http status, finish_reason summary, error
    expectations.md     verbatim expectations from the battery, for reference
                        during analysis

This script does NOT modify production state. It only POSTs to the
gateway's /v1/chat/completions and reads /admin/status. Safe to re-run.

The list of models to probe is read from callosum's /admin/status
(only litellm_gateway-kind backends, since those are what we don't
know the output shape of). The remote credential_proxy / codex
backends use the standard Responses API and are not probed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib import error, request as urlrequest

REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY_PATH = REPO_ROOT / "tests" / "integration" / "probes" / "battery.json"
OUTPUT_ROOT = REPO_ROOT / "tests" / "integration" / "fixtures" / "probed"

GATEWAY_URL = os.environ.get(
    "CALLOSUM_LITELLM_GATEWAY_URL", "http://127.0.0.1:4000"
)
CALLOSUM_URL = os.environ.get(
    "CALLOSUM_BASE_URL", "http://127.0.0.1:8765"
)

# Per-request timeout. Generous because some local models on first
# load (cold start) take 60s+ to respond. Loop-prone models will hit
# this ceiling and we'll capture finish_reason='length' or a timeout
# error — both are useful signals.
TIMEOUT_S = 180.0


def _slugify(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _http_request(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = TIMEOUT_S,
) -> tuple[int, bytes]:
    headers = dict(headers or {})
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, method=method, headers=headers, data=data)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except error.HTTPError as e:
        return e.code, e.read()


def list_local_models() -> list[str]:
    """Pull every model advertised by a litellm_gateway-kind backend from
    callosum's live /status. De-duplicates across multiple gateway backends."""
    status_code, body = _http_request("GET", f"{CALLOSUM_URL}/status")
    if status_code != 200:
        sys.exit(f"failed to GET /status from callosum: {status_code} {body[:200]!r}")
    data = json.loads(body)
    models: set[str] = set()
    for b in data.get("backends", []):
        if b.get("kind") != "litellm_gateway":
            continue
        for m in b.get("advertised_models", []):
            models.add(m)
    return sorted(models)


def run_one_probe(
    model: str, scenario: dict, output_dir: Path
) -> dict:
    """Run a single (model, scenario) pair against the gateway. Returns a
    metadata summary; writes captures to disk under output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    request_body = {**scenario["request"], "model": model, "stream": False}
    request_path = output_dir / "request.json"
    request_path.write_text(json.dumps(request_body, indent=2))

    master_key = os.environ.get("CALLOSUM_LITELLM_MASTER_KEY", "")
    headers = {}
    if master_key:
        headers["Authorization"] = f"Bearer {master_key}"

    t0 = time.time()
    status_code, raw = _http_request(
        "POST",
        f"{GATEWAY_URL}/v1/chat/completions",
        body=request_body,
        headers=headers,
    )
    latency_ms = int((time.time() - t0) * 1000)

    (output_dir / "response.raw").write_bytes(raw)

    parsed: dict | None = None
    parse_error: str | None = None
    try:
        parsed = json.loads(raw.decode("utf-8", errors="replace"))
        (output_dir / "response.json").write_text(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as exc:
        parse_error = f"JSON decode failed: {exc}"

    # Pull useful summary fields if the response parsed.
    finish_reason: str | None = None
    content_preview: str | None = None
    tool_calls: list | None = None
    if isinstance(parsed, dict):
        choices = parsed.get("choices") or []
        if choices and isinstance(choices[0], dict):
            finish_reason = choices[0].get("finish_reason")
            msg = choices[0].get("message", {}) or {}
            raw_content = msg.get("content")
            if isinstance(raw_content, str):
                content_preview = raw_content[:500]
            tc = msg.get("tool_calls")
            if isinstance(tc, list):
                tool_calls = tc

    metadata = {
        "model": model,
        "scenario_id": scenario["id"],
        "scenario_description": scenario["description"],
        "category": scenario.get("category", "?"),
        "http_status": status_code,
        "latency_ms": latency_ms,
        "parse_error": parse_error,
        "finish_reason": finish_reason,
        "content_preview": content_preview,
        "tool_calls_present": tool_calls is not None,
        "tool_calls": tool_calls,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    expectations = scenario.get("expectations", [])
    (output_dir / "expectations.md").write_text(
        f"# {scenario['id']}\n\n"
        f"{scenario['description']}\n\n"
        "## Expectations\n\n"
        + "\n".join(f"- {e}" for e in expectations)
        + "\n"
    )

    return metadata


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model", default=None,
        help="Restrict to a single model slug.",
    )
    p.add_argument(
        "--scenario", default=None,
        help="Restrict to a single scenario id.",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List the models and scenarios that WOULD run, then exit.",
    )
    args = p.parse_args()

    if not BATTERY_PATH.exists():
        sys.exit(f"battery not found at {BATTERY_PATH}")
    battery = json.loads(BATTERY_PATH.read_text())
    scenarios: list[dict] = []
    for cat in ("generic", "codex"):
        for s in battery.get(cat, []):
            scenarios.append({**s, "category": cat})
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            sys.exit(f"scenario {args.scenario!r} not found in battery")

    models = list_local_models()
    if args.model:
        if args.model not in models:
            sys.exit(
                f"model {args.model!r} not in current routing pool. "
                f"available: {models}"
            )
        models = [args.model]

    if args.list:
        print(f"Models ({len(models)}):")
        for m in models:
            print(f"  {m}")
        print(f"Scenarios ({len(scenarios)}):")
        for s in scenarios:
            print(f"  [{s['category']}] {s['id']}")
        return 0

    print(f"Probing {len(models)} model(s) × {len(scenarios)} scenario(s) = {len(models) * len(scenarios)} runs")
    summary_rows: list[dict] = []
    for model in models:
        slug = _slugify(model)
        for scenario in scenarios:
            output_dir = OUTPUT_ROOT / slug / scenario["id"]
            print(f"  {model} :: {scenario['id']} ... ", end="", flush=True)
            try:
                meta = run_one_probe(model, scenario, output_dir)
                summary_rows.append(meta)
                fr = meta.get("finish_reason")
                hs = meta.get("http_status")
                tc = "tool_calls" if meta.get("tool_calls_present") else ""
                print(f"http={hs} finish={fr} {tc} ({meta['latency_ms']}ms)")
            except Exception as exc:
                print(f"ERROR: {type(exc).__name__}: {exc}")
                summary_rows.append({
                    "model": model,
                    "scenario_id": scenario["id"],
                    "error": f"{type(exc).__name__}: {exc}",
                })

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
