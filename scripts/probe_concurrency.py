#!/usr/bin/env python3
"""Concurrency probe: fire N parallel requests at a model endpoint and
report success rate + latency percentiles + cross-talk indicators.

Single-script intentionally — separate from the scenario battery
because the question is different: "how does the serving path behave
under concurrent load" rather than "what does the model emit for
prompt X."

Outputs to tests/integration/fixtures/probed/_concurrency/<model>__<path>__N<n>/
    summary.json     aggregated metrics
    requests/        per-request capture (request body + response + latency)

Usage:
    CODEX_PROXY_TOKEN=... python scripts/probe_concurrency.py \\
        --model model-a0a0 \\
        --path proxy-direct \\
        --n 10

Safe defaults: N=5 to avoid load-spiking the host. Pass --n higher
explicitly when you want to stress-test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "tests" / "integration" / "fixtures" / "probed" / "_concurrency"
GATEWAY_URL = os.environ.get("CALLOSUM_LITELLM_GATEWAY_URL", "http://127.0.0.1:4000")
CALLOSUM_URL = os.environ.get("CALLOSUM_BASE_URL", "http://127.0.0.1:8765")

# A small, deterministic prompt — keeps the test reproducible and the
# response shapes comparable across runs.
DEFAULT_PROMPT_TEXT = "Reply with exactly: OK"


def _slugify(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def resolve_proxy_endpoint(model: str) -> str | None:
    try:
        proc = subprocess.run(
            ["local-llm", "models", "local", "--json"],
            capture_output=True, text=True, check=False, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    for entry in data.get("entries", []):
        m = entry.get("model", {}) or {}
        if m.get("id") == model:
            return m.get("endpoint")
    return None


def build_url_body_headers(
    model: str, path: str, prompt: str
) -> tuple[str, dict, dict]:
    body_responses: dict = {
        "input": [
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": prompt}]}
        ],
        "max_output_tokens": 50,
        "temperature": 0,
        "stream": False,
    }
    body_chat: dict = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50,
        "temperature": 0,
        "stream": False,
        "model": model,
    }
    headers: dict = {}

    if path == "gateway":
        body_chat["model"] = model
        key = os.environ.get("CALLOSUM_LITELLM_MASTER_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return f"{GATEWAY_URL}/v1/chat/completions", body_chat, headers

    if path == "proxy-direct":
        endpoint = resolve_proxy_endpoint(model)
        if endpoint is None:
            raise RuntimeError(f"no proxy endpoint registered for {model!r}")
        body_responses["model"] = (
            model.replace("-ollama-responses-proxy", "-local")
                 .replace("-responses-proxy-q4_k_m", "-local")
        )
        return f"{endpoint.rstrip('/')}/v1/responses", body_responses, headers

    if path == "callosum-v1":
        body_responses["model"] = model
        token = os.environ.get("CODEX_PROXY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return f"{CALLOSUM_URL}/v1/responses", body_responses, headers

    if path == "callosum-codex":
        body_responses["model"] = model
        token = os.environ.get("CODEX_PROXY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return f"{CALLOSUM_URL}/codex", body_responses, headers

    raise ValueError(f"unknown path: {path}")


async def fire_one(
    client: httpx.AsyncClient,
    *,
    url: str,
    body: dict,
    headers: dict,
    req_id: int,
    output_dir: Path,
) -> dict:
    t0 = time.time()
    try:
        response = await client.post(
            url, json=body, headers=headers, timeout=240.0,
        )
        latency_ms = int((time.time() - t0) * 1000)
        raw = response.content
        (output_dir / f"req_{req_id:03d}_response.raw").write_bytes(raw)
        try:
            parsed = response.json()
        except Exception:
            parsed = None
        # Extract a content excerpt + any response id from either shape.
        content_excerpt = None
        upstream_response_id = None
        if isinstance(parsed, dict):
            # chat-completions shape
            ch = (parsed.get("choices") or [{}])[0]
            if isinstance(ch, dict):
                msg = ch.get("message") or {}
                if isinstance(msg.get("content"), str):
                    content_excerpt = msg["content"][:100]
            upstream_response_id = parsed.get("id")
            # responses-API shape
            for item in parsed.get("output", []) or []:
                if isinstance(item, dict) and item.get("type") == "message":
                    for c in item.get("content", []) or []:
                        if (
                            isinstance(c, dict)
                            and c.get("type") == "output_text"
                            and not content_excerpt
                        ):
                            content_excerpt = c.get("text", "")[:100]
        return {
            "req_id": req_id,
            "http_status": response.status_code,
            "latency_ms": latency_ms,
            "upstream_response_id": upstream_response_id,
            "content_excerpt": content_excerpt,
            "error": None,
        }
    except Exception as exc:
        latency_ms = int((time.time() - t0) * 1000)
        return {
            "req_id": req_id,
            "http_status": None,
            "latency_ms": latency_ms,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def run_concurrency(
    *, model: str, path: str, n: int, prompt: str
) -> dict:
    url, body, headers = build_url_body_headers(model, path, prompt)
    output_dir = (
        OUTPUT_ROOT / f"{_slugify(model)}__{path}__N{n}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "requests").mkdir(exist_ok=True)

    # Use a fresh async client per run. Shared connection-pool intentional
    # so concurrent requests do share TCP connections — that's part of
    # what we're stress-testing.
    async with httpx.AsyncClient(http2=False) as client:
        coros = [
            fire_one(
                client, url=url, body=body, headers=headers,
                req_id=i, output_dir=output_dir / "requests",
            )
            for i in range(n)
        ]
        wall_start = time.time()
        results = await asyncio.gather(*coros, return_exceptions=False)
        wall_ms = int((time.time() - wall_start) * 1000)

    succeeded = [r for r in results if r.get("http_status") == 200]
    failed = [r for r in results if r.get("http_status") != 200]
    latencies = [r["latency_ms"] for r in succeeded]
    distinct_response_ids = {
        r.get("upstream_response_id") for r in succeeded
        if r.get("upstream_response_id")
    }
    distinct_content = {
        r.get("content_excerpt") for r in succeeded
        if r.get("content_excerpt")
    }
    summary: dict = {
        "model": model,
        "path": path,
        "n": n,
        "wall_ms": wall_ms,
        "success_count": len(succeeded),
        "failure_count": len(failed),
        "success_rate": len(succeeded) / n if n > 0 else 0.0,
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "latency_ms_mean": int(statistics.mean(latencies)) if latencies else None,
        "latency_ms_p50": int(statistics.median(latencies)) if latencies else None,
        "latency_ms_p90": (
            int(sorted(latencies)[int(len(latencies) * 0.9)])
            if len(latencies) >= 10 else None
        ),
        "distinct_response_id_count": len(distinct_response_ids),
        "distinct_content_excerpt_count": len(distinct_content),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Cross-talk warning: if any two requests got the same upstream
    # response id (possible if a cache layer or session-binding incorrectly
    # shares state), surface it.
    cross_talk_warning = None
    if len(succeeded) > 1:
        seen: dict[str, list[int]] = {}
        for r in succeeded:
            rid = r.get("upstream_response_id")
            if rid:
                seen.setdefault(rid, []).append(r["req_id"])
        dupes = {k: v for k, v in seen.items() if len(v) > 1}
        if dupes:
            cross_talk_warning = {
                "duplicate_response_ids": dupes,
                "implication": "two or more concurrent requests got the same upstream response id; investigate for stale-cache or session-leak",
            }
    if cross_talk_warning:
        summary["cross_talk_warning"] = cross_talk_warning
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--path", default="proxy-direct",
                   choices=["gateway", "proxy-direct", "callosum-v1", "callosum-codex"])
    p.add_argument("--n", type=int, default=5,
                   help="Number of concurrent requests (default 5)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT_TEXT)
    args = p.parse_args()

    print(f"Firing N={args.n} concurrent requests to {args.model} via {args.path}")
    summary = asyncio.run(run_concurrency(
        model=args.model, path=args.path, n=args.n, prompt=args.prompt,
    ))
    print(json.dumps({
        "wall_ms": summary["wall_ms"],
        "success_rate": summary["success_rate"],
        "success_count": summary["success_count"],
        "failure_count": summary["failure_count"],
        "latency_p50_ms": summary.get("latency_ms_p50"),
        "latency_p90_ms": summary.get("latency_ms_p90"),
        "distinct_response_ids": summary["distinct_response_id_count"],
        "distinct_content_excerpts": summary["distinct_content_excerpt_count"],
        "cross_talk_warning": summary.get("cross_talk_warning"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
