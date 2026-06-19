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

Sweep mode also writes a top-level sweep_report.json across all N values
so the degradation curve is visible at a glance.

Usage:
    CALLOSUM_TOKEN=... python scripts/probe_concurrency.py \\
        --model model-a0a0 \\
        --path proxy-direct \\
        --n 10

    # Sweep across multiple N to find the degradation knee.
    CALLOSUM_TOKEN=... python scripts/probe_concurrency.py \\
        --model model-a0a0 \\
        --path proxy-direct \\
        --sweep 1,5,10,20

    # Vary prompts across the burst so we aren't just sending N copies of
    # the same string (which lets caches/serialization mask real behavior).
    CALLOSUM_TOKEN=... python scripts/probe_concurrency.py \\
        --model model-a0a0 \\
        --path proxy-direct --n 10 --prompt-source varied

Safe defaults: N=5 single-burst, prompt-source=static. Pass higher N or
--sweep explicitly when you want to stress-test.
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

# Varied prompts — short, low-token, cover a range of shapes. Used when
# --prompt-source=varied so the burst exercises the cache/serialization
# layer instead of N copies of one string.
VARIED_PROMPTS: list[str] = [
    "Reply with exactly: OK",
    "What is 2+2? One number only.",
    "Say the word: ping",
    "Reply with exactly: ACK",
    "Output the single character: A",
    "What color is the sky? One word.",
    "Reply with exactly: pong",
    "Count to three. Comma-separated.",
    "What is 5*5? One number only.",
    "Say the word: ready",
]


def _slugify(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def resolve_proxy_endpoint(model: str) -> str | None:
    try:
        proc = subprocess.run(
            ["local-llm", "models", "local", "--json"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
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


def build_url_body_headers(model: str, path: str, prompt: str) -> tuple[str, dict, dict]:
    body_responses: dict = {
        "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}],
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
        body_responses["model"] = model.replace("-ollama-responses-proxy", "-local").replace(
            "-responses-proxy-q4_k_m", "-local"
        )
        return f"{endpoint.rstrip('/')}/v1/responses", body_responses, headers

    if path == "callosum-v1":
        body_responses["model"] = model
        token = os.environ.get("CALLOSUM_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return f"{CALLOSUM_URL}/v1/responses", body_responses, headers

    if path == "callosum-codex":
        body_responses["model"] = model
        token = os.environ.get("CALLOSUM_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return f"{CALLOSUM_URL}/codex", body_responses, headers

    raise ValueError(f"unknown path: {path}")


def _classify_failure(result: dict) -> str:
    """Bucket a failed request into a coarse class for the summary."""
    err = result.get("error") or ""
    if err.startswith("TimeoutException") or "Timeout" in err:
        return "timeout"
    if err.startswith("ConnectError") or "Connect" in err:
        return "network"
    status = result.get("http_status")
    if isinstance(status, int):
        if 500 <= status < 600:
            return f"http_5xx ({status})"
        if 400 <= status < 500:
            return f"http_4xx ({status})"
    if err:
        return "exception"
    return "unknown"


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
            url,
            json=body,
            headers=headers,
            timeout=240.0,
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
                        if isinstance(c, dict) and c.get("type") == "output_text" and not content_excerpt:
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


def _percentile(sorted_vals: list[int], pct: float) -> int | None:
    """Return the value at the given percentile of a pre-sorted list.

    Uses the nearest-rank method. Returns None if there is no data.
    Computes any percentile down to one sample (p50 of a single sample
    is that sample); callers should still display p95/p99 only when
    the sample size is large enough to make those values meaningful.
    """
    if not sorted_vals:
        return None
    k = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
    return int(sorted_vals[k])


async def run_concurrency(
    *,
    model: str,
    path: str,
    n: int,
    prompt: str,
    prompt_source: str,
    output_root: Path | None = None,
) -> dict:
    output_dir = (output_root or OUTPUT_ROOT) / f"{_slugify(model)}__{path}__N{n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "requests").mkdir(exist_ok=True)

    # Build per-request prompts. For "static" all N requests share the
    # same prompt (lets caches/serialization mask behavior). For "varied"
    # we cycle through VARIED_PROMPTS so the burst is heterogeneous and
    # any per-prompt caching can't trivially serialize requests.
    if prompt_source == "varied":
        per_req_prompts = [VARIED_PROMPTS[i % len(VARIED_PROMPTS)] for i in range(n)]
    else:
        per_req_prompts = [prompt for _ in range(n)]

    per_req_targets: list[tuple[str, dict, dict]] = [build_url_body_headers(model, path, p) for p in per_req_prompts]

    # Use a fresh async client per run. Shared connection-pool intentional
    # so concurrent requests do share TCP connections — that's part of
    # what we're stress-testing.
    async with httpx.AsyncClient(http2=False) as client:
        coros = [
            fire_one(
                client,
                url=per_req_targets[i][0],
                body=per_req_targets[i][1],
                headers=per_req_targets[i][2],
                req_id=i,
                output_dir=output_dir / "requests",
            )
            for i in range(n)
        ]
        wall_start = time.time()
        results = await asyncio.gather(*coros, return_exceptions=False)
        wall_ms = int((time.time() - wall_start) * 1000)

    succeeded = [r for r in results if r.get("http_status") == 200]
    failed = [r for r in results if r.get("http_status") != 200]
    latencies = sorted([r["latency_ms"] for r in succeeded])
    distinct_response_ids = {r.get("upstream_response_id") for r in succeeded if r.get("upstream_response_id")}
    distinct_content = {r.get("content_excerpt") for r in succeeded if r.get("content_excerpt")}
    # Bucket failures by class — most-common-first.
    failure_classes: dict[str, int] = {}
    for r in failed:
        cls = _classify_failure(r)
        failure_classes[cls] = failure_classes.get(cls, 0) + 1
    summary: dict = {
        "model": model,
        "path": path,
        "n": n,
        "prompt_source": prompt_source,
        "wall_ms": wall_ms,
        "success_count": len(succeeded),
        "failure_count": len(failed),
        "failure_classes": failure_classes,
        "success_rate": len(succeeded) / n if n > 0 else 0.0,
        "latency_ms_min": min(latencies) if latencies else None,
        "latency_ms_max": max(latencies) if latencies else None,
        "latency_ms_mean": int(statistics.mean(latencies)) if latencies else None,
        "latency_ms_p50": _percentile(latencies, 50),
        "latency_ms_p90": _percentile(latencies, 90),
        "latency_ms_p95": _percentile(latencies, 95),
        "latency_ms_p99": _percentile(latencies, 99),
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
                "implication": (
                    "two or more concurrent requests got the same upstream response id; "
                    "investigate for stale-cache or session-leak"
                ),
            }
    if cross_talk_warning:
        summary["cross_talk_warning"] = cross_talk_warning
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    return summary


def _print_summary_brief(summary: dict) -> None:
    """One compact line per-burst for the sweep console output."""
    n = summary["n"]
    ok = summary["success_count"]
    rate = summary["success_rate"]
    p50 = summary.get("latency_ms_p50") or 0
    p95 = summary.get("latency_ms_p95") or 0
    p99 = summary.get("latency_ms_p99") or 0
    wall = summary["wall_ms"]
    fail_class = summary.get("failure_classes") or {}
    fc_str = ", ".join(f"{k}={v}" for k, v in fail_class.items()) if fail_class else "-"
    cross = "cross-talk!" if summary.get("cross_talk_warning") else "no-cross-talk"
    print(
        f"N={n:>3}  ok={ok}/{n}  rate={rate:.2f}  "
        f"p50={p50}ms p95={p95}ms p99={p99}ms  "
        f"wall={wall}ms  fail=[{fc_str}]  {cross}"
    )


async def run_sweep(
    *,
    model: str,
    path: str,
    n_values: list[int],
    prompt: str,
    prompt_source: str,
) -> dict:
    """Run a sequence of bursts at increasing N. Writes a sweep_report
    alongside the per-N summaries."""
    sweep_root = OUTPUT_ROOT / (f"_sweep__{_slugify(model)}__{path}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    sweep_root.mkdir(parents=True, exist_ok=True)

    bursts: list[dict] = []
    print(f"Sweeping {model} via {path} across N={n_values} (prompt_source={prompt_source})")
    for n in n_values:
        summary = await run_concurrency(
            model=model,
            path=path,
            n=n,
            prompt=prompt,
            prompt_source=prompt_source,
            output_root=sweep_root,
        )
        _print_summary_brief(summary)
        # Strip the per-request results from the sweep-level rollup — the
        # per-burst summary.json already has them. Keep the sweep report
        # compact.
        compact = {k: v for k, v in summary.items() if k != "results"}
        bursts.append(compact)

    # Identify the degradation knee: smallest N where success_rate < 1.0.
    knee_n: int | None = None
    for b in bursts:
        if b["success_rate"] < 1.0:
            knee_n = b["n"]
            break

    sweep_report = {
        "model": model,
        "path": path,
        "prompt_source": prompt_source,
        "n_values": n_values,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "knee_n": knee_n,
        "bursts": bursts,
    }
    (sweep_root / "sweep_report.json").write_text(json.dumps(sweep_report, indent=2))
    print()
    print(f"Sweep report: {sweep_root / 'sweep_report.json'}")
    if knee_n is not None:
        print(f"Degradation knee: N={knee_n} (first N with success_rate < 1.0)")
    else:
        print("No degradation knee within the sweep — all N values ran clean.")
    return sweep_report


def _parse_sweep_arg(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            n = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid N value in sweep: {part!r}") from exc
        if n <= 0:
            raise argparse.ArgumentTypeError(f"sweep N values must be positive, got {n}")
        out.append(n)
    if not out:
        raise argparse.ArgumentTypeError("sweep must contain at least one N value")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument(
        "--path", default="proxy-direct", choices=["gateway", "proxy-direct", "callosum-v1", "callosum-codex"]
    )
    p.add_argument(
        "--n", type=int, default=5, help="Number of concurrent requests (default 5). Ignored if --sweep is set."
    )
    p.add_argument(
        "--sweep",
        type=_parse_sweep_arg,
        default=None,
        help="Comma-separated N values to sweep (e.g. '1,5,10,20'). "
        "Runs each burst in sequence and writes a sweep_report.json.",
    )
    p.add_argument("--prompt", default=DEFAULT_PROMPT_TEXT)
    p.add_argument(
        "--prompt-source",
        default="static",
        choices=["static", "varied"],
        help="static: send N copies of --prompt. varied: cycle through "
        "a built-in list of short prompts so the burst is heterogeneous "
        "(avoids cache-masking serialization).",
    )
    args = p.parse_args()

    if args.sweep is not None:
        asyncio.run(
            run_sweep(
                model=args.model,
                path=args.path,
                n_values=args.sweep,
                prompt=args.prompt,
                prompt_source=args.prompt_source,
            )
        )
        return 0

    print(f"Firing N={args.n} concurrent requests to {args.model} via {args.path}")
    summary = asyncio.run(
        run_concurrency(
            model=args.model,
            path=args.path,
            n=args.n,
            prompt=args.prompt,
            prompt_source=args.prompt_source,
        )
    )
    print(
        json.dumps(
            {
                "wall_ms": summary["wall_ms"],
                "success_rate": summary["success_rate"],
                "success_count": summary["success_count"],
                "failure_count": summary["failure_count"],
                "failure_classes": summary["failure_classes"],
                "latency_p50_ms": summary.get("latency_ms_p50"),
                "latency_p90_ms": summary.get("latency_ms_p90"),
                "latency_p95_ms": summary.get("latency_ms_p95"),
                "latency_p99_ms": summary.get("latency_ms_p99"),
                "distinct_response_ids": summary["distinct_response_id_count"],
                "distinct_content_excerpts": summary["distinct_content_excerpt_count"],
                "cross_talk_warning": summary.get("cross_talk_warning"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
