#!/usr/bin/env python3
"""Black-box model probing harness (B2 multi-path + streaming).

Drives the battery defined at tests/integration/probes/battery.json
against every routable local model along multiple dispatch paths and
captures verbatim wire data plus a structured fitness summary.

PATHS
-----
gateway          POST to the litellm gateway at $CALLOSUM_LITELLM_GATEWAY_URL
                 (chat-completions; only works for gateway-served models)
proxy-direct     POST to the per-model dedicated proxy endpoint resolved
                 from local LLM gateway's registry (responses-API; works for
                 *-responses-proxy models)
callosum-v1      POST to callosum's /v1/chat/completions or /v1/responses
                 (path picked by the scenario's request_shape)
callosum-codex   POST to callosum's /codex (responses-only)

MODES
-----
non_stream  buffered request, JSON response
stream      streaming request, parsed SSE events

Outputs land at:
    tests/integration/fixtures/probed/<model>/<scenario>/<path>__<mode>/
        request.json     the exact body posted
        response.raw     raw response bytes (or full SSE byte stream)
        response.json    parsed JSON (non_stream) or list of SSE events (stream)
        metadata.json    latency, http status, finish reason, error
        expectations.md  scenario expectations
        sse_events.txt   readable summary of SSE events (stream only)

SAFETY
------
Read-only against everything except its own captures dir. Each probe is
idempotent — re-running overwrites prior captures for that
(model, scenario, path, mode) cell.
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
ADMIN_TOKEN_PATH = Path("~/.config/callosum/admin_token").expanduser()

PATHS = ("gateway", "proxy-direct", "callosum-v1", "callosum-codex")
MODES = ("non_stream", "stream")
TIMEOUT_S = 240.0


def _slugify(s: str) -> str:
    return s.replace("/", "_").replace(":", "_")


def _admin_token() -> str | None:
    if ADMIN_TOKEN_PATH.exists():
        t = ADMIN_TOKEN_PATH.read_text().strip()
        if t:
            return t
    return None


def _http_request(
    method: str,
    url: str,
    *,
    body: dict | None = None,
    headers: dict | None = None,
    timeout: float = TIMEOUT_S,
) -> tuple[int, bytes, dict[str, str]]:
    headers = dict(headers or {})
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urlrequest.Request(url, method=method, headers=headers, data=data)
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), dict(resp.headers)
    except error.HTTPError as e:
        return e.code, e.read(), dict(e.headers or {})


def _stream_request(
    url: str, *, body: dict, headers: dict, timeout: float = TIMEOUT_S
) -> tuple[int, bytes, list[str], dict[str, str]]:
    """Stream an SSE POST; return (status, raw_bytes, list_of_events, headers)."""
    headers = dict(headers)
    headers["Content-Type"] = "application/json"
    headers["Accept"] = "text/event-stream"
    data = json.dumps(body).encode("utf-8")
    req = urlrequest.Request(url, method="POST", headers=headers, data=data)
    raw = bytearray()
    events: list[str] = []
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            buf = bytearray()
            while True:
                chunk = resp.read(4096)
                if not chunk:
                    break
                raw.extend(chunk)
                buf.extend(chunk)
                # Split SSE by blank-line separators.
                while b"\n\n" in buf:
                    block, _, rest = buf.partition(b"\n\n")
                    buf[:] = rest
                    text = block.decode("utf-8", errors="replace")
                    events.append(text)
            if buf:
                events.append(buf.decode("utf-8", errors="replace"))
            return status, bytes(raw), events, dict(resp.headers)
    except error.HTTPError as e:
        body_bytes = e.read()
        return e.code, body_bytes, [
            body_bytes.decode("utf-8", errors="replace")
        ], dict(e.headers or {})


def list_local_models() -> list[str]:
    """Enumerate every model advertised by a litellm_gateway-kind backend
    on callosum. Returns sorted unique slugs."""
    status, body, _ = _http_request("GET", f"{CALLOSUM_URL}/status")
    if status != 200:
        sys.exit(f"failed to GET /status from callosum: {status} {body[:200]!r}")
    data = json.loads(body)
    out: set[str] = set()
    for b in data.get("backends", []):
        if b.get("kind") != "litellm_gateway":
            continue
        for m in b.get("advertised_models", []):
            out.add(m)
    return sorted(out)


def resolve_proxy_endpoint(model: str) -> str | None:
    """Look up the per-model dedicated endpoint from local LLM gateway's CLI
    registry. Returns None when the model isn't a responses-proxy or no
    endpoint is registered."""
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


def _convert_chat_to_responses(chat_body: dict) -> dict:
    """Best-effort translate a chat-completions body to Responses-API
    shape so chat-only scenarios can be sent via callosum /v1/responses
    and /codex. Translation handles message text only; tool definitions
    and tool_calls are NOT translated (those scenarios should declare
    request_shape: responses directly)."""
    input_items = []
    for m in chat_body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content)
        input_items.append({
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": content}],
        })
    out: dict[str, Any] = {
        "input": input_items,
    }
    if "max_tokens" in chat_body:
        out["max_output_tokens"] = chat_body["max_tokens"]
    if "temperature" in chat_body:
        out["temperature"] = chat_body["temperature"]
    return out


def build_request_for_path(
    scenario: dict, model: str, path: str
) -> tuple[str, dict, dict[str, str]]:
    """Return (url, body, headers) for the (scenario, model, path) combo."""
    shape = scenario.get("request_shape", "chat")
    body = {**scenario["request"], "model": model}
    headers: dict[str, str] = {}

    if path == "gateway":
        if shape == "chat":
            url = f"{GATEWAY_URL}/v1/chat/completions"
        else:
            url = f"{GATEWAY_URL}/v1/responses"
        key = os.environ.get("CALLOSUM_LITELLM_MASTER_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return url, body, headers

    if path == "proxy-direct":
        endpoint = resolve_proxy_endpoint(model)
        if endpoint is None:
            raise RuntimeError(
                f"no proxy endpoint registered for {model!r}"
            )
        # Per-model proxies speak the Responses API. Convert chat scenarios.
        if shape == "chat":
            body = _convert_chat_to_responses(body)
        body["model"] = model.replace("-ollama-responses-proxy", "-local").replace(
            "-responses-proxy-q4_k_m", "-local"
        )
        # ^ A best-effort runtime-model rewrite; the registry's runtime_model
        # field would be the authoritative source. Good enough for the
        # known suffix patterns in this lab.
        url = f"{endpoint.rstrip('/')}/v1/responses"
        return url, body, headers

    if path == "callosum-v1":
        # Callosum /v1/* is gated by the bearer middleware against the
        # auth-service api_keys table. The operator's client.env (e.g.
        # ~/.config/codex-proxy/client.env) defines CODEX_PROXY_TOKEN —
        # source it before running the probe so this env lookup picks
        # it up. The probe never reads the token from disk itself.
        token = os.environ.get("CODEX_PROXY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if shape == "chat":
            url = f"{CALLOSUM_URL}/v1/chat/completions"
        else:
            url = f"{CALLOSUM_URL}/v1/responses"
        return url, body, headers

    if path == "callosum-codex":
        # /codex is Responses-only; convert chat scenarios.
        token = os.environ.get("CODEX_PROXY_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if shape == "chat":
            body = _convert_chat_to_responses(body)
            body["model"] = model
        url = f"{CALLOSUM_URL}/codex"
        return url, body, headers

    raise ValueError(f"unknown path: {path}")


def run_probe(
    scenario: dict, model: str, *, path: str, mode: str
) -> dict[str, Any]:
    """Execute one (scenario, model, path, mode) probe; capture to disk;
    return a metadata summary."""
    output_dir = (
        OUTPUT_ROOT / _slugify(model) / scenario["id"] / f"{path}__{mode}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        url, body, headers = build_request_for_path(scenario, model, path)
    except Exception as exc:
        meta = {
            "scenario_id": scenario["id"],
            "model": model,
            "path": path,
            "mode": mode,
            "skipped_reason": f"build_request failed: {exc}",
        }
        (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    # Pin stream mode.
    if mode == "stream":
        body = {**body, "stream": True}
    else:
        body = {**body, "stream": False}

    (output_dir / "request.json").write_text(json.dumps(body, indent=2))

    t0 = time.time()
    if mode == "stream":
        status, raw, events, resp_headers = _stream_request(
            url, body=body, headers=headers,
        )
        (output_dir / "response.raw").write_bytes(raw)
        # Save a readable event summary for grep-ability.
        (output_dir / "sse_events.txt").write_text("\n---\n".join(events))
        # Try to parse the events list as JSON (each data: line is JSON-ish).
        parsed_events: list[dict] = []
        for ev in events:
            for line in ev.splitlines():
                line = line.strip()
                if line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        parsed_events.append({"_marker": "DONE"})
                        continue
                    try:
                        parsed_events.append(json.loads(payload))
                    except json.JSONDecodeError:
                        pass
        (output_dir / "response.json").write_text(
            json.dumps(parsed_events, indent=2)
        )
        finish_reason = None
        for ev in parsed_events:
            if not isinstance(ev, dict):
                continue
            for ch in ev.get("choices", []) or []:
                if ch.get("finish_reason"):
                    finish_reason = ch.get("finish_reason")
            if ev.get("type") == "response.completed":
                finish_reason = "completed"
        meta_extra = {
            "event_count": len(parsed_events),
            "saw_done_marker": any(
                ev.get("_marker") == "DONE" for ev in parsed_events
                if isinstance(ev, dict)
            ),
            "finish_reason": finish_reason,
        }
    else:
        status, raw, resp_headers = _http_request(
            "POST", url, body=body, headers=headers,
        )
        (output_dir / "response.raw").write_bytes(raw)
        parsed: dict | None = None
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
            (output_dir / "response.json").write_text(
                json.dumps(parsed, indent=2)
            )
        except json.JSONDecodeError:
            parsed = None

        finish_reason = None
        tool_calls_present = False
        content_preview = None
        if isinstance(parsed, dict):
            ch = (parsed.get("choices") or [{}])[0]
            if isinstance(ch, dict):
                finish_reason = ch.get("finish_reason")
                msg = ch.get("message") or {}
                if isinstance(msg.get("tool_calls"), list):
                    tool_calls_present = True
                if isinstance(msg.get("content"), str):
                    content_preview = msg["content"][:300]
            # Responses-API shape
            if "output" in parsed:
                for item in parsed.get("output", []) or []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "function_call":
                        tool_calls_present = True
                    if item.get("type") == "message":
                        for c in item.get("content", []) or []:
                            if isinstance(c, dict) and c.get("type") == "output_text":
                                content_preview = (content_preview or "") + (
                                    c.get("text", "")[:300 - len(content_preview or "")]
                                    if content_preview is None or len(content_preview) < 300
                                    else ""
                                )
                if not finish_reason:
                    finish_reason = parsed.get("status")
        meta_extra = {
            "finish_reason": finish_reason,
            "tool_calls_present": tool_calls_present,
            "content_preview": content_preview,
        }

    latency_ms = int((time.time() - t0) * 1000)
    meta = {
        "scenario_id": scenario["id"],
        "model": model,
        "path": path,
        "mode": mode,
        "http_status": status,
        "latency_ms": latency_ms,
        "url": url,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **meta_extra,
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

    expectations = scenario.get("expectations", [])
    (output_dir / "expectations.md").write_text(
        f"# {scenario['id']}\n\n{scenario['description']}\n\n## Expectations\n\n"
        + "\n".join(f"- {e}" for e in expectations)
        + "\n"
    )
    return meta


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=None, help="Filter to one scenario id.")
    p.add_argument("--model", default=None, help="Filter to one model slug.")
    p.add_argument(
        "--path", default="callosum-v1", choices=PATHS + ("all",),
        help="Dispatch path. 'all' runs every applicable path.",
    )
    p.add_argument(
        "--mode", default="non_stream", choices=MODES + ("all", "scenario"),
        help="non_stream | stream | all | scenario (use the modes declared by the scenario)",
    )
    p.add_argument(
        "--category", default=None,
        help="Filter by battery category (generic|codex|noteapp|format|stress)",
    )
    p.add_argument(
        "--list", action="store_true",
        help="Print what would run and exit.",
    )
    args = p.parse_args()

    if not BATTERY_PATH.exists():
        sys.exit(f"battery not found at {BATTERY_PATH}")
    battery = json.loads(BATTERY_PATH.read_text())
    categories = ("generic", "codex", "noteapp", "format", "stress")
    scenarios: list[dict] = []
    for cat in categories:
        if args.category and cat != args.category:
            continue
        for s in battery.get(cat, []):
            scenarios.append({**s, "category": cat})
    if args.scenario:
        scenarios = [s for s in scenarios if s["id"] == args.scenario]
        if not scenarios:
            sys.exit(f"scenario {args.scenario!r} not found")

    models = list_local_models()
    if args.model:
        if args.model not in models:
            sys.exit(
                f"model {args.model!r} not in routing pool. "
                f"available: {models}"
            )
        models = [args.model]

    paths = list(PATHS) if args.path == "all" else [args.path]

    if args.list:
        print(f"Models ({len(models)}):")
        for m in models:
            print(f"  {m}")
        print(f"Scenarios ({len(scenarios)}):")
        for s in scenarios:
            print(f"  [{s['category']}] {s['id']}  shape={s.get('request_shape')}  modes={s.get('modes')}")
        print(f"Paths: {paths}")
        return 0

    print(f"Probing models={len(models)} scenarios={len(scenarios)} paths={paths} mode={args.mode}")
    summary: list[dict] = []
    for model in models:
        for scenario in scenarios:
            # Pick modes
            if args.mode == "scenario":
                modes = scenario.get("modes", ["non_stream"])
            elif args.mode == "all":
                modes = list(MODES)
            else:
                modes = [args.mode]
            for path in paths:
                for mode in modes:
                    print(
                        f"  {model[:40]:40s} :: {scenario['id']:35s} "
                        f"path={path:14s} mode={mode:11s} ",
                        end="", flush=True,
                    )
                    try:
                        meta = run_probe(scenario, model, path=path, mode=mode)
                        summary.append(meta)
                        if "skipped_reason" in meta:
                            print(f"SKIP: {meta['skipped_reason'][:80]}")
                        else:
                            tc = "tc" if meta.get("tool_calls_present") else ""
                            print(
                                f"http={meta.get('http_status'):<3} "
                                f"finish={str(meta.get('finish_reason'))[:8]:8s} "
                                f"{meta.get('latency_ms'):>6}ms {tc}"
                            )
                    except Exception as exc:
                        print(f"ERROR: {type(exc).__name__}: {str(exc)[:80]}")
                        summary.append({
                            "scenario_id": scenario["id"], "model": model,
                            "path": path, "mode": mode,
                            "error": f"{type(exc).__name__}: {exc}",
                        })

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
