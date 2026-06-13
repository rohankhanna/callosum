#!/usr/bin/env python3
"""Black-box model probing harness (B2 multi-path + streaming).

Drives the battery defined at tests/integration/probes/battery.json
against every routable local model along multiple dispatch paths and
captures verbatim wire data plus a structured fitness summary.

PATHS
-----
gateway                    POST to the litellm gateway at $CALLOSUM_LITELLM_GATEWAY_URL
                           (chat-completions; only works for gateway-served models)
proxy-direct               POST to the per-model dedicated proxy endpoint resolved
                           from local LLM gateway's registry (responses-API; works for
                           *-responses-proxy models)
callosum-v1                POST to callosum's /v1/chat/completions or /v1/responses
                           (path picked by the scenario's request_shape)
callosum-codex             POST to callosum's /codex (responses-only)
callosum-admin-cell-call   POST to callosum's /admin/cell-call targeting one cell
                           explicitly. Bypasses the router (mode filter, denylist,
                           capability filter) AND the response-transform pipeline.
                           Verifies cell reachability + backend-level wiring
                           through callosum's process WITHOUT requiring the
                           operator to flip routing mode. Admin-token-gated.
                           Non-stream only (admin endpoint doesn't stream).

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
import atexit
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error
from urllib import request as urlrequest

REPO_ROOT = Path(__file__).resolve().parent.parent
BATTERY_PATH = REPO_ROOT / "tests" / "integration" / "probes" / "battery.json"
OUTPUT_ROOT = REPO_ROOT / "tests" / "integration" / "fixtures" / "probed"

GATEWAY_URL = os.environ.get("CALLOSUM_LITELLM_GATEWAY_URL", "http://127.0.0.1:4000")
CALLOSUM_URL = os.environ.get("CALLOSUM_BASE_URL", "http://127.0.0.1:8765")
ADMIN_TOKEN_PATH = Path("~/.config/callosum/admin_token").expanduser()

PATHS = (
    "gateway",
    "proxy-direct",
    "callosum-v1",
    "callosum-codex",
    "callosum-admin-cell-call",
)
MODES = ("non_stream", "stream")
TIMEOUT_S = 240.0

# Per-capability model lists. Add a model slug here once it's verified to
# support the capability — the runner consults this to gate
# requires_capability-tagged scenarios. Empty default means the multimodal
# (M) battery scenarios skip cleanly against the current routable pool
# while staying ready for the day a vision-capable local model is
# registered.
MODEL_CAPABILITIES: dict[str, frozenset[str]] = {
    # "<model-slug>": frozenset({"vision", "audio", ...}),
}


def _model_supports(model: str, capability: str) -> bool:
    caps = MODEL_CAPABILITIES.get(model)
    return bool(caps and capability in caps)


# Override toggled by --ignore-capability-gate. When True, the gate inside
# run_probe is bypassed and requires_capability scenarios run anyway.
_FORCE_RUN_CAPABILITY_SCENARIOS: bool = False


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
        return e.code, body_bytes, [body_bytes.decode("utf-8", errors="replace")], dict(e.headers or {})


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
        input_items.append(
            {
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }
        )
    out: dict[str, Any] = {
        "input": input_items,
    }
    if "max_tokens" in chat_body:
        out["max_output_tokens"] = chat_body["max_tokens"]
    if "temperature" in chat_body:
        out["temperature"] = chat_body["temperature"]
    return out


def build_request_for_path(scenario: dict, model: str, path: str) -> tuple[str, dict, dict[str, str]]:
    """Return (url, body, headers) for the (scenario, model, path) combo."""
    shape = scenario.get("request_shape", "chat")
    body = {**scenario["request"], "model": model}
    headers: dict[str, str] = {}

    if path == "gateway":
        url = f"{GATEWAY_URL}/v1/chat/completions" if shape == "chat" else f"{GATEWAY_URL}/v1/responses"
        key = os.environ.get("CALLOSUM_LITELLM_MASTER_KEY", "")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return url, body, headers

    if path == "proxy-direct":
        endpoint = resolve_proxy_endpoint(model)
        if endpoint is None:
            raise RuntimeError(f"no proxy endpoint registered for {model!r}")
        # Per-model proxies speak the Responses API. Convert chat scenarios.
        if shape == "chat":
            body = _convert_chat_to_responses(body)
        body["model"] = model.replace("-ollama-responses-proxy", "-local").replace("-responses-proxy-q4_k_m", "-local")
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
        url = f"{CALLOSUM_URL}/v1/chat/completions" if shape == "chat" else f"{CALLOSUM_URL}/v1/responses"
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

    if path == "callosum-admin-cell-call":
        # /admin/cell-call accepts {"model": ..., "body": <responses-API>} and
        # targets one cell explicitly, bypassing the router AND the response-
        # transform pipeline. Useful for verifying cell reachability through
        # callosum's process without requiring an operator routing-mode flip.
        admin_token = _admin_token()
        if not admin_token:
            raise RuntimeError("no admin token found at ~/.config/callosum/admin_token; cannot drive /admin/cell-call")
        headers["Authorization"] = f"Bearer {admin_token}"
        # /admin/cell-call only speaks Responses; convert chat scenarios.
        inner = body
        if shape == "chat":
            inner = _convert_chat_to_responses(body)
        # Drop "model" from the inner body — the admin endpoint forces it
        # to match the wrapper's `model` field.
        inner = {k: v for k, v in inner.items() if k != "model"}
        cell_body = {"model": model, "body": inner}
        url = f"{CALLOSUM_URL}/admin/cell-call"
        return url, cell_body, headers

    raise ValueError(f"unknown path: {path}")


def run_probe(scenario: dict, model: str, *, path: str, mode: str) -> dict[str, Any]:
    """Execute one (scenario, model, path, mode) probe; capture to disk;
    return a metadata summary."""
    output_dir = OUTPUT_ROOT / _slugify(model) / scenario["id"] / f"{path}__{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # /admin/cell-call is non-stream only — the admin endpoint awaits the
    # full response. Skip stream-mode rows for this path so the fitness
    # analyzer doesn't count them as failures.
    if path == "callosum-admin-cell-call" and mode == "stream":
        meta = {
            "scenario_id": scenario["id"],
            "model": model,
            "path": path,
            "mode": mode,
            "skipped_reason": "callosum-admin-cell-call is non-stream only",
        }
        (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

    # Capability gating: if the scenario declares a required capability
    # (e.g. "vision") and the target model doesn't advertise it via
    # MODEL_CAPABILITIES, mark the row skipped with a clean reason rather
    # than attempting the request and producing a noisy failure. The
    # scenarios stay defined and ready for the day a capable model is
    # registered. Override with --ignore-capability-gate to force.
    req_cap = scenario.get("requires_capability")
    if req_cap and not _model_supports(model, req_cap) and not _FORCE_RUN_CAPABILITY_SCENARIOS:
        meta = {
            "scenario_id": scenario["id"],
            "model": model,
            "path": path,
            "mode": mode,
            "skipped_reason": (
                f"model does not advertise {req_cap!r} capability (register in MODEL_CAPABILITIES to enable)"
            ),
        }
        (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        return meta

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
    body = {**body, "stream": True} if mode == "stream" else {**body, "stream": False}

    (output_dir / "request.json").write_text(json.dumps(body, indent=2))

    t0 = time.time()
    if mode == "stream":
        status, raw, events, resp_headers = _stream_request(
            url,
            body=body,
            headers=headers,
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
                    payload = line[len("data:") :].strip()
                    if payload == "[DONE]":
                        parsed_events.append({"_marker": "DONE"})
                        continue
                    with contextlib.suppress(json.JSONDecodeError):
                        parsed_events.append(json.loads(payload))
        (output_dir / "response.json").write_text(json.dumps(parsed_events, indent=2))
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
            "saw_done_marker": any(ev.get("_marker") == "DONE" for ev in parsed_events if isinstance(ev, dict)),
            "finish_reason": finish_reason,
        }
    else:
        status, raw, resp_headers = _http_request(
            "POST",
            url,
            body=body,
            headers=headers,
        )
        (output_dir / "response.raw").write_bytes(raw)
        parsed: dict | None = None
        try:
            parsed = json.loads(raw.decode("utf-8", errors="replace"))
            (output_dir / "response.json").write_text(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            parsed = None

        # /admin/cell-call returns a wrapper:
        #   {"status": "ok|error", "served_by": ..., "response": <upstream>, "error": ...}
        # Unwrap it so the existing finish_reason / tool_calls extraction
        # operates on the same shape it does for the other paths. Persist
        # the wrapper metadata (served_by, wrapper_error) onto the row.
        admin_served_by: str | None = None
        admin_wrapper_error: str | None = None
        if path == "callosum-admin-cell-call" and isinstance(parsed, dict):
            admin_served_by = parsed.get("served_by")
            admin_wrapper_error = parsed.get("error")
            inner = parsed.get("response")
            if isinstance(inner, dict):
                parsed = inner

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
                                    c.get("text", "")[: 300 - len(content_preview or "")]
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
        if path == "callosum-admin-cell-call":
            meta_extra["admin_served_by"] = admin_served_by
            meta_extra["admin_wrapper_error"] = admin_wrapper_error

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


# Routing-restore guards. When --flip-routing-to is in effect, these
# module-level pieces ensure that the original routing mode is restored
# from EVERY exit path: normal completion (try/finally), an exception
# (try/finally), a SystemExit raised by code (atexit), Ctrl-C (signal
# handler converts SIGINT to SystemExit), and external SIGTERM (signal
# handler does the same). Without this, a SIGTERM mid-run leaves the
# operator's routing on whatever value we flipped to.
_RESTORE_TARGET: str | None = None
_RESTORE_DONE: bool = False


def _restore_routing() -> None:
    """Idempotent restore. Safe to call multiple times — the flag
    prevents double-restore even if atexit + try/finally + signal
    handler all fire on the same exit path."""
    global _RESTORE_DONE
    if _RESTORE_DONE or _RESTORE_TARGET is None:
        return
    _RESTORE_DONE = True
    admin_token = _admin_token()
    headers = {"Authorization": f"Bearer {admin_token}"} if admin_token else {}
    try:
        status, body, _ = _http_request(
            "POST",
            f"{CALLOSUM_URL}/admin/routing",
            body={"routing": _RESTORE_TARGET},
            headers=headers,
        )
    except Exception as exc:
        print(
            f"\n!!! ERROR restoring routing to {_RESTORE_TARGET!r}: "
            f"{type(exc).__name__}: {exc}. "
            f"Manually run: callosum-ctl routing set {_RESTORE_TARGET}",
            file=sys.stderr,
        )
        return
    if status != 200:
        print(
            f"\n!!! WARNING: failed to restore routing to "
            f"{_RESTORE_TARGET!r}: {status} {body[:200]!r}. "
            f"Manually run: callosum-ctl routing set {_RESTORE_TARGET}",
            file=sys.stderr,
        )
    else:
        print(f"Routing restored to {_RESTORE_TARGET!r}")


def _restore_signal_handler(signum: int, frame: object) -> None:
    """Convert SIGTERM/SIGINT into a SystemExit so the cleanup chain
    (atexit + finally) runs. Without this, default SIGTERM behavior
    silently terminates Python and leaves the routing flip stuck."""
    print(
        f"\n!!! Received signal {signum}; restoring routing before exit",
        file=sys.stderr,
    )
    _restore_routing()
    sys.exit(128 + signum)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default=None, help="Filter to one scenario id.")
    p.add_argument("--model", default=None, help="Filter to one model slug.")
    p.add_argument(
        "--path",
        default="callosum-v1",
        choices=PATHS + ("all",),
        help="Dispatch path. 'all' runs every applicable path.",
    )
    p.add_argument(
        "--mode",
        default="non_stream",
        choices=MODES + ("all", "scenario"),
        help="non_stream | stream | all | scenario (use the modes declared by the scenario)",
    )
    p.add_argument(
        "--category",
        default=None,
        help="Filter by battery category (generic|codex|noteapp|format|stress|multimodal)",
    )
    p.add_argument(
        "--ignore-capability-gate",
        action="store_true",
        help=(
            "Run requires_capability scenarios even against models that "
            "don't advertise the capability (instead of marking skipped). "
            "Useful for confirming the runner emits the right wire shape "
            "even when no model can answer."
        ),
    )
    p.add_argument(
        "--flip-routing-to",
        default=None,
        choices=["auto", "offline", "local-only", "remote-only"],
        help=(
            "Briefly flip callosum's routing mode to the given value for "
            "the duration of this run, then restore the original on exit. "
            "Required when probing local models via /v1/responses or /codex "
            "while the operator's standing mode is remote-only. The restore "
            "fires from a try/finally so it runs even if the probes raise. "
            "Reads the admin token from ~/.config/callosum/admin_token."
        ),
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="Print what would run and exit.",
    )
    args = p.parse_args()

    global _FORCE_RUN_CAPABILITY_SCENARIOS
    _FORCE_RUN_CAPABILITY_SCENARIOS = bool(args.ignore_capability_gate)

    if not BATTERY_PATH.exists():
        sys.exit(f"battery not found at {BATTERY_PATH}")
    battery = json.loads(BATTERY_PATH.read_text())
    categories = ("generic", "codex", "noteapp", "format", "stress", "multimodal")
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
            sys.exit(f"model {args.model!r} not in routing pool. available: {models}")
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

    # Routing-mode flip wrapper. Records current state up-front, then
    # arms the signal/atexit guards BEFORE the flip POST so the cleanup
    # chain is in place even if the POST itself is interrupted.
    original_routing: str | None = None
    if args.flip_routing_to is not None:
        admin_token = _admin_token()
        if not admin_token:
            sys.exit("no admin token at ~/.config/callosum/admin_token; cannot perform --flip-routing-to")
        headers = {"Authorization": f"Bearer {admin_token}"}
        status, body, _ = _http_request(
            "GET",
            f"{CALLOSUM_URL}/admin/routing",
            headers=headers,
        )
        if status != 200:
            sys.exit(f"failed to GET /admin/routing: {status} {body[:200]!r}")
        original_routing = json.loads(body).get("routing")
        if not original_routing:
            sys.exit("could not parse current routing from /admin/routing")
        # Arm the guards as soon as we know what to restore to. Pre-flip
        # arming means a signal between this point and the POST still
        # results in the right restore target being committed (an
        # idempotent re-POST of the original value if the flip already
        # landed on the server, a no-op if it didn't).
        global _RESTORE_TARGET
        _RESTORE_TARGET = original_routing
        atexit.register(_restore_routing)
        signal.signal(signal.SIGTERM, _restore_signal_handler)
        signal.signal(signal.SIGINT, _restore_signal_handler)
        print(f"Routing flip: current={original_routing!r}, flipping to {args.flip_routing_to!r}; will restore on exit")
        status, body, _ = _http_request(
            "POST",
            f"{CALLOSUM_URL}/admin/routing",
            body={"routing": args.flip_routing_to},
            headers=headers,
        )
        if status != 200:
            sys.exit(f"failed to flip routing to {args.flip_routing_to!r}: {status} {body[:200]!r}")

    summary: list[dict] = []
    try:
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
                            f"  {model[:40]:40s} :: {scenario['id']:35s} path={path:14s} mode={mode:11s} ",
                            end="",
                            flush=True,
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
                            summary.append(
                                {
                                    "scenario_id": scenario["id"],
                                    "model": model,
                                    "path": path,
                                    "mode": mode,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
    finally:
        # Delegated to _restore_routing(). The _RESTORE_DONE flag means
        # this is a no-op if the signal handler or atexit already ran.
        _restore_routing()

    summary_path = OUTPUT_ROOT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
