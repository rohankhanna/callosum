"""callosum CLI — operator commands against a running proxy.

The proxy exposes an admin HTTP surface under /admin/*. This CLI is a
thin client that reads the admin token (from
~/.config/callosum/admin_token, written on first proxy start) and
sends JSON requests.

Subcommands:

    callosum status
    callosum params {get|set|clear|list} ...
    callosum denylist {add|remove|list} ...
    callosum mode {get|set} ...
    callosum reload

All output is JSON unless --pretty is given (default: pretty for
status/list; raw for setters).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://127.0.0.1:8765"


def _admin_token() -> str:
    """Resolve the admin token. Order:
      1. CALLOSUM_ADMIN_TOKEN env var
      2. ~/.config/callosum/admin_token file (written by the proxy on first start)
    """
    env = os.environ.get("CALLOSUM_ADMIN_TOKEN")
    if env:
        return env.strip()
    path = Path("~/.config/callosum/admin_token").expanduser()
    if path.exists():
        text = path.read_text().strip()
        if text:
            return text
    sys.exit(
        "callosum CLI: no admin token. Start the proxy at least once to "
        "generate ~/.config/callosum/admin_token, or set CALLOSUM_ADMIN_TOKEN."
    )


def _request(
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 10.0,
) -> Any:
    """Send one HTTP request to the proxy's admin endpoint.

    Default timeout is short (10s) because most admin endpoints are
    state-snapshot reads. Long-running endpoints (e.g. /admin/probe-tools
    which sequentially probes every advertised cell) override this
    explicitly — callers should pass a value sized to the worst-case
    workload, not the typical case.
    """
    base = os.environ.get("CALLOSUM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}{path}"
    headers = {"Authorization": f"Bearer {_admin_token()}"}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            payload = resp.read()
    except error.HTTPError as e:
        body_text = e.read().decode("utf-8", errors="replace")
        sys.exit(f"callosum CLI: HTTP {e.code} from {url}\n{body_text}")
    except error.URLError as e:
        sys.exit(
            f"callosum CLI: cannot reach {url} ({e.reason}). "
            "Is the proxy running?"
        )
    if not payload:
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return payload.decode("utf-8", errors="replace")


def _print(obj: Any, *, pretty: bool = True) -> None:
    if isinstance(obj, (dict, list)):
        if pretty:
            print(json.dumps(obj, indent=2, sort_keys=True))
        else:
            print(json.dumps(obj))
    else:
        print(obj)


# ---------- subcommand handlers -----------------------------------------


def cmd_status(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/status"))
    return 0


def cmd_params_list(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/params"))
    return 0


def cmd_params_get(args: argparse.Namespace) -> int:
    all_overrides = _request("GET", "/admin/params") or []
    for row in all_overrides:
        if row.get("model") == args.model:
            _print(row)
            return 0
    print(json.dumps({"model": args.model, "params": {}, "force": False}))
    return 0


def cmd_params_set(args: argparse.Namespace) -> int:
    params: dict[str, Any] = {}
    for kv in args.params:
        if "=" not in kv:
            sys.exit(f"params must be key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        v = v.strip()
        # Coerce common JSON values.
        if v.lower() in ("true", "false"):
            parsed: Any = v.lower() == "true"
        elif v.lower() in ("null", "none"):
            parsed = None
        else:
            try:
                parsed = int(v)
            except ValueError:
                try:
                    parsed = float(v)
                except ValueError:
                    parsed = v
        params[k.strip()] = parsed
    body = {"action": "set", "model": args.model, "params": params, "force": args.force}
    _print(_request("POST", "/admin/params", body), pretty=False)
    return 0


def cmd_params_clear(args: argparse.Namespace) -> int:
    body = {"action": "clear", "model": args.model}
    _print(_request("POST", "/admin/params", body), pretty=False)
    return 0


def cmd_denylist_list(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/denylist"))
    return 0


def cmd_denylist_add(args: argparse.Namespace) -> int:
    body: dict[str, Any] = {"action": "add", "model": args.model}
    if args.reason:
        body["reason"] = args.reason
    _print(_request("POST", "/admin/denylist", body), pretty=False)
    return 0


def cmd_denylist_remove(args: argparse.Namespace) -> int:
    body = {"action": "remove", "model": args.model}
    _print(_request("POST", "/admin/denylist", body), pretty=False)
    return 0


def cmd_mode_get(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/mode"))
    return 0


def cmd_mode_set(args: argparse.Namespace) -> int:
    _print(_request("POST", "/admin/mode", {"mode": args.mode}), pretty=False)
    return 0


def cmd_autonomy_show(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/autonomy"))
    return 0


def cmd_autonomy_promote(args: argparse.Namespace) -> int:
    _print(_request("POST", "/admin/autonomy/promote", {}))
    return 0


def cmd_autonomy_demote(args: argparse.Namespace) -> int:
    body: dict[str, Any] = {}
    if args.reason:
        body["reason"] = args.reason
    _print(_request("POST", "/admin/autonomy/demote", body))
    return 0


def cmd_autonomy_set(args: argparse.Namespace) -> int:
    body: dict[str, Any] = {"level": int(args.level)}
    if args.reason:
        body["reason"] = args.reason
    _print(_request("POST", "/admin/autonomy/set", body))
    return 0


def cmd_autonomy_history(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/autonomy/history"))
    return 0


def cmd_autonomy_audit(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/autonomy/audit"))
    return 0


def cmd_retention_show(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/retention"))
    return 0


def cmd_retention_status(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/retention/status"))
    return 0


def cmd_retention_preview(args: argparse.Namespace) -> int:
    # Retention is invoked synchronously, so the run could take seconds
    # on a large requests table. The status/preview is faster — but
    # share the same generous timeout to keep one ceiling for both.
    _print(_request("POST", "/admin/retention/preview", {}, timeout=300.0))
    return 0


def cmd_retention_run(args: argparse.Namespace) -> int:
    _print(_request("POST", "/admin/retention/run", {}, timeout=600.0))
    return 0


def cmd_self_assessment_history(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/self-assessment/history"))
    return 0


def cmd_self_assessment_preview(args: argparse.Namespace) -> int:
    _print(_request("POST", "/admin/self-assessment/preview", {}, timeout=60.0))
    return 0


def cmd_self_assessment_run(args: argparse.Namespace) -> int:
    _print(_request("POST", "/admin/self-assessment/run", {}, timeout=60.0))
    return 0


def cmd_probe_tools(args: argparse.Namespace) -> int:
    """Run the tool-call verification probe and print per-cell results.

    The proxy iterates every advertised cell across loaded backends,
    sends a canonical single-tool request, and reports whether the
    response carries a structured `function_call` output item per the
    OpenAI Responses-API shape. Cells that emit text-as-JSON (the
    model-a0e5 quirk) or refuse to use the tool fail; the operator can
    follow up with `callosum-ctl denylist add <model>` to keep them
    out of tool-using routing decisions.

    For a pool of ~5-10 cells with 26B-class local models, expect
    total wall time of 30s-3min. Streaming progress is a follow-up.
    """
    body: dict[str, Any] = {}
    if args.models:
        body["models"] = args.models
    # Long timeout because the endpoint serially probes every advertised
    # cell. 26B-class local models on CPU/single-GPU can take 60s+ each.
    # 30 min ceiling is generous enough for a 10-cell pool of slow
    # local models without ever spinning forever.
    result = _request(
        "POST", "/admin/probe-tools", body, timeout=1800.0
    )
    _print(result)
    return 0


# ---------- parser -----------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="callosum",
        description="Operator commands for a running callosum proxy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Print a consolidated operator-state snapshot.").set_defaults(func=cmd_status)

    p_params = sub.add_parser("params", help="Per-cell inference parameter overrides.")
    pp = p_params.add_subparsers(dest="subcommand", required=True)
    pp.add_parser("list", help="List all overrides.").set_defaults(func=cmd_params_list)
    g = pp.add_parser("get", help="Show the override for one model.")
    g.add_argument("model")
    g.set_defaults(func=cmd_params_get)
    s = pp.add_parser(
        "set",
        help="Set inference parameters for one model. Accepts key=value pairs "
             "(values auto-cast to bool/int/float/string).",
    )
    s.add_argument("model")
    s.add_argument("params", nargs="+", help='e.g. think=false temperature=0.0')
    s.add_argument(
        "--force",
        action="store_true",
        help="Authoritative: operator values overwrite client request body. "
             "Default false (preserve client intent).",
    )
    s.set_defaults(func=cmd_params_set)
    c = pp.add_parser("clear", help="Remove the override for one model.")
    c.add_argument("model")
    c.set_defaults(func=cmd_params_clear)

    p_deny = sub.add_parser("denylist", help="Cells excluded from routing.")
    pd = p_deny.add_subparsers(dest="subcommand", required=True)
    pd.add_parser("list", help="List denied cells.").set_defaults(func=cmd_denylist_list)
    a = pd.add_parser("add", help="Add a cell to the denylist.")
    a.add_argument("model")
    a.add_argument("--reason", help="Free-form note for future reference.")
    a.set_defaults(func=cmd_denylist_add)
    r = pd.add_parser("remove", help="Remove a cell from the denylist.")
    r.add_argument("model")
    r.set_defaults(func=cmd_denylist_remove)

    p_mode = sub.add_parser("mode", help="Operator routing mode (auto / offline / local-only / remote-only).")
    pm = p_mode.add_subparsers(dest="subcommand", required=True)
    pm.add_parser("get", help="Print the current mode.").set_defaults(func=cmd_mode_get)
    sm = pm.add_parser("set", help="Set the mode.")
    sm.add_argument("mode", choices=["auto", "offline", "local-only", "remote-only"])
    sm.set_defaults(func=cmd_mode_set)

    p_auto = sub.add_parser(
        "autonomy",
        help="Earned-autonomy ladder for the dev-loop pipeline (L1..L5).",
    )
    pa = p_auto.add_subparsers(dest="subcommand", required=True)
    pa.add_parser(
        "show",
        help="Current level, streak, and promotion eligibility.",
    ).set_defaults(func=cmd_autonomy_show)
    pa.add_parser(
        "promote",
        help="Advance one rung if clean_streak meets threshold; "
             "errors with the missing condition otherwise.",
    ).set_defaults(func=cmd_autonomy_promote)
    ad = pa.add_parser(
        "demote",
        help="Drop one rung immediately. Floor-clamped at L1.",
    )
    ad.add_argument("--reason", help="Free-form note recorded in history.")
    ad.set_defaults(func=cmd_autonomy_demote)
    aset = pa.add_parser(
        "set",
        help="Force-set the level (e.g. for emergency reset to L1=1). "
             "Resets streak and ops counters at the new level.",
    )
    aset.add_argument(
        "level", type=int, choices=[1, 2, 3, 4, 5],
        help="1=manual, 2=auto-invoke, 3=auto-merge+soak, "
             "4=sunset, 5=architecture",
    )
    aset.add_argument("--reason", help="Free-form note recorded in history.")
    aset.set_defaults(func=cmd_autonomy_set)
    pa.add_parser(
        "history",
        help="Recent level transitions (most recent first).",
    ).set_defaults(func=cmd_autonomy_history)
    pa.add_parser(
        "audit",
        help="Recent dev-loop actions: invoke / merge / soak outcomes. "
             "This is what Tier C's weekly self-assessment reads.",
    ).set_defaults(func=cmd_autonomy_audit)

    p_ret = sub.add_parser(
        "retention",
        help="Tier G state retention: archive-and-delete old rows / "
             "branches / files per policy.",
    )
    pr = p_ret.add_subparsers(dest="subcommand", required=True)
    pr.add_parser(
        "show",
        help="List the configured retention policies and archive dir.",
    ).set_defaults(func=cmd_retention_show)
    pr.add_parser(
        "status",
        help="Per-policy current state (row counts, oldest entry, etc.).",
    ).set_defaults(func=cmd_retention_status)
    pr.add_parser(
        "preview",
        help="Dry-run: report what would be archived/deleted without "
             "modifying anything.",
    ).set_defaults(func=cmd_retention_preview)
    pr.add_parser(
        "run",
        help="Execute retention: archive matching rows to a tarball "
             "under the archive dir, then delete them from the live "
             "table. Intended to be invoked weekly from cron.",
    ).set_defaults(func=cmd_retention_run)

    p_sa = sub.add_parser(
        "self-assessment",
        help="Tier C automation agent self-assessment: weekly metrics over "
             "the autonomy audit log + usage log; auto-demotes on "
             "bad signals, suggests promotion on clean.",
    )
    ps = p_sa.add_subparsers(dest="subcommand", required=True)
    ps.add_parser(
        "history",
        help="Past assessments (most recent first).",
    ).set_defaults(func=cmd_self_assessment_history)
    ps.add_parser(
        "preview",
        help="Dry-run: compute metrics + decision without persisting "
             "or firing demote. Safe to run idly to inspect what the "
             "next real cycle would do.",
    ).set_defaults(func=cmd_self_assessment_preview)
    ps.add_parser(
        "run",
        help="Execute one self-assessment cycle. Persists a row, may "
             "auto-demote, writes a feedback artifact. Intended to be "
             "invoked weekly from cron.",
    ).set_defaults(func=cmd_self_assessment_run)

    p_probe = sub.add_parser(
        "probe-tools",
        help="Run tool-call verification probe against advertised cells.",
    )
    p_probe.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Restrict probing to this subset of model names. Default probes all.",
    )
    p_probe.set_defaults(func=cmd_probe_tools)

    # Auth-rotate wizard. Wired here so the help surface lists it
    # alongside the other operator commands.
    from callosum.auth_rotate import add_subparser as _add_auth_rotate
    _add_auth_rotate(sub)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
