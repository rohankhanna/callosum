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


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """Send one HTTP request to the proxy's admin endpoint."""
    base = os.environ.get("CALLOSUM_BASE_URL", DEFAULT_BASE_URL).rstrip("/")
    url = f"{base}{path}"
    headers = {"Authorization": f"Bearer {_admin_token()}"}
    data: bytes | None = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = request.Request(url, method=method, headers=headers, data=data)
    try:
        with request.urlopen(req, timeout=10) as resp:
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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
