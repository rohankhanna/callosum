"""callosum CLI — operator commands against a running proxy.

`callosum serve` starts the daemon. The other subcommands administer a
running instance: the proxy exposes an admin HTTP surface under
/admin/*, and this CLI is a thin client that reads the admin token
(from ~/.config/callosum/admin_token, written on first proxy start)
and sends JSON requests.

Run `callosum -h` for the authoritative, always-current list of
subcommands — argparse generates it from the parser, so it never drifts
from the code. The current top-level surface is: serve, status,
version, review, params, denylist, routing, autonomy, retention,
self-assessment, probe-tools, service, auth-rotate.

Most output is JSON; `review` / `service` print human-readable text.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_SYSTEMD_UNIT = "system-dependency-callosum.service"


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
        sys.exit(f"callosum CLI: cannot reach {url} ({e.reason}). Is the proxy running?")
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


def cmd_version(args: argparse.Namespace) -> int:
    """Print the installed version and the git SHA of the source tree this
    CLI imports from, plus whether that source is a live working tree
    (editable install) or a copied install.

    This is a staleness probe: a pipx copy made before newer subcommands
    landed will silently omit them from `-h`. An editable install tracks
    live source, so in-module edits appear without reinstall; a copied
    install can drift and needs `pipx install --force` to refresh. Paths
    are derived from this module's own location — nothing is hardcoded.
    """
    import importlib.metadata as _md

    try:
        version = _md.version("callosum")
    except _md.PackageNotFoundError:
        version = "0.1.0"

    src_dir = Path(__file__).resolve().parent

    def _git(*args_: str) -> str | None:
        try:
            return (
                subprocess.check_output(
                    ["git", "-C", str(src_dir), *args_],
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError, OSError):
            return None

    sha = _git("rev-parse", "--short", "HEAD")
    if sha is not None:
        info: dict[str, str | None] = {
            "version": version,
            "mode": "editable",
            "source_sha": sha,
            "source": str(src_dir),
            "note": "CLI tracks live source; in-module edits appear without reinstall.",
        }
    else:
        repo_root = _git("rev-parse", "--show-toplevel") or "<repo>"
        info = {
            "version": version,
            "mode": "copied",
            "source_sha": None,
            "source": str(src_dir),
            "note": (
                "CLI is a copied install and can go stale. To track live source: "
                f"pipx install --force --editable {repo_root}"
            ),
        }
    _print(info)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/status"))
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    """Human-readable summary of dev-loop dispatcher state: last run
    outcome, today's marker, and any auto/dev-loop-* branches
    awaiting manual review/merge. The point is to make pending review
    work visible — `callosum status` returns JSON; this prints a
    paste-friendly checklist with git commands inline."""
    payload = _request("GET", "/admin/status")
    if not isinstance(payload, dict):
        print("error: unexpected /admin/status payload", file=sys.stderr)
        return 1
    dev_loop = payload.get("dev_loop")
    if not isinstance(dev_loop, dict):
        print("error: /admin/status missing dev_loop section", file=sys.stderr)
        return 1

    last = dev_loop.get("last_run")
    if isinstance(last, dict):
        ts = last.get("timestamp", "(unknown)")
        outcome = last.get("outcome", "(unknown)")
        reason = last.get("reason", "")
        branch = last.get("branch")
        head = f"last dispatcher run: {ts} -> {outcome}"
        if branch:
            head += f" (branch: {branch})"
        print(head)
        if reason:
            print(f"  reason: {reason}")
    else:
        print("last dispatcher run: (never recorded — dispatcher has not fired since visibility surface landed)")

    marker_present = dev_loop.get("today_marker_present")
    marker_path = dev_loop.get("today_marker_path", "")
    if marker_present:
        print(f"today's marker: present ({marker_path}) — dispatcher will skip further fires today")
    else:
        print(f"today's marker: ABSENT ({marker_path}) — next dispatcher fire is eligible")

    branches = dev_loop.get("pending_review_branches") or []
    count = dev_loop.get("pending_review_count", len(branches))
    print()
    if not branches:
        print("pending review branches: 0 (clean queue)")
        return 0
    print(f"pending review branches: {count}")
    print()
    for b in branches:
        name = b.get("name", "?")
        sha = b.get("sha", "?")
        subject = b.get("subject", "?")
        age = b.get("age_days", 0.0)
        print(f"  {name} ({age:.1f}d old)")
        print(f"    {sha}  {subject}")
        print(f"    inspect: git log {name} --stat")
        print(f"    accept:  git checkout main && git merge --no-ff {name}")
        print(f"    reject:  git branch -D {name}")
        print()
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


def cmd_routing_get(args: argparse.Namespace) -> int:
    _print(_request("GET", "/admin/routing"))
    return 0


def cmd_routing_set(args: argparse.Namespace) -> int:
    _print(
        _request("POST", "/admin/routing", {"routing": args.routing}),
        pretty=False,
    )
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
    result = _request("POST", "/admin/probe-tools", body, timeout=1800.0)
    _print(result)
    return 0


def _run_service_command(argv: list[str]) -> int:
    try:
        return subprocess.run(argv, check=False).returncode
    except FileNotFoundError:
        sys.exit(f"callosum CLI: command not found: {argv[0]}")


def cmd_service_status(args: argparse.Namespace) -> int:
    return _run_service_command(["systemctl", "--user", "status", args.unit])


def cmd_service_logs(args: argparse.Namespace) -> int:
    argv = ["journalctl", "--user", "-u", args.unit, "--no-pager"]
    if args.follow:
        argv.append("-f")
    if args.lines is not None:
        argv.extend(["-n", str(args.lines)])
    return _run_service_command(argv)


def cmd_service_restart(args: argparse.Namespace) -> int:
    return _run_service_command(["systemctl", "--user", "restart", args.unit])


def cmd_service_stop(args: argparse.Namespace) -> int:
    return _run_service_command(["systemctl", "--user", "stop", args.unit])


def cmd_service_start(args: argparse.Namespace) -> int:
    return _run_service_command(["systemctl", "--user", "start", args.unit])


# ---------- parser -----------------------------------------------------


class _HelpOnErrorParser(argparse.ArgumentParser):
    """ArgumentParser variant that prints --help BEFORE emitting the
    error message and exiting non-zero.

    Default argparse behavior on a missing subcommand or an unknown
    argument is to print the usage one-liner + the error and exit. That
    leaves operators staring at "argument command: required" with no
    indication of what valid subcommands exist. This subclass prints the
    full help text on every error path so the recovery hint is always
    present alongside the error.

    The `add_subparsers` override propagates the class to nested
    subparsers automatically — without it, only the top-level parser
    would inherit the behavior and inner `routing` / `denylist` /
    `autonomy` / etc. would revert to argparse defaults.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.print_help(sys.stderr)
        sys.stderr.write(f"\nerror: {message}\n")
        sys.exit(2)

    def add_subparsers(self, **kwargs: Any) -> Any:
        kwargs.setdefault("parser_class", _HelpOnErrorParser)
        return super().add_subparsers(**kwargs)


def _cmd_serve(args: argparse.Namespace) -> int:
    """Dispatch the `callosum serve` subcommand: start the proxy
    daemon. Delegates to the server-startup function in __main__.py.
    The systemd unit that runs the daemon should call this as
    `callosum serve`; bare `callosum` is now reserved for the unified
    help surface.
    """
    from callosum.__main__ import serve_with_args

    serve_with_args(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = _HelpOnErrorParser(
        prog="callosum",
        description=(
            "Callosum proxy CLI. `callosum serve` starts the daemon; "
            "the other subcommands administer a running instance "
            "(status, routing mode, denylist, autonomy ladder, "
            "retention, self-assessment, capability probing, "
            "auth-vault rotation)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser(
        "serve",
        help="Start the callosum proxy daemon (FastAPI + uvicorn). "
        "Loads config, opens backends, binds the configured port.",
        description=(
            "Start the callosum proxy daemon (FastAPI + uvicorn). "
            "Loads config from ~/.config/callosum/config.toml (or "
            "--config), wires backends, and serves /v1/responses, "
            "/v1/chat/completions, /codex, /admin/*, /events/routing "
            "on the configured host:port. The systemd unit that runs "
            "the daemon should invoke `callosum serve`."
        ),
    )
    from pathlib import Path as _Path

    p_serve.add_argument(
        "--config",
        type=_Path,
        default=None,
        help="Path to config.toml (default: ~/.config/callosum/config.toml).",
    )
    p_serve.add_argument("--host", default=None, help="Override [server].host from config.")
    p_serve.add_argument("--port", type=int, default=None, help="Override [server].port from config.")
    p_serve.set_defaults(func=_cmd_serve)

    sub.add_parser("status", help="Print a consolidated operator-state snapshot.").set_defaults(func=cmd_status)
    sub.add_parser(
        "version",
        help="Print installed version + source git SHA; flags whether the CLI tracks live source.",
        description=(
            "Print the installed package version and the git SHA of the "
            "source tree this CLI imports from, plus whether it is an "
            "editable install (tracks live source) or a copied install "
            "(can go stale — e.g. a pipx copy made before newer "
            "subcommands landed). Use it to confirm `callosum -h` reflects "
            "the current code."
        ),
    ).set_defaults(func=cmd_version)
    sub.add_parser(
        "review",
        help="Show last dispatcher run + any pending auto/dev-loop-* branches awaiting manual review.",
        description=(
            "Human-readable summary of dev-loop dispatcher state: "
            "last run outcome with timestamp + reason, whether today's "
            "marker has been written, and the full list of "
            "auto/dev-loop-* branches the dispatcher has created that "
            "haven't been merged or deleted yet. For each pending "
            "branch the output includes copy-paste git commands to "
            "inspect / accept / reject."
        ),
    ).set_defaults(func=cmd_review)

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
    s.add_argument("params", nargs="+", help="e.g. think=false temperature=0.0")
    s.add_argument(
        "--force",
        action="store_true",
        help="Authoritative: operator values overwrite client request body. Default false (preserve client intent).",
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

    p_routing = sub.add_parser(
        "routing",
        help="Backend routing mode (auto / offline / local-only / "
        "remote-only). Distinct from `callosum-ctl autonomy` "
        "which governs dev-loop pipeline autonomy.",
    )
    pr = p_routing.add_subparsers(dest="subcommand", required=True)
    pr.add_parser(
        "get",
        help="Print the current routing mode.",
    ).set_defaults(func=cmd_routing_get)
    sr = pr.add_parser("set", help="Set the routing mode.")
    sr.add_argument(
        "routing",
        choices=["auto", "offline", "local-only", "remote-only"],
    )
    sr.set_defaults(func=cmd_routing_set)

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
        help="Advance one rung if clean_streak meets threshold; errors with the missing condition otherwise.",
    ).set_defaults(func=cmd_autonomy_promote)
    ad = pa.add_parser(
        "demote",
        help="Drop one rung immediately. Floor-clamped at L1.",
    )
    ad.add_argument("--reason", help="Free-form note recorded in history.")
    ad.set_defaults(func=cmd_autonomy_demote)
    aset = pa.add_parser(
        "set",
        help="Force-set the level (e.g. for emergency reset to L1=1). Resets streak and ops counters at the new level.",
    )
    aset.add_argument(
        "level",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="1=manual, 2=auto-invoke, 3=auto-merge+soak, 4=sunset, 5=architecture",
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
        help="Tier G state retention: archive-and-delete old rows / branches / files per policy.",
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
        help="Dry-run: report what would be archived/deleted without modifying anything.",
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

    p_service = sub.add_parser(
        "service",
        help="Manage the systemd user service that runs Callosum.",
    )
    p_service.add_argument(
        "--unit",
        default=DEFAULT_SYSTEMD_UNIT,
        help=f"systemd user unit name (default: {DEFAULT_SYSTEMD_UNIT}).",
    )
    psrv = p_service.add_subparsers(dest="subcommand", required=True)
    psrv.add_parser(
        "status",
        help="Show systemd status for the managed Callosum service.",
    ).set_defaults(func=cmd_service_status)
    logs = psrv.add_parser(
        "logs",
        help="Show journal logs for the managed Callosum service.",
    )
    logs.add_argument(
        "--follow",
        "-f",
        action="store_true",
        help="Follow new log lines until interrupted.",
    )
    logs.add_argument(
        "-n",
        "--lines",
        type=int,
        default=None,
        help="Number of recent log lines to show.",
    )
    logs.set_defaults(func=cmd_service_logs)
    psrv.add_parser(
        "restart",
        help="Restart the managed Callosum service.",
    ).set_defaults(func=cmd_service_restart)
    psrv.add_parser(
        "stop",
        help="Stop the managed Callosum service.",
    ).set_defaults(func=cmd_service_stop)
    psrv.add_parser(
        "start",
        help="Start the managed Callosum service.",
    ).set_defaults(func=cmd_service_start)

    # Auto-dev ticket backlog (PO intake). Opens the ticket SQLite store
    # directly so filing works while the serving daemon is down. Intake +
    # storage only — the dispatcher that consumes tickets is a later slice
    # (see docs/adr/2026-06-27-auto-dev-ticket-queue-statistical-promotion.md).
    from callosum.dev_loop.tickets import KINDS, SOURCES, STATUSES

    p_ticket = sub.add_parser("ticket", help="Auto-dev defect/ticket backlog (PO intake).")
    pt = p_ticket.add_subparsers(dest="ticket_cmd", required=True)
    t_add = pt.add_parser("add", help="File a ticket into the backlog.")
    t_add.add_argument("title")
    t_add.add_argument("--kind", choices=KINDS, default="bug")
    t_add.add_argument("--source", choices=SOURCES, default="po")
    t_add.add_argument("--repro", default=None, help="path to a repro prompt/payload file")
    t_add.add_argument(
        "--acceptance",
        default=None,
        help="machine-checkable acceptance test id (e.g. a pytest node) — sets the ticket triaged",
    )
    t_add.add_argument("--priority", type=int, default=0)
    t_add.set_defaults(func=cmd_ticket_add)
    t_list = pt.add_parser("list", help="List tickets (optionally by status).")
    t_list.add_argument("--status", choices=STATUSES, default=None)
    t_list.set_defaults(func=cmd_ticket_list)
    t_show = pt.add_parser("show", help="Show one ticket and its event trail.")
    t_show.add_argument("id")
    t_show.set_defaults(func=cmd_ticket_show)
    t_triage = pt.add_parser(
        "triage",
        help="Attach an acceptance test and move the ticket to ready (the eligibility wall).",
    )
    t_triage.add_argument("id")
    t_triage.add_argument("--acceptance", required=True, help="machine-checkable acceptance test id")
    t_triage.set_defaults(func=cmd_ticket_triage)

    # Auth-rotate wizard. Wired here so the help surface lists it
    # alongside the other operator commands.
    from callosum.auth_rotate import add_subparser as _add_auth_rotate

    _add_auth_rotate(sub)

    return parser


def cmd_ticket_add(args: argparse.Namespace) -> int:
    from callosum.dev_loop.tickets import TicketStore, default_work_mirror

    store = TicketStore(mirror=default_work_mirror)
    try:
        ticket = store.add(
            kind=args.kind,
            title=args.title,
            source=args.source,
            repro_ref=args.repro,
            acceptance_test=args.acceptance,
            priority=args.priority,
        )
    finally:
        store.close()
    _print(ticket.to_dict())
    return 0


def cmd_ticket_list(args: argparse.Namespace) -> int:
    from callosum.dev_loop.tickets import TicketStore

    store = TicketStore()
    try:
        tickets = store.list_tickets(status=args.status)
    finally:
        store.close()
    _print([t.to_dict() for t in tickets])
    return 0


def cmd_ticket_show(args: argparse.Namespace) -> int:
    from callosum.dev_loop.tickets import TicketStore

    store = TicketStore()
    try:
        ticket = store.get(args.id)
        if ticket is None:
            print(json.dumps({"error": f"no ticket {args.id}"}))
            return 1
        payload = ticket.to_dict()
        payload["events"] = store.events(args.id)
    finally:
        store.close()
    _print(payload)
    return 0


def cmd_ticket_triage(args: argparse.Namespace) -> int:
    from callosum.dev_loop.tickets import TicketStore

    store = TicketStore()
    try:
        ticket = store.triage(args.id, args.acceptance)
    finally:
        store.close()
    _print(ticket.to_dict())
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
