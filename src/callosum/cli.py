"""callosum CLI — operator commands against a running proxy.

`callosum serve` starts the daemon. The other subcommands administer a
running instance: the proxy exposes an admin HTTP surface under
/admin/*, and this CLI is a thin client that reads the admin token
(from ~/.config/callosum/admin_token, written on first proxy start)
and sends JSON requests.

Run `callosum -h` for the authoritative, always-current list of
subcommands — argparse generates it from the parser, so it never drifts
from the code. The current top-level surface is: serve, status,
version, params, denylist, routing, probe-tools, service,
auth-rotate.

Most output is JSON; `service` prints human-readable text.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from callosum.gate.tier2 import Tier2Config
from urllib import error, request

from callosum.config import load_config
from callosum.usage_diagnostic import (
    render_compounding_cost_json,
    render_recent_turns_json,
    render_token_time_series_json,
    run_usage_live,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
DEFAULT_SYSTEMD_UNIT = "system-dependency-callosum.service"
DEFAULT_RUNTIME_ROOT = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser() / "callosum" / "runtime"
SERVICE_SOURCE_ENV = "CALLOSUM_SERVICE_SOURCE"


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


def _repo_root_from_cwd() -> Path:
    try:
        return Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        sys.exit("callosum CLI: not inside a git checkout; pass --repo-root explicitly.")


def _runtime_root_from_args(args: argparse.Namespace) -> Path:
    raw = getattr(args, "runtime_root", None)
    if raw is None:
        return DEFAULT_RUNTIME_ROOT
    return Path(raw).expanduser()


def _runtime_callosum_binary(runtime_root: Path) -> Path:
    return runtime_root / "venv" / "bin" / "callosum"


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


def _tier2_config_from_args(args: argparse.Namespace) -> Tier2Config:
    """Build a Tier2Config from `callosum gate --tier2-*` flags.

    Defaults mirror Tier2Config field defaults (30 samples, 0.9 threshold,
    behavior-v1 suite) so an unflagged invocation leaves Tier 2 at pending
    (the empty-config guard in ResumableTier2Runner refuses a vacuous
    green over zero cells). --tier2-model accepts model_id or
    model_id:weight_identity; an omitted weight becomes None (the
    matrix keys it as "null").
    """
    from callosum.gate.tier2 import Tier2Config

    models: list[tuple[str, str | None]] = []
    for spec in args.tier2_models or ():
        if ":" in spec:
            model_id, weight = spec.split(":", 1)
            models.append((model_id, weight or None))
        else:
            models.append((spec, None))
    return Tier2Config(
        expected_tests=tuple(args.tier2_expected_tests or ()),
        models=tuple(models),
        min_samples=args.tier2_min_samples,
        threshold=args.tier2_threshold,
        suite_version=args.tier2_suite_version,
    )


def cmd_gate(args: argparse.Namespace) -> int:
    """Run the tiered merge/promotion gate (work tracker ).

    Tier 1 (deterministic CPU code tests + ruff + mypy --strict) is the only
    merge-blocking tier. Tier 2 (GPU rate matrix) and Tier 3 (shadow canary)
    report their own status but do not block a Tier-1-green merge. Exit 0
    when the gate is green-for-merge (Tier 1 green), 1 when merge-blocked."""
    from callosum.gate.harness import GateConfig, run_gate
    from callosum.gate.tier1 import Tier1Config, resolve_gate_python
    from callosum.gate.types import Tier

    repo_root = Path(args.repo_root).expanduser() if args.repo_root else _repo_root_from_cwd()
    tiers: tuple[Tier, ...]
    if args.tier:
        wanted = {"1": Tier.TIER1, "2": Tier.TIER2, "3": Tier.TIER3}
        tiers = tuple(wanted[t] for t in args.tier if t in wanted)
    else:
        tiers = (Tier.TIER1, Tier.TIER2, Tier.TIER3)
    config = GateConfig(
        tier1=Tier1Config(
            python=resolve_gate_python(str(repo_root)),
            full_suite=not args.no_full_suite,
        ),
        tier2=_tier2_config_from_args(args),
        repo_root=str(repo_root),
    )
    report = run_gate(config, tiers=tiers)
    if args.json:
        import json
        from dataclasses import asdict

        print(
            json.dumps(
                {
                    "merge_blocked": report.merge_blocked,
                    "green_for_merge": report.green,
                    "summary": report.summary,
                    "tiers": [
                        {
                            "tier": r.tier.value,
                            "status": r.status.value,
                            "reason": r.reason,
                            "checks": [asdict(c) for c in r.checks],
                        }
                        for r in report.results
                    ],
                },
                indent=2,
            )
        )
    else:
        print(report.summary)
        for r in report.results:
            print(f"  {r.tier.value}: {r.status.value} — {r.reason}")
            for c in r.checks:
                print(f"      {c.name}: {c.status.value} (rc={c.returncode}, {c.duration_s:.1f}s)")
    return 0 if report.green else 1


def _default_config_path() -> Path:
    return Path("~/.config/callosum/config.toml").expanduser()


def cmd_usage_recent(args: argparse.Namespace) -> int:
    config_path = args.config if args.config is not None else _default_config_path()
    if not config_path.exists():
        sys.exit(f"callosum CLI: config not found: {config_path}")
    cfg = load_config(config_path)
    usage_path = cfg.usage_log.path
    if usage_path is None:
        sys.exit("callosum CLI: usage_log.path is not configured.")
    payload = render_recent_turns_json(usage_path, limit=args.limit)
    _print(payload, pretty=not args.compact)
    return 0


def cmd_usage_series(args: argparse.Namespace) -> int:
    config_path = args.config if args.config is not None else _default_config_path()
    if not config_path.exists():
        sys.exit(f"callosum CLI: config not found: {config_path}")
    cfg = load_config(config_path)
    usage_path = cfg.usage_log.path
    if usage_path is None:
        sys.exit("callosum CLI: usage_log.path is not configured.")
    payload = render_token_time_series_json(
        usage_path,
        bucket=args.bucket,
        limit=args.limit,
        group_by=args.group_by,
    )
    _print(payload, pretty=not args.compact)
    return 0


def cmd_usage_compounding(args: argparse.Namespace) -> int:
    config_path = args.config if args.config is not None else _default_config_path()
    if not config_path.exists():
        sys.exit(f"callosum CLI: config not found: {config_path}")
    cfg = load_config(config_path)
    usage_path = cfg.usage_log.path
    if usage_path is None:
        sys.exit("callosum CLI: usage_log.path is not configured.")
    payload = render_compounding_cost_json(
        usage_path,
        session_id=args.session_id,
        limit_sessions=args.limit_sessions,
        min_turns=args.min_turns,
    )
    _print(payload, pretty=not args.compact)
    return 0


def cmd_usage_live(args: argparse.Namespace) -> int:
    config_path = args.config if args.config is not None else _default_config_path()
    if not config_path.exists():
        sys.exit(f"callosum CLI: config not found: {config_path}")
    cfg = load_config(config_path)
    usage_path = cfg.usage_log.path
    if usage_path is None:
        sys.exit("callosum CLI: usage_log.path is not configured.")
    try:
        return run_usage_live(
            usage_path,
            bucket=args.bucket,
            group_by=args.group_by,
            series_limit=args.series_limit,
            recent_limit=args.recent_limit,
            interval_s=args.interval,
            out_stream=sys.stdout,
        )
    except FileNotFoundError as exc:
        sys.exit(f"callosum CLI: {exc}")
    except ValueError as exc:
        sys.exit(f"callosum CLI: {exc}")


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


def cmd_build_runtime(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser() if args.repo_root else _repo_root_from_cwd()
    script = repo_root / "scripts" / "install_runtime_venv.sh"
    if not script.exists():
        sys.exit(f"callosum CLI: runtime install script not found: {script}")
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/callosum-uv-cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/callosum-xdg-cache")
    env["CALLOSUM_RUNTIME_ROOT"] = str(_runtime_root_from_args(args))
    return subprocess.run([str(script)], cwd=str(repo_root), env=env, check=False).returncode


def cmd_source_serve(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser() if args.repo_root else _repo_root_from_cwd()
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", "/tmp/callosum-uv-cache")
    env.setdefault("XDG_CACHE_HOME", "/tmp/callosum-xdg-cache")
    argv = ["uv", "run", "callosum", "serve"]
    if args.config is not None:
        argv.extend(["--config", str(args.config)])
    if args.host is not None:
        argv.extend(["--host", args.host])
    if args.port is not None:
        argv.extend(["--port", str(args.port)])
    return subprocess.run(argv, cwd=str(repo_root), env=env, check=False).returncode


def _run_service_command(argv: list[str]) -> int:
    try:
        return subprocess.run(argv, check=False).returncode
    except FileNotFoundError:
        sys.exit(f"callosum CLI: command not found: {argv[0]}")


def _set_service_source_mode(enabled: bool) -> int:
    argv = ["systemctl", "--user"]
    if enabled:
        argv.extend(["set-environment", f"{SERVICE_SOURCE_ENV}=1"])
    else:
        argv.extend(["unset-environment", SERVICE_SOURCE_ENV])
    return _run_service_command(argv)


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
    rc = _set_service_source_mode(args.source)
    if rc != 0:
        return rc
    return _run_service_command(["systemctl", "--user", "restart", args.unit])


def cmd_service_stop(args: argparse.Namespace) -> int:
    return _run_service_command(["systemctl", "--user", "stop", args.unit])


def cmd_service_start(args: argparse.Namespace) -> int:
    rc = _set_service_source_mode(args.source)
    if rc != 0:
        return rc
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
    if getattr(args, "source", False):
        return cmd_source_serve(args)

    from callosum.__main__ import serve_with_args

    serve_with_args(args)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = _HelpOnErrorParser(
        prog="callosum",
        description=(
            "Callosum proxy CLI. `callosum serve` starts the daemon; "
            "the other subcommands administer a running instance "
            "(status, routing mode, denylist, capability probing, "
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
    p_serve.add_argument(
        "--source",
        action="store_true",
        help="Development mode: run from the repo checkout via `uv run callosum serve`.",
    )
    p_serve.add_argument(
        "--repo-root",
        type=_Path,
        default=None,
        help="Repo checkout for --source (default: current git toplevel).",
    )
    p_serve.set_defaults(func=_cmd_serve)

    p_build = sub.add_parser(
        "build",
        help="Build the project and replace the installed runtime CLI in the runtime venv.",
        description=(
            "Build a wheel from the repo checkout and reinstall it into the "
            "dedicated runtime venv. This updates the installed `callosum` "
            "binary used by the managed service without restarting the service."
        ),
    )
    p_build.add_argument(
        "--repo-root",
        type=_Path,
        default=None,
        help="Path to the callosum checkout (default: current git toplevel).",
    )
    p_build.add_argument(
        "--runtime-root",
        type=_Path,
        default=None,
        help="Runtime root (default: ~/.local/share/callosum/runtime).",
    )
    p_build.set_defaults(func=cmd_build_runtime)
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

    p_usage = sub.add_parser(
        "usage",
        help="Read-only usage-log diagnostics over recent turns.",
        description=(
            "Read the configured usage-log SQLite database directly and "
            "print recent-turn token diagnostics. This surface is "
            "read-only and works even when the daemon is down."
        ),
    )
    pu = p_usage.add_subparsers(dest="subcommand", required=True)
    recent = pu.add_parser(
        "recent",
        help="Show recent turns with approximate per-segment prompt-token attribution.",
    )
    recent.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of recent turns to show (default: 10).",
    )
    recent.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ~/.config/callosum/config.toml).",
    )
    recent.add_argument(
        "--compact",
        action="store_true",
        help="Emit one-line JSON instead of pretty-printed JSON.",
    )
    recent.set_defaults(func=cmd_usage_recent)
    series = pu.add_parser(
        "series",
        help="Show token volume over time, grouped by a provenance axis.",
    )
    series.add_argument(
        "--bucket",
        choices=["hour", "day"],
        default="day",
        help="Time bucket size (default: day).",
    )
    series.add_argument(
        "--group-by",
        choices=["mode", "traffic_kind", "both"],
        default="mode",
        help=(
            "Provenance axis to break tokens down by (default: mode). "
            "'mode' groups by effective_routing_mode (the routing lens); "
            "'traffic_kind' groups by decision purpose — operator, "
            "canary_redirect, min_coverage_quota, peer_quality_capture, "
            "peer_quality_sidecar, legacy — "
            "which separates quota-forced/capture noise from real operator "
            "traffic (REQ-003); 'both' emits both breakdowns."
        ),
    )
    series.add_argument(
        "--limit",
        type=int,
        default=30,
        help="Number of most recent buckets to show (default: 30).",
    )
    series.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ~/.config/callosum/config.toml).",
    )
    series.add_argument(
        "--compact",
        action="store_true",
        help="Emit one-line JSON instead of pretty-printed JSON.",
    )
    series.set_defaults(func=cmd_usage_series)
    compounding = pu.add_parser(
        "compounding",
        help=(
            "Show per-session compounding input-token cost — the O(K^2) "
            "re-send tax across a multi-turn session, with per-turn marginal "
            "growth and tool-turn attribution."
        ),
    )
    compounding.add_argument(
        "--session-id",
        default=None,
        help="Restrict to one session_id (default: all sessions).",
    )
    compounding.add_argument(
        "--limit-sessions",
        type=int,
        default=10,
        help="Number of sessions to show (default: 10, most recently active first).",
    )
    compounding.add_argument(
        "--min-turns",
        type=int,
        default=2,
        help="Minimum turns for a session to be included (default: 2).",
    )
    compounding.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ~/.config/callosum/config.toml).",
    )
    compounding.add_argument(
        "--compact",
        action="store_true",
        help="Emit one-line JSON instead of pretty-printed JSON.",
    )
    compounding.set_defaults(func=cmd_usage_compounding)
    live = pu.add_parser(
        "live",
        help=(
            "Live, auto-refreshing terminal view of token volume over time "
            "and recent turns. Stdlib-only TUI (no new dependency); the F2 "
            "live-visualization path (operator decision 2026-08-23)."
        ),
    )
    live.add_argument(
        "--bucket",
        choices=["hour", "day"],
        default="day",
        help="Time bucket size for the volume-over-time panel (default: day).",
    )
    live.add_argument(
        "--group-by",
        choices=["mode", "traffic_kind", "both"],
        default="mode",
        help="Provenance axis for the volume-over-time panel (default: mode).",
    )
    live.add_argument(
        "--series-limit",
        type=int,
        default=14,
        help="Number of most recent buckets in the volume panel (default: 14).",
    )
    live.add_argument(
        "--recent-limit",
        type=int,
        default=15,
        help="Number of recent turns in the turns panel (default: 15).",
    )
    live.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5).",
    )
    live.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.toml (default: ~/.config/callosum/config.toml).",
    )
    live.set_defaults(func=cmd_usage_live)

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
        help="Backend routing mode (auto / offline / local-only / remote-only).",
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
    p_service.add_argument(
        "--runtime-root",
        type=_Path,
        default=None,
        help="Runtime root for the installed callosum binary (default: ~/.local/share/callosum/runtime).",
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
    restart = psrv.add_parser(
        "restart",
        help="Restart the managed Callosum service.",
    )
    restart.add_argument(
        "--source",
        action="store_true",
        help="Run the managed service from the repo checkout instead of the installed runtime binary.",
    )
    restart.set_defaults(func=cmd_service_restart)
    psrv.add_parser(
        "stop",
        help="Stop the managed Callosum service.",
    ).set_defaults(func=cmd_service_stop)
    start = psrv.add_parser(
        "start",
        help="Start the managed Callosum service.",
    )
    start.add_argument(
        "--source",
        action="store_true",
        help="Run the managed service from the repo checkout instead of the installed runtime binary.",
    )
    start.set_defaults(func=cmd_service_start)

    # Surface-only feedback redirect. Points the
    # operator at the existing external feedback channels (/feedback -> the
    # model provider's telemetry tenant, GitHub 3-cli.yml issue, ChatGPT
    # thumbs) for outputs callosum flagged as bad (quality_score = -1), and
    # records the operator's acknowledge/dismiss decision in a local audit
    # table. callosum does NOT relay feedback upstream; the snippet is
    # scrubbed of secret-shaped strings before display. Auto-send (in-band
    # injection without operator approval) is a separate scoped-A4 decision
    # with the auto-promotion master switch OFF and is NOT wired here.
    p_feedback = sub.add_parser(
        "feedback",
        help="Surface-only feedback redirect: point at external channels for flagged outputs.",
    )
    pf = p_feedback.add_subparsers(dest="feedback_cmd", required=True)
    pf_list = pf.add_parser("list", help="List pending bad-output suggestions (compact).")
    pf_list.add_argument("--limit", type=int, default=50)
    pf_list.set_defaults(func=cmd_feedback_list)
    pf_show = pf.add_parser("show", help="Show the scrubbed snippet + redirect for one request.")
    pf_show.add_argument("request_id", type=int)
    pf_show.set_defaults(func=cmd_feedback_show)
    pf_ack = pf.add_parser("acknowledge", help="Mark a suggestion acknowledged (you filed feedback).")
    pf_ack.add_argument("request_id", type=int)
    pf_ack.set_defaults(func=cmd_feedback_acknowledge)
    pf_dis = pf.add_parser("dismiss", help="Mark a suggestion dismissed (not worth filing).")
    pf_dis.add_argument("request_id", type=int)
    pf_dis.set_defaults(func=cmd_feedback_dismiss)

    # Auth-rotate wizard. Wired here so the help surface lists it
    # alongside the other operator commands.
    from callosum.auth_rotate import add_subparser as _add_auth_rotate

    _add_auth_rotate(sub)

    p_gate = sub.add_parser(
        "gate",
        help="Run the tiered merge/promotion gate: Tier-1 CPU tests (blocking) "
        "+ Tier-2 GPU rate matrix (resumable, external) + Tier-3 shadow canary.",
        description=(
            "Run the tiered merge/promotion gate (work tracker ). "
            "Tier 1 (pytest inaugural bug cases + unit suite + ruff + mypy --strict) "
            "is the only merge-blocking tier. Tier 2 consumes the GPU rate matrix "
            "published by the sibling benchmark suite repo (read-only, resumable). "
            "Tier 3 evaluates the live shadow/canary guard with auto-revert. "
            "Exit 0 when Tier 1 is green, 1 when merge-blocked."
        ),
    )
    p_gate.add_argument(
        "--tier",
        action="append",
        choices=["1", "2", "3"],
        default=None,
        help="Run only this tier (repeatable). Default: all three.",
    )
    p_gate.add_argument(
        "--no-full-suite",
        action="store_true",
        help="Skip the full tests/unit suite; run only the inaugural bug cases + ruff + mypy.",
    )
    p_gate.add_argument(
        "--repo-root",
        type=_Path,
        default=None,
        help="Repo checkout to run against (default: current git toplevel).",
    )
    p_gate.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON report instead of the human-readable summary.",
    )
    p_gate.add_argument(
        "--tier2-expected-test",
        action="append",
        default=None,
        dest="tier2_expected_tests",
        metavar="TEST",
        help="Tier-2 behavior test name to require (repeatable). Must match a test the "
        "sibling benchmark suite rate matrix publishes (e.g. mmlu-pro). Default: none "
        "(Tier-2 stays pending until configured).",
    )
    p_gate.add_argument(
        "--tier2-model",
        action="append",
        default=None,
        dest="tier2_models",
        metavar="MODEL[:WEIGHT]",
        help="Local model cell to evaluate, as model_id or model_id:weight_identity (repeatable). "
        "weight_identity is optional and defaults to null. Default: none.",
    )
    p_gate.add_argument(
        "--tier2-min-samples",
        type=int,
        default=30,
        help="Minimum samples per test per cell for coverage (default: 30).",
    )
    p_gate.add_argument(
        "--tier2-threshold",
        type=float,
        default=0.9,
        help="Wilson lower bound a cell's per-test pass rate must meet (default: 0.9).",
    )
    p_gate.add_argument(
        "--tier2-suite-version",
        default="behavior-v1",
        help="Suite version the published matrix must report (default: behavior-v1).",
    )
    p_gate.set_defaults(func=cmd_gate)

    return parser


def _fmt_ts(ts: float) -> str:
    """Render a request-log epoch float as a local timestamp string."""
    from datetime import datetime

    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return "(unknown)"


def _cell_label(model: str | None, effort: str | None) -> str:
    if not model:
        return "(unknown cell)"
    return f"{model}/{effort}" if effort else model


def cmd_feedback_list(args: argparse.Namespace) -> int:
    """Compact listing of pending bad-output suggestions.

    Prints one line per suggestion (request id, cell, thread, detector,
    timestamp) plus the ready-to-paste redirect message for each. The
    operator reads these, files feedback via the named external channels
    themselves, then `callosum feedback acknowledge <id>` (filed) or
    `callosum feedback dismiss <id>` (not worth filing). callosum never
    relays anything upstream.
    """
    payload = _request("GET", f"/admin/feedback?limit={args.limit}")
    if not isinstance(payload, dict):
        print("error: unexpected /admin/feedback payload", file=sys.stderr)
        return 1
    entries = payload.get("pending") or []
    count = payload.get("count", len(entries))
    if not entries:
        print("pending feedback suggestions: 0 (clean queue)")
        return 0
    print(f"pending feedback suggestions: {count}")
    print()
    for e in entries:
        rid = e.get("request_id", "?")
        cell = _cell_label(e.get("model"), e.get("reasoning_effort"))
        thread = e.get("session_id") or "(none)"
        detector = e.get("detector") or "unknown"
        ts = _fmt_ts(e.get("ts_start", 0.0))
        print(f"  request {rid}  {ts}  cell={cell}  thread={thread}  detector={detector}")
        redirect = e.get("redirect")
        if isinstance(redirect, str) and redirect:
            print()
            print(redirect.rstrip())
        print()
        print(f"    file via external channels, then:  callosum feedback acknowledge {rid}")
        print(f"    or skip:                            callosum feedback dismiss {rid}")
        print()
    return 0


def cmd_feedback_show(args: argparse.Namespace) -> int:
    """Show the scrubbed snippet + redirect for one request id."""
    payload = _request("GET", f"/admin/feedback/{args.request_id}")
    if not isinstance(payload, dict):
        print("error: unexpected /admin/feedback payload", file=sys.stderr)
        return 1
    status = payload.get("status", "pending")
    cell = _cell_label(payload.get("model"), payload.get("reasoning_effort"))
    thread = payload.get("session_id") or "(none)"
    detector = payload.get("detector") or "unknown"
    ts = _fmt_ts(payload.get("ts_start", 0.0))
    print(f"request {payload.get('request_id', args.request_id)}  {ts}  status={status}")
    print(f"  cell={cell}  thread={thread}  detector={detector}")
    print()
    redirect = payload.get("redirect")
    if isinstance(redirect, str) and redirect:
        print(redirect.rstrip())
    else:
        print("(no redirect text)")
    return 0


def cmd_feedback_acknowledge(args: argparse.Namespace) -> int:
    """Mark a suggestion acknowledged (operator filed feedback externally)."""
    payload = _request("POST", f"/admin/feedback/{args.request_id}/acknowledge")
    _print(payload)
    return 0


def cmd_feedback_dismiss(args: argparse.Namespace) -> int:
    """Mark a suggestion dismissed (not worth filing)."""
    payload = _request("POST", f"/admin/feedback/{args.request_id}/dismiss")
    _print(payload)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    rc = args.func(args)
    return int(rc) if rc is not None else 0


if __name__ == "__main__":
    sys.exit(main())
