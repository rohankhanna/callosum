"""`callosum-ctl auth-rotate` — guided rotation of codex_auth_vault auth.json.

When a `codex_auth_vault` backend's refresh chain gets desynced (because
something else used its refresh token first, or because the access
token expired while the local vault state was stale) callosum returns
502 with a `refresh rejected: 401` error. The fix is to re-authenticate
into the same ChatGPT account and replace the vault's auth.json with
the fresh OAuth bundle.

This command turns that recovery procedure into a repeatable wizard:

  1. Reads callosum's config to discover all codex_auth_vault backends
     and their vault_path values.
  2. For each backend (or one specified by `--backend`), creates an
     isolated CODEX_HOME directory, prints the exact `codex login`
     command the operator should run in a separate terminal, and
     waits for the operator to confirm completion.
  3. Validates that the fresh auth.json materialized correctly and has
     a refresh_token in the expected shape.
  4. Backs up the existing vault file with a timestamped .bak suffix.
  5. Installs the fresh auth.json into the vault path with mode 0600.
  6. Wipes the temp CODEX_HOME so the auth.json copy doesn't linger.

The command never reads or prints token values. Validation checks
structure only (key presence, valid JSON), not content.

Operator safety:

  * Different accounts: the wizard reminds the operator at each step
    that backends must NOT share ChatGPT accounts. Two backends with
    the same account recreates the race condition that caused the
    rotation to be needed in the first place.
  * Backup-first: existing vault files are copied to `.bak.<timestamp>`
    BEFORE the new file is installed. If anything goes wrong the
    rollback is one `cp` command.
  * No restart by default: the wizard does NOT restart callosum after
    rotating. callosum's AuthVault has mtime-based reload, so new
    files should be picked up on next use anyway. Pass `--restart` to
    explicitly bounce the systemd unit after rotation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Standard callosum config locations. The wizard walks these in order
# and uses the first one that exists. Operators with non-standard
# configs can pass `--config` to override.
_DEFAULT_CONFIG_PATHS: tuple[Path, ...] = (
    Path.home() / ".config" / "callosum" / "config.toml",
)


# systemd unit name to restart when --restart is given. Matches the
# system-managed naming used by the current install. Operators with
# different unit naming pass --systemd-unit to override.
_DEFAULT_SYSTEMD_UNIT = "system-dependency-callosum.service"


@dataclass(frozen=True)
class VaultBackend:
    """One codex_auth_vault backend's identity for rotation purposes.
    Just the id + vault_path; we don't load the auth.json itself here
    because we have no reason to read its contents."""

    id: str
    vault_path: Path


def _load_config_path(explicit: str | None) -> Path:
    """Resolve which config.toml to read. Explicit arg wins; otherwise
    walk the standard locations and pick the first that exists."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"config not found at {path}")
        return path
    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("no callosum config.toml found in standard locations; pass --config")


def _parse_vault_backends(config_path: Path) -> list[VaultBackend]:
    """Parse the config.toml and return every codex_auth_vault backend.

    Uses the stdlib `tomllib` (Python 3.11+); callosum already requires
    >=3.11 per pyproject so this is safe.

    Backends without a vault_path (e.g. credential_proxy / litellm
    gateway backends) are skipped silently; this command only acts on
    codex_auth_vault rows.
    """
    import tomllib

    payload = tomllib.loads(config_path.read_text())
    backends = payload.get("backends") or []
    out: list[VaultBackend] = []
    for b in backends:
        # callosum's BackendConfig field is `type` (not `kind`) and
        # defaults to "codex_auth_vault" when omitted — see
        # callosum.config.BackendConfig. So an entry without an
        # explicit type IS a codex_auth_vault entry. Only skip when
        # the type is explicitly something else.
        backend_type = b.get("type", "codex_auth_vault")
        if backend_type != "codex_auth_vault":
            continue
        backend_id = b.get("id")
        vault_path = b.get("vault_path")
        if not backend_id or not vault_path:
            continue
        out.append(VaultBackend(id=backend_id, vault_path=Path(vault_path)))
    return out


def _validate_fresh_auth(path: Path) -> tuple[bool, str]:
    """Confirm the newly-minted auth.json looks structurally correct.

    Returns (ok, reason). Validation checks STRUCTURE only — never
    prints or returns token content. The bool gates the install step;
    the string explains failures to the operator without leaking
    anything.
    """
    if not path.exists():
        return False, f"no auth.json at {path} (login did not produce one)"
    try:
        size = path.stat().st_size
    except OSError as exc:
        return False, f"could not stat {path}: {exc}"
    if size == 0:
        return False, f"{path} is empty"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return False, f"{path} is not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return False, f"{path} is not a JSON object"
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return False, f"{path} has no `tokens` object"
    # The fields callosum's AuthVault actually consumes. We don't print
    # them; we only confirm they exist.
    for field in ("access_token", "refresh_token"):
        v = tokens.get(field)
        if not isinstance(v, str) or not v:
            return False, f"{path} missing or empty `tokens.{field}`"
    return True, "ok"


def _prompt_continue(message: str) -> None:
    """Block until the operator presses Enter. Standard pattern for
    'do this in another terminal, then come back here.'"""
    try:
        input(f"\n{message}\nPress Enter to continue (or Ctrl-C to abort)... ")
    except KeyboardInterrupt:
        print("\nAborted by operator. No changes made to vault files.", file=sys.stderr)
        sys.exit(130)


def _rotate_one(backend: VaultBackend, *, dry_run: bool) -> bool:
    """Walk the operator through rotating one backend's vault. Returns
    True on successful install, False if the operator aborts or the
    fresh auth.json fails validation."""
    print(f"\n{'=' * 70}")
    print(f"Rotating vault for backend  : {backend.id}")
    print(f"Vault file (callosum reads) : {backend.vault_path}")
    if not backend.vault_path.parent.exists():
        print(
            f"  WARNING: parent directory {backend.vault_path.parent} "
            "does not exist. The install step will fail unless you "
            "create it first.",
            file=sys.stderr,
        )

    isolated_home = Path(tempfile.mkdtemp(prefix=f"codex-fresh-{backend.id}-"))
    os.chmod(isolated_home, 0o700)
    print(f"Temp CODEX_HOME             : {isolated_home}")

    print(
        f"""
Steps for THIS backend ({backend.id}):

  1. Open a SEPARATE terminal.
  2. Run:

         CODEX_HOME={isolated_home} codex login --device-auth

     (Omit `--device-auth` if you have a graphical browser handy.)

  3. Log in with the ChatGPT account dedicated to backend
     `{backend.id}`. THIS MUST BE A DIFFERENT ACCOUNT FROM EVERY
     OTHER BACKEND. Logging into the same account twice will recreate
     the race condition that caused this rotation to be needed.

  4. When the login completes successfully, return to THIS terminal.
"""
    )

    if dry_run:
        print("(dry-run: would wait here for operator confirmation)")
        print(f"(dry-run: would validate, back up {backend.vault_path}, install fresh auth.json)")
        # In dry-run we still clean up the temp dir.
        with contextlib.suppress(OSError):
            isolated_home.rmdir()
        return True

    _prompt_continue(f"After step 4 (login completed for backend `{backend.id}`):")

    fresh_path = isolated_home / "auth.json"
    ok, reason = _validate_fresh_auth(fresh_path)
    if not ok:
        print(f"  ✗ Fresh auth.json validation FAILED: {reason}", file=sys.stderr)
        print(
            "  Aborting this backend. Re-run the command after fixing "
            "the issue. (Login may have failed silently — check the "
            "other terminal for errors.)",
            file=sys.stderr,
        )
        return False
    print("  ✓ Fresh auth.json validated (structure looks correct).")

    # Backup
    ts = time.strftime("%Y%m%dT%H%M%S")
    backup_path = backend.vault_path.with_name(f"{backend.vault_path.name}.bak.{ts}")
    if backend.vault_path.exists():
        shutil.copy2(backend.vault_path, backup_path)
        print(f"  ✓ Backed up existing vault to {backup_path}")
    else:
        print("  ! No existing vault file to back up — first-time rotation for this backend?")

    # Install new vault file with mode 0600. shutil.copyfile copies
    # contents; we set mode explicitly to be safe.
    shutil.copyfile(fresh_path, backend.vault_path)
    os.chmod(backend.vault_path, 0o600)
    print(f"  ✓ Fresh auth.json installed at {backend.vault_path} (mode 0600)")

    # Wipe the temp home. The auth.json there is a duplicate of what
    # we just installed; leaving it around is unnecessary exposure.
    with contextlib.suppress(OSError):
        fresh_path.unlink()
    try:
        isolated_home.rmdir()
    except OSError:
        # Other files might be present (codex sometimes writes session
        # data alongside auth.json). Walk to remove what we can.
        for p in isolated_home.iterdir():
            with contextlib.suppress(OSError):
                p.unlink()
        with contextlib.suppress(OSError):
            isolated_home.rmdir()

    return True


def _restart_callosum(unit: str) -> bool:
    """Optional `--restart` step. Returns True on success."""
    print(f"\nRestarting systemd unit {unit}...")
    try:
        result = subprocess.run(
            ["systemctl", "--user", "restart", unit],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        print("  ✗ systemctl not found. Restart callosum manually.", file=sys.stderr)
        return False
    if result.returncode != 0:
        print(
            f"  ✗ systemctl restart failed (exit {result.returncode})\n  stderr: {result.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    print("  ✓ Restart issued. Health check in 5s...")
    time.sleep(5)
    health = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"  unit status: {health.stdout.strip()}")
    return health.returncode == 0


def cmd_auth_rotate(args: argparse.Namespace) -> int:
    """Entry point invoked by `callosum-ctl auth-rotate`."""
    try:
        config_path = _load_config_path(args.config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Reading config: {config_path}")
    try:
        backends = _parse_vault_backends(config_path)
    except Exception as exc:
        print(f"error parsing config: {exc}", file=sys.stderr)
        return 2

    if not backends:
        print(
            "No codex_auth_vault backends found in config. Nothing to rotate.",
            file=sys.stderr,
        )
        return 1

    if args.backend is not None:
        backends = [b for b in backends if b.id == args.backend]
        if not backends:
            print(
                f"No codex_auth_vault backend with id={args.backend!r} found.",
                file=sys.stderr,
            )
            return 2

    print(f"Backends to rotate: {[b.id for b in backends]}")
    if args.dry_run:
        print("(--dry-run: no files will be modified)")

    if not args.skip_stop and not args.dry_run:
        print(
            f"\nBEFORE STARTING: stop callosum to avoid race during "
            "file replacement. Recommended:\n"
            f"  systemctl --user stop {args.systemd_unit}\n"
            "Pass --skip-stop to skip this advisory (e.g. if callosum "
            "is already stopped, or if you accept the brief race window).",
        )
        _prompt_continue("Once callosum is stopped (or you accept the risk):")

    results: dict[str, bool] = {}
    for backend in backends:
        results[backend.id] = _rotate_one(backend, dry_run=args.dry_run)

    print(f"\n{'=' * 70}")
    print("Rotation summary:")
    for bid, ok in results.items():
        status = "rotated" if ok else "skipped / failed"
        print(f"  {bid:12s}  {status}")

    if args.restart and not args.dry_run:
        _restart_callosum(args.systemd_unit)
    elif not args.dry_run:
        print(
            f"\nNext step: restart callosum so it reloads the vault state.\n"
            f"  systemctl --user restart {args.systemd_unit}\n"
            f"Then verify with `callosum-ctl status`."
        )

    return 0 if all(results.values()) else 1


def add_subparser(sub: Any) -> None:
    """Wire the `auth-rotate` subcommand into the existing argparse
    subparser tree. Invoked from callosum.cli.build_parser()."""
    p = sub.add_parser(
        "auth-rotate",
        help="Guided rotation of codex_auth_vault auth.json files.",
        description=(
            "Walks the operator through re-authenticating each "
            "codex_auth_vault backend's ChatGPT account and replacing "
            "the vault's auth.json with the fresh OAuth bundle. Used "
            "when callosum starts returning 502 with 'refresh "
            "rejected: 401' errors because the refresh chain got "
            "desynced."
        ),
    )
    p.add_argument(
        "--config",
        help="Path to callosum config.toml (default: auto-detect under "
        "~/.config/callosum/).",
    )
    p.add_argument(
        "--backend",
        help="Rotate only this backend id (default: rotate every codex_auth_vault backend).",
    )
    p.add_argument(
        "--restart",
        action="store_true",
        help="Restart the callosum systemd unit after rotation. Off "
        "by default; callosum's AuthVault has mtime-based reload "
        "and may not need a restart.",
    )
    p.add_argument(
        "--systemd-unit",
        default=_DEFAULT_SYSTEMD_UNIT,
        help=f"systemd unit name for stop/restart messages and --restart (default: {_DEFAULT_SYSTEMD_UNIT}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without writing files or prompting for login completion.",
    )
    p.add_argument(
        "--skip-stop",
        action="store_true",
        help="Skip the 'stop callosum first' advisory prompt.",
    )
    p.set_defaults(func=cmd_auth_rotate)
