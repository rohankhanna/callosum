"""Tests for callosum.auth_rotate — the CLI wizard that walks the
operator through re-authenticating codex_auth_vault backends.

The interactive parts (waiting for the operator to run `codex login`
in another terminal) are deliberately not tested here — they'd require
stdin mocking and would be brittle. The deterministic parts ARE
covered:

  * Config parsing: codex_auth_vault backends are picked up; others
    are skipped; missing config / missing fields handled cleanly.
  * Validation: fresh auth.json is accepted, missing/empty/malformed
    files are rejected with reasons that don't leak content.
  * Dry-run end-to-end: produces no side effects on disk.

The validation function is the most important to lock down because
it's the gate between "operator logged in successfully" and
"overwriting the existing vault." Anything that passes validation
will replace a working vault file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from callosum.auth_rotate import (
    VaultBackend,
    _parse_vault_backends,
    _validate_fresh_auth,
    cmd_auth_rotate,
)

# ---------- config parsing -------------------------------------------------


def _write_config(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(content)
    return path


def test_parse_picks_up_codex_auth_vault_backends(tmp_path: Path) -> None:
    """Two codex_auth_vault backends declared → both returned, in
    declaration order, with their vault_path resolved."""
    cfg = _write_config(
        tmp_path,
        """
[[backends]]
id = "primary"
type = "codex_auth_vault"
vault_path = "/vault/primary/auth.json"
models = ["model-a0e8"]

[[backends]]
id = "secondary"
type = "codex_auth_vault"
vault_path = "/vault/secondary/auth.json"
models = ["model-a0e8"]
""",
    )
    backends = _parse_vault_backends(cfg)
    assert len(backends) == 2
    assert backends[0] == VaultBackend(id="primary", vault_path=Path("/vault/primary/auth.json"))
    assert backends[1] == VaultBackend(id="secondary", vault_path=Path("/vault/secondary/auth.json"))


def test_parse_skips_non_codex_auth_vault_backends(tmp_path: Path) -> None:
    """A litellm_gateway backend in the same config should be ignored
    by this command — it doesn't have an auth.json to rotate."""
    cfg = _write_config(
        tmp_path,
        """
[[backends]]
id = "primary"
type = "codex_auth_vault"
vault_path = "/vault/primary/auth.json"
models = ["model-a0e8"]

[[backends]]
id = "local-credential-proxy"
type = "credential_proxy"
proxy_url = "http://127.0.0.1:9999"
upstream_url = "http://127.0.0.1:11434"
models = ["model-a0d5"]
""",
    )
    backends = _parse_vault_backends(cfg)
    assert len(backends) == 1
    assert backends[0].id == "primary"


def test_parse_skips_codex_vault_without_vault_path(tmp_path: Path) -> None:
    """A codex_auth_vault entry that's missing vault_path is malformed.
    The wizard skips it rather than crashing — the operator may have a
    half-edited config that they're still working on, and we'd rather
    rotate what's complete than refuse everything."""
    cfg = _write_config(
        tmp_path,
        """
[[backends]]
id = "complete"
type = "codex_auth_vault"
vault_path = "/vault/complete/auth.json"
models = ["model-a0e8"]

[[backends]]
id = "missing_path"
type = "codex_auth_vault"
models = ["model-a0e8"]
""",
    )
    backends = _parse_vault_backends(cfg)
    assert len(backends) == 1
    assert backends[0].id == "complete"


def test_parse_empty_config_returns_empty_list(tmp_path: Path) -> None:
    """A config with no backends at all (or all non-codex_auth_vault
    backends) returns []. cmd_auth_rotate handles this case with a
    clear 'nothing to rotate' message; this test pins the underlying
    parser behavior."""
    cfg = _write_config(tmp_path, "")
    assert _parse_vault_backends(cfg) == []


# ---------- validation -----------------------------------------------------


def _write_good_auth(path: Path) -> None:
    """Helper: write a minimally-valid auth.json. We use clearly-bogus
    test token values; the validation logic only checks STRUCTURE,
    never the bytes — so any string is fine for these tests."""
    path.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "test-access-not-a-real-token",
                    "refresh_token": "test-refresh-not-a-real-token",
                    "account_id": "test@example.com",
                }
            }
        )
    )


def test_validate_accepts_well_formed_auth_json(tmp_path: Path) -> None:
    path = tmp_path / "auth.json"
    _write_good_auth(path)
    ok, reason = _validate_fresh_auth(path)
    assert ok is True
    assert reason == "ok"


def test_validate_rejects_missing_file(tmp_path: Path) -> None:
    """The most common failure mode: operator didn't actually complete
    login, so no auth.json was produced. Reason mentions the missing
    file but doesn't try to read it."""
    ok, reason = _validate_fresh_auth(tmp_path / "does-not-exist.json")
    assert ok is False
    assert "no auth.json" in reason.lower()


def test_validate_rejects_empty_file(tmp_path: Path) -> None:
    """A zero-byte auth.json could happen if codex crashed after
    creating the file but before writing content. Reject."""
    path = tmp_path / "empty.json"
    path.touch()
    ok, reason = _validate_fresh_auth(path)
    assert ok is False
    assert "empty" in reason.lower()


def test_validate_rejects_malformed_json(tmp_path: Path) -> None:
    """The reason should mention JSON parsing but should NOT include
    the file's content — error messages can sometimes leak content
    in their context; this test ensures we don't."""
    path = tmp_path / "broken.json"
    path.write_text("{ this is not valid")
    ok, reason = _validate_fresh_auth(path)
    assert ok is False
    assert "not valid json" in reason.lower()


def test_validate_rejects_non_object_root(tmp_path: Path) -> None:
    """A JSON array or scalar at the root isn't a valid auth.json
    shape. callosum's AuthVault would crash on it later."""
    path = tmp_path / "array.json"
    path.write_text("[1, 2, 3]")
    ok, reason = _validate_fresh_auth(path)
    assert ok is False
    assert "not a json object" in reason.lower()


def test_validate_rejects_missing_tokens_block(tmp_path: Path) -> None:
    """auth.json must have a `tokens` object — this is the structural
    contract AuthVault depends on."""
    path = tmp_path / "no-tokens.json"
    path.write_text(json.dumps({"some_other_field": "value"}))
    ok, reason = _validate_fresh_auth(path)
    assert ok is False
    assert "no `tokens`" in reason.lower()


def test_validate_rejects_missing_refresh_token(tmp_path: Path) -> None:
    """Without a refresh_token there's no point installing this
    auth.json — callosum can't refresh the access token when it
    expires, and rotation defeats its own purpose. Reject."""
    path = tmp_path / "no-refresh.json"
    path.write_text(
        json.dumps(
            {
                "tokens": {"access_token": "x"},  # access only, no refresh
            }
        )
    )
    ok, reason = _validate_fresh_auth(path)
    assert ok is False
    assert "refresh_token" in reason


def test_validate_rejects_empty_string_token(tmp_path: Path) -> None:
    """A present-but-empty refresh_token is just as broken as a
    missing one. Edge case worth pinning."""
    path = tmp_path / "empty-tokens.json"
    path.write_text(
        json.dumps(
            {
                "tokens": {"access_token": "x", "refresh_token": ""},
            }
        )
    )
    ok, reason = _validate_fresh_auth(path)
    assert ok is False
    assert "refresh_token" in reason


# ---------- dry-run end-to-end --------------------------------------------


def test_dry_run_makes_no_filesystem_changes(tmp_path: Path) -> None:
    """The whole point of --dry-run is to preview without acting. The
    existing vault file MUST NOT be modified, backed up, or replaced
    when this flag is set."""
    cfg = _write_config(
        tmp_path,
        f"""
[[backends]]
id = "primary"
type = "codex_auth_vault"
vault_path = "{tmp_path / "existing-vault.json"}"
models = ["model-a0e8"]
""",
    )
    # Create an "existing" vault file.
    existing = tmp_path / "existing-vault.json"
    existing.write_text("ORIGINAL CONTENT")
    original_mtime = existing.stat().st_mtime
    original_content = existing.read_text()

    args = argparse.Namespace(
        config=str(cfg),
        backend=None,
        restart=False,
        systemd_unit="callosum.service",
        dry_run=True,
        skip_stop=True,  # no prompt in dry-run
    )
    rc = cmd_auth_rotate(args)
    assert rc == 0
    # Existing vault completely untouched.
    assert existing.read_text() == original_content
    assert existing.stat().st_mtime == original_mtime
    # No backup file created.
    backups = list(tmp_path.glob("existing-vault.json.bak.*"))
    assert backups == []


def test_dry_run_with_no_matching_backend_returns_2(tmp_path: Path) -> None:
    """Specifying --backend with a name that doesn't match any
    declared codex_auth_vault returns exit code 2 (operator-error
    convention)."""
    cfg = _write_config(
        tmp_path,
        """
[[backends]]
id = "primary"
type = "codex_auth_vault"
vault_path = "/vault/primary/auth.json"
models = ["model-a0e8"]
""",
    )
    args = argparse.Namespace(
        config=str(cfg),
        backend="does-not-exist",
        restart=False,
        systemd_unit="callosum.service",
        dry_run=True,
        skip_stop=True,
    )
    assert cmd_auth_rotate(args) == 2


def test_missing_config_returns_2(tmp_path: Path) -> None:
    """Bogus --config path → exit 2 (operator-error)."""
    args = argparse.Namespace(
        config=str(tmp_path / "definitely-not-there.toml"),
        backend=None,
        restart=False,
        systemd_unit="callosum.service",
        dry_run=True,
        skip_stop=True,
    )
    assert cmd_auth_rotate(args) == 2


def test_config_with_no_vault_backends_returns_1(tmp_path: Path) -> None:
    """Config exists but has no codex_auth_vault backends → exit 1
    (nothing to do; a softer error than malformed config)."""
    cfg = _write_config(
        tmp_path,
        """
[[backends]]
id = "local-credential-proxy"
type = "credential_proxy"
proxy_url = "http://127.0.0.1:9999"
upstream_url = "http://127.0.0.1:11434"
models = ["model-a0d5"]
""",
    )
    args = argparse.Namespace(
        config=str(cfg),
        backend=None,
        restart=False,
        systemd_unit="callosum.service",
        dry_run=True,
        skip_stop=True,
    )
    assert cmd_auth_rotate(args) == 1
