# Rotating codex_auth_vault auth tokens

When callosum's `codex_auth_vault` backends get into a "refresh
rejected: 401" state — usually because another process consumed a
refresh token in parallel — the fix is to re-authenticate into the
relevant ChatGPT accounts and replace each vault's `auth.json` with a
fresh OAuth bundle.

`callosum-ctl auth-rotate` automates this as a guided wizard. The
wizard never reads or prints token content; it only acts on file
paths and structural validation.

## Quick usage

```bash
# See what would happen, without acting:
callosum-ctl auth-rotate --dry-run --skip-stop

# Rotate every codex_auth_vault backend:
callosum-ctl auth-rotate

# Rotate just one:
callosum-ctl auth-rotate --backend primary

# Rotate + restart callosum via systemd afterward:
callosum-ctl auth-rotate --restart
```

## How it works

For each codex_auth_vault backend declared in your config:

1. Creates an isolated `CODEX_HOME` under tempdir (per-backend, so
   logins for different backends don't overwrite each other).
2. Prints the exact `CODEX_HOME=... codex login --device-auth`
   command you should run **in another terminal**, plus an explicit
   reminder that this backend's ChatGPT account must be DIFFERENT
   from every other backend's account.
3. Waits for you to press Enter signaling the login completed.
4. Validates the fresh `auth.json` (file exists, non-empty, valid
   JSON, contains `tokens.access_token` and `tokens.refresh_token`).
   If validation fails, the wizard skips this backend with a clear
   reason and moves on — your existing vault file stays intact.
5. Backs up the existing vault file with a timestamped `.bak` suffix.
6. Atomically installs the fresh `auth.json` at the vault path with
   mode `0600`.
7. Wipes the temp `CODEX_HOME`.

## Critical: different ChatGPT accounts per backend

If two backends end up authenticated against the SAME ChatGPT account
— even with separate vault paths — they will share an OAuth refresh
chain and stomp each other's refresh tokens. That's the race condition
this wizard exists to recover from; running the wizard with the same
account twice would recreate the same bug.

The wizard reminds you of this at every prompt. Make sure you have
one dedicated ChatGPT account per `codex_auth_vault` entry in your
config, and that none of those accounts is one you log into from any
other tool (especially not the interactive `codex` CLI you run in
your terminals).

## Operator safety properties

- **No silent destruction.** Existing vault files are backed up to
  `<path>.bak.<timestamp>` *before* the new file is installed. To
  roll back: `cp -a <path>.bak.<timestamp> <path>` and restart
  callosum.
- **Per-backend rotation.** Failure on one backend doesn't abort the
  others. The end-of-run summary shows which rotated and which
  skipped/failed.
- **No token leakage in output.** Validation messages describe
  structure only ("no `tokens` object", "empty file", "not valid
  JSON") — never the content.
- **`--dry-run`.** Walks the entire wizard without writing files or
  prompting. Use it to confirm the wizard found the backends you
  expected before committing to the live rotation.

## When to use this

- callosum returning 502 with `refresh rejected: 401` errors
  intermittently or consistently.
- You suspect an external process used one of the vault's refresh
  tokens.
- You're setting up callosum for the first time and need to populate
  vault files for the declared backends.
- You're rotating accounts intentionally (account compromised, plan
  change, etc.).
