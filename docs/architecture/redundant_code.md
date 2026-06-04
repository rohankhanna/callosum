# Redundant code

**Definition.** Code that is fully working, fully tested, and not
currently exercised in production — kept on purpose as a fallback for
when the actively-used alternative path breaks.

Distinct from dead code (which has no current or contingent purpose
and should be deleted) and from feature-flagged code (which is
exercised by a subset of traffic). Redundant code is *off* in normal
operation but ready to be flipped on by a config change alone — no
code edits, no migration, no new dependencies.

## Why call it out

Without an explicit category, code that becomes inactive after a
migration tends to either:

  * Get deleted on the assumption it's dead — losing the fallback the
    operator was implicitly relying on.
  * Accumulate as unmarked clutter — future readers can't tell what's
    load-bearing, what's actively maintained, and what's optional.

Naming the category fixes both. Redundant code is documented to exist
on purpose, gets the same test coverage as active code, and survives
refactors with an explicit reason.

## Properties redundant code must keep

  * **Passes its tests.** A redundant fallback that has silently broken
    isn't a fallback — it's a trap. Every CI run must exercise the
    redundant path the same as the active one.
  * **Operator-switchable without code changes.** Activation must be
    a config / runtime flag, not "comment out these lines and recompile."
  * **Documented here.** The list below names every redundant
    component, what it's a fallback for, and what would trigger
    activating it.
  * **Reviewed in PRs.** A change that breaks a redundant path must be
    treated as a real regression, not a "we don't use that anymore"
    skip.

## Current redundant components

### `callosum.backends.codex_auth_vault.CodexAuthVaultBackend`

  * **Active alternative**: `callosum.backends.credential_proxy.CredentialProxyBackend`,
    which routes OAuth refresh through the credential proxy service.
  * **What this fallback provides**: direct file-backed OAuth refresh
    against `auth.openai.com/oauth/token`, with no external service
    dependency. The backend reads / writes `auth.json` on disk and
    manages its own refresh chain.
  * **When you'd activate it**: credential proxy is unreachable, broken, or
    being temporarily decommissioned. Switch a backend's `type` in
    `config.toml` from `credential_proxy` to `codex_auth_vault`,
    point `vault_path` at a fresh `auth.json` (run `callosum-ctl
    auth-rotate` to produce one), restart. No code edits needed.
  * **Why it stays redundant rather than active**: credential proxy centralizes
    OAuth refresh across multiple consumers, eliminating the
    multi-refresher race that plagued the file-backed path. Until and
    unless credential proxy becomes unavailable, the credential_proxy path is
    architecturally preferable.
  * **Maintenance contract**: every test in
    `tests/unit/test_codex_auth_vault_backend.py` continues to run in
    CI. Behavior changes that affect this backend require updating
    those tests, the same as for any active backend.
