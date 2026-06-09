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

### `callosum.backends.litellm_gateway.LiteLLMGatewayBackend`

  * **Active alternative**: `callosum.backends.local_direct.LocalModelRegistryBackend`,
    which discovers local models via the `local-llm` CLI and
    dispatches directly to per-model endpoints (responses-proxy
    lanes on dedicated ports, ollama, vllm).
  * **What this fallback provides**: routing local-cell traffic
    through a LiteLLM gateway (a single OpenAI-compatible endpoint
    that brokers across runtimes via its own config). The catalog
    comes from the gateway's `/v1/models` rather than from the
    `local-llm` registry, so this backend works without the
    `local-llm` CLI on the operator's PATH.
  * **When you'd activate it**: the `local-llm` CLI isn't
    installed, or `LocalModelRegistrySource.is_available()` returns False
    for any other reason (e.g., the CLI is broken, the registry is
    corrupted). Activation is automatic — `__main__.py` registers
    this backend whenever `LocalModelRegistryBackend` couldn't be
    registered AND `CALLOSUM_LITELLM_GATEWAY_ENABLED=1` is set in
    env. A startup `logger.warning` line surfaces the activation
    so the operator knows the fallback is live.
  * **Why it stays redundant rather than active**: the gateway has
    a known limitation around chat-completions for upstream
    runtimes that only natively serve `/v1/responses`
    (`*-responses-proxy` model rows) — its chat-completions handler
    hangs at the client timeout for those. `LocalModelRegistryBackend`
    routes around the gateway entirely via per-model endpoints, so
    it doesn't inherit the limitation. See
    `docs/decisions/...-keep-litellmgatewaybackend-as-fallback...`
    for the historical decision to keep the fallback rather than
    retire it.
  * **Maintenance contract**: every test in
    `tests/unit/test_litellm_gateway.py` continues to run in CI.
    The inline docstring in `LiteLLMGatewayBackend.chat_completions`
    documents the known chat-completions hang for responses-only
    runtimes; future operators who re-activate this backend will see
    both the docstring and the startup warning.
