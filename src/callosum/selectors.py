"""Client-driven routing selectors.

Clients express routing intent per request via Callosum-owned `callosum:`
model ids instead of mutating the global operator routing mode. A selector is
parsed once on the dispatch hot path and turned into a `SelectorDecision`; the
existing per-request routing-mode machinery (see `app.py`, the canary override
hook) then applies the constraint for that request only. Nothing global mutates.

Grammar
-------
Strategy selectors (a *routing engine* that selects within a pool):
    callosum:auto          -> strategy="auto"
    callosum:local-only    -> strategy="local-only"
    callosum:remote-only   -> strategy="remote-only"

Concrete pins (name a specific model; drop the `-only` suffix):
    callosum:remote/<model>[:<effort>]  -> source="remote", pinned_model, pinned_effort?
    callosum:local/<model>[:<effort>]   -> source="local",  pinned_model, pinned_effort?

Notes
-----
- `callosum:offline` is explicitly rejected until no-network semantics can be
  enforced end to end (raises SelectorError -> caller returns 400).
- A local pin may carry `:<effort>` symmetric to a remote pin. The parser is
  deliberately ignorant of effort vocabulary because providers own that
  fast-moving metadata. It accepts any non-empty effort and the *model-specific*
  check ("does THIS model actually expose that level?") lives where the live
  `supported_reasoning_levels` are known: catalog discovery advertises the
  facts and dispatch surfaces a clean 503 when no live cell serves the pinned
  (model, effort) pair.
- Anything not starting with `callosum:` returns None -> legacy pass-through,
  behavior unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Prefix that marks a Callosum-owned selector id.
SELECTOR_PREFIX = "callosum:"

#: Bare strategy selectors (no model pin). `offline` is intentionally absent.
STRATEGY_SELECTORS: frozenset[str] = frozenset({"auto", "local-only", "remote-only"})

#: Concrete-pin source namespaces and the backend kind each constrains to.
#: "remote" excludes the local gateway; "local" keeps only it.
_PIN_SOURCES: frozenset[str] = frozenset({"remote", "local"})


class SelectorError(ValueError):
    """A `callosum:` id was supplied but is malformed or unsupported.

    Callers should surface this as an HTTP 400 — the client asked for something
    in the Callosum namespace that we can parse but cannot honor.
    """


@dataclass(frozen=True, slots=True)
class SelectorDecision:
    """Parsed routing intent from a `callosum:` model id.

    Exactly one of `strategy` or (`source` + `pinned_model`) is populated.
    """

    strategy: str | None = None
    source: str | None = None
    pinned_model: str | None = None
    pinned_effort: str | None = None


def is_selector(model: str | None) -> bool:
    """True if `model` is in the Callosum selector namespace."""
    return isinstance(model, str) and model.startswith(SELECTOR_PREFIX)


def parse_selector(model: str | None) -> SelectorDecision | None:
    """Parse a client-supplied model id into a `SelectorDecision`.

    Returns None for any id that is not in the `callosum:` namespace (legacy
    pass-through). Raises `SelectorError` for malformed/unsupported selectors.
    """
    if not is_selector(model):
        return None
    assert isinstance(model, str)  # narrowed by is_selector
    rest = model[len(SELECTOR_PREFIX) :].strip()
    if not rest:
        raise SelectorError("empty callosum selector")

    # Bare strategy selectors (no '/').
    if "/" not in rest:
        if rest in STRATEGY_SELECTORS:
            return SelectorDecision(strategy=rest)
        if rest == "offline":
            raise SelectorError(
                "callosum:offline is not supported until no-network semantics can be enforced end to end"
            )
        raise SelectorError(f"unknown callosum strategy selector {model!r}")

    # Concrete pin: callosum:<source>/<model>[:<effort>]
    source, _, tail = rest.partition("/")
    if source not in _PIN_SOURCES:
        raise SelectorError(
            f"unknown callosum pin source {source!r} in {model!r} (expected one of {sorted(_PIN_SOURCES)})"
        )
    if not tail:
        raise SelectorError(f"missing model in callosum pin {model!r}")

    pinned_model, sep, effort = tail.partition(":")
    if not pinned_model:
        raise SelectorError(f"missing model in callosum pin {model!r}")

    # Effort is optional. Its vocabulary is provider-owned metadata, not parser
    # policy: accept every non-empty value here and enforce exact per-model
    # support downstream against the live cell grid. This prevents a stale
    # client-side tuple from hiding newly released provider capabilities.
    pinned_effort: str | None = None
    if sep:
        if not effort:
            raise SelectorError(f"missing reasoning effort in {model!r}")
        pinned_effort = effort
    return SelectorDecision(source=source, pinned_model=pinned_model, pinned_effort=pinned_effort)
