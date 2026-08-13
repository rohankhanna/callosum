# Contributing to Callosum

Callosum is a local, adaptive HTTP routing layer that sits in front of
several LLM backends and decides, per request, which underlying model
should handle it. It exposes an OpenAI-compatible endpoint on
`127.0.0.1`. See `README.md` for the full product framing and quick start.

This repository currently has a single maintainer. The guidance below
describes the workflow that keeps the tree green and reviewable; it
applies equally to the maintainer and to any future contributor.

## Prerequisites

- Python **3.11** or later (`requires-python = ">=3.11"`).
- [`uv`](https://docs.astral.sh/uv/) for environment and dependency
  management. All commands below assume `uv` is on `PATH`.

## Getting a working checkout

```bash
git clone <repo-url> callosum
cd callosum
uv sync          # create / refresh the .venv with dev dependencies
```

The dev dependency group (which includes `pytest`, `ruff`, and `mypy`) is
installed by default via `uv sync`.

## Making changes

Work on a feature branch off `main`:

```bash
git checkout main
git checkout -b feat/<short-description>
```

Keep changes small and focused. Prefer one logical change per branch so
review (and history) stays legible.

### Before you commit

Run the full local verification flow — this is the same set of checks
the merge gate runs:

```bash
make verify
```

`make verify` is shorthand for:

```bash
make lint          # ruff check .
make format-check  # ruff format --check .
make type           # python -m mypy src/callosum
make test           # python -m pytest
```

If `format-check` fails, run `make format` (`ruff format .`) to apply
the formatter, then re-run `make verify`.

Individual checks (useful while iterating):

```bash
make lint     # ruff check .
make type     # mypy src/callosum
make test     # pytest
```

> **Note on `make test`:** the target sets `PYTHONPATH=src` and runs
> `uv run python -m pytest`. The default pytest configuration
> **deselects live end-to-end tests** (they require a running local lane
> and real credentials and are opt-in). If you specifically need them,
> run them explicitly per the instructions in `tests/live_e2e/`; they are
> not part of `make verify` or the merge gate.

All four `make verify` steps must pass before a change is merged. A
merge that skips any of them is not green.

## Tests

Callosum has three layers of tests:

- **Unit tests** (`tests/unit/`) — fast, hermetic, no network. The bulk
  of coverage lives here.
- **Contract tests** (`tests/contract/`) — replay-based tests anchored in
  committed upstream captures, plus `ContractFakeBackend`-based synthetic
  behavior tests. These run in the merge gate and are hermetic (no live
  upstream, no tokens).
- **Live end-to-end tests** (`tests/live_e2e/`) — explicitly invoked
  only; deselected by default. They consume real tokens and require a
  running local lane, so they are never part of the automated gate.

When you fix a representable bug, add a hermetic regression (unit or
contract) that turns red if the fix is reverted. A live E2E test may
*supplement* the hermetic case but must not *substitute* for it, because
live E2E is manual, non-blocking, and consumes real tokens.

## Architecture decisions

Significant architectural or design decisions are recorded as ADRs
(Architecture Decision Records) under `docs/adr/`. Each ADR follows the
`Title / Date / Status / Context / Decision / Consequences` structure.
When you make a material design decision, add an ADR rather than leaving
the rationale implicit in a commit message.

Diagrams live under `docs/diagrams/` (and referenced from
`docs/architecture/`). If you change the architecture, update the
diagrams and any prose that describes it so the docs never drift from
the code.

## Commit and branch model

- Branch from `main`; keep `main` always green.
- Write commit messages in the conventional `type(scope): subject` style
  (e.g. `fix(backends): wrap local stream in stall guard`,
  `test(contract): hermetic regression for expires_at`).
- Squash or rebase freely on your own feature branch; do not rewrite
  history that has been merged to `main`.
- Merge to `main` only after `make verify` is green. Use a
  `--no-ff` merge so the feature-branch context is preserved in history.

## Style and conventions

- Format with `ruff format`; lint with `ruff check`. The configuration
  lives in `pyproject.toml` under `[tool.ruff]`.
- Type-check with `mypy` (`--strict` semantics are enforced for
  `src/callosum`). The configuration lives under `[tool.mypy]`.
- Match the surrounding code's style, naming, and idiom. Prefer
  reading the nearest existing module before introducing a new pattern.

## Reporting issues / security

This project does not yet publish a separate security policy. If you
believe you have found a security-sensitive issue, do not open a public
issue — contact the maintainer directly through a private channel.