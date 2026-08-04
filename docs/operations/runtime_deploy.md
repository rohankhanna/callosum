# Artifact-backed runtime deploy

Callosum's managed service should run from an installed artifact, not
directly from the working tree checkout.

## Build and install

Build a wheel, then install it into a dedicated runtime venv:

```bash
callosum build
```

The helper creates a single runtime venv under
`~/.local/share/callosum/runtime/venv` and installs the wheel there **with
the `[embeddings]` extra** (which pulls in `sentence-transformers`). The extra
is currently dormant dead weight — no live config selects an embedding
provider (the BGE prompt-embedding + KNN predictor subsystem was removed; the
quality predictor is now `cell_majority_prior`, which ignores prompt
embeddings) — but it is carried for parity with `uv`'s `default-groups`
(`["dev", "embeddings"]`) that source-mode serving (`uv run callosum serve`)
installs, so the runtime artifact and the source checkout carry the same
dependency surface. Removing the extra (and `sentence-transformers`) from
`pyproject.toml` is a deferred cleanup item that changes the dependency
surface and belongs in a maintenance window.

The canonical deployed CLI is:

```bash
~/.local/share/callosum/runtime/venv/bin/callosum
```

If the runtime venv is first on `PATH`, `which callosum` prints this
binary. Otherwise use the explicit path above. `callosum service status`
also shows the systemd unit command currently in use.

## Service command

The managed user service should execute:

```bash
~/.local/share/callosum/runtime/venv/bin/callosum serve
```

That keeps the deployed runtime stable across source edits in the repo
checkout.

## Operator commands

Use the same installed binary for service control:

```bash
~/.local/share/callosum/runtime/venv/bin/callosum service status
~/.local/share/callosum/runtime/venv/bin/callosum service start
~/.local/share/callosum/runtime/venv/bin/callosum service stop
~/.local/share/callosum/runtime/venv/bin/callosum service restart
```

To start or restart the managed service from the checkout instead of
the installed runtime, use:

```bash
~/.local/share/callosum/runtime/venv/bin/callosum service start --source
~/.local/share/callosum/runtime/venv/bin/callosum service restart --source
```

Those commands set `CALLOSUM_SERVICE_SOURCE=1` in the user systemd
manager before starting the unit. The unit must branch on that variable:
unset means run the installed runtime binary; set to `1` means run
`uv run callosum serve` from the checkout.

## Source checkout serving

To serve directly from the checkout for development, use:

```bash
callosum serve --source
```

## Host-side change

The systemd user unit lives in the dotfiles repo and must be updated
there before the live service switch. This repo only defines the build
and install contract.
