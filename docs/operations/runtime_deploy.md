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

## Durable environment variables (systemd drop-ins)

Some runtime behavior is toggled by environment variables read once at
process start. `systemctl --user set-environment` sets them transiently
(scoped to the user manager; lost on `daemon-reload`/reboot), which is
fine for a bounded capture window (see `dispatch.md`). For a flag that
must survive restarts and reboots, use a durable systemd drop-in under
the unit's `service.d/` directory instead:

```
~/.config/systemd/user/system-dependency-callosum.service.d/<name>.conf
```

```ini
[Service]
Environment=CALLOSUM_SOME_FLAG=1
```

Then `systemctl --user daemon-reload` and
`systemctl --user restart system-dependency-callosum.service`. Verify
with `systemctl --user show system-dependency-callosum.service -p
Environment`. To disable: remove the file (or set the value to `0`),
`daemon-reload`, and restart.

### Ollama Cloud enable (live example)

`CALLOSUM_OLLAMA_CLOUD_ENABLED=1` registers the `OllamaCloudBackend`
(`BackendKind="ollama_cloud"`) — cloud models served by the local
ollama daemon under `ollama signin`. Callosum holds NO credential; the
daemon does. The durable drop-in is
`~/.config/systemd/user/system-dependency-callosum.service.d/ollama-cloud.conf`:

```ini
[Service]
Environment=CALLOSUM_OLLAMA_CLOUD_ENABLED=1
```

After `daemon-reload` + restart, smoke-test that the backend is live:

```bash
curl -s http://127.0.0.1:8765/status | jq '.backends[] | select(.id=="ollama-cloud")'
curl -s http://127.0.0.1:8765/models | jq '.data[] | select(.id|test(":cloud")) | .id'
```

`:cloud`-suffixed models catalog from the daemon's `/api/tags` and are
classified `callosum:remote/...` (e.g.
`callosum:remote/glm-5.1:cloud:default`). The local lane catalogs from
local LLM gateway's `/v1/models` serving endpoint, which excludes unmanaged
`:cloud` models, so there is no double-list. Cloud cells enter the
per-cell minimum-coverage grid, so the quota may route some real
traffic to `glm-5.X:cloud` to satisfy the per-cell floor.

## Host-side change

The systemd user unit lives in the dotfiles repo and must be updated
there before the live service switch. This repo only defines the build
and install contract.
