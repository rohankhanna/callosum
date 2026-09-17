# Artifact-backed runtime deploy

Callosum's managed service should run from an installed artifact, not
directly from the working tree checkout.

## Build and install

Build a wheel, then install it into a dedicated runtime venv:

```bash
callosum build
```

The helper creates a single runtime venv under
`~/.local/share/callosum/runtime/venv` and installs the bare wheel there. The
former `[embeddings]` extra (which pulled in `sentence-transformers` + torch)
was removed when the BGE prompt-embedding + KNN predictor subsystem was ripped
out — embeddings have no runtime role (the quality predictor is
`cell_majority_prior`, which ignores prompt embeddings), so the runtime
artifact no longer drags in the multi-GB torch/huggingface stack. This also
shrinks the build's network-dependent download surface: torch was the
failure-prone step that timed out builds when pip could not reach its cache.
Source-mode serving (`uv run callosum serve`) uses the same dependency surface
(`uv`'s `default-groups` is now just `["dev"]`).

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
(`BackendKind="ollama_cloud"`), a remote backend that calls an
Ollama-compatible endpoint and discovers its catalog through
`/v1/models`, `/api/tags`, and `/api/show`. The endpoint is
operator-supplied and defaults to `https://ollama.com`. The durable drop-in is
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

Use a selector such as
`callosum:remote/glm-5.1:cloud::default` to pin a specific cloud model
and reasoning effort. The local lane catalogs from its own
`/v1/models` endpoint, so cloud and local cells stay separated.


By default `OllamaCloudBackend.usage_snapshot()` returns an
**honest-advisory** snapshot (`remaining_fraction=1.0`,
`weekly_exhausted=false`) — it does NOT cool down on the cloud account
because callosum has no real usage meter. The advisory is honest about
the gap: it simply never claims exhaustion.

Callosum can optionally read real usage data from an operator-supplied
read-only usage endpoint. This path is disabled by default.

Two env flags gate the path. **Both default OFF — there is no
live-routing change at merge.**

```ini
[Service]
# Construct the read-only OllamaCloudUsageSource (shadow-available;
# does NOT change routing by itself).
Environment=CALLOSUM_OLLAMA_CLOUD_USAGE_SOURCE_ENABLED=1
# Project the real session/weekly meters into usage_snapshot() so the
# quota/cool-down path reacts to them. Explicitly enabled by configuration.
# Requires the source flag ON too. Default OFF → honest-advisory unchanged.
Environment=CALLOSUM_OLLAMA_CLOUD_USAGE_LIVE=1
```

Auxiliary env vars (defaults shown):

```ini
Environment=CALLOSUM_OLLAMA_CLOUD_USAGE_URL=http://127.0.0.1:7342
Environment=CALLOSUM_OLLAMA_CLOUD_USAGE_ACCOUNT=primary
Environment=CALLOSUM_OLLAMA_CLOUD_STANDIN_TTL=1800
```

When `CALLOSUM_OLLAMA_CLOUD_USAGE_LIVE` is off, or the usage endpoint is
unavailable, `usage_snapshot()` falls back to the existing
honest-advisory behavior. Routing never errors and never cools down the
cloud lane just because a meter is unavailable.

**Observability:** `callosum usage ollama-cloud` is a read-only
diagnostic that constructs an ephemeral source from the same env vars
and prints the live session/weekly meters without touching routing:

```bash
callosum usage ollama-cloud
```

**Current limitation:** Callosum does not provide a public Ollama usage
contract. The optional usage endpoint is operator-specific and may be
removed if Ollama later ships an official usage API.

## Host-side change

Update the host's systemd user unit before changing the live service.
This repository only defines the build and install contract.
