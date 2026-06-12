# callosum-dev-loop systemd scheduling

This directory contains systemd `--user` unit templates that run the
callosum dev loop on a schedule. The templates are committed; the
*installed* units (with operator-specific paths) live in
`~/.config/systemd/user/` and are not version-controlled.

## What this does

The dev loop is the agentic mechanism that reads callosum's capability
harness output, sees which local models have failing findings, and
proposes (or writes) per-model adapters under `src/callosum/transforms/`.
See `src/callosum/dev_loop/` for the runtime details.

Scheduling: by default, the dev loop fires every 6h, matching the
capability harness's cadence so each iteration has fresh data.
Scheduled invocations run `callosum-dev-loop run --cron`, which means
they consult the autonomy level and record skip/failure/success outcomes
in the autonomy audit log. Manual `run` invocations remain operator-driven
unless you pass `--cron` explicitly.

## Activation

```bash
# Install only (does NOT enable):
./install.sh

# Install AND enable the timer:
./install.sh --enable

# Preview substitutions without writing anything:
./install.sh --dry-run

# Remove everything:
./install.sh --uninstall
```

The install script auto-detects:
- the callosum repo root (two dirs up from itself);
- the mise node bin dir (the cron PATH gotcha fix — see below);
- a default codex agent invocation.

Override any of those by setting `REPO_ROOT`, `PATH_PREPEND`, or
`AGENT_COMMAND` in the environment before running.

After install, manage the timer the normal systemd way:

```bash
systemctl --user list-timers callosum-dev-loop.timer
systemctl --user start callosum-dev-loop.service          # one-off
systemctl --user enable --now callosum-dev-loop.timer     # enable schedule
systemctl --user disable --now callosum-dev-loop.timer    # stop schedule
journalctl --user -u callosum-dev-loop.service -f         # follow logs
```

## The cron / systemd PATH gotcha

systemd `--user` services inherit a minimal PATH that does not include
developer tooling directories (mise, nvm, asdf, etc.). The Codex CLI
is a Node shim with `#!/usr/bin/env node` at the top; even if the
service points at codex by absolute path, the shim STILL needs `node`
discoverable on PATH at exec time. Without that, the service fails
with `env: 'node': No such file or directory`.

The template fixes this two ways, both controlled by the install
script's `PATH_PREPEND` substitution:

1. `Environment="CALLOSUM_DEV_LOOP_PATH_PREPEND=..."` — the dispatcher
   reads this and prepends it to the subprocess PATH when spawning the
   agent. The codex shim then finds node.
2. `Environment="PATH=..."` — sets the service's own PATH so the
   dispatcher's `uv`, `git`, and `pytest` calls also work.

Both point at the same directory: whatever contains your mise-managed
`node` binary. The install script auto-detects this via `mise which node`.

If you're not on mise, set `PATH_PREPEND` explicitly:

```bash
PATH_PREPEND="$HOME/.nvm/versions/node/v22.0.0/bin" ./install.sh
```

## Safety contract

The unit template is intentionally narrow:

- `Type=oneshot` — fires, runs once, exits. Does not retry on failure.
- No `Restart=` — a single bad iteration won't loop and burn quota.
- `TimeoutStartSec=3600` — bounds a stuck iteration at one hour.
- `WorkingDirectory=` is pinned to your callosum checkout so git
  operations and `uv run` see the right repo.

The dispatcher itself never pushes to a remote and never merges to
main. Iterations land on `auto/dev-loop-<timestamp>` branches for
your manual review. See `src/callosum/dev_loop/cli.py` for the full
safety constraints.

## Verifying after first run

```bash
# Check the timer is registered:
systemctl --user list-timers callosum-dev-loop.timer

# Trigger a one-off iteration:
systemctl --user start callosum-dev-loop.service

# Watch what happened:
journalctl --user -u callosum-dev-loop.service --since '5 minutes ago'

# See the auto-generated branches (if any):
git branch | grep '^  auto/dev-loop-'
```

A successful "do nothing" iteration is the most common outcome when
there are no failing harness findings — the dispatcher exits 0 and
no branch is created. That is correct behavior. Scheduled skips should
still appear in the autonomy audit log because the systemd unit runs with
`--cron`. The system only generates code when there is observed work to do.
