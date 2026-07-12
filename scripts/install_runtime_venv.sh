#!/usr/bin/env bash
# Build and install Callosum into a dedicated runtime venv.
#
# The canonical deployed CLI binary is the installed `callosum` under:
#   ~/.local/share/callosum/runtime/venv/bin/callosum
#
# This helper does not touch systemd. It only prepares the runtime
# artifact and prints the exact binary path that the managed service
# should execute.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd "${here}/.." && pwd -P)"
runtime_root="${CALLOSUM_RUNTIME_ROOT:-${XDG_DATA_HOME:-$HOME/.local/share}/callosum/runtime}"
venv_dir="${runtime_root}/venv"

cd "${repo_root}"
uv build

wheel_path="$(ls -1t dist/*.whl | head -n 1)"

# Recreate the venv from scratch. rm -rf first is mandatory: `python3 -m venv`
# on an existing dir without --clear does NOT remove prior site-packages, so a
# build run under a different interpreter (e.g. 3.11 today, 3.12 tomorrow) would
# leave an orphan lib/pythonX.Y/site-packages alongside the new one, and the
# daemon (which imports whatever bin/python3 points at) could keep running the
# old code while pip installs into the new one — a silent split-brain venv.
rm -rf "${venv_dir}"
python3 -m venv "${venv_dir}"
# Install the wheel WITH the [embeddings] extra so the runtime artifact is
# self-sufficient: a host whose config selects the bge embedding provider can
# start the installed binary without a follow-up pip install. This mirrors
# uv's default-groups (["dev", "embeddings"]) used by `uv run` source serving.
# Install via the venv's OWN interpreter (`bin/python3 -m pip`), not `bin/pip`:
# bin/pip's shebang can point at a different interpreter than bin/python3 after
# cross-version rebuilds, which would install into the wrong site-packages.
"${venv_dir}/bin/python3" -m pip install --force-reinstall "${wheel_path}[embeddings]" >/dev/null

cat <<EOF
runtime_venv=${venv_dir}
callosum_binary=${venv_dir}/bin/callosum
service_execstart=${venv_dir}/bin/callosum serve
operator_commands=${venv_dir}/bin/callosum service start|stop|restart|status
EOF
