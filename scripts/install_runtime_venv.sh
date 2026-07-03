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

python3 -m venv "${venv_dir}"
# Install the wheel WITH the [embeddings] extra so the runtime artifact is
# self-sufficient: a host whose config selects the bge embedding provider can
# start the installed binary without a follow-up pip install. This mirrors
# uv's default-groups (["dev", "embeddings"]) used by `uv run` source serving.
"${venv_dir}/bin/pip" install --force-reinstall "${wheel_path}[embeddings]" >/dev/null

cat <<EOF
runtime_venv=${venv_dir}
callosum_binary=${venv_dir}/bin/callosum
service_execstart=${venv_dir}/bin/callosum serve
operator_commands=${venv_dir}/bin/callosum service start|stop|restart|status
EOF
