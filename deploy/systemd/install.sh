#!/usr/bin/env bash
# install.sh — substitute operator-specific values into the dev-loop
# systemd templates and install them to ~/.config/systemd/user/.
#
# Idempotent: re-running overwrites the installed units with the
# current template content + current substitutions. Does NOT enable
# or start the timer — operator does that explicitly.
#
# Usage:
#   ./install.sh                  # auto-detect everything, install only
#   ./install.sh --enable         # install AND enable+start the timer
#   ./install.sh --dry-run        # print what would be written, don't write
#   ./install.sh --uninstall      # remove the units
#
# Environment overrides:
#   REPO_ROOT       defaults to the dir containing this script's ../..
#   PATH_PREPEND    defaults to the mise node bin dir (auto-detected)
#   AGENT_COMMAND   defaults to 'codex exec --file {PROMPT_FILE}'

set -euo pipefail

# `pwd -P` resolves symlinks. Without it, a checkout reached via a
# symlink (e.g. ~/Desktop/codex-proxy → callosum) would produce a
# REPO_ROOT pointing at the symlink, and systemd-set WorkingDirectory
# would dereference inconsistently across boots. Always commit the
# physical path to the unit.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root_default="$(cd "${here}/../.." && pwd -P)"

REPO_ROOT="${REPO_ROOT:-${repo_root_default}}"
AGENT_COMMAND="${AGENT_COMMAND:-codex exec --file {PROMPT_FILE}}"

# Auto-detect the mise node bin dir if PATH_PREPEND wasn't given.
# This is the path that fixes the cron gotcha — the codex Node shim
# needs `node` on PATH, and mise's node lives outside the cron-default
# PATH. If mise isn't installed, the operator must pass PATH_PREPEND
# explicitly.
if [[ -z "${PATH_PREPEND:-}" ]]; then
    if command -v mise >/dev/null 2>&1; then
        node_path="$(mise which node 2>/dev/null || true)"
        if [[ -n "${node_path}" ]]; then
            PATH_PREPEND="$(dirname "${node_path}")"
        fi
    fi
    if [[ -z "${PATH_PREPEND:-}" ]]; then
        echo "error: PATH_PREPEND not set and mise node not found." >&2
        echo "       set PATH_PREPEND to the directory containing the" >&2
        echo "       node binary the codex shim should use." >&2
        exit 2
    fi
fi

dry_run=0
enable=0
uninstall=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) dry_run=1 ;;
        --enable) enable=1 ;;
        --uninstall) uninstall=1 ;;
        --help|-h)
            sed -n '1,/^set -e/p' "$0" | sed -n '/^# /p' | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
    shift
done

unit_dir="${HOME}/.config/systemd/user"
service_dst="${unit_dir}/callosum-dev-loop.service"
timer_dst="${unit_dir}/callosum-dev-loop.timer"

if [[ ${uninstall} -eq 1 ]]; then
    if [[ ${dry_run} -eq 1 ]]; then
        echo "(dry-run) would: systemctl --user disable --now callosum-dev-loop.timer"
        echo "(dry-run) would: rm ${service_dst} ${timer_dst}"
        echo "(dry-run) would: systemctl --user daemon-reload"
    else
        systemctl --user disable --now callosum-dev-loop.timer 2>/dev/null || true
        rm -f "${service_dst}" "${timer_dst}"
        systemctl --user daemon-reload
        echo "uninstalled."
    fi
    exit 0
fi

# Confirm the dispatcher binary actually exists. Better to fail
# loudly here than have systemd surface a confusing error later.
if ! (cd "${REPO_ROOT}" && uv run callosum-dev-loop --help >/dev/null 2>&1); then
    echo "error: 'uv run callosum-dev-loop' did not run successfully from" >&2
    echo "       ${REPO_ROOT}. Run 'uv sync' there first, or set REPO_ROOT" >&2
    echo "       to the directory containing your callosum checkout." >&2
    exit 2
fi

echo "installing callosum-dev-loop systemd units with:"
echo "  REPO_ROOT     = ${REPO_ROOT}"
echo "  PATH_PREPEND  = ${PATH_PREPEND}"
echo "  AGENT_COMMAND = ${AGENT_COMMAND}"
echo "  destination   = ${unit_dir}"

render() {
    local template="$1"
    sed \
        -e "s|@REPO_ROOT@|${REPO_ROOT}|g" \
        -e "s|@PATH_PREPEND@|${PATH_PREPEND}|g" \
        -e "s|@AGENT_COMMAND@|${AGENT_COMMAND}|g" \
        "${template}"
}

if [[ ${dry_run} -eq 1 ]]; then
    echo "--- callosum-dev-loop.service ---"
    render "${here}/callosum-dev-loop.service.template"
    echo "--- callosum-dev-loop.timer ---"
    render "${here}/callosum-dev-loop.timer.template"
    echo "(dry-run) no files written."
    exit 0
fi

mkdir -p "${unit_dir}"
render "${here}/callosum-dev-loop.service.template" >"${service_dst}"
render "${here}/callosum-dev-loop.timer.template" >"${timer_dst}"
systemctl --user daemon-reload
echo "installed:"
echo "  ${service_dst}"
echo "  ${timer_dst}"

if [[ ${enable} -eq 1 ]]; then
    systemctl --user enable --now callosum-dev-loop.timer
    echo
    echo "timer enabled. inspect with:"
    echo "  systemctl --user list-timers callosum-dev-loop.timer"
    echo "  systemctl --user status callosum-dev-loop.service"
    echo "  journalctl --user -u callosum-dev-loop.service -f"
else
    echo
    echo "timer NOT enabled. start it with:"
    echo "  systemctl --user enable --now callosum-dev-loop.timer"
    echo "or trigger a one-off iteration with:"
    echo "  systemctl --user start callosum-dev-loop.service"
fi
