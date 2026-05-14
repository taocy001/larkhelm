#!/usr/bin/env bash
# deploy.sh — reinstall source into pipx venv and restart larkhelm via systemd.
#
# Replaces the legacy nohup+pgrep workflow. systemd is the source of truth for
# process state; this script does not background processes itself.
#
# Prerequisites:
#   - larkhelm.service installed at /etc/systemd/system/larkhelm.service
#     (system-level unit; verify with `systemctl status larkhelm`)
#   - sudo access to restart the unit (passwordless not required; will prompt)
#
# Editable installs (`pipx install -e`) pick up source changes automatically on
# next process start, but pyproject.toml dependency changes still need a real
# install. We run `pip install --no-deps` to refresh entry points and metadata
# without re-resolving the entire dep tree; pass `--with-deps` if you bumped
# any pyproject.toml dependencies.

set -euo pipefail

UNIT="larkhelm.service"
VENV_PYTHON="$HOME/.local/pipx/venvs/larkhelm/bin/python"
SRC="$(cd "$(dirname "$0")" && pwd)"
INSTALL_ARGS=("--no-deps")

# Flags
for arg in "$@"; do
    case "$arg" in
        --with-deps)    INSTALL_ARGS=() ;;        # full resolve, picks up new deps
        --skip-install) INSTALL_ARGS=("--skip") ;;  # restart only
        -h|--help)
            cat <<'EOF'
Usage: ./deploy.sh [--with-deps|--skip-install]
  (default)        Reinstall source into pipx venv with --no-deps, then restart.
  --with-deps      Reinstall and re-resolve dependencies (slower; use when
                   pyproject.toml deps change).
  --skip-install   Skip pip install, only restart the service (useful when you
                   just edited source under an editable install).
EOF
            exit 0
            ;;
    esac
done

# 1) Sanity: systemd unit exists
if ! systemctl cat "$UNIT" >/dev/null 2>&1; then
    echo "[deploy] ERROR: systemd unit '$UNIT' not found." >&2
    echo "[deploy] Install it at /etc/systemd/system/$UNIT first." >&2
    exit 1
fi

# 2) Install (unless --skip-install)
if [ "${INSTALL_ARGS[0]:-}" != "--skip" ]; then
    if [ ! -x "$VENV_PYTHON" ]; then
        echo "[deploy] ERROR: pipx venv python not found at $VENV_PYTHON" >&2
        echo "[deploy] Run 'pipx install -e $SRC' first." >&2
        exit 1
    fi
    echo "[deploy] Installing from $SRC (${INSTALL_ARGS[*]:-full deps})..."
    "$VENV_PYTHON" -m pip install -q "${INSTALL_ARGS[@]}" "$SRC"
fi

# 3) Restart via systemctl. sudo will prompt if no cached creds; that's expected.
echo "[deploy] Restarting $UNIT ..."
sudo systemctl restart "$UNIT"

# 4) Confirm active state
sleep 1
STATE=$(systemctl is-active "$UNIT" 2>/dev/null || true)
if [ "$STATE" != "active" ]; then
    echo "[deploy] ERROR: service is '$STATE' after restart." >&2
    echo "[deploy] Recent journal entries:" >&2
    journalctl -u "$UNIT" --no-pager -n 20 >&2 || true
    exit 1
fi

MAIN_PID=$(systemctl show -p MainPID --value "$UNIT")
echo "[deploy] Service active, MainPID=$MAIN_PID"
