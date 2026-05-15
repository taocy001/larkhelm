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
SRC="$(cd "$(dirname "$0")" && pwd)"

# Locate the pipx venv python. pipx layout changed over versions:
#   * pipx >= 1.4 (current): $HOME/.local/share/pipx/venvs/larkhelm/    (XDG)
#   * pipx <  1.4 (legacy):  $HOME/.local/pipx/venvs/larkhelm/
# Prefer `pipx environment` if available (works across versions); fall
# back to probing the two known paths.
_resolve_venv_python() {
    if command -v pipx >/dev/null 2>&1; then
        local venvs_dir
        venvs_dir=$(pipx environment --value PIPX_LOCAL_VENVS 2>/dev/null || true)
        if [ -n "$venvs_dir" ] && [ -x "$venvs_dir/larkhelm/bin/python" ]; then
            echo "$venvs_dir/larkhelm/bin/python"; return 0
        fi
    fi
    for candidate in \
        "$HOME/.local/share/pipx/venvs/larkhelm/bin/python" \
        "$HOME/.local/pipx/venvs/larkhelm/bin/python"; do
        if [ -x "$candidate" ]; then
            echo "$candidate"; return 0
        fi
    done
    return 1
}
VENV_PYTHON=$(_resolve_venv_python || true)

# Mode is exclusive: default | with-deps | skip-install. Track the selection
# explicitly so we can reject conflicting flags instead of silently last-wins.
MODE="default"

usage() {
    cat <<'EOF'
Usage: ./deploy.sh [--with-deps|--skip-install] [-h|--help]
  (default)        Reinstall source into pipx venv with --no-deps, then restart.
  --with-deps      Reinstall and re-resolve dependencies (slower; use when
                   pyproject.toml deps change).
  --skip-install   Skip pip install, only restart the service (useful when you
                   just edited source under an editable install).
The two mode flags are mutually exclusive.
EOF
}

for arg in "$@"; do
    case "$arg" in
        --with-deps)
            if [ "$MODE" != "default" ] && [ "$MODE" != "with-deps" ]; then
                echo "[deploy] ERROR: --with-deps conflicts with --$MODE." >&2; usage >&2; exit 2
            fi
            MODE="with-deps"
            ;;
        --skip-install)
            if [ "$MODE" != "default" ] && [ "$MODE" != "skip-install" ]; then
                echo "[deploy] ERROR: --skip-install conflicts with --$MODE." >&2; usage >&2; exit 2
            fi
            MODE="skip-install"
            ;;
        -h|--help)
            usage; exit 0 ;;
        --)
            ;;
        *)
            echo "[deploy] ERROR: unknown argument '$arg'." >&2
            usage >&2
            exit 2
            ;;
    esac
done

# 1) Sanity: systemd unit exists
if ! systemctl cat "$UNIT" >/dev/null 2>&1; then
    echo "[deploy] ERROR: systemd unit '$UNIT' not found." >&2
    echo "[deploy] Install it at /etc/systemd/system/$UNIT first." >&2
    exit 1
fi

# 2) Install (skipped when --skip-install was passed)
if [ "$MODE" != "skip-install" ]; then
    if [ -z "$VENV_PYTHON" ] || [ ! -x "$VENV_PYTHON" ]; then
        echo "[deploy] ERROR: pipx venv for 'larkhelm' not found." >&2
        echo "[deploy] Searched:" >&2
        echo "[deploy]   * \$(pipx environment --value PIPX_LOCAL_VENVS)/larkhelm/bin/python" >&2
        echo "[deploy]   * \$HOME/.local/share/pipx/venvs/larkhelm/bin/python" >&2
        echo "[deploy]   * \$HOME/.local/pipx/venvs/larkhelm/bin/python" >&2
        echo "[deploy] Run 'pipx install -e $SRC' first, or pass --skip-install if" >&2
        echo "[deploy] you've already installed and just want to restart the service." >&2
        exit 1
    fi
    if [ "$MODE" = "with-deps" ]; then
        echo "[deploy] Installing from $SRC into $VENV_PYTHON (re-resolving deps)..."
        "$VENV_PYTHON" -m pip install -q "$SRC"
    else
        echo "[deploy] Installing from $SRC into $VENV_PYTHON (--no-deps; pass --with-deps to re-resolve)..."
        "$VENV_PYTHON" -m pip install --no-deps -q "$SRC"
    fi
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
