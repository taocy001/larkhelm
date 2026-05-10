#!/usr/bin/env bash
# deploy.sh — install updated source into the pipx venv and gracefully restart the service
set -euo pipefail

VENV_PYTHON="$HOME/.local/pipx/venvs/larkhelm/bin/python"
START_CMD="$HOME/.local/bin/larkhelm"
CONFIG="--config $HOME/.config/larkhelm/config.json --data-dir $HOME/.local/share/larkhelm"
LOG="$HOME/.local/share/larkhelm/logs/larkhelm.log"
SRC="$(cd "$(dirname "$0")" && pwd)"

echo "[deploy] Installing from $SRC ..."
"$VENV_PYTHON" -m pip install --no-deps -q "$SRC"
echo "[deploy] Install done."

# Send SIGTERM to the running service and wait for graceful shutdown
PID=$(pgrep -f "larkhelm start" | grep -v "$$" | head -1 || true)
if [ -n "$PID" ]; then
    echo "[deploy] Sending SIGTERM to pid $PID, waiting for graceful exit..."
    kill "$PID"
    for i in $(seq 1 90); do
        sleep 1
        kill -0 "$PID" 2>/dev/null || { echo "[deploy] Process exited after ${i}s."; break; }
        if [ "$i" -eq 90 ]; then
            echo "[deploy] Graceful shutdown timed out, force-killing."
            kill -9 "$PID" 2>/dev/null || true
        fi
    done
else
    echo "[deploy] No running service found."
fi

# Start new process
echo "[deploy] Starting service..."
# shellcheck disable=SC2086
nohup "$START_CMD" start $CONFIG >> "$LOG" 2>&1 &
sleep 3
NEW_PID=$(pgrep -f "larkhelm start" | grep -v "$$" | head -1 || true)
if [ -n "$NEW_PID" ]; then
    echo "[deploy] Service started, pid=$NEW_PID"
else
    echo "[deploy] ERROR: service did not start. Check $LOG"
    exit 1
fi
