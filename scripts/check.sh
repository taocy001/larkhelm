#!/usr/bin/env bash
# larkhelm · unified developer check entry point (P2 REQ-11 / AC-07).
#
# Shared between local dev (``make test`` / ``make lint`` / ``make type``)
# and CI (``make ci`` in ``.github/workflows/ci.yml``). Putting the actual
# commands here means a green ``make all`` locally implies a green CI run.
#
# Usage:
#   ./scripts/check.sh test    # pytest (full suite)
#   ./scripts/check.sh lint    # ruff bug-detector subset
#   ./scripts/check.sh type    # mypy strict subset
#
# Extra args after the subcommand are forwarded verbatim:
#   ./scripts/check.sh test tests/test_metrics.py -k prometheus
set -euo pipefail

# Always run from repo root so relative paths (``larkhelm/`` / ``tests/``)
# resolve correctly when invoked from a sub-directory.
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "${SCRIPT_DIR}/.."

if [[ "$#" -lt 1 ]]; then
    echo "usage: $0 {test|lint|type} [extra args]" >&2
    exit 2
fi

cmd="$1"
shift

# Python interpreter: prefer the project venv if active, else fall back to
# ``python3``. CI sets up ``python -m pip install -e .[dev]`` against the
# matrix interpreter so ``python3`` resolves to the matrix Python.
PY="${PYTHON:-python3}"

case "$cmd" in
    test)
        exec "$PY" -m pytest \
            --ignore=tests/test_idle_timeout.py \
            "$@"
        ;;
    lint)
        exec "$PY" -m ruff check larkhelm/ tests/ "$@"
        ;;
    type)
        # ``--follow-imports=silent`` scopes the strict check to the six
        # modules pinned by ``[[tool.mypy.overrides]]``; otherwise mypy
        # transitively checks every import target with the same strictness
        # and explodes to 200+ errors.
        exec "$PY" -m mypy --follow-imports=silent \
            larkhelm/crew_types.py \
            larkhelm/dedup.py \
            larkhelm/concurrency.py \
            larkhelm/log.py \
            larkhelm/token_stats.py \
            larkhelm/chat_state.py \
            "$@"
        ;;
    *)
        echo "unknown subcommand: $cmd (expected test|lint|type)" >&2
        exit 2
        ;;
esac
