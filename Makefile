# larkhelm · developer make entry points (P2 REQ-11 / AC-07).
#
# Local: `make test` / `make lint` / `make type` / `make all`.
# CI invokes `make ci` (alias of `make all`) so the same commands run
# in both environments — a green local `make all` implies a green CI.
#
# Everything is delegated to scripts/check.sh; the Makefile only wires
# names. Extra args can be passed via ARGS:
#   make test ARGS="tests/test_metrics.py -k prometheus"

.PHONY: test lint type all ci help config-sync

help:
	@echo "larkhelm Make targets:"
	@echo "  make test          — pytest full suite (forwards \$$ARGS)"
	@echo "  make lint          — ruff bug-detector subset"
	@echo "  make type          — mypy strict subset"
	@echo "  make all           — test + lint + type + config-sync"
	@echo "  make ci            — alias of make all (used by .github/workflows)"
	@echo "  make config-sync   — verify config.py setdefault keys match example.json"

test:
	@./scripts/check.sh test $(ARGS)

lint:
	@./scripts/check.sh lint $(ARGS)

type:
	@./scripts/check.sh type $(ARGS)

all: test lint type config-sync

ci: all

config-sync:
	@python3 scripts/check_config_sync.py
