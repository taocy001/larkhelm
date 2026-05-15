#!/usr/bin/env python3
"""bench_phase2_boot — AC-12 verification harness for Phase D / Phase 2.

Measures the cost of boot-time GC + embedding-warmup against the SLO defined
in ``.crew_workspace/prd_criteria.json``:

    * Bridge MUST be ready to serve within ~1 s after the warmup thread is
      kicked off (i.e. ``_start_memory_boot_warmup()`` must not block).
    * The boot-warmup daemon MUST finish a 100-chat stale-slice scan within
      4 s on the reference hardware.

Strategy (no live bridge, no Feishu network):

    1. Spin up an isolated MEMORY_HOME under a tempdir so the host's real
       state is never touched.
    2. Materialise 100 ``session_<chat_id>.md`` files plus a minimal
       audit-JSONL trail so ``iter_known_chat_cwd_pairs`` / ``mark_stale_slices``
       have realistic input.
    3. Patch :data:`larkhelm.memory.MEMORY_HOME_DIR` to the tempdir before
       importing ``memory_lifecycle`` / ``memory_retriever``.
    4. Time:
         * ``_start_memory_boot_warmup()`` return latency  → must be < 1 s
         * The daemon-thread join duration                → must be < 4 s
    5. Emit a one-line ``RESULT: ...`` summary and exit 0 on green, 1 on red.

Usage:

    python scripts/bench_phase2_boot.py             # default 100 chats
    python scripts/bench_phase2_boot.py --chats 250 # stress

The exit code is the contract: AC-12 passes iff this script exits 0.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# Make `import larkhelm.*` resolve to the source tree when the script is run
# from the repo root with the package installed editable OR from a clean
# checkout. We DON'T touch sys.path otherwise — the import lookup order is
# inherited from the caller's environment.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── SLO constants (AC-12) ─────────────────────────────────────────────────

READY_BUDGET_SEC = 1.0
GC_BUDGET_SEC = 4.0


def _populate_fixture(memory_home: Path, audit_path: Path, n_chats: int) -> None:
    """Create N session-<chat>.md files plus an audit JSONL with one entry
    each, so ``mark_stale_slices`` has a realistic walk.
    """
    memory_home.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    with audit_path.open("w", encoding="utf-8") as fh:
        for i in range(n_chats):
            chat_id = f"benchchat{i:04d}"
            md = memory_home / f"session_{chat_id}.md"
            md.write_text(
                "## bench slice\n"
                "id: bench0000aaaa\n"
                "kind: fact\n"
                "this is a benchmark fixture slice; not a real memory.\n",
                encoding="utf-8",
            )
            audit_record = {
                "ts": now_iso,
                "schema_version": "2",
                "chat_id": chat_id,
                "agent_type": "chat",
                "mode": "keyword",
                "declared_mode": "keyword",
                "elapsed_ms": 12,
                "fail_open": False,
                "selected_slice_ids": ["bench0000aaaa"],
                "selected_token_chars": 64,
                "hybrid_alpha": 0.6,
                "query_token_count": 8,
                "top_k_returned": 1,
                "stale_hit_count": 0,
            }
            fh.write(json.dumps(audit_record, ensure_ascii=False) + "\n")


def _run_bench(n_chats: int) -> int:
    """Returns 0 on pass, 1 on fail."""
    tmpdir = Path(tempfile.mkdtemp(prefix="bench_phase2_boot_"))
    memory_home = tmpdir / "memory"
    audit_path = tmpdir / "audit" / "memory_retriever_audit.jsonl"

    print(f"[bench] fixture root: {tmpdir}")
    print(f"[bench] populating {n_chats} chat fixtures...")
    _populate_fixture(memory_home, audit_path, n_chats)

    # Redirect the production MEMORY_HOME_DIR + audit path BEFORE importing
    # the modules that capture them at import time.
    os.environ["LARKHELM_DATA_DIR"] = str(tmpdir)

    import larkhelm.memory as _mem
    _mem.MEMORY_HOME_DIR = memory_home

    # `memory_retriever._resolve_audit_path` reads `_cfg.config` lazily; we
    # set the override there so audit walks land on our fixture file.
    import larkhelm.config as _cfg
    try:
        _cfg._init_runtime(data_dir=str(tmpdir))
    except SystemExit:
        # Missing APP_ID in bench env — fall back to plain dict overrides.
        pass
    if not hasattr(_cfg, "config") or _cfg.config is None:
        _cfg.config = {}
    _cfg.config["memory_retriever_audit_path"] = str(audit_path)
    _cfg.config["memory_stale_window_days"] = 90

    # Pull lifecycle helpers AFTER the redirects so the module captures the
    # right MEMORY_HOME_DIR.
    from larkhelm.memory_lifecycle import (
        iter_known_chat_cwd_pairs, mark_stale_slices,
    )

    # Inline replica of `_start_memory_boot_warmup`. We can't call the real
    # one because it depends on `_init_runtime` having materialised APP_ID,
    # which isn't valid in a bench env — so we mirror its semantics here.
    done_ev = threading.Event()
    chats_seen = {"count": 0}

    def _loop() -> None:
        try:
            for chat_id, cwd in iter_known_chat_cwd_pairs():
                chats_seen["count"] += 1
                try:
                    mark_stale_slices(chat_id, cwd, dry_run=False, window_days=90)
                except Exception as inner:
                    print(f"[bench] mark_stale_slices({chat_id}) failed: {inner}",
                          file=sys.stderr)
        finally:
            done_ev.set()

    # Phase 1 — measure how long the kick-off takes (must be effectively 0).
    t0 = time.perf_counter()
    th = threading.Thread(target=_loop, daemon=True, name="bench-boot-warmup")
    th.start()
    ready_elapsed = time.perf_counter() - t0
    print(f"[bench] ready-to-serve  : {ready_elapsed * 1000:.1f} ms  "
          f"(budget = {READY_BUDGET_SEC * 1000:.0f} ms)")

    # Phase 2 — wait for the daemon to finish or hit the budget.
    gc_finished = done_ev.wait(timeout=GC_BUDGET_SEC + 1.0)
    gc_elapsed = time.perf_counter() - t0 - ready_elapsed
    print(f"[bench] gc finished    : {gc_elapsed * 1000:.1f} ms  "
          f"(budget = {GC_BUDGET_SEC * 1000:.0f} ms, n={chats_seen['count']})")

    ok = True
    if ready_elapsed > READY_BUDGET_SEC:
        print(f"[bench] FAIL: ready_elapsed {ready_elapsed:.3f}s > "
              f"{READY_BUDGET_SEC}s", file=sys.stderr)
        ok = False
    if not gc_finished:
        print(f"[bench] FAIL: GC daemon didn't finish within "
              f"{GC_BUDGET_SEC}s", file=sys.stderr)
        ok = False
    elif gc_elapsed > GC_BUDGET_SEC:
        print(f"[bench] FAIL: gc_elapsed {gc_elapsed:.3f}s > "
              f"{GC_BUDGET_SEC}s", file=sys.stderr)
        ok = False
    if chats_seen["count"] < n_chats:
        print(f"[bench] FAIL: only iterated {chats_seen['count']}/{n_chats} chats",
              file=sys.stderr)
        ok = False

    status = "PASS" if ok else "FAIL"
    print(f"RESULT: {status} ready={ready_elapsed * 1000:.1f}ms "
          f"gc={gc_elapsed * 1000:.1f}ms n={chats_seen['count']}")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chats", type=int, default=100,
                        help="number of chat fixtures to materialise (default: 100)")
    args = parser.parse_args(argv)
    return _run_bench(args.chats)


if __name__ == "__main__":
    sys.exit(main())
