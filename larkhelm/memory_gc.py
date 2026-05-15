"""larkhelm · session memory GC daemon (S3 — Phase B)

Background daemon that scans ``MEMORY_HOME_DIR`` once per
``SESSION_GC_INTERVAL_SEC`` (default 1h) and unlinks ``session_*.md`` files
older than ``SESSION_GC_MAX_AGE_DAYS`` (default 7d). Project and global
memory layers are intentionally NOT touched — they have different
lifecycle semantics and a separate user-explicit cleanup path
(``memory.gc_project_memory``).

Test isolation: ``start_memory_gc_thread()`` returns immediately when
``LARKHELM_TEST_MODE`` is set OR ``config.memory_session_gc_enabled`` is
False. Either gate is sufficient — tests don't need the thread, and an
operator can disable GC entirely without restarting the bridge by editing
config and bouncing.

Safety: only filenames matching ``session_*.md`` are even considered for
deletion. Inside the loop we double-check the prefix before calling
``unlink`` so an upstream ``glob`` regression can't accidentally clear
``project_*.md`` or ``global_*.md``.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, info, warn

SESSION_GC_MAX_AGE_DAYS: int = 7
SESSION_GC_INTERVAL_SEC: int = 3600


class MemoryGCRunner:
    """Daemon-thread holder for the session-GC loop.

    Single instance per process; ``start()`` is idempotent. Tests can call
    ``run_once()`` directly without starting the thread.
    """

    def __init__(self,
                 *,
                 max_age_days: int = SESSION_GC_MAX_AGE_DAYS,
                 interval_sec: int = SESSION_GC_INTERVAL_SEC):
        self.max_age_days = max(1, int(max_age_days))
        self.interval_sec = max(60, int(interval_sec))
        self._started = False
        self._thread: threading.Thread | None = None
        self._started_lock = threading.Lock()

    # ── public API ─────────────────────────────────────────────────

    def start(self) -> None:
        """Spin up the daemon loop (idempotent)."""
        with self._started_lock:
            if self._started:
                return
            self._started = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="memory-gc",
        )
        self._thread.start()
        info(f"[MemoryGC] daemon started (max_age_days={self.max_age_days}, "
             f"interval_sec={self.interval_sec})")

    def run_once(self) -> tuple[int, int]:
        """Single sweep — returns ``(scanned, deleted)``.

        Phase 2 additions (run at the tail of the sweep, never raise):

          * :func:`memory_retriever.rotate_audit_files` — daily + 32MiB
            rollover + 30 day unlink of audit JSONL archives.
          * :func:`memory_lifecycle.mark_stale_slices` over every known
            (chat_id, cwd) pair — refreshes ``.meta.json`` sidecars.
        """
        from larkhelm.memory import MEMORY_HOME_DIR
        scanned = 0
        deleted = 0
        try:
            if not MEMORY_HOME_DIR.exists():
                self._phase2_tail()
                return (0, 0)
        except Exception as e:
            _debug_log(f"[MemoryGC] home dir stat failed: {e}")
            return (0, 0)
        now = time.time()
        cutoff = now - self.max_age_days * 86400
        for path in MEMORY_HOME_DIR.glob("session_*.md"):
            scanned += 1
            try:
                if not self._should_delete(path, now, cutoff):
                    continue
                # Defence-in-depth: re-validate the filename against the
                # ``session_`` prefix + ``.md`` suffix BEFORE unlink so an
                # upstream glob bug or symlink trick can't clear other layers.
                if not (path.name.startswith("session_") and path.suffix == ".md"):
                    _debug_log(f"[MemoryGC] safety check rejected {path.name}")
                    continue
                path.unlink(missing_ok=True)
                deleted += 1
                _debug_log(f"[MemoryGC] deleted {path.name}")
            except Exception as e:
                _debug_log(f"[MemoryGC] unlink failed for {path.name}: {e}")
        if scanned or deleted:
            info(f"[MemoryGC] sweep complete scanned={scanned} deleted={deleted}")
        self._phase2_tail()
        return (scanned, deleted)

    def _phase2_tail(self) -> None:
        """Run Phase 2 housekeeping hooks; failures are swallowed."""
        try:
            from larkhelm.memory_retriever import rotate_audit_files
            rotate_audit_files()
        except Exception as e:
            _debug_log(f"[MemoryGC] rotate_audit_files failed: {e}")
        try:
            from larkhelm.memory_lifecycle import (
                iter_known_chat_cwd_pairs,
                mark_stale_slices,
            )
            cfg = getattr(_cfg, "config", None) or {}
            window_days = int(cfg.get("memory_stale_window_days", 90) or 90)
            count = 0
            for chat_id, cwd in iter_known_chat_cwd_pairs():
                try:
                    mark_stale_slices(chat_id, cwd, dry_run=False, window_days=window_days)
                    count += 1
                except Exception as inner:
                    _debug_log(
                        f"[MemoryGC] mark_stale_slices({chat_id}) failed: {inner}"
                    )
            if count:
                _debug_log(f"[MemoryGC] stale sweep covered {count} chat(s)")
        except Exception as e:
            _debug_log(f"[MemoryGC] stale sweep failed: {e}")

    # ── internals ──────────────────────────────────────────────────

    def _should_delete(self, path: Path, now: float, cutoff: float) -> bool:
        try:
            mtime = path.stat().st_mtime
        except Exception as e:
            _debug_log(f"[MemoryGC] stat failed for {path.name}: {e}")
            return False
        return mtime < cutoff

    def _loop(self) -> None:
        while True:
            try:
                time.sleep(self.interval_sec)
            except Exception:
                time.sleep(self.interval_sec)
            try:
                self.run_once()
            except Exception as e:
                # The daemon must never die — the absence of GC silently leaks
                # session files forever, which is exactly the bug Phase B fixes.
                warn(f"[MemoryGC] loop error (continuing): {e}")


_RUNNER: MemoryGCRunner | None = None
_RUNNER_LOCK = threading.Lock()


def _get_runner() -> MemoryGCRunner:
    global _RUNNER
    with _RUNNER_LOCK:
        if _RUNNER is None:
            cfg = getattr(_cfg, "config", None) or {}
            _RUNNER = MemoryGCRunner(
                max_age_days=int(cfg.get("memory_session_gc_max_age_days",
                                         SESSION_GC_MAX_AGE_DAYS)),
                interval_sec=SESSION_GC_INTERVAL_SEC,
            )
        return _RUNNER


def start_memory_gc_thread() -> None:
    """Idempotent thread starter respecting ``LARKHELM_TEST_MODE`` + config flag.

    Either gate (env var present OR ``memory_session_gc_enabled`` False) skips
    the daemon entirely — no thread is created. Other failure modes (config
    not yet initialised, etc.) also degrade silently.
    """
    if os.environ.get("LARKHELM_TEST_MODE", "").strip():
        _debug_log("[MemoryGC] skipped (LARKHELM_TEST_MODE set)")
        return
    cfg = getattr(_cfg, "config", None) or {}
    if not cfg.get("memory_session_gc_enabled", True):
        _debug_log("[MemoryGC] skipped (memory_session_gc_enabled=false)")
        return
    try:
        _get_runner().start()
    except Exception as e:
        warn(f"[MemoryGC] start failed: {e}")
