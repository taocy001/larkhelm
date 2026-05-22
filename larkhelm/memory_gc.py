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

    P3 REQ-08 / REQ-09: extended with:

    * ``interval_hours`` configurable via ``memory_gc_interval_hours``.
      Stored alongside the legacy ``interval_sec`` for back-compat.
    * ``_ckpt_gc`` composition — :class:`CheckpointGC` is invoked from
      ``run_once`` so its sweep shares the same tick thread (D5).
    * ``stop()`` — ``threading.Event`` wake-up so atexit / tests can
      end the daemon cleanly instead of waiting up to ``interval_sec``.
    """

    def __init__(self,
                 *,
                 max_age_days: int = SESSION_GC_MAX_AGE_DAYS,
                 interval_sec: int = SESSION_GC_INTERVAL_SEC,
                 interval_hours: float | None = None,
                 ckpt_gc: "object | None" = None):
        self.max_age_days = max(1, int(max_age_days))
        if interval_hours is not None and interval_hours > 0:
            self.interval_hours = float(interval_hours)
            self.interval_sec = max(60, int(interval_hours * 3600))
        else:
            self.interval_sec = max(60, int(interval_sec))
            self.interval_hours = self.interval_sec / 3600.0
        self._ckpt_gc = ckpt_gc
        self._started = False
        self._thread: threading.Thread | None = None
        self._started_lock = threading.Lock()
        self._stop_event = threading.Event()

    # ── P3 REQ-09: external composition ────────────────────────────

    def attach_checkpoint_gc(self, ckpt_gc: "object | None") -> None:
        """Register a :class:`CheckpointGC` for the next tick onward."""
        self._ckpt_gc = ckpt_gc

    def stop(self) -> None:
        """Signal the daemon loop to wake up and exit."""
        self._stop_event.set()

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
                # Even when there's no session file to sweep we still run
                # the phase-2 tail (audit rotate + stale recompute) and the
                # checkpoint GC, so an early-deployment instance whose
                # MEMORY_HOME_DIR is empty still gets its tail tasks done.
                self._phase2_tail()
                self._run_checkpoint_gc()
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
        # P3 REQ-09: checkpoint GC piggy-backs on the same tick when wired.
        # Shared helper with ``_tick`` (NIT-1 dedup follow-up).
        self._run_checkpoint_gc()
        return (scanned, deleted)

    # ── P3 REQ-08 (class-diagram aliases) ───────────────────────────
    # The design.md class diagram names two internal hooks that the
    # original Phase-B impl folded into ``_phase2_tail``. We expose them
    # as discrete methods so tests can call them directly and so the
    # implementation matches the design contract verbatim.

    def _rotate_audit_jsonl(self) -> None:
        """Delegate to ``memory_retriever.rotate_audit_files`` (32MiB / 30d)."""
        try:
            from larkhelm.memory_retriever import rotate_audit_files
            rotate_audit_files()
        except Exception as e:
            _debug_log(f"[MemGcDaemon] rotate_audit_jsonl failed: {e}")

    def _recompute_stale_slices_incremental(self) -> None:
        """Iterate known (chat_id, cwd) pairs and refresh stale sidecars."""
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
                        f"[MemGcDaemon] mark_stale_slices({chat_id}) failed: {inner}"
                    )
            if count:
                _debug_log(f"[MemGcDaemon] stale sweep covered {count} chat(s)")
        except Exception as e:
            _debug_log(f"[MemGcDaemon] stale sweep failed: {e}")

    def _run_checkpoint_gc(self) -> None:
        """Single source of truth for the checkpoint sub-GC call. Both
        ``run_once`` and ``_tick`` used to inline the same try/except block;
        NIT-1 in P3 review flagged the duplication."""
        if self._ckpt_gc is None:
            return
        try:
            removed = self._ckpt_gc.scan_once()
            if removed:
                _debug_log(f"[MemGcDaemon] checkpoint GC removed {removed} files")
        except Exception as e:
            _debug_log(f"[MemGcDaemon] checkpoint GC failed: {e}")

    def _tick(self) -> None:
        """Single full tick — rotate, recompute stale, run checkpoint GC.

        P3 REQ-08 / REQ-09: this is the canonical method named in the
        design class diagram. ``run_once`` retains the session-file
        sweep + counter return so legacy callers and tests still work.
        """
        self._rotate_audit_jsonl()
        self._recompute_stale_slices_incremental()
        self._run_checkpoint_gc()

    def _phase2_tail(self) -> None:
        """Run Phase 2 + P3 housekeeping hooks; failures are swallowed.

        NIT-1 follow-up: previously this method inlined the audit-rotate
        and stale-recompute logic that also lived in ``_rotate_audit_jsonl``
        / ``_recompute_stale_slices_incremental``. Now delegates to those
        two so there's exactly one implementation per step.
        """
        self._rotate_audit_jsonl()
        self._recompute_stale_slices_incremental()

    # ── internals ──────────────────────────────────────────────────

    def _should_delete(self, path: Path, now: float, cutoff: float) -> bool:
        try:
            mtime = path.stat().st_mtime
        except Exception as e:
            _debug_log(f"[MemoryGC] stat failed for {path.name}: {e}")
            return False
        if mtime >= cutoff:
            return False
        # MEM-H9: skip sessions whose chat is currently loaded in memory
        # (turn_count > 0 means the chat has been active since last restart).
        stem = path.stem  # "session_<chat_id>"
        if stem.startswith("session_"):
            chat_id_candidate = stem[len("session_"):]
            try:
                from larkhelm.chat_state import _get_turn_count
                if _get_turn_count(chat_id_candidate) > 0:
                    return False
            except Exception:
                pass
        return True

    def _loop(self) -> None:
        while True:
            # ``Event.wait`` honours ``stop()`` so atexit / tests don't have
            # to wait up to ``interval_sec`` for the daemon to notice.
            if self._stop_event.wait(self.interval_sec):
                _debug_log("[MemGcDaemon] stop signalled, exiting loop")
                return
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
            interval_hours = float(
                getattr(_cfg, "MEMORY_GC_INTERVAL_HOURS", 0.0)
                or cfg.get("memory_gc_interval_hours", 0.0)
                or 0.0
            )
            _RUNNER = MemoryGCRunner(
                max_age_days=int(cfg.get("memory_session_gc_max_age_days",
                                         SESSION_GC_MAX_AGE_DAYS)),
                interval_sec=SESSION_GC_INTERVAL_SEC,
                interval_hours=interval_hours if interval_hours > 0 else None,
            )
        return _RUNNER


def attach_checkpoint_gc(ckpt_gc: "object | None") -> None:
    """Register a :class:`CheckpointGC` with the singleton runner.

    Called from ``crew/__init__.py`` once DATA_DIR is known. Safe to
    call before :func:`start_memory_gc_thread` — the runner picks up
    the attachment on its next tick.
    """
    try:
        _get_runner().attach_checkpoint_gc(ckpt_gc)
    except Exception as e:
        _debug_log(f"[MemGcDaemon] attach_checkpoint_gc failed: {e}")


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
