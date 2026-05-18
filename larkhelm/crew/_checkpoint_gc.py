"""larkhelm · crew._checkpoint_gc — orphan checkpoint sweeper (P3 REQ-09).

`.crew_workspace/*/crew_checkpoint.json` files accumulate over time as
``/crew`` and ``/dev`` runs land. Most are abandoned mid-task or finish
cleanly; either way nothing prunes them. By 6 months of operation the
DATA_DIR fills with checkpoints from chats that no longer exist.

This module scans the workspaces root and unlinks any checkpoint whose
mtime is older than ``ttl_days`` AND whose owning chat has no live
``_active_crew`` entry. It never raises — the caller is the
``memory_gc`` tick which must stay alive.

Design (see ``.crew_workspace/design.md`` §1.2 D5):

* Shares the ``MemoryGC._tick`` thread with the audit-rotate /
  stale-recompute work; no separate daemon.
* Uses a configurable ``_now`` callable so tests can advance the clock
  without sleeping.
* The "active lock" check imports ``crew._state`` lazily so a partial
  bootstrap doesn't tank the GC.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterable

from larkhelm.log import _debug_log


class CheckpointGC:
    """Sweep orphan ``crew_checkpoint.json`` files older than ``ttl_days``."""

    def __init__(
        self,
        workspaces_root: Path,
        ttl_days: float = 7.0,
    ) -> None:
        self.workspaces_root = Path(workspaces_root)
        self.ttl_days = float(ttl_days)
        self._now: Callable[[], float] = time.time

    # ── public ──────────────────────────────────────────────────────────

    def scan_once(self) -> int:
        """Walk ``workspaces_root`` once and unlink orphan checkpoints.

        Returns the count of removed files. Never raises — every error
        path collapses to a debug log + ``return removed_count`` so the
        outer GC tick stays alive.
        """
        removed = 0
        if not self.workspaces_root.exists():
            return 0
        try:
            candidates = self._iter_checkpoints()
        except Exception as e:
            _debug_log(f"[CkptGc] iter failed: {e}")
            return 0
        for ckpt in candidates:
            try:
                if not self._is_orphan(ckpt):
                    continue
                ckpt.unlink()
                removed += 1
                _debug_log(f"[CkptGc] removed orphan {ckpt}")
            except Exception as e:
                _debug_log(f"[CkptGc] unlink {ckpt} failed: {e}")
        if removed > 0:
            _debug_log(f"[CkptGc] removed {removed} orphan checkpoints")
        return removed

    # ── internals ───────────────────────────────────────────────────────

    def _iter_checkpoints(self) -> Iterable[Path]:
        """Yield every ``crew_checkpoint.json`` under workspaces_root.

        We don't follow symlinks (security: a workspace folder could in
        theory be replaced with a symlink to elsewhere; the unlink
        below would then act on an unintended target).
        """
        for entry in self.workspaces_root.iterdir():
            if not entry.is_dir() or entry.is_symlink():
                continue
            ckpt = entry / "crew_checkpoint.json"
            if ckpt.is_file() and not ckpt.is_symlink():
                yield ckpt

    def _is_orphan(self, ckpt_path: Path) -> bool:
        """True iff the checkpoint is past its TTL AND no live crew claims it."""
        try:
            mtime = ckpt_path.stat().st_mtime
        except OSError:
            return False
        age_sec = self._now() - mtime
        if age_sec < self.ttl_days * 86400:
            return False
        if self._has_active_crew_lock(ckpt_path):
            return False
        return True

    @staticmethod
    def _has_active_crew_lock(ckpt_path: Path) -> bool:
        """Best-effort: is the chat that owns this workspace still active?

        The workspace dir name typically encodes ``crew_id`` (uuid), not
        ``chat_id`` directly. We import the crew state map lazily and
        return ``True`` if *any* active state references this workspace.
        """
        try:
            from larkhelm.crew._state import _active_crew_lock, _active_crew_states
        except Exception:
            return False
        workspace_dir = str(ckpt_path.parent.resolve())
        try:
            with _active_crew_lock:
                states = list(_active_crew_states.values())
        except Exception:
            return False
        for state in states:
            try:
                ws = getattr(state, "workspace_dir", "") or ""
                if ws and str(Path(ws).resolve()) == workspace_dir:
                    return True
            except Exception:
                continue
        return False


__all__ = ["CheckpointGC"]
