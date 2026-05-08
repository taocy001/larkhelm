"""
larkhelm · Crew global state variables + breakpoint registry + git helpers
"""
from __future__ import annotations

import subprocess
import threading
import time

from larkhelm.log import _debug_log
from larkhelm.crew_types import CrewState


# ═══════════════════════════════════════════════════════════════
#  Global state
# ═══════════════════════════════════════════════════════════════

# Per-chat active crew (only one crew per chat at a time)
_active_crew: dict[str, str]        = {}   # chat_id → crew_id
_active_crew_states: dict[str, "CrewState"] = {}  # chat_id → CrewState
_active_crew_lock = threading.Lock()

# Subscribers waiting for crew to finish (protected by _active_crew_lock)
_crew_done_subscribers: dict[str, list[threading.Event]] = {}  # chat_id → [Event]


def is_crew_running(chat_id: str) -> bool:
    with _active_crew_lock:
        return chat_id in _active_crew


def subscribe_crew_done(chat_id: str) -> threading.Event:
    """Return an Event that fires when crew for chat_id finishes.

    Race-safe: if crew already finished by the time this is called, returns a
    pre-set event so the caller never blocks.  Must be called after _set_pending.
    """
    ev = threading.Event()
    with _active_crew_lock:
        if chat_id not in _active_crew:
            ev.set()   # already done
            return ev
        _crew_done_subscribers.setdefault(chat_id, []).append(ev)
    return ev


def _signal_crew_done(chat_id: str) -> None:
    """Called from crew finally blocks (inside _active_crew_lock held by caller)."""
    subscribers = _crew_done_subscribers.pop(chat_id, [])
    for ev in subscribers:
        ev.set()

# Phase 3.1: human-confirmation breakpoint registry
_breakpoint_events:  dict[str, threading.Event] = {}  # crew_id → Event
_breakpoint_results: dict[str, bool]            = {}  # crew_id → confirmed?
_breakpoint_meta = threading.Lock()

# Crew card index: card_mid → {"title", "summary", "chat_id", "ts"}
# Used to inject context when a user replies to a crew card
_crew_card_index: dict[str, dict] = {}  # OrderedDict semantics, max 50 entries
_crew_card_index_lock = threading.Lock()
_CREW_CARD_INDEX_MAX  = 50
# Per chat_id, record the most recently completed crew result (valid for 2h), for context injection
_recent_crew_by_chat: dict[str, dict] = {}  # chat_id → {"title", "summary", "ts"}
_RECENT_CREW_TTL = 7200  # seconds


def get_recent_crew_context(chat_id: str) -> "dict":
    """Called by handlers.py: returns context if this chat has a crew task completed within the last 2h."""
    with _crew_card_index_lock:
        entry = _recent_crew_by_chat.get(chat_id)
    if entry and (time.time() - entry["ts"]) < _RECENT_CREW_TTL:
        return entry
    return None


def clear_recent_crew_context(chat_id: str) -> None:
    """Clear stale sticky context when starting a new crew/dev task, to prevent cross-injection."""
    with _crew_card_index_lock:
        _recent_crew_by_chat.pop(chat_id, None)


# ═══════════════════════════════════════════════════════════════
#  Git helpers (resume snapshots and auto-commit)
# ═══════════════════════════════════════════════════════════════

def _git_head(cwd: str) -> str:
    """Return the current HEAD commit hash, or empty string if not a git repo."""
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5, cwd=cwd)
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git_auto_commit(cwd: str, label: str) -> str:
    """
    Auto-commit all workspace changes and return the commit hash (empty string on failure).
    Only runs when dev_auto_commit=True and there are uncommitted changes.
    """
    import larkhelm.config as _cfg
    if not _cfg.config.get("dev_auto_commit", False):
        return ""
    try:
        # Check whether there are any changes
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5, cwd=cwd)
        if not dirty.stdout.strip():
            return ""
        subprocess.run(["git", "add", "-A"], cwd=cwd, timeout=10,
                       capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "--no-verify", "-m", f"[dev:{label}] auto-checkpoint"],
            cwd=cwd, timeout=10, capture_output=True, check=True,
        )
        return _git_head(cwd)
    except Exception as e:
        _debug_log(f"[Git] auto-commit failed ({label}): {e}")
        return ""


def _register_crew_card(card_mid: str, chat_id: str, title: str, summary: str):
    """Record card_mid after a task completes, so context can be injected when a user replies to the card."""
    entry = {
        "chat_id": chat_id,
        "title":   title,
        "summary": summary[:3000],
        "ts":      time.time(),
    }
    with _crew_card_index_lock:
        _crew_card_index[card_mid] = entry
        while len(_crew_card_index) > _CREW_CARD_INDEX_MAX:
            _crew_card_index.pop(next(iter(_crew_card_index)))
        # Also record per chat_id for injection in non-reply scenarios
        _recent_crew_by_chat[chat_id] = entry


def get_crew_card_context(card_mid: str) -> "dict":
    """Called by handlers.py: returns crew context for the given card_mid, or None if not found."""
    with _crew_card_index_lock:
        return _crew_card_index.get(card_mid)


def signal_breakpoint(crew_id: str, confirmed: bool):
    """Called by handle_card_action: set the breakpoint result after the user clicks confirm/cancel."""
    with _breakpoint_meta:
        _breakpoint_results[crew_id] = confirmed
        ev = _breakpoint_events.get(crew_id)
        if ev:
            ev.set()  # Set inside the lock to avoid a race between pop and set
