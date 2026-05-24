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
#
# Owner-token contract (read by ``describe_active_owner`` below):
#   * ``f"plan:{plan_id}"`` — written by ``cmd_plan._run_plan._hold_slot``
#   * ``"<12-char-hex crew_id>"`` — written by ``crew._commands`` (/crew, /dev)
#     and by ``crew._checkpoint.resume_interrupted_crews``
# Any new writer MUST either reuse one of these formats or extend the helper.
_active_crew: dict[str, str]        = {}   # chat_id → owner_token (see above)
_active_crew_states: dict[str, "CrewState"] = {}  # chat_id → CrewState
_active_crew_lock = threading.Lock()


def describe_active_owner(owner: str) -> str:
    """Render a short human-readable label for an ``_active_crew`` owner token.

    Used by both ``/plan`` and ``/crew`` / ``/dev`` conflict cards so the user
    sees *what* is holding the slot, not just "task in progress". Pre-C3 the
    conflict cards just said "已有任务" with no hint about which command to
    cancel — operator debug nightmare. Pure-string parser, no imports of
    ``cmd_plan`` so this helper is safe to call from any module.
    """
    if not owner:
        return "未知任务"
    if owner.startswith("plan:"):
        plan_id = owner[5:13] or "?"
        return f"`/plan` 任务 (id={plan_id})"
    return f"`/crew` 或 `/dev` 任务 (id={owner[:8]})"


def current_owner(chat_id: str) -> str:
    """Snapshot the ``_active_crew`` owner token for ``chat_id``.

    Returns an empty string when the slot is free. Acquires
    ``_active_crew_lock`` once so callers don't need to import both the
    dict and the lock — keeps cross-module wiring trivial and races out.
    Use ``describe_active_owner`` on the returned token to render a
    user-facing label.
    """
    with _active_crew_lock:
        return _active_crew.get(chat_id, "")


def owner_kind(owner: str) -> str:
    """Classify an owner token: ``"plan"`` / ``"crew"`` / ``""`` (free).

    Pure-string sister of :func:`describe_active_owner` for call sites
    that need to branch on owner type (e.g. card title rendering)
    without parsing the human-readable description. ``""`` lets the
    caller distinguish "no owner" from an unknown future owner class.
    """
    if not owner:
        return ""
    if owner.startswith("plan:"):
        return "plan"
    return "crew"

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
# Per chat_id, record the most recently completed crew result, for context injection.
# TTL is now sourced from ``config.RECENT_CREW_STICKY_TTL_SEC`` (default 1800;
# was hard-coded 7200) — see ``_ttl_sec`` below. Entry shape additionally
# carries ``injection_count`` (P2) for the consume-based dedup path.
_recent_crew_by_chat: dict[str, dict] = {}  # chat_id → entry


def _ttl_sec() -> int:
    """Resolve the sticky-context TTL from config; floor at 60s.

    Read at every call so an operator flipping
    ``recent_crew_sticky_ttl_sec`` in config.json + sending a SIGHUP-style
    config reload picks up the new value without restarting.
    """
    import larkhelm.config as _cfg
    try:
        raw = int(getattr(_cfg, "RECENT_CREW_STICKY_TTL_SEC", 1800) or 1800)
    except (TypeError, ValueError):
        raw = 1800
    return max(60, raw)


def _max_injections() -> int:
    import larkhelm.config as _cfg
    # Note: 0 is a *valid* value (disables eviction; TTL-only mode), so the
    # ``value or default`` idiom would coerce 0 → default here. Read straight
    # and only coerce truly bad inputs (None / non-int).
    raw = getattr(_cfg, "RECENT_CREW_STICKY_MAX_INJECTIONS", 5)
    if raw is None:
        return 5
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 5


def get_recent_crew_context(chat_id: str) -> "dict":
    """Read-only accessor: returns sticky crew context, or None if absent/expired.

    Phase 5 D10: this path is intentionally non-mutating — ``/status``,
    ``/memory diagnose``, and the parent-fetch fallback in ``_query.py`` /
    ``_query_session.py`` use it, and a read should not bump the dedup
    counter (which would double-count a single user turn). Use
    :func:`consume_recent_crew_context` from the primary injection path
    (``handlers/_message.py``).

    On TTL expiry the entry is lazy-removed and
    ``larkhelm_sticky_context_evicted_total{reason='ttl'}`` is bumped.
    """
    ttl = _ttl_sec()
    now = time.time()
    expired_entry = None
    with _crew_card_index_lock:
        entry = _recent_crew_by_chat.get(chat_id)
        if entry and (now - entry["ts"]) >= ttl:
            expired_entry = _recent_crew_by_chat.pop(chat_id, None)
            entry = None
    if expired_entry is not None:
        try:
            from larkhelm.metrics import inc_sticky_context_evicted
            inc_sticky_context_evicted("ttl")
        except Exception as e:
            _debug_log(f"[Crew] sticky-context ttl metric failed: {e}")
        _debug_log(
            f"[Crew] sticky context TTL expired for {chat_id[:12]} "
            f"(title='{(expired_entry.get('title', '') or '')[:20]}')"
        )
    if entry is not None:
        return entry
    return None


def consume_recent_crew_context(chat_id: str) -> "dict | None":
    """Mutating accessor: returns sticky crew context AND bumps the
    per-entry injection counter; evicts the entry when the counter
    reaches ``config.RECENT_CREW_STICKY_MAX_INJECTIONS``.

    Called from the prompt-injection main path
    (``handlers/_message.py:717``). Read-only callers MUST keep using
    :func:`get_recent_crew_context`.

    Returns None when no entry, entry expired (TTL), or entry already
    evicted by an earlier consume call.
    """
    ttl = _ttl_sec()
    cap = _max_injections()
    now = time.time()
    expired = False
    evicted_max = False
    title_for_log = ""
    new_count = 0
    returned_entry: dict | None = None
    with _crew_card_index_lock:
        entry = _recent_crew_by_chat.get(chat_id)
        if entry is None:
            return None
        title_for_log = str(entry.get("title", "") or "")
        if (now - entry["ts"]) >= ttl:
            _recent_crew_by_chat.pop(chat_id, None)
            expired = True
        else:
            new_count = int(entry.get("injection_count", 0) or 0) + 1
            entry["injection_count"] = new_count
            returned_entry = entry
            # cap == 0 disables the per-count eviction (TTL-only mode).
            if cap > 0 and new_count >= cap:
                _recent_crew_by_chat.pop(chat_id, None)
                evicted_max = True
    if expired:
        try:
            from larkhelm.metrics import inc_sticky_context_evicted
            inc_sticky_context_evicted("ttl")
        except Exception as e:
            _debug_log(f"[Crew] sticky-context ttl metric failed: {e}")
        _debug_log(
            f"[Crew] sticky context TTL expired for {chat_id[:12]} "
            f"(title='{title_for_log[:20]}')"
        )
        return None
    if evicted_max:
        try:
            from larkhelm.metrics import inc_sticky_context_evicted
            inc_sticky_context_evicted("max_injections")
        except Exception as e:
            _debug_log(f"[Crew] sticky-context max-inj metric failed: {e}")
        _debug_log(
            f"[Crew] sticky context expired after {new_count} injections "
            f"(chat={chat_id[:12]}, title='{title_for_log[:20]}')"
        )
    return returned_entry


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


def _git_auto_commit(
    cwd: str,
    label: str,
    *,
    add_targets: "list[str] | None" = None,
    commit_message: "str | None" = None,
) -> str:
    """Auto-commit workspace changes and return the new commit hash.

    Two modes (B2):

    - ``add_targets=None`` (legacy): ``git add -A``. Used by per-agent
      checkpoints (implementer / fixer) where the diff is bounded by the
      single agent's work — accepting drift is OK there because the next
      agent will inspect the commit anyway.

    - ``add_targets=[...]`` (B2 finalize path): ``git add -- <list>``. Used
      by ``workspace_finalize`` after ``review APPROVED`` to commit only
      the curated white-list of paths (file_changes.json ∪ verified
      drift). Reviewer-flagged risk: a bare ``-A`` here could pick up
      unrelated dirty state from the user's parallel work.

    Returns the new HEAD short hash, or empty string when:
      - ``dev_auto_commit`` config is false (default)
      - no dirty changes in ``cwd``
      - any subprocess step fails (logged via ``_debug_log``)

    ``commit_message`` overrides the default ``[dev:{label}] auto-checkpoint``
    template (used by the finalize path to embed drift summary).
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
        if add_targets is None:
            subprocess.run(["git", "add", "-A"], cwd=cwd, timeout=10,
                           capture_output=True, check=True)
        else:
            # ``git add -- <paths>`` — empty list short-circuits to no-op.
            if not add_targets:
                _debug_log(f"[Git] auto-commit ({label}): empty add_targets, skipping")
                return ""
            subprocess.run(
                ["git", "add", "--"] + list(add_targets),
                cwd=cwd, timeout=15, capture_output=True, check=True,
            )
        msg = commit_message or f"[dev:{label}] auto-checkpoint"
        subprocess.run(
            ["git", "commit", "--no-verify", "-m", msg],
            cwd=cwd, timeout=15, capture_output=True, check=True,
        )
        return _git_head(cwd)
    except Exception as e:
        _debug_log(f"[Git] auto-commit failed ({label}): {e}")
        return ""


def _register_crew_card(card_mid: str, chat_id: str, title: str, summary: str):
    """Record card_mid after a task completes, so context can be injected when a user replies to the card."""
    entry = {
        "chat_id":         chat_id,
        "title":           title,
        "summary":         summary[:3000],
        "ts":              time.time(),
        # P2 (design.md §3.2): per-entry consume counter used by
        # ``consume_recent_crew_context`` to evict after N injections.
        "injection_count": 0,
    }
    with _crew_card_index_lock:
        _crew_card_index[card_mid] = entry
        while len(_crew_card_index) > _CREW_CARD_INDEX_MAX:
            _crew_card_index.pop(next(iter(_crew_card_index)))
        # Two views share the SAME dict reference: card_mid (reply path) and
        # chat_id (sticky inject path). Mutations from
        # ``consume_recent_crew_context`` (``injection_count += 1``) are
        # visible on the card_mid view too. ``get_crew_card_context`` does
        # not currently read ``injection_count``, so this is benign — but if
        # a future caller on the card_mid path starts reading that field
        # they will see a count that includes sticky-inject ticks.
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
