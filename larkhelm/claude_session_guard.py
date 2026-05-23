"""larkhelm · Claude session auto-reset guard (P0).

When the Claude --resume prefix accumulates 5M cache_read tokens or 50
record_token_usage calls (defaults; both knobs in config), this module:

  1. clears the per-chat ``.sid`` file (next ``_do_query`` starts a fresh session)
  2. zeros the cache_read / turns counters in ``_chat_state_store``
  3. records a milestone so memory captures the boundary
  4. bumps ``larkhelm_session_auto_reset_total{reason}``

The auto-reset only runs when ``model == "claude"`` AND
``config.CLAUDE_SESSION_AUTO_RESET_ENABLED`` is true (default true). All
public entry points swallow exceptions — the token-accounting path must
never break because of a guard failure.

Design ref: ``.crew_workspace/design.md`` §1.1 D2 / §5.1.
"""
from __future__ import annotations

import threading
from typing import Optional

import larkhelm.config as _cfg
from larkhelm.chat_state import (
    _clear_claude_session_counters,
    _clear_sid,
    _get_claude_session_counters,
    _increment_claude_session_counters,
)
from larkhelm.log import _debug_log


# Per-chat "reset in progress" registry. Atomic claim-or-skip prevents two
# concurrent ``record_token_usage`` calls — that both observe a fresh
# threshold crossing in the gap between ``_increment_claude_session_counters``
# (atomic) and ``_check_thresholds`` (non-atomic) — from each triggering
# ``_perform_auto_reset``. Without this guard the loser would double the
# milestone, double the metric, and run ``_clear_sid`` on an already-unlinked
# .sid file (FileNotFoundError swallowed but noisy).
_resetting_chats: set[str] = set()
_resetting_lock = threading.Lock()


def _enabled() -> bool:
    """Honour the operator override (read at every call, not memoised)."""
    return bool(getattr(_cfg, "CLAUDE_SESSION_AUTO_RESET_ENABLED", True))


def _threshold_cache_read() -> int:
    return int(getattr(_cfg, "CLAUDE_SESSION_RESET_CACHE_TOKENS", 5_000_000)
               or 5_000_000)


def _threshold_turns() -> int:
    return int(getattr(_cfg, "CLAUDE_SESSION_RESET_TURNS", 50) or 50)


def _check_thresholds(cache_read_total: int, turn_total: int) -> Optional[str]:
    """Decide which threshold (if any) was crossed.

    cache_read is checked first because the cost signal is the primary
    motivation (each prefix-replay is billed at ~10% of the input price,
    so a 5M-token prefix dominates token cost long before turn 50 even
    in a busy session).
    """
    if cache_read_total >= _threshold_cache_read():
        return "cache_tokens"
    if turn_total >= _threshold_turns():
        return "turns"
    return None


def _perform_auto_reset(chat_id: str, reason: str,
                        cache_read_total: int, turn_total: int) -> None:
    """Best-effort reset: clear sid, zero counters, milestone, metric.

    Each step is wrapped so a downstream failure (e.g. memory module not
    importable in a partial-bootstrap test) cannot prevent the rest.
    """
    try:
        _clear_sid(chat_id, "claude")
    except Exception as e:
        _debug_log(f"[ClaudeSessionGuard] _clear_sid failed for {chat_id[:8]}: {e}")
    try:
        _clear_claude_session_counters(chat_id)
    except Exception as e:
        _debug_log(
            f"[ClaudeSessionGuard] clear counters failed for {chat_id[:8]}: {e}"
        )
    try:
        from larkhelm.memory import record_milestone
        record_milestone(
            chat_id,
            "session_auto_reset",
            f"Auto reset: {reason} "
            f"(cache_read={cache_read_total}, turns={turn_total})",
        )
    except Exception as e:
        _debug_log(
            f"[ClaudeSessionGuard] record_milestone failed for {chat_id[:8]}: {e}"
        )
    try:
        from larkhelm.metrics import inc_session_auto_reset
        inc_session_auto_reset(reason)
    except Exception as e:
        _debug_log(
            f"[ClaudeSessionGuard] inc_session_auto_reset failed: {e}"
        )
    _debug_log(
        f"[ClaudeSessionGuard] auto-reset chat={chat_id[:8]} reason={reason} "
        f"cache_read={cache_read_total} turns={turn_total}"
    )


def maybe_auto_reset_session(
    chat_id: str, model: str, usage: dict,
) -> Optional[str]:
    """Increment counters and trigger auto-reset on threshold cross.

    Hook point: ``token_stats.record_token_usage`` calls this at the end of
    the standard accounting path. No-op unless ``model == "claude"`` AND
    ``CLAUDE_SESSION_AUTO_RESET_ENABLED`` is True. Never raises.

    Returns the reset reason ('cache_tokens' | 'turns') if a reset was
    performed, otherwise None.
    """
    try:
        if model != "claude":
            return None
        if not _enabled():
            return None
        try:
            delta = max(0, int((usage or {}).get("cache_read", 0) or 0))
        except (TypeError, ValueError):
            delta = 0
        cache_read_total, turn_total = _increment_claude_session_counters(
            chat_id, delta,
        )
        reason = _check_thresholds(cache_read_total, turn_total)
        if reason is None:
            return None
        # Atomic claim: if another thread is already running the reset for
        # this chat, bail out without re-running it. The winner's
        # ``_clear_claude_session_counters`` (called inside
        # ``_perform_auto_reset``) zeros the totals before this thread can
        # observe them again, so subsequent calls see "no cross" anyway.
        with _resetting_lock:
            if chat_id in _resetting_chats:
                return None
            _resetting_chats.add(chat_id)
        try:
            _perform_auto_reset(chat_id, reason, cache_read_total, turn_total)
        finally:
            with _resetting_lock:
                _resetting_chats.discard(chat_id)
        return reason
    except Exception as e:
        # Critical: never propagate — token accounting must succeed
        # regardless of guard state.
        try:
            _debug_log(
                f"[ClaudeSessionGuard] auto-reset check failed for "
                f"{chat_id[:8]}: {e}"
            )
        except Exception:
            pass
        return None


def get_session_counters(chat_id: str) -> dict:
    """Snapshot the per-chat session counters for ``/stats`` display.

    Always returns a populated dict (zeros when the chat has no state).
    """
    try:
        cache_read, turns = _get_claude_session_counters(chat_id)
    except Exception as e:
        _debug_log(
            f"[ClaudeSessionGuard] get_session_counters failed for "
            f"{chat_id[:8]}: {e}"
        )
        cache_read, turns = 0, 0
    return {
        "cache_read":           cache_read,
        "turns":                turns,
        "threshold_cache_read": _threshold_cache_read(),
        "threshold_turns":      _threshold_turns(),
        "enabled":              _enabled(),
    }


def clear_session_counters(chat_id: str) -> None:
    """Zero the per-chat session counters. Idempotent."""
    try:
        _clear_claude_session_counters(chat_id)
    except Exception as e:
        _debug_log(
            f"[ClaudeSessionGuard] clear_session_counters failed for "
            f"{chat_id[:8]}: {e}"
        )


__all__ = [
    "maybe_auto_reset_session",
    "get_session_counters",
    "clear_session_counters",
]
