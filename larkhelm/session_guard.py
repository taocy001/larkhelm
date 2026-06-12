"""larkhelm · universal session auto-reset guard (Week-3).

Generalises the Claude-only ``claude_session_guard`` to all backends.
Public API:
  - maybe_auto_reset(chat_id, backend, usage) → Optional[str]
  - get_session_counters(chat_id, backend) → dict
  - clear_session_counters(chat_id, backend) → None

``claude_session_guard`` is now a thin wrapper that calls these with
backend="claude", preserving the existing public API for all callers.
"""
from __future__ import annotations

import threading
from typing import Optional

import larkhelm.config as _cfg
from larkhelm.chat_state import (
    _clear_backend_session_counters,
    _clear_sid,
    _get_backend_session_counters,
    _increment_backend_session_counters,
)
from larkhelm.log import _debug_log

_DEFAULT_POLICIES: dict[str, dict] = {
    "claude":   {"max_cache_read_tokens": 5_000_000, "max_turns": 50},
    "gemini":   {"max_cache_read_tokens": 4_000_000, "max_turns": 40},
    "deepseek": {"max_cache_read_tokens": 2_000_000, "max_turns": 30},
    "kimi":     {"max_cache_read_tokens": 0,          "max_turns": 60},
}

# Per-backend "reset in progress" registry prevents concurrent resets for the
# same (chat_id, backend) pair.
_resetting_chats_by_backend: dict[str, set[str]] = {}
_resetting_lock = threading.Lock()

# Pre-reset settle wait bound (seconds). memory.MEMORY_GENERATION_TIMEOUT is
# 120s; +10s slack covers thread scheduling and log/disk IO.
_SETTLE_WAIT_SEC = 130


def _guard_enabled() -> bool:
    return bool(getattr(_cfg, "SESSION_GUARD_ENABLED", True))


def _get_policy(backend: str) -> dict:
    policies = getattr(_cfg, "SESSION_GUARD_POLICIES", None) or {}
    return policies.get(backend) or _DEFAULT_POLICIES.get(backend) or {}


def _check_thresholds(
    cache_read: int, turns: int, policy: dict,
) -> Optional[str]:
    max_cache = int(policy.get("max_cache_read_tokens", 0) or 0)
    max_turns = int(policy.get("max_turns", 0) or 0)
    if max_cache > 0 and cache_read >= max_cache:
        return "cache_tokens"
    if max_turns > 0 and turns >= max_turns:
        return "turns"
    return None


def _settle_memory_before_reset(chat_id: str) -> bool:
    """Synchronously flush recent context into session memory before reset.

    ``maybe_auto_update`` runs in a background daemon thread; without waiting
    on its ``on_done`` callback the subsequent ``_clear_sid`` races the
    summarizer, and a concurrent post-query update (which holds the per-chat
    update lock) makes the forced settle bail out silently with
    ``already_in_progress``. So we:

      1. wait (up to ``_SETTLE_WAIT_SEC``) for the forced update to finish;
      2. on ``already_in_progress``, wait for the holder to release the
         per-chat update lock, then retry exactly once;
      3. give up (logged) if the retry is also crowded out.

    Blocking trade-off: this runs on the query-finalize path
    (token_stats.record_token_usage → maybe_auto_reset → _perform_reset), so
    waiting here blocks that chat's finalize thread for up to roughly
    2 × ``_SETTLE_WAIT_SEC`` plus one lock wait. That is deliberate:
    auto-reset is a low-frequency event (once per ~50 turns / millions of
    cache tokens) and clearing the sid before the summary lands would lose
    the session tail from memory permanently.

    Returns True if a settle attempt ran to completion (success or terminal
    failure), False if it timed out or was crowded out twice.
    """
    from larkhelm.memory import _get_update_lock, maybe_auto_update

    for attempt in (1, 2):
        done = threading.Event()
        outcome: dict = {}

        def _on_done(success, content, error, _done=done, _outcome=outcome):
            _outcome["error"] = error
            _done.set()

        # MEM-C1: this path (token_stats → maybe_auto_reset) has no access to
        # the triggering user's open_id; pass None explicitly so the global
        # cascade falls back to the legacy chain (and skips when unresolved).
        maybe_auto_update(chat_id, force=True, on_done=_on_done,
                          sender_open_id=None)
        if not done.wait(timeout=_SETTLE_WAIT_SEC):
            _debug_log(
                f"[SessionGuard] settle timed out after {_SETTLE_WAIT_SEC}s "
                f"for {chat_id[:8]} (attempt {attempt})"
            )
            return False
        if outcome.get("error") != "already_in_progress":
            return True
        if attempt == 1:
            # A regular post-query update holds the per-chat update lock;
            # wait for it to drain, then retry once.
            lock = _get_update_lock(chat_id)
            if lock.acquire(timeout=_SETTLE_WAIT_SEC):
                lock.release()
    _debug_log(
        f"[SessionGuard] settle skipped for {chat_id[:8]}: "
        "update lock busy after retry"
    )
    return False


def _perform_reset(
    chat_id: str, backend: str, reason: str,
    cache_read: int, turns: int,
) -> None:
    """Best-effort reset: maybe_auto_update → clear sid + counters → milestone + metrics + card."""
    # REQ-19a: settle context into memory before wiping the session.
    # Synchronous (bounded) wait — see _settle_memory_before_reset for the
    # blocking trade-off rationale.
    try:
        _settle_memory_before_reset(chat_id)
    except Exception as e:
        _debug_log(f"[SessionGuard] settle pre-reset failed for {chat_id[:8]}: {e}")

    try:
        _clear_sid(chat_id, backend)
    except Exception as e:
        _debug_log(f"[SessionGuard] _clear_sid failed for {chat_id[:8]}: {e}")

    try:
        _clear_backend_session_counters(chat_id, backend)
    except Exception as e:
        _debug_log(f"[SessionGuard] clear counters failed for {chat_id[:8]}: {e}")

    try:
        from larkhelm.memory import record_milestone
        record_milestone(
            chat_id,
            "session_auto_reset",
            f"Auto reset [{backend}]: {reason} "
            f"(cache_read={cache_read}, turns={turns})",
        )
    except Exception as e:
        _debug_log(f"[SessionGuard] record_milestone failed for {chat_id[:8]}: {e}")

    try:
        from larkhelm.metrics import inc_session_auto_reset
        inc_session_auto_reset(reason)
    except Exception as e:
        _debug_log(f"[SessionGuard] metrics bump failed for {chat_id[:8]}: {e}")

    # REQ-19b: notify user that the session was auto-reset
    try:
        from larkhelm.lark_client import send_card
        send_card(
            chat_id,
            "♻️ 会话已自动重置",
            f"**{backend}** 会话已自动重置（原因：{reason}）。\n\n"
            "对话历史已保存至记忆，您可以继续提问。",
            color="blue",
        )
    except Exception as e:
        _debug_log(f"[SessionGuard] send_card notification failed for {chat_id[:8]}: {e}")

    _debug_log(
        f"[SessionGuard] auto-reset chat={chat_id[:8]} backend={backend} "
        f"reason={reason} cache_read={cache_read} turns={turns}"
    )


def maybe_auto_reset(
    chat_id: str,
    backend: str,
    usage: dict,
) -> Optional[str]:
    """Increment per-backend session counters and trigger auto-reset if threshold crossed.

    backend ∈ {"claude", "gemini", "deepseek", "kimi"}.
    usage keys: cache_read (int), input_tokens (int).
    Returns reset reason ("cache_tokens" | "turns") or None.
    Never raises — token accounting path must not break.
    """
    try:
        if not _guard_enabled():
            return None
        try:
            delta = max(0, int((usage or {}).get("cache_read", 0) or 0))
        except (TypeError, ValueError):
            delta = 0
        cache_read_total, turn_total = _increment_backend_session_counters(
            chat_id, backend, delta,
        )
        policy = _get_policy(backend)
        if not policy:
            return None
        min_turns = max(1, int(getattr(_cfg, "SESSION_GUARD_MIN_TURNS_BEFORE_RESET", 5)))
        if turn_total < min_turns:
            return None
        reason = _check_thresholds(cache_read_total, turn_total, policy)
        if reason is None:
            return None
        with _resetting_lock:
            chats = _resetting_chats_by_backend.setdefault(backend, set())
            if chat_id in chats:
                return None
            chats.add(chat_id)
        try:
            _perform_reset(chat_id, backend, reason, cache_read_total, turn_total)
        finally:
            with _resetting_lock:
                _resetting_chats_by_backend.get(backend, set()).discard(chat_id)
        return reason
    except Exception as e:
        try:
            _debug_log(
                f"[SessionGuard] auto-reset check failed for {chat_id[:8]}: {e}"
            )
        except Exception:
            pass
        return None


def get_session_counters(
    chat_id: str,
    backend: str = "claude",
) -> dict:
    """Return per-backend session counter snapshot.

    Returns: {
        "cache_read": int,
        "turns": int,
        "threshold_cache_read": int,
        "threshold_turns": int,
        "enabled": bool,
    }
    """
    try:
        cache_read, turns = _get_backend_session_counters(chat_id, backend)
    except Exception as e:
        _debug_log(
            f"[SessionGuard] get_session_counters failed for {chat_id[:8]}: {e}"
        )
        cache_read, turns = 0, 0
    policy = _get_policy(backend)
    return {
        "cache_read":           cache_read,
        "turns":                turns,
        "threshold_cache_read": int(policy.get("max_cache_read_tokens", 0) or 0),
        "threshold_turns":      int(policy.get("max_turns", 0) or 0),
        "enabled":              _guard_enabled(),
    }


def clear_session_counters(chat_id: str, backend: str = "claude") -> None:
    """Zero the per-backend session counters. Idempotent. Never raises."""
    try:
        _clear_backend_session_counters(chat_id, backend)
    except Exception as e:
        _debug_log(
            f"[SessionGuard] clear_session_counters failed for {chat_id[:8]}: {e}"
        )


__all__ = [
    "maybe_auto_reset",
    "get_session_counters",
    "clear_session_counters",
]
