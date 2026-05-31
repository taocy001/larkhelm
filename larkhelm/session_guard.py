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

import json
import threading
import time
from datetime import datetime
from typing import Optional

import larkhelm.config as _cfg
from larkhelm.chat_state import (
    _clear_backend_session_counters,
    _clear_sid,
    _get_backend_session_counters,
    _increment_backend_session_counters,
)
from larkhelm.log import _debug_log
from larkhelm.secure_io import secure_atomic_write

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


def _guard_enabled() -> bool:
    return bool(getattr(_cfg, "SESSION_GUARD_ENABLED", True))


def _checkpoint_enabled() -> bool:
    return bool(getattr(_cfg, "SESSION_GUARD_CHECKPOINT_BEFORE_RESET", True))


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


def _write_anchor(
    chat_id: str, summary: str, backend: str, reason: str,
) -> None:
    try:
        anchor_path = _cfg.SESSION_DIR / f"{chat_id}.anchor.json"
        data = {
            "summary": summary,
            "ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
            "backend": backend,
            "reason": reason,
        }
        secure_atomic_write(anchor_path, json.dumps(data, ensure_ascii=False))
    except Exception as e:
        _debug_log(f"[SessionGuard] _write_anchor failed for {chat_id[:8]}: {e}")


def _perform_reset(
    chat_id: str, backend: str, reason: str,
    cache_read: int, turns: int,
) -> None:
    """Best-effort reset: optional checkpoint → clear sid + counters → milestone + metrics."""
    summary = ""
    if _checkpoint_enabled():
        try:
            from larkhelm.memory import generate_session_checkpoint

            def _gen_checkpoint():
                nonlocal summary
                try:
                    _cp_turns = max(1, int(getattr(_cfg, "SESSION_GUARD_CHECKPOINT_TURNS", 5)))
                    summary = generate_session_checkpoint(chat_id, turns=_cp_turns)
                except Exception as e:
                    _debug_log(
                        f"[SessionGuard] generate_session_checkpoint failed for "
                        f"{chat_id[:8]}: {e}"
                    )

            t = threading.Thread(target=_gen_checkpoint, daemon=True)
            t.start()
            t.join(timeout=60)
        except Exception as e:
            _debug_log(
                f"[SessionGuard] checkpoint thread setup failed for {chat_id[:8]}: {e}"
            )

    if summary:
        _write_anchor(chat_id, summary, backend, reason)

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
        from larkhelm.metrics import inc_session_auto_reset, inc_session_checkpoint
        inc_session_auto_reset(reason)
        inc_session_checkpoint(backend, reason)
    except Exception as e:
        _debug_log(f"[SessionGuard] metrics bump failed for {chat_id[:8]}: {e}")

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
