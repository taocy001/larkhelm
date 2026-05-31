"""larkhelm · Claude session auto-reset guard — thin wrapper (Week-3).

Delegates to ``larkhelm.session_guard`` with ``backend="claude"``.
All callers (``token_stats.record_token_usage``, ``commands._cmd_stats``,
``commands._cmd_reset``) continue to work unchanged.

Original implementation logic has moved to ``session_guard.py``;
only the public API surface is preserved here.
"""
from __future__ import annotations

from typing import Optional

from larkhelm.session_guard import (
    clear_session_counters as _clear_session_counters,
    get_session_counters as _get_session_counters,
    maybe_auto_reset as _maybe_auto_reset,
)


def maybe_auto_reset_session(
    chat_id: str, model: str, usage: dict,
) -> Optional[str]:
    """Increment counters and trigger auto-reset on threshold cross.

    Hook point: ``token_stats.record_token_usage`` calls this at the end of
    the standard accounting path. No-op unless ``model == "claude"`` AND
    ``SESSION_GUARD_ENABLED`` is True. Never raises.

    Returns the reset reason ('cache_tokens' | 'turns') if a reset was
    performed, otherwise None.
    """
    if model != "claude":
        return None
    # Backward-compat: honour legacy per-backend flag set by callers that still
    # use CLAUDE_SESSION_AUTO_RESET_ENABLED rather than SESSION_GUARD_ENABLED.
    try:
        import larkhelm.config as _cfg
        if not getattr(_cfg, "CLAUDE_SESSION_AUTO_RESET_ENABLED", True):
            return None
    except Exception:
        pass
    return _maybe_auto_reset(chat_id, "claude", usage)


def get_session_counters(chat_id: str) -> dict:
    """Snapshot the per-chat Claude session counters for ``/stats`` display.

    Always returns a populated dict (zeros when the chat has no state).
    """
    return _get_session_counters(chat_id, "claude")


def clear_session_counters(chat_id: str) -> None:
    """Zero the per-chat Claude session counters. Idempotent."""
    _clear_session_counters(chat_id, "claude")


__all__ = [
    "maybe_auto_reset_session",
    "get_session_counters",
    "clear_session_counters",
]
