"""larkhelm · API session history persistence for Anthropic/Google/OpenAI backends"""
from __future__ import annotations

import json
import os
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log

# Token budget for API session history (rough estimate: 1 token ≈ 3 chars).
# Stays well under typical 200K context windows after adding system prompt + response headroom.
_MAX_HISTORY_TOKENS = 80_000


def _session_file(provider: str, chat_id: str) -> Path:
    sessions_dir = _cfg.DATA_DIR / "api_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir / f"{provider}_{chat_id}.json"


def _estimate_tokens(msg: dict) -> int:
    """Rough token count for one history message (1 token ≈ 3 chars)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return max(1, len(content) // 3)
    if isinstance(content, list):
        chars = sum(
            len(str(p.get("text", "") or p.get("content", "")))
            for p in content if isinstance(p, dict)
        )
        return max(1, chars // 3)
    return 1


def load_history(provider: str, chat_id: str) -> list[dict]:
    """Load API session history. Returns [] on failure (silent fallback)."""
    try:
        f = _session_file(provider, chat_id)
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        _debug_log(f"[ApiSession] load_history failed {provider}/{chat_id}: {e}")
    return []


def save_history(provider: str, chat_id: str, history: list[dict]) -> None:
    """Atomically write history file. Trims to token budget before writing."""
    try:
        history = truncate_history(history)
        f = _session_file(provider, chat_id)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, f)
    except Exception as e:
        _debug_log(f"[ApiSession] save_history failed {provider}/{chat_id}: {e}")


def truncate_history(history: list[dict]) -> list[dict]:
    """Trim oldest messages until estimated token count is within _MAX_HISTORY_TOKENS.

    Always preserves the leading system message (if any) and keeps at least the
    most recent turn (last 2 messages) so the caller always gets a usable history.
    """
    if not history:
        return history
    has_system = history[0].get("role") == "system"
    system = history[:1] if has_system else []
    rest = list(history[1:] if has_system else history)

    while len(rest) > 2:
        total = sum(_estimate_tokens(m) for m in system + rest)
        if total <= _MAX_HISTORY_TOKENS:
            break
        rest.pop(0)

    return system + rest


def clear_history(provider: str, chat_id: str) -> None:
    """Delete session file (missing_ok=True). Called on /reset."""
    try:
        _session_file(provider, chat_id).unlink(missing_ok=True)
    except Exception as e:
        _debug_log(f"[ApiSession] clear_history failed {provider}/{chat_id}: {e}")
