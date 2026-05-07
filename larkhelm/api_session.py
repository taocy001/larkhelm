"""larkhelm · API session history persistence for Anthropic/Google/OpenAI backends"""
from __future__ import annotations

import json
import os
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log

_MAX_HISTORY = 40


def _session_file(provider: str, chat_id: str) -> Path:
    sessions_dir = _cfg.DATA_DIR / "api_sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    return sessions_dir / f"{provider}_{chat_id}.json"


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
    """Atomically write history file. Truncates if over _MAX_HISTORY entries."""
    try:
        if len(history) > _MAX_HISTORY:
            history = truncate_history(history)
        f = _session_file(provider, chat_id)
        tmp = f.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, f)
    except Exception as e:
        _debug_log(f"[ApiSession] save_history failed {provider}/{chat_id}: {e}")


def truncate_history(history: list[dict]) -> list[dict]:
    """Keep first system message (if any); remove oldest entries until len <= _MAX_HISTORY."""
    if not history:
        return history
    if history[0].get("role") == "system":
        system = [history[0]]
        rest = history[1:]
        while len(system) + len(rest) > _MAX_HISTORY:
            rest = rest[1:]
        return system + rest
    return history[-_MAX_HISTORY:]


def clear_history(provider: str, chat_id: str) -> None:
    """Delete session file (missing_ok=True). Called on /reset."""
    try:
        _session_file(provider, chat_id).unlink(missing_ok=True)
    except Exception as e:
        _debug_log(f"[ApiSession] clear_history failed {provider}/{chat_id}: {e}")
