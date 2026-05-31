"""larkhelm · persistent chat state (cwd/model/crons/voice_lang per-chat override), session IDs, btw message ID tracking, pending doc write"""
from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.secure_io import secure_atomic_write

if TYPE_CHECKING:
    from larkhelm.agent_hub.intent_types import IntentResult

__all__ = [
    "_state_lock", "_chat_state_store",
    "_load_global_state", "_save_state", "_get_chat_state", "_set_chat_field",
    "_sid_file", "_load_sid", "_save_sid", "_clear_sid",
    "_get_cwd", "_set_cwd", "_get_chat_model", "_set_chat_model",
    "_get_turn_count", "_increment_turn_count",
    "_get_backend_id", "_set_backend_id",
    "_get_voice_lang", "_set_voice_lang",
    "_register_btw_msg", "_is_btw_reply",
    "set_pending_doc_write", "pop_pending_doc_write",
    "_set_pending_intent", "_pop_pending_intent",
    "_get_claude_session_counters",
    "_increment_claude_session_counters",
    "_clear_claude_session_counters",
    "_get_backend_session_counters",
    "_increment_backend_session_counters",
    "_clear_backend_session_counters",
    "_pop_chat_field",
    "_flush_save",
]

# ═══════════════════════════════════════════════════
#  Persistent state
# ═══════════════════════════════════════════════════
_state_lock = threading.RLock()  # RLock allows _set_chat_field to call _save_state while holding the lock
_chat_state_store: dict = {}

# Debounced write: coalesce rapid sequential field updates into a single disk write.
_save_timer: threading.Timer | None = None
_save_timer_lock = threading.Lock()
_SAVE_DEBOUNCE_SEC = 2.0


def _schedule_save() -> None:
    """Schedule a debounced _save_state() (fires _SAVE_DEBOUNCE_SEC after the last mutation)."""
    global _save_timer
    with _save_timer_lock:
        if _save_timer is not None:
            _save_timer.cancel()
        t = threading.Timer(_SAVE_DEBOUNCE_SEC, _save_state)
        t.daemon = True
        _save_timer = t
        t.start()


def _flush_save() -> None:
    """Cancel any pending debounced save and write state to disk immediately.

    Used in tests and upgrade paths that call _load_global_state() right
    after mutations — without this the debounce window would swallow the write.
    Thread-safe: uses _save_timer_lock before touching the timer.
    """
    global _save_timer
    with _save_timer_lock:
        if _save_timer is not None:
            _save_timer.cancel()
            _save_timer = None
    _save_state()


def _load_global_state() -> None:
    # Cancel any pending debounced write — it would overwrite the state we're about to load.
    # Safe: this function is only ever called at bridge startup, before concurrent threads
    # are mutating state, so no new timer can be created between the cancel and the load.
    global _save_timer
    with _save_timer_lock:
        if _save_timer is not None:
            _save_timer.cancel()
            _save_timer = None
    # Update in-place rather than rebinding — ensures all modules that imported _chat_state_store see fresh data
    try:
        data = json.loads(_cfg.STATE_FILE.read_text())
    except Exception:
        data = {}
    with _state_lock:
        _chat_state_store.clear()
        _chat_state_store.update(data)


def _save_state() -> None:
    try:
        with _state_lock:
            data = json.dumps(_chat_state_store, ensure_ascii=False, indent=2)
            _cfg.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            secure_atomic_write(_cfg.STATE_FILE, data)
    except Exception as e:
        _debug_log(f"[State] 保存失败: {e}")


def _get_chat_state(chat_id: str) -> dict:
    with _state_lock:
        return copy.deepcopy(_chat_state_store.setdefault(chat_id, {}))


def _set_chat_field(chat_id: str, key: str, value: object) -> None:
    with _state_lock:
        _chat_state_store.setdefault(chat_id, {})[key] = value
    _schedule_save()


def _pop_chat_field(chat_id: str, key: str, default: object = None) -> object:
    """Atomically read and remove a field from the chat state.

    Combining the read and delete under one lock acquisition prevents the
    TOCTOU race where two concurrent callers both observe a truthy value
    and both proceed to act on it (e.g. two concurrent file uploads both
    claiming the pending_memory_import slot).
    """
    with _state_lock:
        value = _chat_state_store.setdefault(chat_id, {}).pop(key, default)
    if value != default:
        _schedule_save()
    return value


# ═══════════════════════════════════════════════════
#  Session files (sid)
# ═══════════════════════════════════════════════════
def _sid_file(chat_id: str, model: str) -> Path:
    prefix = "" if model == "claude" else f"{model}_"
    return _cfg.SESSION_DIR / f"{prefix}{chat_id}.sid"


def _load_sid(chat_id: str, model: str) -> str | None:
    try:
        v = _sid_file(chat_id, model).read_text().strip()
        return v or None
    except FileNotFoundError:
        return None


def _save_sid(chat_id: str, sid: str, model: str) -> None:
    try:
        _cfg.SESSION_DIR.mkdir(parents=True, exist_ok=True)
        secure_atomic_write(_sid_file(chat_id, model), sid)
    except Exception as e:
        _debug_log(f"[Session] 保存失败: {e}")


def _clear_sid(chat_id: str, model: str) -> None:
    _sid_file(chat_id, model).unlink(missing_ok=True)


# ═══════════════════════════════════════════════════
#  Working directory management
# ═══════════════════════════════════════════════════
def _get_cwd(chat_id: str) -> str:
    cwd = _get_chat_state(chat_id).get("cwd", _cfg.DEFAULT_CWD)
    return cwd if Path(cwd).is_dir() else _cfg.DEFAULT_CWD


def _set_cwd(chat_id: str, path: str) -> None:
    _set_chat_field(chat_id, "cwd", path)


# ═══════════════════════════════════════════════════
#  Model preference
# ═══════════════════════════════════════════════════
def _get_chat_model(chat_id: str) -> str:
    return _get_chat_state(chat_id).get("model", _cfg.DEFAULT_MODEL)


def _set_chat_model(chat_id: str, model: str) -> None:
    _set_chat_field(chat_id, "model", model)


# ═══════════════════════════════════════════════════
#  Turn count (for memory auto-update trigger)
# ═══════════════════════════════════════════════════
def _get_turn_count(chat_id: str) -> int:
    return int(_get_chat_state(chat_id).get("turn_count", 0))


def _increment_turn_count(chat_id: str) -> int:
    with _state_lock:
        # Access _chat_state_store directly to avoid nested _get_chat_state()
        # call inside the same RLock (defensive: keeps correct if lock type changes).
        state = _chat_state_store.setdefault(chat_id, {})
        new_count = state.get("turn_count", 0) + 1
        state["turn_count"] = new_count
    _schedule_save()
    return new_count


# ═══════════════════════════════════════════════════
#  Claude session counters (P0: cache-bleed auto-reset)
# ═══════════════════════════════════════════════════
# Two counters live alongside the normal per-chat state under
# ``_chat_state_store[chat_id]``:
#   * ``claude_session_cache_read`` — accumulated ``usage.cache_read`` tokens
#     since the last reset
#   * ``claude_session_turns``      — count of ``record_token_usage(model="claude")``
#     calls since the last reset
# Missing keys read as 0 so existing .feishu_state.json files are byte-compat
# (NFR-2). See ``larkhelm.claude_session_guard.maybe_auto_reset_session`` for
# the threshold-check + reset side-effect logic.
def _get_claude_session_counters(chat_id: str) -> tuple[int, int]:
    """Return ``(cache_read_total, turn_total)``; defaults to ``(0, 0)``."""
    with _state_lock:
        state = _chat_state_store.get(chat_id) or {}
        cache_read = int(state.get("claude_session_cache_read", 0) or 0)
        turns = int(state.get("claude_session_turns", 0) or 0)
    return cache_read, turns


def _increment_claude_session_counters(
    chat_id: str,
    cache_read_delta: int,
) -> tuple[int, int]:
    """Atomic ``+cache_read_delta`` and ``+1 turn``. Returns new totals."""
    try:
        delta = max(0, int(cache_read_delta or 0))
    except (TypeError, ValueError):
        delta = 0
    with _state_lock:
        state = _chat_state_store.setdefault(chat_id, {})
        new_cache = int(state.get("claude_session_cache_read", 0) or 0) + delta
        new_turns = int(state.get("claude_session_turns", 0) or 0) + 1
        state["claude_session_cache_read"] = new_cache
        state["claude_session_turns"] = new_turns
    _schedule_save()
    return new_cache, new_turns


def _clear_claude_session_counters(chat_id: str) -> None:
    """Zero both Claude session counters; idempotent on fresh chats."""
    with _state_lock:
        state = _chat_state_store.setdefault(chat_id, {})
        state["claude_session_cache_read"] = 0
        state["claude_session_turns"] = 0
    _schedule_save()


# ═══════════════════════════════════════════════════
#  Generic backend session counters (Week-3)
# ═══════════════════════════════════════════════════
# Uses key format "{backend}_session_cache_read" / "{backend}_session_turns".
# When backend="claude" these are identical to the claude-specific keys above,
# so byte-compat with existing state files is preserved (NFR-2).

def _get_backend_session_counters(chat_id: str, backend: str) -> tuple[int, int]:
    """Return (cache_read_total, turn_total); defaults to (0, 0)."""
    cache_key = f"{backend}_session_cache_read"
    turns_key = f"{backend}_session_turns"
    with _state_lock:
        state = _chat_state_store.get(chat_id) or {}
        cache_read = int(state.get(cache_key, 0) or 0)
        turns = int(state.get(turns_key, 0) or 0)
    return cache_read, turns


def _increment_backend_session_counters(
    chat_id: str,
    backend: str,
    cache_read_delta: int,
) -> tuple[int, int]:
    """Atomic +cache_read_delta and +1 turn. Returns new totals."""
    try:
        delta = max(0, int(cache_read_delta or 0))
    except (TypeError, ValueError):
        delta = 0
    cache_key = f"{backend}_session_cache_read"
    turns_key = f"{backend}_session_turns"
    with _state_lock:
        state = _chat_state_store.setdefault(chat_id, {})
        new_cache = int(state.get(cache_key, 0) or 0) + delta
        new_turns = int(state.get(turns_key, 0) or 0) + 1
        state[cache_key] = new_cache
        state[turns_key] = new_turns
    _schedule_save()
    return new_cache, new_turns


def _clear_backend_session_counters(chat_id: str, backend: str) -> None:
    """Zero both session counters for the specified backend. Idempotent."""
    cache_key = f"{backend}_session_cache_read"
    turns_key = f"{backend}_session_turns"
    with _state_lock:
        state = _chat_state_store.setdefault(chat_id, {})
        state[cache_key] = 0
        state[turns_key] = 0
    _schedule_save()


# ═══════════════════════════════════════════════════
#  Backend preference (backend_id field)
# ═══════════════════════════════════════════════════
def _get_backend_id(chat_id: str) -> str | None:
    return _get_chat_state(chat_id).get("backend_id")


def _set_backend_id(chat_id: str, backend_id: str) -> None:
    _set_chat_field(chat_id, "backend_id", backend_id)


# ═══════════════════════════════════════════════════
#  Voice language preference (per-chat override of VOICE_DEFAULT_LANG)
# ═══════════════════════════════════════════════════
def _get_voice_lang(chat_id: str) -> str:
    return _get_chat_state(chat_id).get("voice_lang", _cfg.VOICE_DEFAULT_LANG)


def _set_voice_lang(chat_id: str, lang: str) -> None:
    _set_chat_field(chat_id, "voice_lang", lang)


# ═══════════════════════════════════════════════════
#  btw side-note state
# ═══════════════════════════════════════════════════
_BTW_MSG_ID_CAP = 50     # maximum btw message IDs tracked per chat
_BTW_CHAT_ID_CAP = 500   # maximum distinct chat_ids tracked (outer dict LRU cap)
_btw_msg_ids: dict[str, set[str]] = {}
_btw_msg_ids_meta = threading.Lock()


def _register_btw_msg(chat_id: str, msg_id: str) -> None:
    """Record a btw reply message ID so that subsequent threaded replies can be identified."""
    if not msg_id:
        return
    with _btw_msg_ids_meta:
        if chat_id not in _btw_msg_ids:
            # Evict oldest entry when outer dict exceeds cap
            if len(_btw_msg_ids) >= _BTW_CHAT_ID_CAP:
                _btw_msg_ids.pop(next(iter(_btw_msg_ids)), None)
            _btw_msg_ids[chat_id] = set()
        s = _btw_msg_ids[chat_id]
        s.add(msg_id)
        if len(s) > _BTW_MSG_ID_CAP:
            remove = list(s)[:len(s) // 2]
            for m in remove:
                s.discard(m)


def _is_btw_reply(chat_id: str, parent_id: str | None) -> bool:
    """Return True if the message's parent_id points to a known btw reply."""
    if not parent_id:
        return False
    with _btw_msg_ids_meta:
        return parent_id in _btw_msg_ids.get(chat_id, set())


# ═══════════════════════════════════════════════════
#  /doc write pending confirmation staging area
# ═══════════════════════════════════════════════════
_pending_doc_writes: dict[str, dict] = {}  # chat_id → {url, content, ref, expire_ts}
_pending_doc_writes_lock = threading.Lock()


def set_pending_doc_write(chat_id: str, url: str, content: str, ref: object) -> None:
    """Stage a document write operation that awaits user confirmation (5-minute TTL)."""
    now = time.time()
    with _pending_doc_writes_lock:
        # Evict any expired entries to prevent unbounded accumulation of doc content strings
        expired = [k for k, v in _pending_doc_writes.items() if v["expire_ts"] < now]
        for k in expired:
            del _pending_doc_writes[k]
        _pending_doc_writes[chat_id] = {
            "url":       url,
            "content":   content,
            "ref":       ref,
            "expire_ts": now + 300,
        }


def pop_pending_doc_write(chat_id: str) -> dict | None:
    """Pop and remove the pending write operation; returns None if expired or absent."""
    with _pending_doc_writes_lock:
        entry = _pending_doc_writes.pop(chat_id, None)
    if entry and time.time() <= entry["expire_ts"]:
        return entry
    return None


# ═══════════════════════════════════════════════════
#  Pending intent — per-chat IntentResult handoff
# ═══════════════════════════════════════════════════
# Phase D: ``handlers/_message.py`` resolves an IntentResult before deciding
# whether to dispatch via AgentDispatcher or fall through to ``_do_query`` for
# the chat agent_type. When falling through, ``_do_query`` needs the same
# IntentResult so that ``get_memory_context_v2`` can apply the per-agent
# memory injection policy. The dict below is the handoff channel: set by the
# upstream resolver, popped (and cleared) by ``_do_query``. Pop semantics
# avoid leaking an old intent into the next user turn.
_pending_intents: dict[str, "IntentResult"] = {}
_pending_intents_lock = threading.Lock()


def _set_pending_intent(chat_id: str, intent: "IntentResult | None") -> None:
    """Stage the resolved IntentResult for the next ``_do_query`` on this chat.

    No-op when ``intent`` is None. Thread-safe via a dedicated lock so it
    does not contend with the persistent ``_state_lock``."""
    if intent is None:
        return
    with _pending_intents_lock:
        _pending_intents[chat_id] = intent


def _pop_pending_intent(chat_id: str) -> "IntentResult | None":
    """Atomically pop and return the staged intent; returns None when absent."""
    with _pending_intents_lock:
        return _pending_intents.pop(chat_id, None)
