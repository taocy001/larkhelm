"""larkhelm · persistent chat state (cwd/model/crons), session IDs, btw message ID tracking, pending doc write"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log

__all__ = [
    "_state_lock", "_chat_state_store",
    "_load_global_state", "_save_state", "_get_chat_state", "_set_chat_field",
    "_sid_file", "_load_sid", "_save_sid", "_clear_sid",
    "_get_cwd", "_set_cwd", "_get_chat_model", "_set_chat_model",
    "_register_btw_msg", "_is_btw_reply",
    "set_pending_doc_write", "pop_pending_doc_write",
]

# ═══════════════════════════════════════════════════
#  Persistent state
# ═══════════════════════════════════════════════════
_state_lock = threading.Lock()
_chat_state_store: dict = {}


def _load_global_state() -> None:
    # Update in-place rather than rebinding — ensures all modules that imported _chat_state_store see fresh data
    try:
        data = json.loads(_cfg.STATE_FILE.read_text())
    except Exception:
        data = {}
    with _state_lock:
        _chat_state_store.clear()
        _chat_state_store.update(data)


def _save_state() -> None:
    # Caller must hold _state_lock before calling this function
    try:
        data = json.dumps(_chat_state_store, ensure_ascii=False, indent=2)
        tmp = _cfg.STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, _cfg.STATE_FILE)
    except Exception as e:
        _debug_log(f"[State] 保存失败: {e}")


def _get_chat_state(chat_id: str) -> dict:
    with _state_lock:
        return _chat_state_store.setdefault(chat_id, {})


def _set_chat_field(chat_id: str, key: str, value: object) -> None:
    with _state_lock:
        _chat_state_store.setdefault(chat_id, {})[key] = value
        _save_state()


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
        _sid_file(chat_id, model).write_text(sid)
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
#  btw side-note state
# ═══════════════════════════════════════════════════
_BTW_MSG_ID_CAP = 50  # maximum btw message IDs tracked per chat
_btw_msg_ids: dict[str, set[str]] = {}
_btw_msg_ids_meta = threading.Lock()


def _register_btw_msg(chat_id: str, msg_id: str) -> None:
    """Record a btw reply message ID so that subsequent threaded replies can be identified."""
    if not msg_id:
        return
    with _btw_msg_ids_meta:
        s = _btw_msg_ids.setdefault(chat_id, set())
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
    with _pending_doc_writes_lock:
        _pending_doc_writes[chat_id] = {
            "url":       url,
            "content":   content,
            "ref":       ref,
            "expire_ts": time.time() + 300,
        }


def pop_pending_doc_write(chat_id: str) -> dict | None:
    """Pop and remove the pending write operation; returns None if expired or absent."""
    with _pending_doc_writes_lock:
        entry = _pending_doc_writes.pop(chat_id, None)
    if entry and time.time() <= entry["expire_ts"]:
        return entry
    return None
