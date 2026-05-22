"""larkhelm · Feishu event deduplication (LRU eviction + disk persistence)

On startup the last known event/message IDs are loaded from DATA_DIR/.feishu_dedup.json
so that events seen before a restart are not reprocessed. State is flushed to disk at
most once per _SAVE_INTERVAL seconds (debounced) to bound disk I/O.

Only entries younger than _PERSIST_TTL seconds are restored on load — entries older
than 24h are beyond Feishu's retry window and safe to discard.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict

__all__ = ["DEDUP_CAP", "_is_duplicate"]

DEDUP_CAP = 500           # LRU eviction threshold
_SAVE_INTERVAL = 60.0     # debounce: flush at most once per minute
_PERSIST_TTL = 86400      # discard entries older than 24h on load

_seen_event_ids: OrderedDict[str, float] = OrderedDict()  # event_id → timestamp
_seen_msg_ids:   OrderedDict[str, float] = OrderedDict()  # message_id → timestamp
_seen_lock = threading.Lock()

_loaded = False
_save_timer: threading.Timer | None = None
_save_timer_lock = threading.Lock()


def _dedup_path():
    try:
        import larkhelm.config as _cfg
        return _cfg.DATA_DIR / ".feishu_dedup.json"
    except Exception:
        return None


def _load_from_disk() -> None:
    """Load persisted dedup IDs from disk (called once before first check)."""
    p = _dedup_path()
    if p is None or not p.exists():
        return
    try:
        data = json.loads(p.read_text())
        cutoff = time.time() - _PERSIST_TTL
        # Sort by timestamp ascending so OrderedDict LRU order reflects actual age.
        event_items = sorted(
            ((k, v) for k, v in data.get("event_ids", {}).items() if v > cutoff),
            key=lambda x: x[1],
        )
        msg_items = sorted(
            ((k, v) for k, v in data.get("msg_ids", {}).items() if v > cutoff),
            key=lambda x: x[1],
        )
        with _seen_lock:
            for eid, ts in event_items:
                _seen_event_ids[eid] = ts
            for mid, ts in msg_items:
                _seen_msg_ids[mid] = ts
    except Exception:
        pass


def _save_to_disk() -> None:
    """Write current dedup state to disk; invoked by the debounce timer."""
    global _save_timer
    with _save_timer_lock:
        _save_timer = None
    p = _dedup_path()
    if p is None:
        return
    try:
        with _seen_lock:
            data = {
                "event_ids": dict(_seen_event_ids),
                "msg_ids": dict(_seen_msg_ids),
            }
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, ensure_ascii=False))
    except Exception:
        pass


def _schedule_save() -> None:
    """Schedule a debounced disk flush (no-op if one is already pending)."""
    global _save_timer
    with _save_timer_lock:
        if _save_timer is not None:
            return
        t = threading.Timer(_SAVE_INTERVAL, _save_to_disk)
        t.daemon = True
        _save_timer = t
        t.start()


def _is_duplicate(event_id: str, message_id: str = "") -> bool:
    """Dual deduplication on event_id + message_id. LRU eviction when DEDUP_CAP is exceeded.

    Feishu documentation notes that the same message can generate retry events with different event_ids,
    making message_id deduplication more reliable.
    """
    global _loaded
    if not _loaded:
        _load_from_disk()
        _loaded = True

    is_dup = False
    with _seen_lock:
        if event_id in _seen_event_ids:
            is_dup = True
        elif message_id and message_id in _seen_msg_ids:
            is_dup = True
        else:
            _seen_event_ids[event_id] = time.time()
            if len(_seen_event_ids) > DEDUP_CAP:
                _seen_event_ids.popitem(last=False)
            if message_id:
                _seen_msg_ids[message_id] = time.time()
                if len(_seen_msg_ids) > DEDUP_CAP:
                    _seen_msg_ids.popitem(last=False)

    if not is_dup:
        _schedule_save()
    return is_dup
