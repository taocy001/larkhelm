"""larkhelm · Feishu event deduplication (LRU eviction)"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict

__all__ = ["DEDUP_CAP", "_is_duplicate"]

DEDUP_CAP = 500  # LRU eviction threshold

_seen_event_ids: OrderedDict[str, float] = OrderedDict()  # event_id → timestamp
_seen_msg_ids:   OrderedDict[str, float] = OrderedDict()  # message_id → timestamp
_seen_lock = threading.Lock()


def _is_duplicate(event_id: str, message_id: str = "") -> bool:
    """Dual deduplication on event_id + message_id. LRU eviction when DEDUP_CAP is exceeded.

    Feishu documentation notes that the same message can generate retry events with different event_ids,
    making message_id deduplication more reliable.
    """
    with _seen_lock:
        if event_id in _seen_event_ids:
            return True
        if message_id and message_id in _seen_msg_ids:
            return True
        _seen_event_ids[event_id] = time.time()
        if len(_seen_event_ids) > DEDUP_CAP:
            _seen_event_ids.popitem(last=False)
        if message_id:
            _seen_msg_ids[message_id] = time.time()
            if len(_seen_msg_ids) > DEDUP_CAP:
                _seen_msg_ids.popitem(last=False)
        return False
