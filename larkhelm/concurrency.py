"""larkhelm · per-chat concurrency primitives: locks, cancel events, graceful shutdown, pending message slots, cron lock"""
import threading
import time
from collections import OrderedDict

__all__ = [
    "BTW_TIMEOUT",
    "_chat_locks", "_get_chat_lock",
    "_btw_locks", "_get_btw_lock",
    "_cancel_events", "_cancel_events_ts", "_get_cancel_event", "_trigger_cancel", "_reset_cancel",
    "_replace_cancel_event",
    "_shutting_down", "set_shutting_down", "is_shutting_down", "wait_for_idle",
    "_pending_msg", "_set_pending", "_pop_pending", "_update_pending_card_mid",
    "_cron_lock",
    "get_busy_chat_ids",
]

BTW_TIMEOUT = 120  # /btw quick-answer timeout (seconds)

_LOCK_CACHE_MAX = 500  # LRU cap to prevent _chat_locks / _btw_locks from growing unboundedly

_chat_locks: OrderedDict[str, threading.Lock] = OrderedDict()
_chat_locks_meta = threading.Lock()

_btw_locks: OrderedDict[str, threading.Lock] = OrderedDict()
_btw_locks_meta = threading.Lock()

_cancel_events: dict[str, threading.Event] = {}
_cancel_events_ts: dict[str, float] = {}  # chat_id → last access timestamp
_CANCEL_EVENT_TTL = 3600  # clean up after 1 hour of inactivity to prevent memory growth in long-running processes
_cancel_meta = threading.Lock()

_pending_msg: dict = {}  # chat_id → (message, model, user_msg_id, queue_card_mid)
_pending_meta = threading.Lock()

_cron_lock = threading.Lock()

_shutting_down = False


def _get_chat_lock(chat_id: str) -> threading.Lock:
    with _chat_locks_meta:
        if chat_id in _chat_locks:
            _chat_locks.move_to_end(chat_id)
        else:
            _chat_locks[chat_id] = threading.Lock()
            if len(_chat_locks) > _LOCK_CACHE_MAX:
                _chat_locks.popitem(last=False)
        return _chat_locks[chat_id]


def _get_btw_lock(chat_id: str) -> threading.Lock:
    with _btw_locks_meta:
        if chat_id in _btw_locks:
            _btw_locks.move_to_end(chat_id)
        else:
            _btw_locks[chat_id] = threading.Lock()
            if len(_btw_locks) > _LOCK_CACHE_MAX:
                _btw_locks.popitem(last=False)
        return _btw_locks[chat_id]


def _get_cancel_event(chat_id: str) -> threading.Event:
    with _cancel_meta:
        now = time.time()
        # Evict event objects for chats that have been inactive beyond TTL
        expired = [cid for cid, ts in _cancel_events_ts.items() if now - ts > _CANCEL_EVENT_TTL]
        for cid in expired:
            _cancel_events.pop(cid, None)
            _cancel_events_ts.pop(cid, None)
        if chat_id not in _cancel_events:
            _cancel_events[chat_id] = threading.Event()
        _cancel_events_ts[chat_id] = now
        return _cancel_events[chat_id]


def _trigger_cancel(chat_id: str) -> None:
    _get_cancel_event(chat_id).set()


def _reset_cancel(chat_id: str) -> None:
    _get_cancel_event(chat_id).clear()


def _replace_cancel_event(chat_id: str) -> None:
    """For soft-timeout use only: replace the per-chat cancel event with a fresh object.
    Old task's _watch() thread still holds a reference to the old object and is unaffected.
    New tasks obtain the new object via _get_cancel_event; /cancel only affects the new task.
    """
    with _cancel_meta:
        _cancel_events[chat_id] = threading.Event()
        _cancel_events_ts[chat_id] = time.time()


def set_shutting_down() -> None:
    global _shutting_down
    _shutting_down = True


def is_shutting_down() -> bool:
    return _shutting_down


def wait_for_idle(timeout: float = 120.0) -> bool:
    """Wait until all per-chat locks are released and no AI subprocesses are active, up to timeout seconds.
    Checks both per-chat locks (normal queries) and the global subprocess semaphore (including soft-timeout background processes).
    Returns True if all tasks finished, False on timeout.
    """
    from larkhelm.ai_runner import active_proc_count
    deadline = time.time() + timeout
    while time.time() < deadline:
        with _chat_locks_meta:
            locks = list(_chat_locks.values())
        busy_count = 0
        acquired = []
        for lock in locks:
            if lock.acquire(blocking=False):
                acquired.append(lock)
            else:
                busy_count += 1
        for lock in acquired:
            lock.release()
        if busy_count == 0 and active_proc_count() == 0:
            return True
        time.sleep(1.0)
    return False


def get_busy_chat_ids() -> list[str]:
    """Return chat_ids whose per-chat lock is currently held (non-blocking probe)."""
    with _chat_locks_meta:
        snapshot = list(_chat_locks.items())
    busy: list[str] = []
    for chat_id, lock in snapshot:
        if lock.acquire(blocking=False):
            lock.release()
        else:
            busy.append(chat_id)
    return busy


def _set_pending(chat_id: str, message: str, model: str, user_msg_id: str) -> str:
    """Set a queued message. Replaces any existing queued message and returns the old queue card message_id (for patch)."""
    with _pending_meta:
        existing = _pending_msg.get(chat_id)
        old_mid = existing[3] if existing else None
        _pending_msg[chat_id] = (message, model, user_msg_id, None)
        return old_mid


def _update_pending_card_mid(chat_id: str, mid: str) -> None:
    """Write the queue card message_id into the pending slot for later patching."""
    with _pending_meta:
        if chat_id in _pending_msg:
            msg, model, umid, _ = _pending_msg[chat_id]
            _pending_msg[chat_id] = (msg, model, umid, mid)


def _pop_pending(chat_id: str) -> tuple:
    with _pending_meta:
        return _pending_msg.pop(chat_id, None)
