"""larkhelm · thread-safe LRU cache with optional TTL"""
import threading
import time
from collections import OrderedDict
from typing import Any


class LRUCache:
    """Thread-safe LRU cache with optional per-entry TTL.

    Parameters
    ----------
    capacity:
        Maximum number of entries. Values < 0 are clamped to 0 (no storage).
    ttl:
        Optional time-to-live in seconds. Entries older than this are
        treated as absent on the next access.
    """

    def __init__(self, capacity: int, ttl: float = None):
        self.capacity = max(0, capacity)
        self._ttl = ttl
        self._cache: OrderedDict = OrderedDict()  # key → (value, expire_at | None)
        self._lock = threading.Lock()

    def put(self, key: Any, value: Any) -> None:
        if self.capacity == 0:
            return
        expire_at = time.monotonic() + self._ttl if self._ttl is not None else None
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            elif len(self._cache) >= self.capacity:
                self._cache.popitem(last=False)  # evict least-recently-used
            self._cache[key] = (value, expire_at)

    def get(self, key: Any) -> Any:
        with self._lock:
            if key not in self._cache:
                return None
            value, expire_at = self._cache[key]
            if expire_at is not None and time.monotonic() > expire_at:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, key: Any) -> bool:
        with self._lock:
            if key not in self._cache:
                return False
            _, expire_at = self._cache[key]
            if expire_at is not None and time.monotonic() > expire_at:
                del self._cache[key]
                return False
            return True
