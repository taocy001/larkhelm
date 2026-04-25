import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Dict

class LRUCache:
    """
    A thread-safe LRU cache with TTL support.
    
    Fixed issues:
    1. __contains__ handles value=None correctly.
    2. __len__ is O(1).
    3. capacity < 0 is handled gracefully.
    4. Full type annotations added.
    """
    def __init__(self, capacity: int, ttl: int = 300):
        # Review 3: Handle capacity < 0
        self.capacity: int = max(0, capacity)
        self.ttl: int = ttl
        self.cache: OrderedDict[Any, Any] = OrderedDict()
        self.timestamps: Dict[Any, float] = {}
        self.lock: threading.Lock = threading.Lock()
    
    def get(self, key: Any) -> Optional[Any]:
        # Review 4: Type annotations
        with self.lock:
            if key not in self.cache:
                return None
            # Check expiration
            if time.monotonic() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return None
            self.cache.move_to_end(key)
            return self.cache[key]
    
    def put(self, key: Any, value: Any) -> None:
        # Review 4: Type annotations
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            self.timestamps[key] = time.monotonic()
            
            # Review 3: Ensure capacity constraint
            while len(self.cache) > self.capacity:
                oldest = next(iter(self.cache))
                del self.cache[oldest]
                del self.timestamps[oldest]

    def __contains__(self, key: Any) -> bool:
        # Review 1: Handle value=None correctly by checking presence and TTL, not using get()
        with self.lock:
            if key not in self.cache:
                return False
            if time.monotonic() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                return False
            return True

    def __len__(self) -> int:
        # Review 2: O(1) traversal
        with self.lock:
            return len(self.cache)
