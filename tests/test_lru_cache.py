import time
import threading
import pytest
from larkhelm.lru_cache import LRUCache

def test_basic_ops():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    
    assert cache.get("a") == 1
    assert cache.get("b") == 2
    assert cache.get("c") is None
    assert len(cache) == 2

def test_lru_eviction():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    
    # Access "a" to make "b" the least recently used
    cache.get("a")
    
    # Adding "c" should evict "b"
    cache.put("c", 3)
    
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None
    assert len(cache) == 2

def test_ttl_expiration():
    # Use a short TTL for testing
    cache = LRUCache(capacity=10, ttl=1)
    cache.put("a", 1)
    
    assert cache.get("a") == 1
    
    # Wait for expiration
    time.sleep(1.1)
    
    assert cache.get("a") is None
    assert "a" not in cache
    # Note: __len__ is O(1) and might return expired items until they are accessed or overwritten
    # but in this implementation, __contains__ also cleans up.
    # So if we call "a" not in cache, it should clean it up.
    assert len(cache) == 0

def test_none_value_handling():
    # Review 1: Handle value=None
    cache = LRUCache(capacity=10)
    cache.put("a", None)
    
    # get() should return None (which is the value)
    assert cache.get("a") is None
    # __contains__ should return True because key exists
    assert "a" in cache
    
    # Check with expiration
    cache = LRUCache(capacity=10, ttl=0.1)
    cache.put("b", None)
    time.sleep(0.2)
    assert "b" not in cache
    assert cache.get("b") is None

def test_len_o1():
    # Review 2: __len__ should be O(1)
    cache = LRUCache(capacity=100)
    for i in range(50):
        cache.put(i, i)
    assert len(cache) == 50

def test_capacity_edge_cases():
    # Review 3: capacity < 0
    cache_neg = LRUCache(capacity=-1)
    assert cache_neg.capacity == 0
    cache_neg.put("a", 1)
    assert len(cache_neg) == 0
    assert cache_neg.get("a") is None
    
    cache_zero = LRUCache(capacity=0)
    cache_zero.put("a", 1)
    assert len(cache_zero) == 0
    assert cache_zero.get("a") is None

def test_concurrency():
    cache = LRUCache(capacity=100)
    num_threads = 10
    num_ops = 500
    
    def worker(worker_id):
        for i in range(num_ops):
            key = f"{worker_id}_{i}"
            cache.put(key, i)
            cache.get(key)
            # Occasional check
            _ = key in cache
            
    threads = []
    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    assert len(cache) == 100

def test_update_existing_key():
    cache = LRUCache(capacity=2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("a", 10) # Should update and move to end
    
    cache.put("c", 3) # Should evict "b"
    assert cache.get("b") is None
    assert cache.get("a") == 10
    assert cache.get("c") == 3
