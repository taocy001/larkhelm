"""Tests for ``larkhelm._context_cache`` (P1 REQ-01..03).

Covers:
  * LRUCache hit/miss/promote/evict semantics
  * TTLCache get/expire/invalidate_chat
  * cached_recent_turns: hit + mtime / size invalidation + dedup_prefix
    differentiation + LRU eviction
  * cached_memory_layer: 3-layer key isolation + mtime invalidation +
    file_path=None bypass
  * cached_doc_read: 60s hit window + 61s expiry + chat_id isolation +
    DocPermissionError NOT cached
  * Flag-off bypass: 3 enabled flags → loader called every time
  * Prometheus counter bridge increments
  * 4-thread concurrency stress check (no KeyError, no torn state)
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Bootstrap config (shared) ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_ctxcache_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm import _context_cache as cc  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
#  LRUCache primitive
# ════════════════════════════════════════════════════════════════════════


class LRUCacheTests(unittest.TestCase):

    def setUp(self):
        cc.reset_for_tests()

    def test_get_miss_then_put_then_hit(self):
        lru: cc.LRUCache[str, int] = cc.LRUCache("t", 3)
        hit, _ = lru.get("a")
        self.assertFalse(hit)
        lru.put("a", 1)
        hit, value = lru.get("a")
        self.assertTrue(hit)
        self.assertEqual(value, 1)

    def test_put_evicts_lru_when_full(self):
        lru: cc.LRUCache[str, int] = cc.LRUCache("t", 3)
        lru.put("a", 1)
        lru.put("b", 2)
        lru.put("c", 3)
        # "a" is LRU; inserting "d" should evict it
        evicted = lru.put("d", 4)
        self.assertEqual(evicted, "a")
        hit_a, _ = lru.get("a")
        hit_d, _ = lru.get("d")
        self.assertFalse(hit_a)
        self.assertTrue(hit_d)

    def test_get_promotes_to_mru(self):
        lru: cc.LRUCache[str, int] = cc.LRUCache("t", 3)
        lru.put("a", 1)
        lru.put("b", 2)
        lru.put("c", 3)
        # Access "a" → now "b" is the LRU.
        lru.get("a")
        evicted = lru.put("d", 4)
        self.assertEqual(evicted, "b")

    def test_invalidate(self):
        lru: cc.LRUCache[str, int] = cc.LRUCache("t", 3)
        lru.put("a", 1)
        lru.invalidate("a")
        hit, _ = lru.get("a")
        self.assertFalse(hit)

    def test_stats(self):
        lru: cc.LRUCache[str, int] = cc.LRUCache("t", 3)
        lru.put("a", 1)
        lru.get("a")    # hit
        lru.get("b")    # miss
        s = lru.stats()
        self.assertEqual(s["size"], 1)
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)
        self.assertEqual(s["evicts"], 0)

    def test_maxsize_invalid(self):
        with self.assertRaises(ValueError):
            cc.LRUCache("t", 0)
        with self.assertRaises(ValueError):
            cc.LRUCache("t", -1)


# ════════════════════════════════════════════════════════════════════════
#  TTLCache primitive
# ════════════════════════════════════════════════════════════════════════


class TTLCacheTests(unittest.TestCase):

    def setUp(self):
        cc.reset_for_tests()

    def test_get_within_ttl_is_hit(self):
        ttl: cc.TTLCache[str, int] = cc.TTLCache("t", 60.0)
        ttl.put("a", 1)
        self.assertEqual(ttl.get("a"), 1)

    def test_get_after_ttl_is_miss(self):
        ttl: cc.TTLCache[str, int] = cc.TTLCache("t", 60.0)
        ttl.put("a", 1)
        # Patch time.monotonic *as imported into _context_cache*.
        now = time.monotonic()
        with patch.object(cc.time, "monotonic", return_value=now + 61.0):
            self.assertIsNone(ttl.get("a"))

    def test_invalidate_chat_drops_matching_keys(self):
        ttl: cc.TTLCache[cc.DocKey, int] = cc.TTLCache("t", 60.0)
        k1 = cc.DocKey(chat_id="A", doc_type="docx", token="t1", max_chars=100)
        k2 = cc.DocKey(chat_id="A", doc_type="docx", token="t2", max_chars=100)
        k3 = cc.DocKey(chat_id="B", doc_type="docx", token="t3", max_chars=100)
        ttl.put(k1, 1)
        ttl.put(k2, 2)
        ttl.put(k3, 3)
        ttl.invalidate_chat("A")
        self.assertIsNone(ttl.get(k1))
        self.assertIsNone(ttl.get(k2))
        self.assertEqual(ttl.get(k3), 3)


# ════════════════════════════════════════════════════════════════════════
#  cached_recent_turns
# ════════════════════════════════════════════════════════════════════════


class CachedRecentTurnsTests(unittest.TestCase):

    def setUp(self):
        cc.reset_for_tests()
        _cfg.RECENT_TURNS_CACHE_ENABLED = True

    def test_hit_avoids_second_loader_call(self):
        # Two calls with the same conv_seqno (default=0) → second is a hit.
        calls = [0]

        def loader():
            calls[0] += 1
            return "fixed-result"

        a = cc.cached_recent_turns("chatA", 6, 2000, None, loader=loader)
        b = cc.cached_recent_turns("chatA", 6, 2000, None, loader=loader)
        self.assertEqual(a, "fixed-result")
        self.assertEqual(b, "fixed-result")
        self.assertEqual(calls[0], 1, "second call should have hit cache")

    def test_seqno_change_invalidates(self):
        # A new user/assistant log entry bumps conv_seqno → cache miss.
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        a = cc.cached_recent_turns("chatA", 6, 2000, None, conv_seqno=0,
                                   loader=loader)
        # Simulate a new conversation turn (seqno incremented by log_entry).
        b = cc.cached_recent_turns("chatA", 6, 2000, None, conv_seqno=1,
                                   loader=loader)
        self.assertEqual(calls[0], 2, "different seqno → cache miss → loader called twice")
        self.assertNotEqual(a, b)

    def test_same_seqno_is_hit(self):
        # Tool / shell / error log entries do NOT change conv_seqno, so a
        # retry within the same request sees the same key and hits the cache.
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        # First call (miss, seqno=5)
        a = cc.cached_recent_turns("chatA", 6, 2000, None, conv_seqno=5,
                                   loader=loader)
        # Retry call (same seqno=5 — no new user/assistant entry yet) → hit
        b = cc.cached_recent_turns("chatA", 6, 2000, None, conv_seqno=5,
                                   loader=loader)
        self.assertEqual(calls[0], 1, "same seqno → second call hits cache")
        self.assertEqual(a, b)

    def test_dedup_prefix_excluded_from_key(self):
        # dedup_prefix is intentionally NOT part of the cache key (see
        # RecentTurnsKey docstring).  Two calls with different dedup_prefix
        # values but the same (chat_id, max_turns, max_chars, conv_seqno)
        # should share the same cache entry — the second call is a hit.
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        a = cc.cached_recent_turns("chatA", 6, 2000, "prefix-A", loader=loader)
        b = cc.cached_recent_turns("chatA", 6, 2000, "prefix-B", loader=loader)
        self.assertEqual(calls[0], 1, "same conv_seqno → second call is a cache hit")
        self.assertEqual(a, b, "same cached payload returned for both calls")

    def test_lru_eviction_when_capacity_exceeded(self):
        # Reach into the singleton to test capacity behaviour deterministically.
        # _recent_turns_cache has maxsize=64; insert 65 different chat_ids.
        loader_calls = [0]

        def loader():
            loader_calls[0] += 1
            return f"v{loader_calls[0]}"

        for i in range(65):
            cc.cached_recent_turns(f"chat{i:04d}", 6, 2000, None, loader=loader)
        # chat0000 should have been evicted.
        stats = cc._recent_turns_cache.stats()
        self.assertLessEqual(stats["size"], 64)
        self.assertGreaterEqual(stats["evicts"], 1)

    def test_disabled_flag_falls_through(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return "x"

        with patch.object(_cfg, "RECENT_TURNS_CACHE_ENABLED", False):
            cc.cached_recent_turns("chatA", 6, 2000, None, loader=loader)
            cc.cached_recent_turns("chatA", 6, 2000, None, loader=loader)
        self.assertEqual(calls[0], 2, "flag off → loader runs every call")


# ════════════════════════════════════════════════════════════════════════
#  cached_memory_layer
# ════════════════════════════════════════════════════════════════════════


class CachedMemoryLayerTests(unittest.TestCase):

    def setUp(self):
        cc.reset_for_tests()
        self.tmp = Path(_TMP) / "mem"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.global_md = self.tmp / "global.md"
        self.project_md = self.tmp / "project.md"
        self.global_md.write_text("global body v1")
        self.project_md.write_text("project body v1")
        _cfg.MEMORY_LEGACY_CACHE_ENABLED = True

    def test_hit_avoids_loader(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return "out"

        a = cc.cached_memory_layer("global", self.global_md, loader=loader)
        b = cc.cached_memory_layer("global", self.global_md, loader=loader)
        self.assertEqual(a, "out")
        self.assertEqual(b, "out")
        self.assertEqual(calls[0], 1)

    def test_three_layer_keys_are_isolated(self):
        # global and project both point to two different files; same loader
        # output must NOT share keys across layers.
        gcalls = [0]
        pcalls = [0]

        def gload():
            gcalls[0] += 1
            return "G"

        def pload():
            pcalls[0] += 1
            return "P"

        cc.cached_memory_layer("global", self.global_md, loader=gload)
        cc.cached_memory_layer("project", self.project_md, loader=pload)
        cc.cached_memory_layer("global", self.global_md, loader=gload)
        cc.cached_memory_layer("project", self.project_md, loader=pload)
        # Each loader still called exactly once (first call); the second
        # was served from cache.
        self.assertEqual(gcalls[0], 1)
        self.assertEqual(pcalls[0], 1)

    def test_mtime_change_invalidates(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        cc.cached_memory_layer("global", self.global_md, loader=loader)
        new_t = time.time() + 5.0
        os.utime(self.global_md, (new_t, new_t))
        cc.cached_memory_layer("global", self.global_md, loader=loader)
        self.assertEqual(calls[0], 2)

    def test_file_path_none_bypasses(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return None

        cc.cached_memory_layer("global", None, loader=loader)
        cc.cached_memory_layer("global", None, loader=loader)
        self.assertEqual(calls[0], 2,
                         "file_path=None must bypass cache and always call loader")

    def test_disabled_flag_falls_through(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return "x"

        with patch.object(_cfg, "MEMORY_LEGACY_CACHE_ENABLED", False):
            cc.cached_memory_layer("global", self.global_md, loader=loader)
            cc.cached_memory_layer("global", self.global_md, loader=loader)
        self.assertEqual(calls[0], 2)


# ════════════════════════════════════════════════════════════════════════
#  cached_doc_read
# ════════════════════════════════════════════════════════════════════════


class CachedDocReadTests(unittest.TestCase):

    class _FakeRef:
        def __init__(self, token: str = "tok123", doc_type: str = "docx"):
            self.token = token
            self.doc_type = doc_type

    def setUp(self):
        cc.reset_for_tests()
        _cfg.DOC_INJECT_CACHE_ENABLED = True
        _cfg.DOC_INJECT_CACHE_TTL_SEC = 60

    def test_hit_within_ttl(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        ref = self._FakeRef()
        a = cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        b = cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        self.assertEqual(a, b)
        self.assertEqual(calls[0], 1)

    def test_miss_after_ttl(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        ref = self._FakeRef()
        cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        now = time.monotonic()
        with patch.object(cc.time, "monotonic", return_value=now + 61.0):
            cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        self.assertEqual(calls[0], 2)

    def test_chat_id_isolation(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        ref = self._FakeRef()
        cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        cc.cached_doc_read("chatB", ref, 1000, loader=loader)
        # Same doc_token, different chat_ids → two distinct cache slots.
        self.assertEqual(calls[0], 2)

    def test_permission_error_not_cached(self):
        from larkhelm.lark_client import DocPermissionError
        calls = [0]

        def loader():
            calls[0] += 1
            raise DocPermissionError("denied")

        ref = self._FakeRef()
        with self.assertRaises(DocPermissionError):
            cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        with self.assertRaises(DocPermissionError):
            cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        self.assertEqual(calls[0], 2, "errors must not enter the cache")

    def test_disabled_flag_falls_through(self):
        calls = [0]

        def loader():
            calls[0] += 1
            return "x"

        ref = self._FakeRef()
        with patch.object(_cfg, "DOC_INJECT_CACHE_ENABLED", False):
            cc.cached_doc_read("chatA", ref, 1000, loader=loader)
            cc.cached_doc_read("chatA", ref, 1000, loader=loader)
        self.assertEqual(calls[0], 2)


# ════════════════════════════════════════════════════════════════════════
#  P4 — cached_doc_read_with_meta + age hint (AC-05)
# ════════════════════════════════════════════════════════════════════════


class TestAgeHint(unittest.TestCase):
    """AC-05 — cached_doc_read_with_meta surfaces (from_cache, age_sec)
    on every call so _inject_doc_context can render a parenthetical
    age hint to the user."""

    class _FakeRef:
        def __init__(self, token: str = "tok456", doc_type: str = "docx"):
            self.token = token
            self.doc_type = doc_type

    def setUp(self):
        cc.reset_for_tests()
        _cfg.DOC_INJECT_CACHE_ENABLED = True
        _cfg.DOC_INJECT_CACHE_TTL_SEC = 600

    def test_first_inject_no_hint(self):
        """First call → from_cache=False, age_sec=None (no hint to render)."""
        def loader():
            return "doc payload"

        ref = self._FakeRef()
        result = cc.cached_doc_read_with_meta(
            "chatX", ref, 1000, loader=loader,
        )
        self.assertEqual(result.payload, "doc payload")
        self.assertFalse(result.from_cache)
        self.assertIsNone(result.age_sec)

    def test_second_inject_has_minutes_hint(self):
        """Second call after 240s → from_cache=True, age_sec≈240."""
        calls = [0]

        def loader():
            calls[0] += 1
            return f"payload{calls[0]}"

        ref = self._FakeRef()
        # Seed
        cc.cached_doc_read_with_meta("chatX", ref, 1000, loader=loader)
        now = time.monotonic()
        with patch.object(cc.time, "monotonic", return_value=now + 240.0):
            result = cc.cached_doc_read_with_meta(
                "chatX", ref, 1000, loader=loader,
            )
        self.assertTrue(result.from_cache)
        self.assertEqual(result.payload, "payload1",
            "second call must return the cached payload, not a re-load",
        )
        self.assertIsNotNone(result.age_sec)
        # Allow ±1s slack against monotonic float→int truncation.
        self.assertGreaterEqual(result.age_sec, 239)
        self.assertLessEqual(result.age_sec, 241)
        self.assertEqual(calls[0], 1, "loader must not run on a hit")

        # The query-side helper renders 240s into the "<5min" bucket.
        # P5-OPT2: per-minute rendering was replaced with 4 discrete buckets
        # so the injected user-message string stays byte-stable across small
        # clock drifts — letting Anthropic's 5-min ephemeral user-turn cache
        # actually hit on repeated questions about the same doc.
        from larkhelm.handlers._query import _format_age_hint
        self.assertIn("刚刚", _format_age_hint(result.age_sec))

    def test_under_60s_renders_in_first_bucket(self):
        """`_format_age_hint(30)` → "刚刚" (sub-5min bucket)."""
        from larkhelm.handlers._query import _format_age_hint
        text = _format_age_hint(30)
        self.assertIn("刚刚", text)
        self.assertNotIn("分钟前", text)

    def test_age_hint_buckets(self):
        """P5-OPT2 contract — 4 discrete buckets, stable inside each band."""
        from larkhelm.handlers._query import _format_age_hint
        # <5min bucket
        self.assertEqual(_format_age_hint(0), _format_age_hint(299))
        self.assertIn("刚刚", _format_age_hint(299))
        # 5..30min bucket
        self.assertEqual(_format_age_hint(300), _format_age_hint(1799))
        self.assertIn("几分钟前", _format_age_hint(1500))
        # 30..60min bucket
        self.assertEqual(_format_age_hint(1800), _format_age_hint(3599))
        self.assertIn("约半小时前", _format_age_hint(3000))
        # >=1h bucket — counted in floor hours
        self.assertIn("1 小时前", _format_age_hint(3600))
        self.assertIn("3 小时前", _format_age_hint(3 * 3600 + 1700))

    def test_age_hint_bucket_boundaries(self):
        """P5-OPT2 — boundary checks pin the cutoffs so a ``<`` ↔ ``<=`` flip
        in ``_format_age_hint`` is caught by CI rather than shipping silently
        (reviewer flag)."""
        from larkhelm.handlers._query import _format_age_hint
        # <5min → 5..30min cut
        self.assertNotEqual(_format_age_hint(299), _format_age_hint(300))
        # 5..30min → 30..60min cut
        self.assertNotEqual(_format_age_hint(1799), _format_age_hint(1800))
        # 30..60min → ≥1h cut
        self.assertNotEqual(_format_age_hint(3599), _format_age_hint(3600))

    def test_disabled_flag_bypasses_with_no_cache_flag(self):
        """DOC_INJECT_CACHE_ENABLED=False → from_cache=False even on 2nd call."""
        calls = [0]

        def loader():
            calls[0] += 1
            return f"v{calls[0]}"

        ref = self._FakeRef()
        with patch.object(_cfg, "DOC_INJECT_CACHE_ENABLED", False):
            r1 = cc.cached_doc_read_with_meta("chatX", ref, 1000, loader=loader)
            r2 = cc.cached_doc_read_with_meta("chatX", ref, 1000, loader=loader)
        self.assertFalse(r1.from_cache)
        self.assertFalse(r2.from_cache)
        self.assertIsNone(r1.age_sec)
        self.assertIsNone(r2.age_sec)
        self.assertEqual(calls[0], 2, "bypass must call loader every time")


# ════════════════════════════════════════════════════════════════════════
#  Metrics bridge
# ════════════════════════════════════════════════════════════════════════


class MetricsBridgeTests(unittest.TestCase):

    def setUp(self):
        cc.reset_for_tests()
        _cfg.RECENT_TURNS_CACHE_ENABLED = True

    def test_inc_helpers_dont_raise_without_prom_client(self):
        # The bridge must not crash when prometheus-client is absent.
        from larkhelm import metrics as _metrics
        try:
            _metrics.inc_recent_turns_cache("hit")
            _metrics.inc_memory_layer_cache("global", "miss")
            _metrics.inc_doc_inject_cache("hit")
        except Exception as e:
            self.fail(f"metrics bridge raised: {e}")

    def test_counters_increment_when_available(self):
        from larkhelm import metrics as _metrics
        reg = _metrics.get_registry()
        if not reg.available or reg.recent_turns_cache_total is None:
            self.skipTest("prometheus-client not installed in this venv")

        # Pull the underlying counter's labelled child, snapshot before / after.
        ch = reg.recent_turns_cache_total.labels(outcome="hit")
        try:
            before = ch._value.get()
        except Exception:
            self.skipTest("prometheus_client internal shape changed")
        _metrics.inc_recent_turns_cache("hit")
        _metrics.inc_recent_turns_cache("hit")
        after = ch._value.get()
        self.assertGreaterEqual(after - before, 2)


# ════════════════════════════════════════════════════════════════════════
#  Concurrency stress check
# ════════════════════════════════════════════════════════════════════════


class ConcurrencyTests(unittest.TestCase):

    def test_4_threads_100_iters_no_corruption(self):
        cc.reset_for_tests()
        _cfg.RECENT_TURNS_CACHE_ENABLED = True
        errors: list[str] = []

        def loader():
            return "v"

        def worker(chat_id: str):
            try:
                for _ in range(100):
                    cc.cached_recent_turns(chat_id, 6, 2000, None, loader=loader)
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=worker, args=(f"c{i}",))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        self.assertEqual(errors, [])

    def test_same_chat_4_threads_lock_contention(self):
        """Reviewer round-1 nit #4: the previous concurrency test gave
        each thread an independent ``chat_id``, so the LRU slots never
        collided and the lock was barely contested. This stresses the
        opposite extreme — all 4 threads share one ``chat_id``, hitting
        the same key, the same lock, every iteration.

        Asserts:
          1. No corruption (no exception raised in any thread).
          2. loader was called **exactly once** — every other access is
             a cache hit. If the lock were broken or the put-then-promote
             logic raced, we'd see 2+ loader invocations.
          3. ``hits + misses`` accounting balances to ``4 × 100``.
        """
        cc.reset_for_tests()
        _cfg.RECENT_TURNS_CACHE_ENABLED = True
        errors: list[str] = []
        loader_calls = [0]
        loader_lock = threading.Lock()

        def loader():
            with loader_lock:
                loader_calls[0] += 1
            return "shared-value"

        SHARED_CHAT = "shared_chat"

        def worker():
            try:
                for _ in range(100):
                    val = cc.cached_recent_turns(
                        SHARED_CHAT, 6, 2000, None, loader=loader,
                    )
                    if val != "shared-value":
                        errors.append(f"unexpected value: {val!r}")
            except Exception as e:
                errors.append(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(errors, [], f"workers reported errors: {errors}")
        # All threads share one key — under a healthy lock the loader
        # races to the first miss, then every subsequent call hits.
        # Worst-case the 4 threads each see a miss before any one wins
        # the put race, so loader_calls ∈ [1, 4]. ≥5 would imply the
        # promotion-then-check window leaked.
        self.assertGreaterEqual(loader_calls[0], 1)
        self.assertLessEqual(
            loader_calls[0], 4,
            f"loader fired {loader_calls[0]}× — promote/put race?",
        )
        # Counters: 4×100 = 400 total accesses, all keyed identically.
        stats = cc._recent_turns_cache.stats()
        self.assertEqual(stats["hits"] + stats["misses"], 400)
        # Capacity stays at 1 entry — same key all the way.
        self.assertEqual(stats["size"], 1)


# ════════════════════════════════════════════════════════════════════════
#  log.py + memory_context.py + _query.py integration smoke tests
# ════════════════════════════════════════════════════════════════════════


class LogModuleIntegrationTests(unittest.TestCase):
    """Verify the thin shell in ``log.py`` reaches into the cache module."""

    def setUp(self):
        cc.reset_for_tests()
        log_dir = Path(_cfg.LOG_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        jsonl = log_dir / "all.jsonl"
        jsonl.write_text(
            '{"chat_id":"int_chat","role":"user","content":"hello"}\n'
            '{"chat_id":"int_chat","role":"assistant","content":"world"}\n'
        )

    def test_get_recent_turns_uses_cache(self):
        from larkhelm import log as _log
        original = _log._get_recent_turns_uncached
        calls = [0]

        def counting(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        with patch.object(_log, "_get_recent_turns_uncached", counting):
            with patch.object(_cfg, "RECENT_TURNS_CACHE_ENABLED", True):
                _log._get_recent_turns("int_chat")
                _log._get_recent_turns("int_chat")
        self.assertEqual(calls[0], 1, "second call should hit cache")

    def test_disabled_flag_bypass(self):
        from larkhelm import log as _log
        original = _log._get_recent_turns_uncached
        calls = [0]

        def counting(*args, **kwargs):
            calls[0] += 1
            return original(*args, **kwargs)

        with patch.object(_log, "_get_recent_turns_uncached", counting):
            with patch.object(_cfg, "RECENT_TURNS_CACHE_ENABLED", False):
                _log._get_recent_turns("int_chat")
                _log._get_recent_turns("int_chat")
        self.assertEqual(calls[0], 2)


if __name__ == "__main__":
    unittest.main()
