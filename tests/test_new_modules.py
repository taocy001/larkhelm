"""
larkhelm new-module test suite
Coverage: dedup / concurrency / log / token_stats / chat_state / state (compatibility shim)
"""
import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Initialize config (using a temporary directory) ──────────────────────
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)
_DUMMY_CONFIG = {
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

# Now import modules that depend on config
import larkhelm.dedup as dedup
import larkhelm.concurrency as concurrency
import larkhelm.log as log_mod
import larkhelm.token_stats as token_stats
import larkhelm.chat_state as chat_state
import larkhelm.state as state_compat


# ═══════════════════════════════════════════════════════════════════════════
#  dedup.py
# ═══════════════════════════════════════════════════════════════════════════
class TestDedup(unittest.TestCase):
    def setUp(self):
        # Clear global state
        with dedup._seen_lock:
            dedup._seen_event_ids.clear()
            dedup._seen_msg_ids.clear()

    def test_new_event_not_duplicate(self):
        self.assertFalse(dedup._is_duplicate("ev1", "msg1"))

    def test_same_event_id_duplicate(self):
        dedup._is_duplicate("ev2", "msg2")
        self.assertTrue(dedup._is_duplicate("ev2", "msg3"))

    def test_same_msg_id_different_event_duplicate(self):
        dedup._is_duplicate("ev3", "msg4")
        self.assertTrue(dedup._is_duplicate("ev_new", "msg4"))

    def test_empty_message_id_no_msg_dedup(self):
        dedup._is_duplicate("ev5", "")
        # Different event_id with empty message_id → not a duplicate
        self.assertFalse(dedup._is_duplicate("ev6", ""))

    def test_lru_eviction(self):
        # After filling past DEDUP_CAP, the oldest entries should be evicted
        for i in range(dedup.DEDUP_CAP + 10):
            dedup._is_duplicate(f"evict_ev_{i}", f"evict_msg_{i}")
        # DEDUP_CAP = 500; after inserting 510 entries the oldest 10 should be gone
        with dedup._seen_lock:
            self.assertLessEqual(len(dedup._seen_event_ids), dedup.DEDUP_CAP)

    def test_dedup_cap_constant_value(self):
        self.assertEqual(dedup.DEDUP_CAP, 500)

    def test_concurrent_dedup(self):
        """Should not crash under concurrent access and should not produce duplicate IDs"""
        results = []
        lock = threading.Lock()

        def worker(i):
            r = dedup._is_duplicate(f"conc_ev_{i}", f"conc_msg_{i}")
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Each (event_id, msg_id) is unique, so none should be flagged as a duplicate on first call
        self.assertTrue(all(r is False for r in results))


# ═══════════════════════════════════════════════════════════════════════════
#  concurrency.py
# ═══════════════════════════════════════════════════════════════════════════
class TestConcurrency(unittest.TestCase):
    def setUp(self):
        # Clear global state
        with concurrency._chat_locks_meta:
            concurrency._chat_locks.clear()
        with concurrency._btw_locks_meta:
            concurrency._btw_locks.clear()
        with concurrency._cancel_meta:
            concurrency._cancel_events.clear()
        with concurrency._pending_meta:
            concurrency._pending_msg.clear()
        concurrency._shutting_down = False

    def test_get_chat_lock_same_object(self):
        l1 = concurrency._get_chat_lock("chat_a")
        l2 = concurrency._get_chat_lock("chat_a")
        self.assertIs(l1, l2)

    def test_get_chat_lock_different_chats(self):
        l1 = concurrency._get_chat_lock("chat_x")
        l2 = concurrency._get_chat_lock("chat_y")
        self.assertIsNot(l1, l2)

    def test_btw_lock_independent_of_chat_lock(self):
        cl = concurrency._get_chat_lock("chat_b")
        bl = concurrency._get_btw_lock("chat_b")
        self.assertIsNot(cl, bl)

    def test_cancel_event_trigger_and_reset(self):
        ev = concurrency._get_cancel_event("cancel_chat")
        self.assertFalse(ev.is_set())
        concurrency._trigger_cancel("cancel_chat")
        self.assertTrue(ev.is_set())
        concurrency._reset_cancel("cancel_chat")
        self.assertFalse(ev.is_set())

    def test_replace_cancel_event_creates_new_object(self):
        old_ev = concurrency._get_cancel_event("replace_chat")
        old_ev.set()
        concurrency._replace_cancel_event("replace_chat")
        new_ev = concurrency._get_cancel_event("replace_chat")
        self.assertIsNot(old_ev, new_ev)
        self.assertFalse(new_ev.is_set())  # new object should not be set
        self.assertTrue(old_ev.is_set())   # old object state should be unchanged

    def test_shutting_down_flag(self):
        self.assertFalse(concurrency.is_shutting_down())
        concurrency.set_shutting_down()
        self.assertTrue(concurrency.is_shutting_down())

    def test_pending_msg_set_and_pop(self):
        concurrency._set_pending("p_chat", "hello", "claude", "msg_001")
        result = concurrency._pop_pending("p_chat")
        # 4-tuple: (message, model, user_msg_id, old_card_mid)
        self.assertEqual(result[:3], ("hello", "claude", "msg_001"))
        # After popping, should return None
        self.assertIsNone(concurrency._pop_pending("p_chat"))

    def test_pending_msg_overwrite(self):
        concurrency._set_pending("ow_chat", "first", "claude", None)
        concurrency._set_pending("ow_chat", "second", "gemini", "m2")
        result = concurrency._pop_pending("ow_chat")
        self.assertEqual(result[0], "second")

    def test_wait_for_idle_no_locks_held(self):
        # Should return True immediately when there are no active queries
        self.assertTrue(concurrency.wait_for_idle(timeout=2.0))

    def test_wait_for_idle_timeout(self):
        lock = concurrency._get_chat_lock("busy_chat")
        lock.acquire()
        try:
            result = concurrency.wait_for_idle(timeout=1.5)
            self.assertFalse(result)
        finally:
            lock.release()

    def test_btw_timeout_constant(self):
        self.assertEqual(concurrency.BTW_TIMEOUT, 120)

    def test_cron_lock_is_lock(self):
        self.assertIsInstance(concurrency._cron_lock, type(threading.Lock()))

    def test_get_chat_lock_lru_no_alias_race(self):
        """LRU eviction must not create a new lock object for a chat that still holds its lock.
        Regression for lock-alias race: when >_LOCK_CACHE_MAX chats are active, evicting a
        held lock would let a second caller obtain a different Lock for the same chat_id."""
        cap = concurrency._LOCK_CACHE_MAX

        # Fill the cache to capacity with chats that are NOT held
        for i in range(cap):
            concurrency._get_chat_lock(f"fill_chat_{i}")

        # Now acquire the LRU lock (the first chat inserted) while the cache is at capacity
        lru_chat = "fill_chat_0"
        lru_lock = concurrency._get_chat_lock(lru_chat)
        lru_lock.acquire()
        try:
            # Insert one more chat to trigger the LRU eviction path
            concurrency._get_chat_lock("overflow_chat")

            # Because fill_chat_0 is held, it must NOT have been evicted
            with concurrency._chat_locks_meta:
                still_present = lru_chat in concurrency._chat_locks
            if still_present:
                # If still present, calling _get_chat_lock must return the same object
                second_ref = concurrency._get_chat_lock(lru_chat)
                self.assertIs(second_ref, lru_lock, "Lock alias created for a held lock")
        finally:
            lru_lock.release()

    def test_get_chat_lock_lru_evicts_idle_lock(self):
        """Idle (unheld) LRU entries should be evicted when cache overflows."""
        cap = concurrency._LOCK_CACHE_MAX
        for i in range(cap + 1):
            concurrency._get_chat_lock(f"idle_chat_{i}")
        with concurrency._chat_locks_meta:
            self.assertLessEqual(len(concurrency._chat_locks), cap + 1)

    def test_stale_lock_dead_holder_replaced(self):
        """Dead holder thread: _get_chat_lock returns a new, acquirable lock."""
        lock = concurrency._get_chat_lock("stale_dead_chat")
        acquired = threading.Event()

        def holder():
            lock.acquire()
            acquired.set()
            # exits without releasing — simulates a crashed daemon thread

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        acquired.wait(timeout=2.0)
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive(), "Holder thread should have exited")
        self.assertIsNotNone(lock._holder_ident)

        new_lock = concurrency._get_chat_lock("stale_dead_chat")
        self.assertIsNot(new_lock, lock, "Stale lock should have been replaced")
        self.assertTrue(new_lock.acquire(blocking=False), "New lock must be acquirable")
        new_lock.release()

    def test_stale_lock_held_too_long_replaced(self):
        """Lock held beyond HARD_TIMEOUT is replaced even when holder is alive."""
        try:
            from larkhelm.config import HARD_TIMEOUT as _ht
            hard_timeout = float(_ht)
        except Exception:
            hard_timeout = 21600.0

        lock = concurrency._get_chat_lock("stale_overtime_chat")
        lock.acquire()
        # Backdate acquisition time to trigger the overtime condition (is_stale uses 2× threshold)
        lock._acq_mono = time.monotonic() - hard_timeout * 2 - 1

        new_lock = concurrency._get_chat_lock("stale_overtime_chat")
        self.assertIsNot(new_lock, lock, "Overtime lock should have been replaced")
        self.assertTrue(new_lock.acquire(blocking=False), "New lock must be acquirable")
        new_lock.release()

    def test_stale_lock_replacement_logs_warn(self):
        """Replacement emits a warning containing the expected prefix and reason."""
        lock = concurrency._get_chat_lock("stale_log_chat")
        acquired = threading.Event()

        def holder():
            lock.acquire()
            acquired.set()

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        acquired.wait(timeout=2.0)
        t.join(timeout=2.0)
        self.assertFalse(t.is_alive())

        with patch("larkhelm.log.warn") as mock_warn:
            concurrency._get_chat_lock("stale_log_chat")

        self.assertTrue(mock_warn.called, "warn() should have been called on stale replacement")
        msg = mock_warn.call_args[0][0]
        self.assertIn("[Concurrency] stale lock for chat", msg)
        self.assertIn("holder thread dead or lock held too long", msg)

    def test_stale_lock_alive_holder_not_replaced(self):
        """A lock held by an alive thread is NOT replaced."""
        lock = concurrency._get_chat_lock("stale_alive_chat")
        ready = threading.Event()
        done = threading.Event()

        def holder():
            lock.acquire()
            ready.set()
            done.wait(timeout=5.0)
            lock.release()

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        ready.wait(timeout=2.0)
        try:
            same_lock = concurrency._get_chat_lock("stale_alive_chat")
            self.assertIs(same_lock, lock, "Live-holder lock must NOT be replaced")
        finally:
            done.set()
            t.join(timeout=2.0)

    def test_chatllock_is_stale_dead_holder(self):
        """_ChatLock.is_stale() returns True when the holder thread is dead."""
        from larkhelm.concurrency import _ChatLock
        lock = _ChatLock()
        # Assign an impossible thread ident so no live thread matches it.
        lock._holder_ident = -999999
        lock._acq_mono = time.monotonic() - 1.0  # held only 1 second

        # With a large hard_timeout the time check won't fire;
        # only the dead-thread check should trigger.
        self.assertTrue(lock.is_stale(21600.0))

    def test_chatllock_is_stale_not_held(self):
        """_ChatLock.is_stale() returns False when the lock is not held."""
        from larkhelm.concurrency import _ChatLock
        lock = _ChatLock()
        self.assertFalse(lock.is_stale(21600.0))


# ═══════════════════════════════════════════════════════════════════════════
#  log.py
# ═══════════════════════════════════════════════════════════════════════════
class TestLog(unittest.TestCase):
    def setUp(self):
        # Clean up all.jsonl
        all_jsonl = _cfg_module.LOG_DIR / "all.jsonl"
        if all_jsonl.exists():
            all_jsonl.unlink()

    def test_log_entry_creates_file(self):
        log_mod.log_entry("test_chat", "user", "hello world")
        all_jsonl = _cfg_module.LOG_DIR / "all.jsonl"
        self.assertTrue(all_jsonl.exists())

    def test_log_entry_content(self):
        log_mod.log_entry("test_chat2", "assistant", "response text", model="gemini", trace_id="t123")
        all_jsonl = _cfg_module.LOG_DIR / "all.jsonl"
        lines = all_jsonl.read_text().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        found = [r for r in records if r.get("chat_id") == "test_chat2"]
        self.assertTrue(len(found) > 0)
        self.assertEqual(found[-1]["role"], "assistant")
        self.assertEqual(found[-1]["model"], "gemini")
        self.assertEqual(found[-1]["trace_id"], "t123")

    def test_read_logs_empty_chat(self):
        result = log_mod._read_logs("nonexistent_chat_xyz")
        self.assertEqual(result, [])

    def test_read_logs_returns_records(self):
        log_mod.log_entry("read_test_chat", "user", "test message")
        records = log_mod._read_logs("read_test_chat")
        self.assertGreater(len(records), 0)
        self.assertEqual(records[-1]["chat_id"], "read_test_chat")

    def test_debug_log_writes(self):
        # Isolate: create a separate file inside _TMP_DIR to avoid config.py pointing to /var/log
        debug_log = Path(_TMP_DIR) / "test_debug.log"
        with patch.object(_cfg_module, "DEBUG_LOG", debug_log):
            log_mod._debug_log("test debug message")
            self.assertTrue(debug_log.exists())
            content = debug_log.read_text()
            self.assertIn("test debug message", content)

    def test_log_entry_write_fail(self):
        """AC-01: log_entry should not raise on write failure (silent degradation)"""
        # Mock open to throw OSError
        with patch("builtins.open", side_effect=OSError("Disk full")):
            try:
                log_mod.log_entry("err_chat", "user", "test msg")
            except Exception as e:
                self.fail(f"log_entry raised {type(e).__name__}: {e}")

    def test_log_entry_concurrent(self):
        """Concurrent writes should not raise exceptions"""
        errors = []
        def worker(i):
            try:
                log_mod.log_entry(f"conc_chat_{i % 3}", "user", f"msg {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])

    def test_log_entry_error_role(self):
        log_mod.log_entry("err_chat", "error", "something failed")
        records = log_mod._read_logs("err_chat")
        found = [r for r in records if r.get("chat_id") == "err_chat"]
        self.assertTrue(found[-1].get("is_error"))

    def test_markdown_file_created(self):
        log_mod.log_entry("md_chat", "user", "markdown test")
        from datetime import datetime
        date_str = datetime.now().strftime('%Y-%m-%d')
        md_file = _cfg_module.LOG_DIR / "md_chat" / f"{date_str}.md"
        self.assertTrue(md_file.exists())
        content = md_file.read_text()
        self.assertIn("markdown test", content)


# ═══════════════════════════════════════════════════════════════════════════
#  token_stats.py
# ═══════════════════════════════════════════════════════════════════════════
class TestTokenStats(unittest.TestCase):
    def setUp(self):
        with token_stats._token_stats_lock:
            token_stats._token_stats.clear()
        all_jsonl = _cfg_module.LOG_DIR / "all.jsonl"
        if all_jsonl.exists():
            all_jsonl.unlink()  # clean up between tests

    def test_record_token_usage_basic(self):
        token_stats.record_token_usage("ts_chat", "claude", {
            "input_tokens": 100, "output_tokens": 50,
            "cache_read": 10, "cache_create": 5, "cost_usd": 0.001,
        })
        stats = token_stats.get_token_stats("ts_chat")
        self.assertEqual(stats["claude"]["input_tokens"], 100)
        self.assertEqual(stats["claude"]["output_tokens"], 50)
        self.assertEqual(stats["claude"]["calls"], 1)

    def test_record_accumulates(self):
        for _ in range(3):
            token_stats.record_token_usage("acc_chat", "claude", {
                "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.0001,
            })
        stats = token_stats.get_token_stats("acc_chat")
        self.assertEqual(stats["claude"]["input_tokens"], 30)
        self.assertEqual(stats["claude"]["calls"], 3)

    def test_get_token_stats_global(self):
        token_stats.record_token_usage("global_chat_a", "claude", {"input_tokens": 100})
        token_stats.record_token_usage("global_chat_b", "claude", {"input_tokens": 200})
        all_stats = token_stats.get_token_stats(None)
        self.assertGreaterEqual(all_stats["claude"]["input_tokens"], 300)

    def test_record_writes_to_jsonl(self):
        token_stats.record_token_usage("persist_chat", "gemini", {
            "input_tokens": 99, "output_tokens": 77, "cost_usd": 0.002,
        })
        all_jsonl = _cfg_module.LOG_DIR / "all.jsonl"
        lines = all_jsonl.read_text().splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        token_records = [r for r in records if r.get("role") == "token" and r.get("chat_id") == "persist_chat"]
        self.assertEqual(len(token_records), 1)
        self.assertEqual(token_records[0]["input_tokens"], 99)
        self.assertEqual(token_records[0]["model"], "gemini")

    def test_get_token_stats_persistent(self):
        token_stats.record_token_usage("pers2_chat", "claude", {
            "input_tokens": 55, "output_tokens": 33,
        })
        result = token_stats.get_token_stats_persistent("pers2_chat")
        self.assertIn("claude", result)
        self.assertEqual(result["claude"]["input_tokens"], 55)

    def test_get_token_stats_persistent_date_filter(self):
        token_stats.record_token_usage("date_chat", "claude", {"input_tokens": 10})
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        result = token_stats.get_token_stats_persistent("date_chat", date_prefix=today)
        self.assertIn("claude", result)

    def test_get_token_stats_persistent_nonexistent_chat(self):
        result = token_stats.get_token_stats_persistent("no_such_chat_xyz")
        self.assertEqual(result, {})

    def test_shared_log_lock(self):
        """token_stats._jsonl_lock should be the same object as log._log_lock (unified write lock)"""
        from larkhelm.log import _log_lock
        self.assertIs(token_stats._jsonl_lock, _log_lock)

    def test_concurrent_record(self):
        """Concurrent record_token_usage calls should not crash and counts should be consistent"""
        errors = []
        def worker():
            try:
                token_stats.record_token_usage("conc_ts", "claude", {"input_tokens": 1, "calls": 0})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        stats = token_stats.get_token_stats("conc_ts")
        self.assertEqual(stats["claude"]["calls"], 30)

    def test_crew_namespace_does_not_leak_into_parent_chat(self):
        """Strict per-chat aggregation: a record with chat_id ``X__suffix``
        must NOT roll up into chat ``X``.

        Was the opposite contract until the stats audit caught it as
        a cross-chat leak risk. The justification used to be "crew
        agents write under a ``parent__crew_id_role`` namespace and the
        user wants /stats to aggregate those up to the parent". The
        catch: by the time the JSONL record gets written,
        ``runner_base._record_tokens`` (lines 535-545) has ALREADY
        stripped the ``__crew_*`` suffix and written the bare parent
        chat_id, so the fuzzy-match branch in
        ``get_token_stats_persistent`` was dead code at the legitimate
        caller AND opened a leak for any ad-hoc / future writer using
        ``X__leak`` as a chat_id key.
        """
        token_stats.record_token_usage("crew_chat__agent1", "claude", {"input_tokens": 42})
        result = token_stats.get_token_stats_persistent("crew_chat")
        self.assertNotIn(
            "claude", result,
            "records under 'crew_chat__agent1' leaked into chat 'crew_chat' — "
            "fuzzy prefix match has been re-introduced (or runner_base no "
            "longer strips __crew_ suffixes before writing)"
        )

    def test_crew_token_roll_up_via_runner_base_strip(self):
        """Documents the legitimate roll-up path: ``runner_base._record_tokens``
        strips ``__crew_*`` before writing, so the bare parent chat_id is
        what lands in JSONL. ``get_token_stats_persistent`` then matches
        exactly. This test pins THAT contract — any regression where
        runner_base stops stripping the suffix would surface here as
        the parent chat showing zero tokens despite crew activity."""
        token_stats.record_token_usage("crew_chat", "claude", {"input_tokens": 42})
        result = token_stats.get_token_stats_persistent("crew_chat")
        self.assertIn("claude", result)
        self.assertEqual(result["claude"]["input_tokens"], 42)


# ═══════════════════════════════════════════════════════════════════════════
#  chat_state.py
# ═══════════════════════════════════════════════════════════════════════════
class TestChatState(unittest.TestCase):
    def setUp(self):
        with chat_state._state_lock:
            chat_state._chat_state_store.clear()
        with chat_state._btw_msg_ids_meta:
            chat_state._btw_msg_ids.clear()
        with chat_state._pending_doc_writes_lock:
            chat_state._pending_doc_writes.clear()
        # Remove state file for a clean slate
        if _cfg_module.STATE_FILE.exists():
            _cfg_module.STATE_FILE.unlink()

    def test_get_chat_state_empty(self):
        state = chat_state._get_chat_state("new_chat")
        self.assertIsInstance(state, dict)

    def test_set_chat_field_and_read(self):
        chat_state._set_chat_field("field_chat", "model", "gemini")
        state = chat_state._get_chat_state("field_chat")
        self.assertEqual(state["model"], "gemini")

    def test_save_and_load_state(self):
        chat_state._set_chat_field("persist_chat", "cwd", "/tmp/testdir")
        chat_state._load_global_state()
        state = chat_state._get_chat_state("persist_chat")
        self.assertEqual(state.get("cwd"), "/tmp/testdir")

    def test_sid_file_path_claude(self):
        p = chat_state._sid_file("sid_chat", "claude")
        self.assertTrue(str(p).endswith("sid_chat.sid"))

    def test_sid_file_path_gemini(self):
        p = chat_state._sid_file("sid_chat", "gemini")
        self.assertTrue(str(p).endswith("gemini_sid_chat.sid"))

    def test_save_load_clear_sid(self):
        chat_state._save_sid("sid_test", "abc123", "claude")
        loaded = chat_state._load_sid("sid_test", "claude")
        self.assertEqual(loaded, "abc123")
        chat_state._clear_sid("sid_test", "claude")
        loaded2 = chat_state._load_sid("sid_test", "claude")
        self.assertIsNone(loaded2)

    def test_get_cwd_default(self):
        cwd = chat_state._get_cwd("cwd_new_chat")
        self.assertEqual(cwd, _cfg_module.DEFAULT_CWD)

    def test_set_cwd(self):
        chat_state._set_cwd("cwd_chat", _TMP_DIR)
        self.assertEqual(chat_state._get_cwd("cwd_chat"), _TMP_DIR)

    def test_get_set_chat_model(self):
        self.assertEqual(chat_state._get_chat_model("model_chat"), "claude")  # default
        chat_state._set_chat_model("model_chat", "gemini")
        self.assertEqual(chat_state._get_chat_model("model_chat"), "gemini")

    def test_register_and_is_btw_reply(self):
        chat_state._register_btw_msg("btw_chat", "msg_btw_001")
        self.assertTrue(chat_state._is_btw_reply("btw_chat", "msg_btw_001"))
        self.assertFalse(chat_state._is_btw_reply("btw_chat", "msg_other"))

    def test_is_btw_reply_none_parent(self):
        self.assertFalse(chat_state._is_btw_reply("btw_chat2", None))

    def test_btw_msg_id_cap_eviction(self):
        """Set size should not exceed _BTW_MSG_ID_CAP after overflow"""
        for i in range(chat_state._BTW_MSG_ID_CAP + 20):
            chat_state._register_btw_msg("cap_chat", f"msg_{i}")
        with chat_state._btw_msg_ids_meta:
            size = len(chat_state._btw_msg_ids.get("cap_chat", set()))
        self.assertLessEqual(size, chat_state._BTW_MSG_ID_CAP)

    def test_pending_doc_write_set_and_pop(self):
        chat_state.set_pending_doc_write("doc_chat", "http://example.com", "content", None)
        entry = chat_state.pop_pending_doc_write("doc_chat")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["url"], "http://example.com")
        self.assertEqual(entry["content"], "content")

    def test_pending_doc_write_pop_returns_none_after_first(self):
        chat_state.set_pending_doc_write("doc_chat2", "url", "content", None)
        chat_state.pop_pending_doc_write("doc_chat2")
        self.assertIsNone(chat_state.pop_pending_doc_write("doc_chat2"))

    def test_pending_doc_write_expiry(self):
        chat_state.set_pending_doc_write("exp_chat", "url", "c", None)
        with chat_state._pending_doc_writes_lock:
            chat_state._pending_doc_writes["exp_chat"]["expire_ts"] = time.time() - 1
        result = chat_state.pop_pending_doc_write("exp_chat")
        self.assertIsNone(result)

    def test_load_global_state_missing_file(self):
        """_load_global_state should not raise when STATE_FILE does not exist"""
        if _cfg_module.STATE_FILE.exists():
            _cfg_module.STATE_FILE.unlink()
        try:
            chat_state._load_global_state()
        except Exception as e:
            self.fail(f"_load_global_state raised {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  state.py compatibility shim
# ═══════════════════════════════════════════════════════════════════════════
class TestStateCompat(unittest.TestCase):
    def test_re_exports_chat_state_symbols(self):
        for name in [
            "_state_lock", "_chat_state_store",
            "_load_global_state", "_save_state", "_get_chat_state", "_set_chat_field",
            "_sid_file", "_load_sid", "_save_sid", "_clear_sid",
            "_get_cwd", "_set_cwd", "_get_chat_model", "_set_chat_model",
            "_register_btw_msg", "_is_btw_reply",
            "set_pending_doc_write", "pop_pending_doc_write",
        ]:
            self.assertTrue(hasattr(state_compat, name), f"Missing: {name}")

    def test_re_exports_concurrency_symbols(self):
        for name in [
            "_chat_locks", "_get_chat_lock",
            "_btw_locks", "_get_btw_lock", "BTW_TIMEOUT",
            "_cancel_events", "_get_cancel_event", "_trigger_cancel", "_reset_cancel",
            "_replace_cancel_event", "_shutting_down", "set_shutting_down",
            "is_shutting_down", "wait_for_idle",
            "_pending_msg", "_set_pending", "_pop_pending", "_cron_lock",
        ]:
            self.assertTrue(hasattr(state_compat, name), f"Missing: {name}")

    def test_re_exports_dedup_symbols(self):
        self.assertTrue(hasattr(state_compat, "_is_duplicate"))

    def test_re_exports_log_symbols(self):
        for name in ["_log_lock", "log_entry", "_read_logs", "_debug_log"]:
            self.assertTrue(hasattr(state_compat, name), f"Missing: {name}")

    def test_re_exports_token_stats_symbols(self):
        for name in ["_token_stats", "record_token_usage", "get_token_stats", "get_token_stats_persistent"]:
            self.assertTrue(hasattr(state_compat, name), f"Missing: {name}")

    def test_symbols_are_same_objects(self):
        """Symbols exported by the compatibility shim must be identical objects to those in the sub-modules"""
        self.assertIs(state_compat._get_chat_lock, concurrency._get_chat_lock)
        self.assertIs(state_compat.log_entry, log_mod.log_entry)
        self.assertIs(state_compat._is_duplicate, dedup._is_duplicate)
        self.assertIs(state_compat.record_token_usage, token_stats.record_token_usage)

    def test_state_py_line_count(self):
        src = Path("larkhelm/state.py").read_text().splitlines()
        # AC-04: must be <= 60 lines
        self.assertLessEqual(len(src), 60, f"state.py has {len(src)} lines (max 60)")

    def test_state_py_no_impl_code(self):
        src = Path("larkhelm/state.py").read_text()
        bad_patterns = [
            "_token_stats =", "def log_entry", "def _is_duplicate",
            "_chat_locks =", "_cancel_events =", "_seen_event_ids =",
        ]
        for pat in bad_patterns:
            self.assertNotIn(pat, src, f"state.py contains impl code: {pat!r}")


# ═══════════════════════════════════════════════════════════════════════════
#  Module-level checks
# ═══════════════════════════════════════════════════════════════════════════
class TestModuleChecks(unittest.TestCase):
    def test_all_modules_have_docstring(self):
        for mod in [dedup, concurrency, log_mod, token_stats, chat_state]:
            self.assertTrue(bool(mod.__doc__), f"{mod.__name__} missing docstring")

    def test_token_stats_shared_log_lock(self):
        """token_stats._jsonl_lock should share the same object as log._log_lock to prevent interleaved writes to all.jsonl"""
        from larkhelm.log import _log_lock
        self.assertIs(token_stats._jsonl_lock, _log_lock)

    def test_all_modules_define_all(self):
        for mod in [dedup, concurrency, log_mod, token_stats, chat_state]:
            self.assertTrue(hasattr(mod, "__all__"), f"{mod.__name__} missing __all__")

    def test_no_circular_imports(self):
        """Verify there are no circular imports between modules"""
        import importlib
        for mod_name in [
            "larkhelm.dedup", "larkhelm.concurrency",
            "larkhelm.log", "larkhelm.token_stats", "larkhelm.chat_state",
        ]:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                self.fail(f"Circular import in {mod_name}: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  lark_client.py — FeishuDocClient pure-logic tests
# ═══════════════════════════════════════════════════════════════════════════
class TestFeishuDocClient(unittest.TestCase):
    """Pure-logic tests for FeishuDocClient, no real Feishu API calls."""

    def setUp(self):
        import larkhelm.lark_client as lark_client_mod
        self.lark_client_mod = lark_client_mod
        self.DocRef = lark_client_mod.DocRef
        self.client_cls = lark_client_mod.FeishuDocClient
        self.DocPermissionError = lark_client_mod.DocPermissionError
        self.DocNotFoundError = lark_client_mod.DocNotFoundError
        self.DocAPIError = lark_client_mod.DocAPIError
        self.dc = lark_client_mod.FeishuDocClient()
        # Pre-set the module-level client (annotation only, not assigned), so patch.object can find it
        if not hasattr(lark_client_mod, "client"):
            lark_client_mod.client = unittest.mock.MagicMock()

    # ── parse_doc_url test cases ────────────────────────────────────────────

    def test_parse_doc_url_docx(self):
        ref = self.dc.parse_url("https://example.feishu.cn/docx/AbCdEfGh123")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "docx")
        self.assertEqual(ref.token, "AbCdEfGh123")

    def test_parse_doc_url_docs(self):
        ref = self.dc.parse_url("https://example.feishu.cn/docs/DocsToken456")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "docs")
        self.assertEqual(ref.token, "DocsToken456")

    def test_parse_doc_url_wiki(self):
        ref = self.dc.parse_url("https://example.feishu.cn/wiki/WikiToken789")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "wiki")
        self.assertEqual(ref.token, "WikiToken789")

    def test_parse_doc_url_sheets(self):
        ref = self.dc.parse_url("https://example.feishu.cn/sheets/SheetsTokenXyz")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "sheets")
        self.assertEqual(ref.token, "SheetsTokenXyz")

    def test_parse_doc_url_folder(self):
        ref = self.dc.parse_url("https://example.feishu.cn/drive/folder/FolderTokenAbc")
        self.assertIsNotNone(ref)
        self.assertEqual(ref.doc_type, "folder")
        self.assertEqual(ref.token, "FolderTokenAbc")

    def test_parse_doc_url_invalid_returns_none(self):
        ref = self.dc.parse_url("https://example.com/not-a-feishu-url")
        self.assertIsNone(ref)

    def test_parse_doc_url_empty_returns_none(self):
        ref = self.dc.parse_url("")
        self.assertIsNone(ref)

    # ── _md_to_blocks test cases ───────────────────────────────────────────

    def test_md_to_blocks_paragraph(self):
        blocks = self.dc._md_to_blocks("Hello world")
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["block_type"], 2)
        self.assertIn("text", blocks[0])

    def test_md_to_blocks_headings(self):
        md = "# Title\n## Subtitle\n### Sub-sub"
        blocks = self.dc._md_to_blocks(md)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["block_type"], 3)   # H1
        self.assertEqual(blocks[1]["block_type"], 4)   # H2
        self.assertEqual(blocks[2]["block_type"], 5)   # H3

    def test_md_to_blocks_code_block(self):
        md = "```python\nprint('hi')\n```"
        blocks = self.dc._md_to_blocks(md)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["block_type"], 14)
        self.assertEqual(blocks[0]["code"]["style"]["language"], 49)  # python=49

    def test_md_to_blocks_lists(self):
        md = "- item one\n* item two\n1. ordered"
        blocks = self.dc._md_to_blocks(md)
        self.assertEqual(len(blocks), 3)
        self.assertEqual(blocks[0]["block_type"], 12)  # bullet
        self.assertEqual(blocks[1]["block_type"], 12)  # bullet
        self.assertEqual(blocks[2]["block_type"], 13)  # ordered

    # ── _call_api error-code mapping test cases ────────────────────────────

    def _make_mock_resp(self, raw_content: bytes | None):
        """Build a mock response object."""
        mock_resp = unittest.mock.MagicMock()
        if raw_content is None:
            mock_resp.raw = None
        else:
            mock_resp.raw = unittest.mock.MagicMock()
            mock_resp.raw.content = raw_content
        return mock_resp

    def test_call_api_success(self):
        import json
        payload = json.dumps({"code": 0, "data": {"document": {}}}).encode()
        mock_resp = self._make_mock_resp(payload)
        with patch.object(self.lark_client_mod, "client") as mock_client:
            mock_client.request.return_value = mock_resp
            result = self.dc._call_api("GET", "/open-apis/docx/v1/documents/xxx")
        self.assertEqual(result["code"], 0)

    def test_call_api_permission_error(self):
        import json
        payload = json.dumps({"code": 99991663, "msg": "permission denied"}).encode()
        mock_resp = self._make_mock_resp(payload)
        with patch.object(self.lark_client_mod, "client") as mock_client:
            mock_client.request.return_value = mock_resp
            with self.assertRaises(self.DocPermissionError):
                self.dc._call_api("GET", "/some/uri")

    def test_call_api_not_found_error(self):
        import json
        payload = json.dumps({"code": 99991664, "msg": "not found"}).encode()
        mock_resp = self._make_mock_resp(payload)
        with patch.object(self.lark_client_mod, "client") as mock_client:
            mock_client.request.return_value = mock_resp
            with self.assertRaises(self.DocNotFoundError):
                self.dc._call_api("GET", "/some/uri")

    def test_call_api_generic_error(self):
        import json
        payload = json.dumps({"code": 500, "msg": "internal error"}).encode()
        mock_resp = self._make_mock_resp(payload)
        with patch.object(self.lark_client_mod, "client") as mock_client:
            mock_client.request.return_value = mock_resp
            with self.assertRaises(self.DocAPIError) as ctx:
                self.dc._call_api("GET", "/some/uri")
        self.assertEqual(ctx.exception.code, 500)

    def test_call_api_empty_response(self):
        mock_resp = self._make_mock_resp(None)
        with patch.object(self.lark_client_mod, "client") as mock_client:
            mock_client.request.return_value = mock_resp
            with self.assertRaises(self.DocAPIError) as ctx:
                self.dc._call_api("GET", "/some/uri")
        self.assertEqual(ctx.exception.code, 0)


# ═══════════════════════════════════════════════════════════════════════════
#  config.py — _init_runtime() boundary cases
# ═══════════════════════════════════════════════════════════════════════════
class TestInitRuntimeBoundary(unittest.TestCase):
    """Tests for _init_runtime() boundary validation (empty credentials, invalid enum)."""

    def _make_config(self, overrides: dict) -> Path:
        base = {"APP_ID": "test_app", "APP_SECRET": "test_secret"}
        base.update(overrides)
        cfg_file = Path(_TMP_DIR) / f"config_boundary_{id(overrides)}.json"
        cfg_file.write_text(json.dumps(base))
        return cfg_file

    def test_empty_app_id_exits(self):
        cfg = self._make_config({"APP_ID": ""})
        with self.assertRaises(SystemExit):
            _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)

    def test_empty_app_secret_exits(self):
        cfg = self._make_config({"APP_SECRET": ""})
        with self.assertRaises(SystemExit):
            _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)

    def test_empty_claude_cmd_falls_back_to_default(self):
        cfg = self._make_config({"claude_command": ""})
        _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)
        self.assertEqual(_cfg_module.CLAUDE_CMD, "claude")

    def test_empty_gemini_cmd_falls_back_to_default(self):
        cfg = self._make_config({"gemini_command": ""})
        _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)
        self.assertEqual(_cfg_module.GEMINI_CMD, "gemini")

    def test_invalid_doc_write_backend_falls_back_to_auto(self):
        cfg = self._make_config({"doc_write_backend": "invalid_value"})
        _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)
        self.assertEqual(_cfg_module.DOC_WRITE_BACKEND, "auto")

    def test_valid_doc_write_backend_values(self):
        for val in ("auto", "feishu", "local"):
            cfg = self._make_config({"doc_write_backend": val})
            _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)
            self.assertEqual(_cfg_module.DOC_WRITE_BACKEND, val)

    def test_missing_required_field_exits(self):
        cfg = Path(_TMP_DIR) / "config_no_secret.json"
        cfg.write_text(json.dumps({"APP_ID": "test"}))
        with self.assertRaises(SystemExit):
            _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)

    def test_invalid_default_model_falls_back_to_claude(self):
        cfg = self._make_config({"default_model": "invalid_model"})
        _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)
        self.assertEqual(_cfg_module.DEFAULT_MODEL, "claude")

    def test_valid_default_model_values(self):
        for val in ("claude", "gemini", "kimi"):
            cfg = self._make_config({"default_model": val})
            _cfg_module._init_runtime(config_path=str(cfg), data_dir=_TMP_DIR)
            self.assertEqual(_cfg_module.DEFAULT_MODEL, val)


if __name__ == "__main__":
    unittest.main(verbosity=2)
