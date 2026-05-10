"""Tests for ``larkhelm.voice.merge`` (M3.2 commit 4).

Coverage map → PRD acceptance criteria:

| AC    | Test                          |
| ----- | ----------------------------- |
| AC-01 | TestPublicAPI                 |
| AC-02 | TestTimerAutoFlush            |
| AC-03 | TestCapImmediateFlush         |
| AC-04 | TestZeroWindowNoTimer         |
| AC-05 | TestMultiChatIsolation        |
| AC-06 | TestDispatchIsDaemon          |
| AC-07 | TestSinglePromptNoJoiner      |

Mocking strategy
----------------
* ``merge._Timer`` is replaced by a ``FakeTimer`` that records the
  ``(interval, callback, args)`` triple but whose ``start()`` is a
  no-op.  Tests advance the simulated timer by calling
  ``merge._on_timer(chat_id)`` directly.  This keeps the suite
  deterministic and < 0.5 s end-to-end.
* ``merge._dispatch`` is patched to a synchronous collector so the
  daemon thread never actually runs ``_do_query`` (which would pull
  in chat_state / lark_client and require a fully-bootstrapped bridge).
* The one test that intentionally exercises the daemon-thread path
  (``TestDispatchIsDaemon``) patches ``threading.Thread`` instead and
  inspects the ``daemon=True`` kwarg on the captured call.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Config init (mirrors tests/test_voice_transcribe.py:33-47) ─────────────
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_voice_merge_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_DUMMY_CONFIG = {
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

import larkhelm.voice.merge as merge


class FakeTimer:
    """Drop-in replacement for ``threading.Timer`` used in tests.

    Records ``(interval, callback, args)`` so assertions can verify the
    Timer was scheduled with the expected window.  ``start()`` is a
    no-op — tests fire the callback manually via ``merge._on_timer``
    to keep timing deterministic.
    """

    instances: "list[FakeTimer]" = []

    def __init__(self, interval, function, args=None, kwargs=None):
        self.interval = interval
        self.function = function
        self.args = tuple(args or ())
        self.kwargs = dict(kwargs or {})
        self.started = False
        self.cancelled = False
        self.daemon = False
        FakeTimer.instances.append(self)

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


class _MergeTestBase(unittest.TestCase):
    """Shared bootstrap: reset module state, swap in FakeTimer, snapshot cfg."""

    def setUp(self) -> None:
        # Reset module-level state — without this a prior test's leftover
        # buffer/timer would leak into the next case.
        merge._buffers.clear()
        merge._timers.clear()
        FakeTimer.instances.clear()

        # Snapshot voice cfg so tests can mutate freely.
        self._saved_window = _cfg.VOICE_MERGE_WINDOW_SEC
        self._saved_cap = _cfg.VOICE_MAX_MERGE

        # Default mocks: FakeTimer for the Timer seam, collector for dispatch.
        self._timer_patch = patch.object(merge, "_Timer", FakeTimer)
        self._timer_patch.start()

        self.dispatched: "list[tuple[str, str, str, object, object]]" = []

        def _collect(prompt, chat_id, model, user_msg_id, parent_id):
            self.dispatched.append((prompt, chat_id, model, user_msg_id, parent_id))

        self._dispatch_patch = patch.object(merge, "_dispatch", _collect)
        self._dispatch_patch.start()

    def tearDown(self) -> None:
        self._dispatch_patch.stop()
        self._timer_patch.stop()
        _cfg.VOICE_MERGE_WINDOW_SEC = self._saved_window
        _cfg.VOICE_MAX_MERGE = self._saved_cap
        merge._buffers.clear()
        merge._timers.clear()
        FakeTimer.instances.clear()


class TestPublicAPI(unittest.TestCase):
    """AC-01 — ``add_voice`` is importable and callable."""

    def test_public_api_importable(self) -> None:
        from larkhelm.voice.merge import add_voice
        self.assertTrue(callable(add_voice))
        # Module-level __all__ should expose only add_voice.
        self.assertEqual(merge.__all__, ["add_voice"])


class TestTimerAutoFlush(_MergeTestBase):
    """AC-02 — window-based auto-flush joins items with ``\\n\\n``."""

    def test_timer_auto_flush_joins(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 2
        _cfg.VOICE_MAX_MERGE = 10

        merge.add_voice("chat_X", "a", "claude", user_msg_id="m1", parent_id="p1")
        merge.add_voice("chat_X", "b", "claude", user_msg_id="m2", parent_id="p1")

        # No flush yet — Timer hasn't fired and cap not reached.
        self.assertEqual(self.dispatched, [])
        self.assertIn("chat_X", merge._buffers)
        self.assertIn("chat_X", merge._timers)

        # Manually fire the Timer callback (FakeTimer.start was a no-op).
        merge._on_timer("chat_X")

        self.assertEqual(len(self.dispatched), 1)
        prompt, chat_id, model, user_msg_id, parent_id = self.dispatched[0]
        self.assertEqual(prompt, "a\n\nb")
        self.assertEqual(chat_id, "chat_X")
        self.assertEqual(model, "claude")
        # Metadata is taken from the first buffered item.
        self.assertEqual(user_msg_id, "m1")
        self.assertEqual(parent_id, "p1")
        # Buffer + timer drained.
        self.assertNotIn("chat_X", merge._buffers)
        self.assertNotIn("chat_X", merge._timers)


class TestCapImmediateFlush(_MergeTestBase):
    """AC-03 — reaching VOICE_MAX_MERGE flushes synchronously."""

    def test_cap_immediate_flush(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 10
        _cfg.VOICE_MAX_MERGE = 2

        merge.add_voice("chat_Y", "first", "claude")
        # First add: buffer=1 < cap, Timer armed, no dispatch yet.
        self.assertEqual(self.dispatched, [])
        self.assertIn("chat_Y", merge._timers)

        merge.add_voice("chat_Y", "second", "claude")
        # Second add: buffer=2 == cap → flush immediately, no Timer left.
        self.assertEqual(len(self.dispatched), 1)
        prompt, _, _, _, _ = self.dispatched[0]
        self.assertEqual(prompt, "first\n\nsecond")
        self.assertNotIn("chat_Y", merge._timers)
        self.assertNotIn("chat_Y", merge._buffers)


class TestZeroWindowNoTimer(_MergeTestBase):
    """AC-04 — VOICE_MERGE_WINDOW_SEC=0 disables Timer entirely."""

    def test_zero_window_no_timer(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        _cfg.VOICE_MAX_MERGE = 5

        # Stop the FakeTimer patch so we can install a Mock that fails the
        # test if it gets called.  setUp's _timer_patch already wraps
        # _Timer; layer a stricter assertion on top.
        from unittest.mock import MagicMock
        sentinel_timer = MagicMock(side_effect=AssertionError(
            "Timer must NOT be constructed when VOICE_MERGE_WINDOW_SEC=0"
        ))
        with patch.object(merge, "_Timer", sentinel_timer):
            merge.add_voice("chat_Z", "hello", "claude")

        # Dispatched once, no Timer ever instantiated, no leftover state.
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0][0], "hello")
        self.assertFalse(sentinel_timer.called)
        self.assertNotIn("chat_Z", merge._timers)
        self.assertNotIn("chat_Z", merge._buffers)


class TestMultiChatIsolation(_MergeTestBase):
    """AC-05 — buffers and timers are per-chat, never bleed across chats."""

    def test_multi_chat_isolation(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 5
        _cfg.VOICE_MAX_MERGE = 10

        merge.add_voice("chat_A", "a1", "claude")
        merge.add_voice("chat_B", "b1", "claude")
        merge.add_voice("chat_A", "a2", "claude")
        merge.add_voice("chat_B", "b2", "claude")

        # Both chats have their own buffer + timer; nothing dispatched yet.
        self.assertEqual(self.dispatched, [])
        self.assertEqual([it.text for it in merge._buffers["chat_A"]], ["a1", "a2"])
        self.assertEqual([it.text for it in merge._buffers["chat_B"]], ["b1", "b2"])
        self.assertIn("chat_A", merge._timers)
        self.assertIn("chat_B", merge._timers)
        # Timers are distinct instances.
        self.assertIsNot(merge._timers["chat_A"], merge._timers["chat_B"])

        # Firing chat_A's Timer must not touch chat_B.
        timer_b_before = merge._timers["chat_B"]
        merge._on_timer("chat_A")

        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.dispatched[0][0], "a1\n\na2")
        self.assertEqual(self.dispatched[0][1], "chat_A")
        # chat_B remains intact; its Timer is the same object as before.
        self.assertIn("chat_B", merge._buffers)
        self.assertIn("chat_B", merge._timers)
        self.assertIs(merge._timers["chat_B"], timer_b_before)
        self.assertFalse(timer_b_before.cancelled)


class TestDispatchIsDaemon(_MergeTestBase):
    """AC-06 — the dispatch thread is created with ``daemon=True``."""

    def test_dispatch_is_daemon(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        _cfg.VOICE_MAX_MERGE = 5

        # Stop the dispatch collector so the real _flush_locked → Thread
        # path runs.  We then patch threading.Thread to capture kwargs.
        self._dispatch_patch.stop()
        try:
            captured: "list[dict]" = []

            class _CaptureThread:
                def __init__(self, *args, **kwargs):
                    captured.append({"args": args, "kwargs": kwargs})

                def start(self):
                    pass

            with patch.object(threading, "Thread", _CaptureThread):
                merge.add_voice("chat_D", "ping", "claude")

            self.assertEqual(len(captured), 1)
            self.assertIs(captured[0]["kwargs"].get("daemon"), True)
            # Sanity: the real _dispatch is the target.
            self.assertIs(captured[0]["kwargs"].get("target"), merge._dispatch)
        finally:
            # Restart the dispatch patch so tearDown's stop() doesn't fail.
            self._dispatch_patch = patch.object(merge, "_dispatch", lambda *a, **k: None)
            self._dispatch_patch.start()


class TestSinglePromptNoJoiner(_MergeTestBase):
    """AC-07 — a single buffered item dispatches with no separator added."""

    def test_single_prompt_no_joiner(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        _cfg.VOICE_MAX_MERGE = 5

        merge.add_voice("chat_S", "hello", "claude")

        self.assertEqual(len(self.dispatched), 1)
        prompt = self.dispatched[0][0]
        self.assertEqual(prompt, "hello")
        # No leading / trailing separator — exact equality is the contract.
        self.assertFalse(prompt.startswith("\n"))
        self.assertFalse(prompt.endswith("\n"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
