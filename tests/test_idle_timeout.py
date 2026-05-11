"""
Tests for the idle-clock timeout semantics in
``runner_base.BaseProcessRunner._watch`` and ``runner_deepseek._watch``.

Pre-fix bug: ``hard_deadline = time.time() + HARD_TIMEOUT`` was set at
process start, so any task running ≥ HARD_TIMEOUT wall-clock seconds was
killed regardless of whether it was actively producing output.

Post-fix invariant:
  * A subprocess that keeps emitting output (calls ``_touch_activity``)
    must never trip the hard timeout — only sustained silence does.
  * The soft timeout fires after RESPONSE_TIMEOUT of silence, not from
    process start.
  * Both timeouts measure idle time from the most recent activity stamp.

These tests poke ``_watch`` directly with a mock proc + fake clock so we
don't have to spawn real subprocesses or wait real minutes.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_TMP = tempfile.mkdtemp(prefix="larkhelm_idle_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "response_timeout": 5,   # 5s soft for the test
    "hard_timeout": 10,      # 10s hard for the test (config.py floor adjusts if hard ≤ soft)
}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

# Sanity: the config above kept hard > soft, so floor logic didn't kick in.
assert _cfg.HARD_TIMEOUT > _cfg.RESPONSE_TIMEOUT


def _make_base_runner():
    """Build a minimal BaseProcessRunner subclass instance for _watch tests."""
    from larkhelm.runner_base import BaseProcessRunner

    class _StubRunner(BaseProcessRunner):
        def build_args(self): return ["true"]
        def build_stdin(self): return None
        def parse_stdout_event(self, ev): return False
        def cleanup_extra(self): pass

    r = _StubRunner.__new__(_StubRunner)
    BaseProcessRunner.__init__(
        r, backend_name="stub", chat_id="c1", message="m", sid=None, cwd="/tmp",
    )
    r._proc = MagicMock()
    r._proc.kill = MagicMock()
    return r


class IdleWatchTests(unittest.TestCase):
    """``BaseProcessRunner._watch`` — idle-based hard/soft timeouts."""

    def test_continuous_activity_never_trips_hard_timeout(self):
        """The whole point of the fix: long-running task that keeps emitting
        output never gets killed even after total wall-clock > HARD_TIMEOUT."""
        r = _make_base_runner()
        on_soft = MagicMock()
        r.on_soft_timeout = on_soft

        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()

        # Simulate stable output for ~3× HARD_TIMEOUT, ticking activity well
        # below the timeout windows. With the OLD wall-clock implementation
        # the proc.kill() call would fire around HARD_TIMEOUT seconds in.
        deadline = time.time() + _cfg.HARD_TIMEOUT * 3
        while time.time() < deadline:
            r._touch_activity()
            time.sleep(0.2)
        r._completed.set()
        watcher.join(timeout=2)

        r._proc.kill.assert_not_called()
        on_soft.assert_not_called()

    def test_silence_trips_soft_timeout(self):
        r = _make_base_runner()
        on_soft = MagicMock()
        r.on_soft_timeout = on_soft

        # Force "no activity since last_activity_ts" to be ≥ RESPONSE_TIMEOUT
        # by backdating last_activity_ts manually.
        with r._activity_lock:
            r._last_activity_ts = time.time() - (_cfg.RESPONSE_TIMEOUT + 1)

        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()
        # Give the watcher at most ~2s to notice (it polls every 0.3s).
        for _ in range(20):
            if r._soft_timeout_flag.is_set():
                break
            time.sleep(0.1)
        r._completed.set()
        watcher.join(timeout=2)
        self.assertTrue(r._soft_timeout_flag.is_set())
        on_soft.assert_called_once()
        r._proc.kill.assert_not_called()  # hard hasn't elapsed yet

    def test_silence_beyond_hard_kills_process(self):
        r = _make_base_runner()

        with r._activity_lock:
            r._last_activity_ts = time.time() - (_cfg.HARD_TIMEOUT + 1)

        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()
        for _ in range(20):
            if r._proc.kill.called:
                break
            time.sleep(0.1)
        watcher.join(timeout=2)
        r._proc.kill.assert_called()

    def test_activity_after_soft_fire_does_not_re_arm_soft(self):
        """Soft fires at most once per run. Re-arming would release the
        chat lock twice, confusing the outer ``_do_query`` flow."""
        r = _make_base_runner()
        on_soft = MagicMock()
        r.on_soft_timeout = on_soft

        with r._activity_lock:
            r._last_activity_ts = time.time() - (_cfg.RESPONSE_TIMEOUT + 1)
        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()
        for _ in range(20):
            if r._soft_timeout_flag.is_set():
                break
            time.sleep(0.1)
        # Resume activity; soft must not fire again.
        for _ in range(15):
            r._touch_activity()
            time.sleep(0.1)
        r._completed.set()
        watcher.join(timeout=2)
        self.assertEqual(on_soft.call_count, 1)

    def test_cancel_event_short_circuits(self):
        r = _make_base_runner()
        cancel_ev = threading.Event()
        r.cancel_ev = cancel_ev

        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()
        time.sleep(0.5)
        cancel_ev.set()
        watcher.join(timeout=2)
        self.assertTrue(r._cancelled_flag.is_set())
        r._proc.kill.assert_called()


class RunStdoutLoopTouchesActivityTests(unittest.TestCase):
    """``run()`` main stdout loop must call ``_touch_activity`` per line."""

    def test_stdout_iteration_refreshes_activity(self):
        from larkhelm.runner_base import BaseProcessRunner

        # We don't actually call run() (it would Popen a real process);
        # we exercise the *contract* by inspecting the source for the
        # _touch_activity invocation in the stdout loop. This is a
        # cheap regression guard against someone removing the call.
        import inspect
        src = inspect.getsource(BaseProcessRunner.run)
        self.assertIn("_touch_activity()", src,
            "run() must call _touch_activity() on each stdout line")

    def test_drain_stderr_refreshes_activity(self):
        from larkhelm.runner_base import BaseProcessRunner
        import inspect
        src = inspect.getsource(BaseProcessRunner._drain_stderr)
        self.assertIn("_touch_activity()", src,
            "_drain_stderr must call _touch_activity() on each stderr line")


class DeepSeekIdleWatchTests(unittest.TestCase):
    """DeepSeek runner has its own _watch (no Popen) — verify it too uses idle."""

    def _make_ds_runner(self):
        from larkhelm.runner_deepseek import DeepSeekRunner
        r = DeepSeekRunner.__new__(DeepSeekRunner)
        # Skip __init__ — it requires a real spec / HTTP setup. Set only
        # the attributes _watch / _touch_activity touch.
        r.backend_name = "DeepSeek"
        r.chat_id = "c1"
        r.cancel_ev = None
        r.on_soft_timeout = None
        r._completed = threading.Event()
        r._cancelled_flag = threading.Event()
        r._soft_timeout_flag = threading.Event()
        r._activity_lock = threading.Lock()
        r._last_activity_ts = time.time()
        return r

    def test_deepseek_continuous_activity_never_cancels(self):
        r = self._make_ds_runner()
        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()
        deadline = time.time() + _cfg.HARD_TIMEOUT * 2
        while time.time() < deadline:
            r._touch_activity()
            time.sleep(0.2)
        r._completed.set()
        watcher.join(timeout=2)
        self.assertFalse(r._cancelled_flag.is_set())

    def test_deepseek_silence_beyond_hard_cancels(self):
        r = self._make_ds_runner()
        with r._activity_lock:
            r._last_activity_ts = time.time() - (_cfg.HARD_TIMEOUT + 1)
        watcher = threading.Thread(target=r._watch, daemon=True)
        watcher.start()
        for _ in range(20):
            if r._cancelled_flag.is_set():
                break
            time.sleep(0.1)
        watcher.join(timeout=2)
        self.assertTrue(r._cancelled_flag.is_set())


if __name__ == "__main__":
    unittest.main()
