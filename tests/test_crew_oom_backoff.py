"""Regression guards for OOM-aware retry backoff in ``_run_agent_wrapper``.

History
-------
Pre-fix: ``except Exception → time.sleep(1) → retry``. For
non-resource failures (HTTP transient, parse error, etc.) 1s is fine.
For OOM-class failures (cgroup OOM-killer, V8 heap exhaustion), 1s is
NOT enough — the kernel hasn't reclaimed page cache yet, the cgroup
is still near memory.max, and the second attempt typically OOMs the
same way. Confirmed in commit b3a116f's morning /dev run: implementer
attempt 1 OOM-killed, attempt 2 attempted within 1s and failed before
even getting to the work.

Post-fix: detect OOM-shape errors via known message markers (the two
sources we actually emit — runner_base._on_kill_signal SIGKILL wrap
+ V8 "heap limit" stderr), and back off 8s instead of 1s. Also
snapshot cgroup memory state into the debug log for later forensics.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Config bootstrap
_TMP = tempfile.mkdtemp(prefix="larkhelm_oom_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.crew import _runner as cr


class IsLikelyOomErrorTests(unittest.TestCase):
    """``_is_likely_oom_error`` matches the OOM signatures emitted
    by the runner_base subprocess wrapper + V8 directly."""

    def test_matches_runner_base_killed_by_os(self):
        """The exact wording ``runner_base._on_kill_signal`` produces
        when the OS sent SIGKILL — covers the cgroup OOM path."""
        e = RuntimeError("claude killed by OS (rc=-9, likely cgroup OOM). "
                         "Check dmesg for 'task=claude ... oom-kill'.")
        self.assertTrue(cr._is_likely_oom_error(e))

    def test_matches_v8_heap_limit_phrase(self):
        e = RuntimeError(
            "abnormal exit rc=134\n"
            "FATAL ERROR: Reached heap limit Allocation failed - "
            "JavaScript heap out of memory"
        )
        self.assertTrue(cr._is_likely_oom_error(e))

    def test_matches_v8_ineffective_mark_compacts(self):
        e = RuntimeError(
            "FATAL ERROR: Ineffective mark-compacts near heap limit"
        )
        self.assertTrue(cr._is_likely_oom_error(e))

    def test_matches_case_insensitive(self):
        """Production logs sometimes capitalise differently across
        versions of node / claude CLI. Matcher must be lower-case."""
        e = RuntimeError("Process KILLED BY OS (rc=-9)")
        self.assertTrue(cr._is_likely_oom_error(e))

    def test_does_not_match_unrelated_failures(self):
        cases = [
            RuntimeError("HTTPSConnectionPool: SSL EOF"),
            RuntimeError("JSON parse error at line 3"),
            TimeoutError("query timeout after 300s"),
            ValueError("expected dict, got str"),
            RuntimeError("backend 'gemini' is disabled in config"),
        ]
        for e in cases:
            self.assertFalse(cr._is_likely_oom_error(e),
                f"non-OOM error {e!r} must NOT match — would back off "
                "8s unnecessarily on every transient failure")

    def test_garbage_exception_does_not_raise(self):
        """A weird exception whose __str__ raises shouldn't make the
        detector crash — retry path is hot, must stay defensive."""
        class WeirdExc(Exception):
            def __str__(self):
                raise RuntimeError("don't call me")
        # Falls through to False (default conservative behaviour)
        self.assertFalse(cr._is_likely_oom_error(WeirdExc()))


class LogOomDiagnosticsTests(unittest.TestCase):
    """``_log_oom_diagnostics`` snapshots cgroup memory into debug
    log. Fail-soft on any IO problem."""

    def test_missing_cgroup_dir_is_silent_no_op(self):
        """Non-systemd host / dev box without the larkhelm.service
        cgroup → silently skip rather than raise."""
        with patch("pathlib.Path.is_dir", return_value=False):
            # Must not raise
            cr._log_oom_diagnostics("test_agent")

    def test_disk_error_does_not_propagate(self):
        """A perms problem reading memory.current shouldn't compound
        the failure we're trying to diagnose."""
        with patch.object(cr, "_debug_log") as log_mock, \
             patch("pathlib.Path.is_dir", return_value=True), \
             patch("pathlib.Path.read_text",
                   side_effect=PermissionError("nope")):
            cr._log_oom_diagnostics("test_agent")
        # Should have logged with "?" placeholders for unreadable files,
        # not crashed
        self.assertTrue(log_mock.called,
            "diagnostic helper must still emit a log line even when "
            "every file is unreadable — the agent_id + 'OOM diagnostic' "
            "marker is the grep anchor")
        # All values should be '?' placeholders
        body = log_mock.call_args.args[0]
        self.assertIn("OOM diagnostic", body)
        self.assertIn("memory.current=?", body)


class BackoffBehaviourTests(unittest.TestCase):
    """End-to-end: an OOM-class exception on attempt 1 must trigger
    the 8s backoff, not 1s. Other exceptions keep the 1s behaviour."""

    def _make_state(self):
        """Minimal CrewState scaffolding for the wrapper. Uses real
        types so the function under test sees the same shape as
        production."""
        from larkhelm.crew_types import (
            AgentSpec, AgentState, AgentStatus, CrewState, CrewPlan,
        )
        import threading
        spec = AgentSpec(id="impl", role="r", model="claude",
                         system="", prompt="", depends_on=[], timeout=60,
                         output_file="changes.md")
        return CrewState(
            crew_id="t", chat_id="oc_t",
            plan=CrewPlan(title="t", agents=[spec]),
            agents={"impl": AgentState(spec=spec, status=AgentStatus.PENDING)},
            cancel_ev=threading.Event(),
            lock=threading.Lock(),
        )

    def _run_with_first_attempt_raising(self, exc: Exception):
        """Run ``_run_agent_wrapper`` with first call raising ``exc``
        and second call returning ``"ok"``. Return (sleep_duration,
        diag_called) tuples."""
        state = self._make_state()
        call_count = {"n": 0}

        def _fake_run_agent(state, agent_id):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise exc
            return "ok"

        slept: list[float] = []
        # ``_git_auto_commit`` is imported from ``crew._state`` inside the
        # function body — patch at its real module location, not on the
        # _runner namespace (where the symbol doesn't exist).
        import larkhelm.crew._state as _state_mod
        with patch.object(cr, "_run_agent", side_effect=_fake_run_agent), \
             patch.object(cr.time, "sleep", side_effect=slept.append), \
             patch.object(cr, "_log_oom_diagnostics") as diag, \
             patch.object(cr, "_sync_output_file", return_value=""), \
             patch.object(_state_mod, "_git_auto_commit", return_value=""), \
             patch.object(cr, "_crew_update_card"), \
             patch.object(cr, "_detect_fail_marker", return_value=False):
            cr._run_agent_wrapper(state, "impl")
        return slept, diag.called

    def test_oom_error_triggers_8s_backoff(self):
        oom_exc = RuntimeError(
            "claude killed by OS (rc=-9, likely cgroup OOM). "
            "Check `systemctl status larkhelm` for details."
        )
        slept, diag_called = self._run_with_first_attempt_raising(oom_exc)
        self.assertEqual(slept, [8],
            f"OOM-class failure should back off 8s; got slept={slept!r}")
        self.assertTrue(diag_called,
            "OOM path must call _log_oom_diagnostics for forensics")

    def test_non_oom_error_keeps_1s_backoff(self):
        unrelated = RuntimeError("HTTPSConnectionPool: SSL EOF")
        slept, diag_called = self._run_with_first_attempt_raising(unrelated)
        self.assertEqual(slept, [1],
            f"non-OOM failure should keep legacy 1s backoff; got slept={slept!r}")
        self.assertFalse(diag_called,
            "non-OOM path must NOT spam OOM diagnostics — those are "
            "forensics for memory issues, not general transient retries")


if __name__ == "__main__":
    unittest.main()
