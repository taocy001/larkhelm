"""Coverage for Phase 4 of the logging unification:

  * ``LARKHELM_LOG_LEVEL`` env-var parsing (DEBUG/INFO/WARN/ERROR + bad values).
  * ``_log_at`` gate behavior — writes below ``_min_level`` are dropped.
  * ``info`` / ``warn`` / ``error`` helpers prepend ``<LEVEL>`` tag while
    ``_debug_log`` keeps the legacy untagged format (so 250+ existing call
    sites remain grep-compatible).
  * ``lark_client._fetch_bot_open_id`` no longer double-writes to stderr;
    the failure paths now route through ``warn(...)`` only.
"""
from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from larkhelm import log as larkhelm_log


def _capture_log_lines(callable_, *args, **kwargs) -> list[str]:
    """Run ``callable_(*args, **kwargs)`` against an isolated DEBUG_LOG file
    and return the lines that were written. Patches ``_cfg.DEBUG_LOG`` to a
    tempfile so the developer's real log isn't touched by tests."""
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "larkhelm.log"
        with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True):
            callable_(*args, **kwargs)
            if not log_path.exists():
                return []
            return log_path.read_text(encoding="utf-8").splitlines()


# ── Level parsing ──────────────────────────────────────────────────────


class TestResolveLevelFromEnv(unittest.TestCase):

    def _reload_at(self, env_value: str | None) -> larkhelm_log.Level:
        """Re-evaluate ``_resolve_level_from_env`` with a controlled env."""
        env = dict(os.environ)
        if env_value is None:
            env.pop("LARKHELM_LOG_LEVEL", None)
        else:
            env["LARKHELM_LOG_LEVEL"] = env_value
        with patch.dict(os.environ, env, clear=True):
            return larkhelm_log._resolve_level_from_env()

    def test_unset_defaults_to_debug(self):
        self.assertIs(self._reload_at(None), larkhelm_log.Level.DEBUG)

    def test_empty_defaults_to_debug(self):
        self.assertIs(self._reload_at(""), larkhelm_log.Level.DEBUG)

    def test_each_canonical_value(self):
        for name in ("DEBUG", "INFO", "WARN", "ERROR"):
            self.assertIs(self._reload_at(name), larkhelm_log.Level(name))

    def test_lowercase_normalized(self):
        self.assertIs(self._reload_at("warn"), larkhelm_log.Level.WARN)

    def test_whitespace_stripped(self):
        self.assertIs(self._reload_at("  ERROR  "), larkhelm_log.Level.ERROR)

    def test_unknown_value_falls_back_to_debug_with_stderr_warning(self):
        # Capture stderr writes via a tiny spy that records each chunk
        # print() emits. print() calls .write() one or more times per
        # invocation, so we concatenate all captured chunks before asserting.
        captured: list[str] = []

        class _StderrSpy:
            def write(self, s):
                captured.append(s)

            def flush(self):
                pass

        with patch("sys.stderr", _StderrSpy()):
            level = self._reload_at("VERBOSE")
        self.assertIs(level, larkhelm_log.Level.DEBUG)

        # The fallback path MUST emit an operator-visible warning so a
        # typo'd LARKHELM_LOG_LEVEL doesn't silently default to DEBUG
        # without trace. Assert both the bad value and the fallback
        # mention appear in stderr.
        joined = "".join(captured)
        self.assertIn("VERBOSE", joined,
                      f"stderr should echo the bad value; got: {joined!r}")
        self.assertIn("falling back to DEBUG", joined,
                      f"stderr should announce the fallback; got: {joined!r}")

    def test_current_log_level_is_module_level(self):
        """``current_log_level()`` returns the module's resolved level
        (frozen at import time, intentionally NOT live-updated)."""
        self.assertIsInstance(larkhelm_log.current_log_level(), larkhelm_log.Level)


# ── Gate behavior ──────────────────────────────────────────────────────


class TestLogLevelGate(unittest.TestCase):

    def setUp(self):
        # Reset write counter for predictable output sizes.
        larkhelm_log._debug_write_count = 0

    def _set_level(self, level: larkhelm_log.Level):
        return patch.object(larkhelm_log, "_min_level", level)

    def test_debug_gate_drops_below_threshold(self):
        with self._set_level(larkhelm_log.Level.WARN):
            lines = _capture_log_lines(larkhelm_log._debug_log, "[Foo] noisy")
        self.assertEqual(lines, [], "DEBUG must be filtered out at WARN gate")

    def test_warn_emitted_at_warn_threshold(self):
        with self._set_level(larkhelm_log.Level.WARN):
            lines = _capture_log_lines(larkhelm_log.warn, "[Foo] hot")
        self.assertEqual(len(lines), 1)
        self.assertIn("<WARN>", lines[0])
        self.assertIn("[Foo] hot", lines[0])

    def test_info_below_warn_threshold_dropped(self):
        with self._set_level(larkhelm_log.Level.WARN):
            lines = _capture_log_lines(larkhelm_log.info, "[Foo] background")
        self.assertEqual(lines, [])

    def test_error_passes_at_error_threshold(self):
        with self._set_level(larkhelm_log.Level.ERROR):
            lines = _capture_log_lines(larkhelm_log.error, "[Foo] kaboom")
        self.assertEqual(len(lines), 1)
        self.assertIn("<ERROR>", lines[0])

    def test_warn_dropped_at_error_threshold(self):
        with self._set_level(larkhelm_log.Level.ERROR):
            lines = _capture_log_lines(larkhelm_log.warn, "[Foo] mostly OK")
        self.assertEqual(lines, [])

    def test_debug_threshold_lets_all_through(self):
        with self._set_level(larkhelm_log.Level.DEBUG):
            lines_d = _capture_log_lines(larkhelm_log._debug_log, "[Foo] d")
            lines_i = _capture_log_lines(larkhelm_log.info, "[Foo] i")
            lines_w = _capture_log_lines(larkhelm_log.warn, "[Foo] w")
            lines_e = _capture_log_lines(larkhelm_log.error, "[Foo] e")
        self.assertTrue(lines_d and lines_i and lines_w and lines_e)


# ── Output format ──────────────────────────────────────────────────────


class TestOutputFormat(unittest.TestCase):

    def setUp(self):
        larkhelm_log._debug_write_count = 0

    def test_debug_format_is_untagged_for_grep_compat(self):
        """The 250+ existing _debug_log call sites assume the line format
        ``[HH:MM:SS] [Module] msg``. Phase 4 must NOT prepend <DEBUG> or
        any new tag that would invalidate ``grep '\\[Memory\\]' DEBUG_LOG``
        and similar tooling."""
        with patch.object(larkhelm_log, "_min_level", larkhelm_log.Level.DEBUG):
            lines = _capture_log_lines(larkhelm_log._debug_log, "[Memory] test")
        self.assertEqual(len(lines), 1)
        line = lines[0]
        # Format: [HH:MM:SS] [Module] msg  — no level tag.
        self.assertRegex(line, r"^\[\d\d:\d\d:\d\d\] \[Memory\] test$")
        self.assertNotIn("<DEBUG>", line)

    def test_info_warn_error_carry_explicit_tag(self):
        with patch.object(larkhelm_log, "_min_level", larkhelm_log.Level.DEBUG):
            for fn, expected in [
                (larkhelm_log.info, "<INFO>"),
                (larkhelm_log.warn, "<WARN>"),
                (larkhelm_log.error, "<ERROR>"),
            ]:
                lines = _capture_log_lines(fn, "[Test] sample")
                self.assertEqual(len(lines), 1)
                self.assertIn(expected, lines[0])
                self.assertIn("[Test] sample", lines[0])

    def test_safe_log_and_lazy_debug_log_still_route_through_gate(self):
        """The two no-raise variants both delegate to ``_debug_log``; the
        gate must apply equally so an operator can silence them too."""
        with patch.object(larkhelm_log, "_min_level", larkhelm_log.Level.WARN):
            lines_safe = _capture_log_lines(larkhelm_log.safe_log, "[Test] should be filtered")
            lines_lazy = _capture_log_lines(larkhelm_log.lazy_debug_log, "[Test] also filtered")
        self.assertEqual(lines_safe, [])
        self.assertEqual(lines_lazy, [])


# ── lark_client _fetch_bot_open_id no longer prints to stderr ──────────


class TestLarkClientWarnConsolidation(unittest.TestCase):
    """The 3 ``print(..., file=sys.stderr)`` sites in
    ``_fetch_bot_open_id`` (paired with redundant ``_debug_log``) are now
    a single ``warn(...)`` call. Verify that:
      1. No bare print() to stderr fires from those paths.
      2. The DEBUG_LOG receives exactly ONE entry per failure (was two)
         tagged ``<WARN>``.
    """

    def _stub_client(self, **resp_attrs):
        """Build a fake lark.Client.request that returns the given resp."""
        from larkhelm import lark_client
        fake_resp = MagicMock(**resp_attrs)
        fake_client = MagicMock()
        fake_client.request.return_value = fake_resp
        return patch.object(lark_client, "client", fake_client)

    def _capture(self, fn) -> tuple[list[str], list[str]]:
        """Run ``fn()`` capturing both stderr writes and DEBUG_LOG lines."""
        from larkhelm import lark_client
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "larkhelm.log"
            stderr_writes: list[str] = []

            class _StderrSpy:
                def write(self, s):
                    stderr_writes.append(s)

                def flush(self):
                    pass

            # NOTE: lark_client no longer imports ``sys`` (it used to, for
            # the ``print(file=sys.stderr)`` calls that this PR consolidated
            # into ``warn(...)``). So we only need to patch the global
            # ``sys.stderr``; there's no module-local ``lark_client.sys``
            # reference to spy on anymore. If a future change re-introduces
            # ``import sys`` in lark_client, the static check
            # ``test_no_print_to_stderr_anywhere_in_lark_client`` still
            # catches the regression.
            with patch.object(larkhelm_log._cfg, "DEBUG_LOG", log_path, create=True), \
                 patch.object(larkhelm_log, "_min_level", larkhelm_log.Level.DEBUG), \
                 patch("sys.stderr", _StderrSpy()):
                fn()
            log_lines = log_path.read_text().splitlines() if log_path.exists() else []
        return stderr_writes, log_lines

    def test_no_open_id_in_response_writes_warn_only(self):
        from larkhelm import lark_client

        bad_resp = MagicMock()
        bad_resp.raw = MagicMock()
        bad_resp.raw.content = b'{"bot": {}}'
        with self._stub_client(), \
             patch.object(lark_client, "client") as fc:
            fc.request.return_value = bad_resp
            stderr_writes, log_lines = self._capture(lark_client._fetch_bot_open_id)

        # No stderr print (the print(file=sys.stderr) is gone).
        self.assertNotIn("BOT_OPEN_ID fetch failed", "".join(stderr_writes))
        # Exactly ONE WARN line in the log (was previously: 1 stderr + 1 _debug_log).
        warn_lines = [l for l in log_lines if "<WARN>" in l and "BOT_OPEN_ID" in l]
        self.assertEqual(len(warn_lines), 1, f"got: {log_lines}")

    def test_request_exception_writes_warn_only(self):
        from larkhelm import lark_client

        with patch.object(lark_client, "client") as fc:
            fc.request.side_effect = RuntimeError("API down")
            stderr_writes, log_lines = self._capture(lark_client._fetch_bot_open_id)

        self.assertNotIn("BOT_OPEN_ID fetch", "".join(stderr_writes))
        warn_lines = [l for l in log_lines if "<WARN>" in l and "exception" in l]
        self.assertEqual(len(warn_lines), 1, f"got: {log_lines}")

    def test_no_print_to_stderr_anywhere_in_lark_client(self):
        """Static check: ``lark_client.py`` must contain zero
        ``print(..., file=sys.stderr)`` references after the cleanup.
        Catches future regressions where someone re-introduces the
        double-write idiom."""
        src = Path("larkhelm/lark_client.py").read_text(encoding="utf-8")
        self.assertNotIn("file=sys.stderr", src,
                         "lark_client must route warnings through warn(), "
                         "not print(file=sys.stderr)")


if __name__ == "__main__":
    unittest.main()
