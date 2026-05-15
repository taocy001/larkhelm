"""Bug 1 — probe subprocess timeout is INDETERMINATE, not healthy.

Pre-fix (the bug this file pins): ``_probe_gemini`` and ``_probe_claude``
returned ``(True, "")`` on ``subprocess.TimeoutExpired`` with rationale
"slow start = model exists". In practice this masked real outages: a
hung Gemini CLI (because the upstream API is down or rate-limited)
took 12 s to time out, then got reported as healthy.

Post-fix: timeout returns ``(None, "subprocess timeout")``.
``BackendRegistry.set_probe_result(spec_id, None, ...)`` accepts the
sentinel and does NOT mutate ``healthy`` — only refreshes
``last_probed_at``. Real-traffic ``record_call_failure`` becomes the
authoritative health signal until the next definitive probe result.
"""
from __future__ import annotations

import atexit
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="larkhelm_probe_indet_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm import model_probe
from larkhelm.backend_registry import BackendSpec, BackendRegistry


class ProbeTimeoutReturnsIndeterminateTests(unittest.TestCase):
    """All three CLI probes (gemini / claude / kimi) must return
    ``(None, ...)`` — NOT ``(True, "")`` — when the subprocess times
    out. The sentinel comment ``# slow start = model exists`` was the
    rationale for the bug; this test pins the corrected behaviour."""

    def _patch_subprocess_timeout(self):
        """Force ``subprocess.run`` to raise ``TimeoutExpired``."""
        return patch.object(
            model_probe.subprocess, "run",
            side_effect=subprocess.TimeoutExpired(cmd=["x"], timeout=12),
        )

    def test_gemini_timeout_returns_none(self):
        spec = BackendSpec(
            id="gemini", provider="gemini_cli", display_name="Gemini",
            role="orchestrator", tags=["tools"], command="gemini",
        )
        with self._patch_subprocess_timeout():
            ok, err = model_probe._probe_gemini(spec)
        self.assertIsNone(ok,
                          f"timeout must be INDETERMINATE (None), not True/False. got {(ok, err)!r}")
        self.assertIn("timeout", err.lower())

    def test_claude_timeout_returns_none(self):
        spec = BackendSpec(
            id="claude", provider="claude_cli", display_name="Claude",
            role="orchestrator", tags=["vision", "tools"], command="claude",
        )
        with self._patch_subprocess_timeout():
            ok, err = model_probe._probe_claude(spec)
        self.assertIsNone(ok, f"got {(ok, err)!r}")
        self.assertIn("timeout", err.lower())

    def test_kimi_timeout_returns_none(self):
        spec = BackendSpec(
            id="kimi", provider="kimi_cli", display_name="Kimi",
            role="worker", tags=["vision", "tools"], command="kimi",
        )
        with self._patch_subprocess_timeout():
            ok, err = model_probe._probe_kimi(spec)
        self.assertIsNone(ok, f"got {(ok, err)!r}")
        self.assertIn("timeout", err.lower())


class SetProbeResultNoneIsNonMutatingTests(unittest.TestCase):
    """``BackendRegistry.set_probe_result(spec_id, None, ...)`` must
    NOT mutate the healthy flag — only update ``last_probed_at``.
    """

    def _registry_with_spec(self, healthy: bool) -> tuple[BackendRegistry, BackendSpec]:
        reg = BackendRegistry()
        spec = BackendSpec(
            id="gemini-test", provider="gemini_cli", display_name="Gemini",
            role="orchestrator", tags=["tools"], command="gemini",
            healthy=healthy, enabled=True,
        )
        with reg._lock:
            reg._specs[spec.id] = spec
        return reg, spec

    def test_none_does_not_flip_healthy_to_false(self):
        """A probe timeout on a currently-healthy backend must KEEP it
        healthy until real-traffic signals say otherwise."""
        reg, spec = self._registry_with_spec(healthy=True)
        reg.set_probe_result(spec.id, None, "subprocess timeout")
        self.assertTrue(spec.healthy,
                        "indeterminate probe must not flip healthy=True → False")
        self.assertIn("indeterminate", (spec.last_error or "").lower(),
                      "last_error must surface the indeterminate state for /status")

    def test_none_does_not_resurrect_unhealthy(self):
        """A probe timeout on an unhealthy backend must NOT silently
        re-mark it healthy — this is the exact bug we're fixing."""
        reg, spec = self._registry_with_spec(healthy=False)
        spec.last_error = "previous auth failure"
        reg.set_probe_result(spec.id, None, "subprocess timeout")
        self.assertFalse(spec.healthy,
                         "indeterminate probe must not flip healthy=False → True")

    def test_none_still_updates_last_probed_at(self):
        """The probe tick DID run — staleness timestamp must advance
        so the recover thread doesn't keep re-probing in a tight loop."""
        import time as _time
        reg, spec = self._registry_with_spec(healthy=True)
        spec.last_probed_at = 0.0
        before = _time.time()
        reg.set_probe_result(spec.id, None, "subprocess timeout")
        self.assertGreaterEqual(spec.last_probed_at, before,
                                "indeterminate probe must still update last_probed_at")

    def test_true_clears_indeterminate_error(self):
        """When a real probe succeeds AFTER a previous indeterminate
        timeout, last_error must clear (not retain the stale
        'probe indeterminate: ...' string)."""
        reg, spec = self._registry_with_spec(healthy=True)
        reg.set_probe_result(spec.id, None, "subprocess timeout")
        self.assertIsNotNone(spec.last_error)
        reg.set_probe_result(spec.id, True, "")
        self.assertIsNone(spec.last_error)
        self.assertTrue(spec.healthy)


# ─────────────────────────────────────────────────────────────────────────
# Round-1 review #2: kimi rc=0 with no stream-json events also INDETERMINATE
# ─────────────────────────────────────────────────────────────────────────


class KimiProbeNoEventIsIndeterminateTests(unittest.TestCase):
    """`_probe_kimi`'s Step-3 "rc=0 but no events" path was a second
    false-positive vector (same bug class as the TimeoutExpired path).
    A kimi-cli that hangs internally then exits clean with no output
    used to be silently marked healthy."""

    def test_rc_zero_with_no_events_returns_none(self):
        spec = BackendSpec(
            id="kimi", provider="kimi_cli", display_name="Kimi",
            role="worker", tags=["tools"], command="kimi",
        )
        # Mock subprocess.run to return (rc=0, stdout="", stderr="") —
        # the exact production shape that pre-fix returned (True, "").
        class _FakeResult:
            returncode = 0
            stdout = ""
            stderr = ""
        with patch.object(model_probe.subprocess, "run", return_value=_FakeResult()):
            ok, err = model_probe._probe_kimi(spec)
        self.assertIsNone(ok,
                          f"rc=0 with no stream-json must be INDETERMINATE, "
                          f"not True. got {(ok, err)!r}")
        self.assertIn("no stream-json event", err)

    def test_rc_zero_WITH_events_still_returns_true(self):
        """Sanity: rc=0 with at least one structured event is the
        happy path and must still mark healthy."""
        spec = BackendSpec(
            id="kimi", provider="kimi_cli", display_name="Kimi",
            role="worker", tags=["tools"], command="kimi",
        )
        class _FakeResult:
            returncode = 0
            stdout = '{"type": "system", "msg": "hello"}\n'
            stderr = ""
        with patch.object(model_probe.subprocess, "run", return_value=_FakeResult()):
            ok, err = model_probe._probe_kimi(spec)
        self.assertTrue(ok, f"valid stream-json event must mark healthy. got {(ok, err)!r}")


# ─────────────────────────────────────────────────────────────────────────
# Round-1 review #3: API SDK probe timeouts INDETERMINATE (symmetric w/ CLI)
# ─────────────────────────────────────────────────────────────────────────


class ClassifyAPIExceptionTests(unittest.TestCase):
    """The new ``_classify_api_exception`` helper maps SDK exceptions
    to probe-result tuples. Timeout / network exceptions go to
    INDETERMINATE (None); auth / quota / schema stay DEFINITE (False).
    """

    def test_timeout_exceptions_are_indeterminate(self):
        cases = [
            TimeoutError("read timeout after 12s"),
            RuntimeError("APITimeoutError: deadline exceeded"),
            ConnectionError("operation timed out"),
        ]
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                ok, err = model_probe._classify_api_exception(exc)
                self.assertIsNone(ok,
                                  f"timeout-shape {exc!r} → must be INDETERMINATE, got {(ok, err)!r}")

    def test_network_exceptions_are_indeterminate(self):
        cases = [
            ConnectionRefusedError("Connection refused"),
            RuntimeError("[Errno -2] Name or service not known"),
            RuntimeError("SSL handshake failed"),
        ]
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                ok, err = model_probe._classify_api_exception(exc)
                self.assertIsNone(ok,
                                  f"network-shape {exc!r} → must be INDETERMINATE, got {(ok, err)!r}")

    def test_auth_quota_exceptions_stay_definite(self):
        cases = [
            RuntimeError("401 Unauthorized: invalid api_key"),
            RuntimeError("429 Too Many Requests"),
            ValueError("model not found: gpt-9000"),
            RuntimeError("HTTP 500 internal server error"),
        ]
        for exc in cases:
            with self.subTest(exc=type(exc).__name__):
                ok, err = model_probe._classify_api_exception(exc)
                self.assertFalse(ok,
                                 f"non-timeout {exc!r} → must be DEFINITE False, got {(ok, err)!r}")


if __name__ == "__main__":
    unittest.main()
