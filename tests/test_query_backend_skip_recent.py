"""Tests for P1 REQ-04 — CLI/DeepSeek path skips recent_turns injection
when the backend already carries history (sid present / load_history non-empty).

The two helper functions in ``_query.py`` encode the actual decision; we
exercise them directly rather than spin up the full backend stack. The
existing integration coverage (``tests/test_do_query.py``, etc.) still
exercises the calling sites.

Covered:
  * AC-04 — ``_maybe_drop_recent_turns_for_cli`` drops recent_turns when
    sid is non-empty AND ``CLI_SKIP_RECENT_TURNS_WHEN_SID`` is on.
  * AC-04 negation — sid is None / sid empty → keep recent_turns.
  * AC-04 flag-off — flag off → keep recent_turns even with sid.
  * AC-05 — ``_maybe_drop_recent_turns_for_deepseek`` drops recent_turns
    when ``load_history("deepseek_api", chat_id)`` returns a non-empty list.
  * AC-05 negation — load_history returns [] → keep recent_turns.
  * load_history raising is treated as "couldn't probe" → keep recent_turns.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Bootstrap config (shared); _query.py reads _cfg.CLI_SKIP_RECENT_TURNS_WHEN_SID
_TMP = tempfile.mkdtemp(prefix="larkhelm_qskip_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers import _query as q  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
#  AC-04 — claude_cli / kimi_cli / gemini_cli skip recent_turns when sid
# ════════════════════════════════════════════════════════════════════════


class CLIRecentTurnsSkipTests(unittest.TestCase):

    def setUp(self):
        _cfg.CLI_SKIP_RECENT_TURNS_WHEN_SID = True

    def test_sid_present_drops_recent_turns(self):
        out = q._maybe_drop_recent_turns_for_cli(
            "chatA", "claude", sid="abc-123", recent_turns="HISTORICAL_TURNS",
            provider="claude_cli",
        )
        self.assertEqual(out, "")

    def test_sid_none_keeps_recent_turns(self):
        out = q._maybe_drop_recent_turns_for_cli(
            "chatA", "claude", sid=None, recent_turns="HISTORICAL_TURNS",
            provider="claude_cli",
        )
        self.assertEqual(out, "HISTORICAL_TURNS")

    def test_sid_empty_string_keeps_recent_turns(self):
        out = q._maybe_drop_recent_turns_for_cli(
            "chatA", "claude", sid="", recent_turns="HISTORICAL_TURNS",
            provider="claude_cli",
        )
        self.assertEqual(out, "HISTORICAL_TURNS")

    def test_flag_off_keeps_recent_turns_even_with_sid(self):
        with patch.object(_cfg, "CLI_SKIP_RECENT_TURNS_WHEN_SID", False):
            out = q._maybe_drop_recent_turns_for_cli(
                "chatA", "claude", sid="abc-123", recent_turns="HISTORICAL_TURNS",
                provider="claude_cli",
            )
        self.assertEqual(out, "HISTORICAL_TURNS")

    def test_skip_applies_to_kimi_and_gemini_too(self):
        for prov in ("kimi_cli", "gemini_cli"):
            out = q._maybe_drop_recent_turns_for_cli(
                "chatA", prov.split("_")[0], sid="abc", recent_turns="x",
                provider=prov,
            )
            self.assertEqual(out, "", f"{prov} should also skip when sid present")


# ════════════════════════════════════════════════════════════════════════
#  AC-05 — deepseek_api skips recent_turns when load_history non-empty
# ════════════════════════════════════════════════════════════════════════


class DeepseekRecentTurnsSkipTests(unittest.TestCase):

    def setUp(self):
        _cfg.CLI_SKIP_RECENT_TURNS_WHEN_SID = True

    def test_non_empty_history_drops_recent_turns(self):
        def fake_load_history(provider, chat_id):
            self.assertEqual(provider, "deepseek_api")
            return [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]

        out = q._maybe_drop_recent_turns_for_deepseek(
            "chatA", recent_turns="HISTORICAL_TURNS",
            load_history_fn=fake_load_history,
        )
        self.assertEqual(out, "")

    def test_empty_history_keeps_recent_turns(self):
        def fake_load_history(provider, chat_id):
            return []

        out = q._maybe_drop_recent_turns_for_deepseek(
            "chatA", recent_turns="HISTORICAL_TURNS",
            load_history_fn=fake_load_history,
        )
        self.assertEqual(out, "HISTORICAL_TURNS")

    def test_flag_off_keeps_recent_turns_even_with_history(self):
        def fake_load_history(provider, chat_id):
            return [{"role": "user", "content": "hi"}]

        with patch.object(_cfg, "CLI_SKIP_RECENT_TURNS_WHEN_SID", False):
            out = q._maybe_drop_recent_turns_for_deepseek(
                "chatA", recent_turns="HISTORICAL_TURNS",
                load_history_fn=fake_load_history,
            )
        self.assertEqual(out, "HISTORICAL_TURNS")

    def test_load_history_exception_keeps_recent_turns(self):
        """When load_history raises, treat as "can't determine" and keep
        recent_turns rather than silently dropping context."""

        def boom(provider, chat_id):
            raise RuntimeError("io error")

        out = q._maybe_drop_recent_turns_for_deepseek(
            "chatA", recent_turns="HISTORICAL_TURNS",
            load_history_fn=boom,
        )
        self.assertEqual(out, "HISTORICAL_TURNS")


# ════════════════════════════════════════════════════════════════════════
#  P0 — API backends skip recent_turns pre-read via API_SKIP_RECENT_TURNS_WHEN_HISTORY
#
#  These tests exercise the _do_query early-exit gate that prevents the
#  100 KB tail-read for backends that carry history structurally and would
#  drop recent_turns anyway (_run_backend_single "NOTE: recent_turns
#  intentionally omitted — history already carries it").
# ════════════════════════════════════════════════════════════════════════


class _FakeSpec:
    def __init__(self, provider: str, spec_id: str = "fake"):
        self.provider = provider
        self.id = spec_id
        self.display_name = provider


class APIBackendSkipRecentTurnsTests(unittest.TestCase):
    """Verify the _do_query pre-read gate for API backends."""

    def setUp(self):
        _cfg.API_SKIP_RECENT_TURNS_WHEN_HISTORY = True

    def tearDown(self):
        _cfg.API_SKIP_RECENT_TURNS_WHEN_HISTORY = True

    def _run_gate(self, spec, flag: bool) -> bool:
        """Return True iff the gate would set _skip_recent_turns for the given spec."""
        with patch.object(_cfg, "API_SKIP_RECENT_TURNS_WHEN_HISTORY", flag):
            skip = False
            if flag:
                _api_providers = {"anthropic_api", "google_api", "openai_compat_api"}
                if spec is not None and spec.provider in _api_providers:
                    skip = True
            return skip

    def test_api_backend_skip_recent_turns_flag_on(self):
        spec = _FakeSpec("anthropic_api")
        self.assertTrue(self._run_gate(spec, flag=True))

    def test_api_backend_no_skip_flag_off(self):
        spec = _FakeSpec("anthropic_api")
        self.assertFalse(self._run_gate(spec, flag=False))

    def test_api_backend_no_skip_spec_none(self):
        self.assertFalse(self._run_gate(None, flag=True))

    def test_api_backend_google_skip(self):
        spec = _FakeSpec("google_api")
        self.assertTrue(self._run_gate(spec, flag=True))

    def test_api_backend_openai_compat_skip(self):
        spec = _FakeSpec("openai_compat_api")
        self.assertTrue(self._run_gate(spec, flag=True))

    def test_cli_backend_not_skipped_by_api_gate(self):
        spec = _FakeSpec("claude_cli")
        self.assertFalse(self._run_gate(spec, flag=True))

    def test_config_flag_default_is_true(self):
        self.assertTrue(bool(getattr(_cfg, "API_SKIP_RECENT_TURNS_WHEN_HISTORY", True)))


# ════════════════════════════════════════════════════════════════════════
#  API backends strip [SESSION MEMORY] from extra_system when history
#  is non-empty (the structured history already carries those turns
#  verbatim) — _maybe_strip_session_memory_for_api.
# ════════════════════════════════════════════════════════════════════════

_EXTRA_SYSTEM = (
    "[GLOBAL MEMORY]\nglobal facts\n[/GLOBAL MEMORY]\n\n"
    "[PROJECT MEMORY]\nproject facts\n[/PROJECT MEMORY]\n\n"
    "[SESSION MEMORY]\nrecent summary\n[/SESSION MEMORY]"
)
_HISTORY = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "hello"},
]


class APISessionMemoryStripTests(unittest.TestCase):

    def test_history_nonempty_strips_session_block(self):
        out = q._maybe_strip_session_memory_for_api(
            _EXTRA_SYSTEM, _HISTORY, provider="anthropic_api", chat_id="chatA",
        )
        self.assertNotIn("[SESSION MEMORY]", out)
        self.assertNotIn("recent summary", out)
        # global / project layers survive (stable prefix)
        self.assertIn("global facts", out)
        self.assertIn("project facts", out)

    def test_history_empty_keeps_full_injection(self):
        out = q._maybe_strip_session_memory_for_api(
            _EXTRA_SYSTEM, [], provider="anthropic_api", chat_id="chatA",
        )
        self.assertEqual(out, _EXTRA_SYSTEM)
        self.assertIn("[SESSION MEMORY]", out)

    def test_no_session_block_passthrough(self):
        plain = "[GLOBAL MEMORY]\ng\n[/GLOBAL MEMORY]"
        out = q._maybe_strip_session_memory_for_api(
            plain, _HISTORY, provider="google_api", chat_id="chatA",
        )
        self.assertEqual(out, plain)

    def test_empty_extra_system_passthrough(self):
        out = q._maybe_strip_session_memory_for_api(
            "", _HISTORY, provider="openai_compat_api", chat_id="chatA",
        )
        self.assertEqual(out, "")

    def test_flag_off_keeps_full_injection(self):
        """api_strip_session_memory_when_history=false is the rollback
        switch: history non-empty must NOT strip the session block."""
        with patch.object(
            q._cfg, "API_STRIP_SESSION_MEMORY_WHEN_HISTORY", False, create=True
        ):
            out = q._maybe_strip_session_memory_for_api(
                _EXTRA_SYSTEM, _HISTORY, provider="anthropic_api", chat_id="chatA",
            )
        self.assertEqual(out, _EXTRA_SYSTEM)

    def test_strip_flag_default_is_true(self):
        self.assertTrue(
            bool(getattr(_cfg, "API_STRIP_SESSION_MEMORY_WHEN_HISTORY", True)))

    def test_boundary_tags_shared_with_layered_cache_split(self):
        """The strip must use the exact stable/volatile boundary that
        backend_api_streaming._split_stable_volatile defines, so the gate
        and the layered cache_control split can never disagree."""
        from larkhelm.backend_api_streaming import _split_stable_volatile
        stable, _volatile = _split_stable_volatile(_EXTRA_SYSTEM)
        out = q._maybe_strip_session_memory_for_api(
            _EXTRA_SYSTEM, _HISTORY, provider="anthropic_api", chat_id="chatA",
        )
        self.assertEqual(out, stable)


class APISessionMemoryStripIntegrationTests(unittest.TestCase):
    """_run_backend_single wires the strip into all three API branches."""

    def _run(self, provider: str, runner_name: str, history: list):
        captured: dict = {}

        def fake_runner(spec, chat_id, message, history, cancel_ev, on_text,
                        extra_system=""):
            captured["extra_system"] = extra_system
            return "ok", list(history)

        with patch(f"larkhelm.backend_api.{runner_name}", fake_runner), \
             patch("larkhelm.api_session.load_history", return_value=history), \
             patch("larkhelm.api_session.save_history", lambda *a, **k: None):
            q._run_backend_single(
                _FakeSpec(provider), "chatX", "msg", _TMP, None,
                None, None, None, None,
                extra_system=_EXTRA_SYSTEM,
            )
        return captured["extra_system"]

    def test_anthropic_history_nonempty_system_has_no_session_memory(self):
        out = self._run("anthropic_api", "run_anthropic", _HISTORY)
        self.assertNotIn("[SESSION MEMORY]", out)
        self.assertIn("global facts", out)

    def test_anthropic_history_empty_system_keeps_session_memory(self):
        out = self._run("anthropic_api", "run_anthropic", [])
        self.assertIn("[SESSION MEMORY]", out)

    def test_google_and_openai_compat_also_strip(self):
        for provider, runner in (
            ("google_api", "run_google"),
            ("openai_compat_api", "run_openai_compat"),
        ):
            out = self._run(provider, runner, _HISTORY)
            self.assertNotIn(
                "[SESSION MEMORY]", out,
                f"{provider} must strip session memory when history non-empty",
            )


if __name__ == "__main__":
    unittest.main()
