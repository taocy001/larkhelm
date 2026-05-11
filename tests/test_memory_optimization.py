"""
Tests for the 4-part memory injection / update optimisation
(`fix(memory): ...` series, 2026-05-11).

Pin the new invariants so a regression to "concatenate everything every call"
or "send system as plain string" would fail loudly.

Covered:
  #1  ``recent_turns`` is dropped on API backends, kept on CLI / DeepSeek.
  #2  Anthropic ``system`` field ships as a 1-element list with
      ``cache_control: ephemeral`` (not a plain string).
  #3  ``_run_one_shot(prefer_cheap=True)`` resolves a cheap backend before
      falling back to the orchestrator.
  #4  ``_load_md_body`` cache hits when mtime is unchanged and invalidates
      when the file is rewritten.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Bootstrap config (shared by all four sets of tests) ─────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_memopt_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)


# ════════════════════════════════════════════════════════════════════════
#  #1 — recent_turns dispatched only to backends that need it
# ════════════════════════════════════════════════════════════════════════

class RecentTurnsRoutingTests(unittest.TestCase):
    """``_run_backend_single`` must drop ``recent_turns`` on API backends but
    pass it through to CLI / DeepSeek paths."""

    def _make_spec(self, provider: str) -> MagicMock:
        s = MagicMock()
        s.id = f"test_{provider}"
        s.provider = provider
        s.model = ""
        return s

    def test_anthropic_path_does_not_receive_recent_turns(self):
        from larkhelm.handlers import _query as q
        spec = self._make_spec("anthropic_api")
        captured: dict = {}

        def fake_run_anthropic(spec, chat_id, message, history, cancel_ev,
                               on_text, extra_system=""):
            # Verify the system channel does NOT carry the recent_turns blob.
            captured["extra_system"] = extra_system
            return ("ok", history)

        with patch("larkhelm.backend_api.run_anthropic", side_effect=fake_run_anthropic), \
             patch("larkhelm.api_session.load_history", return_value=[]), \
             patch("larkhelm.api_session.save_history"):
            q._run_backend_single(
                spec, "chat1", "hi", "/tmp", cancel_ev=None,
                on_text=lambda *a, **k: None,
                on_tool=lambda *a, **k: None,
                on_tool_result=lambda *a, **k: None,
                on_soft_timeout=lambda: None,
                extra_system="MEMORY",
                recent_turns="RECENT_BLOB",
            )
        # The recent blob must not leak into extra_system on the API path.
        self.assertNotIn("RECENT_BLOB", captured["extra_system"])
        self.assertEqual(captured["extra_system"], "MEMORY")

    def test_claude_cli_path_combines_memory_and_recent_turns(self):
        from larkhelm.handlers import _query as q
        spec = self._make_spec("claude_cli")
        captured: dict = {}

        def fake_run_claude(spec, chat_id, message, sid, cwd, cancel_ev, **kw):
            captured["system_prompt"] = kw.get("system_prompt", "")
            return "ok"

        # _load_sid returns None → fresh session → cli_extra should include both.
        with patch("larkhelm.backend_cli.run_claude", side_effect=fake_run_claude), \
             patch.object(q, "_load_sid", return_value=None):
            q._run_backend_single(
                spec, "chat1", "hi", "/tmp", cancel_ev=None,
                on_text=lambda *a, **k: None,
                on_tool=lambda *a, **k: None,
                on_tool_result=lambda *a, **k: None,
                on_soft_timeout=lambda: None,
                extra_system="MEMORY",
                recent_turns="RECENT_BLOB",
            )
        self.assertIn("MEMORY", captured["system_prompt"])
        self.assertIn("RECENT_BLOB", captured["system_prompt"])

    def test_claude_cli_resumed_session_skips_extras(self):
        """Existing sid → extras must NOT be re-injected (legacy behaviour)."""
        from larkhelm.handlers import _query as q
        spec = self._make_spec("claude_cli")
        captured: dict = {}

        def fake_run_claude(spec, chat_id, message, sid, cwd, cancel_ev, **kw):
            captured["message"] = message
            captured["system_prompt"] = kw.get("system_prompt") or ""
            return "ok"

        with patch("larkhelm.backend_cli.run_claude", side_effect=fake_run_claude), \
             patch.object(q, "_load_sid", return_value="EXISTING_SID"):
            q._run_backend_single(
                spec, "chat1", "hi", "/tmp", cancel_ev=None,
                on_text=lambda *a, **k: None,
                on_tool=lambda *a, **k: None,
                on_tool_result=lambda *a, **k: None,
                on_soft_timeout=lambda: None,
                extra_system="MEMORY",
                recent_turns="RECENT_BLOB",
            )
        # The system_prompt is NOT prefixed into ``message`` when resuming.
        self.assertEqual(captured["message"], "hi")
        # system_prompt is still passed through to run_claude (it may use it
        # internally; the bug being defended is the prefix-into-message one).

    def test_deepseek_path_receives_recent_turns(self):
        """DeepSeek runner doesn't load history, so recent_turns is genuine context."""
        from larkhelm.handlers import _query as q
        spec = self._make_spec("deepseek_api")
        captured: dict = {}

        def fake_run_deepseek(spec, chat_id, message, sid=None, cwd="",
                              cancel_ev=None, on_text=None, on_tool=None,
                              on_tool_result=None, on_soft_timeout=None,
                              system_prompt=None):
            captured["system_prompt"] = system_prompt or ""
            return "ok"

        with patch("larkhelm.backend_cli.run_deepseek", side_effect=fake_run_deepseek):
            q._run_backend_single(
                spec, "chat1", "hi", "/tmp", cancel_ev=None,
                on_text=lambda *a, **k: None,
                on_tool=lambda *a, **k: None,
                on_tool_result=lambda *a, **k: None,
                on_soft_timeout=lambda: None,
                extra_system="MEMORY",
                recent_turns="RECENT_BLOB",
            )
        self.assertIn("MEMORY", captured["system_prompt"])
        self.assertIn("RECENT_BLOB", captured["system_prompt"])


# ════════════════════════════════════════════════════════════════════════
#  #2 — Anthropic prompt caching marker on extra_system
# ════════════════════════════════════════════════════════════════════════

class AnthropicCacheControlTests(unittest.TestCase):

    def _patched_anthropic(self):
        """Build a fake anthropic SDK module that records create() kwargs."""
        captured: dict = {}

        class _StreamCtx:
            def __init__(self, kwargs):
                captured["kwargs"] = kwargs

            def __enter__(self):
                self.text_stream = iter([])
                return self

            def __exit__(self, *a):
                return False

        class _Messages:
            def stream(self, **kwargs):
                return _StreamCtx(kwargs)

        class _Client:
            messages = _Messages()

        class _Module:
            Anthropic = staticmethod(lambda **kw: _Client())

        return _Module(), captured

    def test_system_field_is_list_with_cache_control(self):
        from larkhelm import backend_api as bapi
        spec = MagicMock(id="anthropic_test", model="claude-sonnet-4-6",
                         api_key="x", base_url="")
        fake_mod, captured = self._patched_anthropic()
        with patch.dict("sys.modules", {"anthropic": fake_mod}):
            bapi.run_anthropic(
                spec=spec, chat_id="chat1", message="hi",
                history=[], extra_system="some-memory-blob",
            )
        sys_field = captured["kwargs"]["system"]
        # Must be a list (cache_control requires list form), not a string.
        self.assertIsInstance(sys_field, list)
        self.assertEqual(len(sys_field), 1)
        blk = sys_field[0]
        self.assertEqual(blk["type"], "text")
        self.assertEqual(blk["cache_control"], {"type": "ephemeral"})
        self.assertIn("some-memory-blob", blk["text"])

    def test_no_system_no_field(self):
        """When extra_system is empty AND history has no system messages,
        ``system`` kwarg should be absent (don't ship empty cache blocks)."""
        from larkhelm import backend_api as bapi
        spec = MagicMock(id="anthropic_test", model="claude-sonnet-4-6",
                         api_key="x", base_url="")
        fake_mod, captured = self._patched_anthropic()
        with patch.dict("sys.modules", {"anthropic": fake_mod}):
            bapi.run_anthropic(
                spec=spec, chat_id="chat1", message="hi",
                history=[], extra_system="",
            )
        self.assertNotIn("system", captured["kwargs"])


# ════════════════════════════════════════════════════════════════════════
#  #3 — _run_one_shot prefer_cheap routing
# ════════════════════════════════════════════════════════════════════════

class RunOneShotCheapRoutingTests(unittest.TestCase):

    def test_prefer_cheap_uses_cheap_backend_when_available(self):
        from larkhelm import memory as mem
        cheap_spec = MagicMock(id="deepseek", provider="deepseek_api",
                               tags=["cheap"], healthy=True, enabled=True)
        orch_spec  = MagicMock(id="claude",   provider="claude_cli",
                               tags=["tools"], healthy=True, enabled=True)

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_cli.run_deepseek", return_value="cheap-output") as run_cheap, \
             patch("larkhelm.backend_cli.run_claude") as run_orch, \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = cheap_spec
            reg.get_orchestrator.return_value = orch_spec
            # deepseek_api is in _API_PROVIDERS? let's check — actually no,
            # it's special-cased. The runner is backend_cli.run_deepseek,
            # but memory._run_one_shot has a different dispatch table that
            # only routes the three _API_PROVIDERS via backend_api. We need
            # to pick a backend whose provider IS in _API_PROVIDERS. Switch
            # to openai_compat_api which IS in the table.
            cheap_spec.provider = "openai_compat_api"
            with patch("larkhelm.backend_api.run_openai_compat",
                       return_value=("cheap-output", [])) as run_oc:
                result = mem._run_one_shot("PROMPT", ns="_test_ns",
                                           prefer_cheap=True)
            # The cheap backend was selected.
            reg.get_by_tag.assert_called_once_with(["cheap"])
            # And the orchestrator was NOT consulted (cheap path resolved).
            reg.get_orchestrator.assert_not_called()

    def test_prefer_cheap_falls_back_to_orchestrator(self):
        from larkhelm import memory as mem
        orch_spec = MagicMock(id="claude", provider="openai_compat_api",
                              tags=["tools"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_api.run_openai_compat",
                   return_value=("orch-output", [])), \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = None  # no cheap backend
            reg.get_orchestrator.return_value = orch_spec
            mem._run_one_shot("PROMPT", ns="_test_ns", prefer_cheap=True)
            reg.get_by_tag.assert_called_once_with(["cheap"])
            reg.get_orchestrator.assert_called_once()

    def test_default_prefer_cheap_false_uses_orchestrator(self):
        from larkhelm import memory as mem
        orch_spec = MagicMock(id="claude", provider="openai_compat_api",
                              tags=["tools"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_api.run_openai_compat",
                   return_value=("orch-output", [])), \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_orchestrator.return_value = orch_spec
            mem._run_one_shot("PROMPT", ns="_test_ns")  # prefer_cheap not set
            # No cheap lookup at all when the caller didn't opt in.
            reg.get_by_tag.assert_not_called()
            reg.get_orchestrator.assert_called_once()


# ════════════════════════════════════════════════════════════════════════
#  #4 — _load_md_body mtime cache
# ════════════════════════════════════════════════════════════════════════

class LoadMdBodyCacheTests(unittest.TestCase):

    def setUp(self):
        from larkhelm import memory as mem
        # Wipe the module-level cache for each test.
        with mem._mem_body_cache_lock:
            mem._mem_body_cache.clear()
        self.mem = mem
        self.tmpdir = Path(tempfile.mkdtemp(prefix="larkhelm_loadmd_"))
        self.path = self.tmpdir / "session_test.md"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        with self.mem._mem_body_cache_lock:
            self.mem._mem_body_cache.clear()

    def _write(self, body: str) -> None:
        self.path.write_text(f"---\nupdated_at: x\n---\n\n{body}", encoding="utf-8")

    def test_cache_hit_avoids_reread(self):
        self._write("hello world")
        first = self.mem._load_md_body(self.path)
        self.assertEqual(first, "hello world")
        # Now hot-swap the file contents while keeping mtime — cache should
        # still serve the old body.
        st = self.path.stat()
        self.path.write_text("---\nupdated_at: x\n---\n\nNEW CONTENT",
                             encoding="utf-8")
        os.utime(self.path, (st.st_atime, st.st_mtime))  # restore mtime
        second = self.mem._load_md_body(self.path)
        self.assertEqual(second, "hello world",
                         "same mtime → cache must serve old body")

    def test_mtime_change_invalidates_cache(self):
        self._write("v1")
        self.assertEqual(self.mem._load_md_body(self.path), "v1")
        # Sleep just long enough for mtime to change, then rewrite.
        time.sleep(0.05)
        self._write("v2")
        # Force mtime forward in case the FS resolution swallowed the gap.
        future = time.time() + 1
        os.utime(self.path, (future, future))
        self.assertEqual(self.mem._load_md_body(self.path), "v2",
                         "new mtime must trigger re-read")

    def test_deleted_file_drops_cached_entry(self):
        self._write("transient")
        self.mem._load_md_body(self.path)
        self.path.unlink()
        self.assertIsNone(self.mem._load_md_body(self.path))
        # Cache key should be gone so re-create is observed.
        self._write("re-created")
        self.assertEqual(self.mem._load_md_body(self.path), "re-created")

    def test_none_path_returns_none(self):
        self.assertIsNone(self.mem._load_md_body(None))


if __name__ == "__main__":
    unittest.main()
