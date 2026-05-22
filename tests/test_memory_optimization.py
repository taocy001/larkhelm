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
            captured["system_prompt"] = kw.get("system_prompt")
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
        # On a resumed session, system_prompt must be None — not re-injected.
        self.assertIsNone(captured["system_prompt"])

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
        bapi.run_anthropic(
            spec=spec, chat_id="chat1", message="hi",
            history=[], extra_system="some-memory-blob",
            _anthropic_module=fake_mod,
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
        bapi.run_anthropic(
            spec=spec, chat_id="chat1", message="hi",
            history=[], extra_system="",
            _anthropic_module=fake_mod,
        )
        self.assertNotIn("system", captured["kwargs"])


# ════════════════════════════════════════════════════════════════════════
#  #3 — _run_one_shot prefer_cheap routing
# ════════════════════════════════════════════════════════════════════════

class RunOneShotCheapRoutingTests(unittest.TestCase):
    """``_run_one_shot`` cheap-backend selection + DeepSeek dispatch + fallback.

    The historically wrong version of this test class used
    ``provider="openai_compat_api"`` to stand in for the cheap backend,
    sidestepping the real production case where ``get_by_tag(["cheap"])``
    returns DeepSeek (``provider="deepseek_api"``). DeepSeek's runner lives
    in ``backend_cli`` (not ``backend_api``), and the dispatch table used to
    miss it — silently routing every cascade call through ``run_claude``
    with a DeepSeek spec attached. The current tests exercise the real
    DeepSeek path so a regression to "cascade quietly uses claude" would
    fail loudly.
    """

    def test_prefer_cheap_dispatches_deepseek_via_backend_cli(self):
        """The high-value production case: cheap = DeepSeek (deepseek_api)."""
        from larkhelm import memory as mem
        cheap_spec = MagicMock(id="deepseek", provider="deepseek_api",
                               tags=["cheap", "fast"], healthy=True, enabled=True)
        orch_spec  = MagicMock(id="claude", provider="claude_cli",
                               tags=["tools"], healthy=True, enabled=True)

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_cli.run_deepseek", return_value="cheap-output") as run_deepseek, \
             patch("larkhelm.backend_cli.run_claude") as run_claude, \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = cheap_spec
            reg.get_orchestrator.return_value = orch_spec
            result = mem._run_one_shot("PROMPT", ns="_test_ns", prefer_cheap=True)
        self.assertEqual(result, "cheap-output")
        # The cheap backend was selected.
        reg.get_by_tag.assert_called_once_with(["cheap"])
        # The orchestrator was NOT consulted (cheap path resolved).
        reg.get_orchestrator.assert_not_called()
        # ▶ Critical regression guard: ``run_deepseek`` was called, NOT
        #   ``run_claude``. The pre-fix bug routed deepseek_api into the CLI
        #   else-branch and spawned claude with a DeepSeek spec.
        self.assertEqual(run_deepseek.call_count, 1,
            "deepseek_api must dispatch to backend_cli.run_deepseek, not run_claude")
        run_claude.assert_not_called()
        # And run_deepseek received the cheap spec, not some downgrade.
        _args, kwargs = run_deepseek.call_args
        self.assertIs(kwargs.get("spec"), cheap_spec)
        self.assertEqual(kwargs.get("chat_id"), "_test_ns")
        self.assertEqual(kwargs.get("message"), "PROMPT")
        self.assertIsNone(kwargs.get("sid"))

    def test_prefer_cheap_dispatches_openai_compat_via_backend_api(self):
        """Non-DeepSeek cheap backend (eg. openai_compat tagged ``cheap``)
        must still go through backend_api, not get redirected to deepseek."""
        from larkhelm import memory as mem
        cheap_spec = MagicMock(id="cheap-oc", provider="openai_compat_api",
                               tags=["cheap"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_api.run_openai_compat",
                   return_value=("oc-output", [])) as run_oc, \
             patch("larkhelm.backend_cli.run_deepseek") as run_ds, \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = cheap_spec
            result = mem._run_one_shot("PROMPT", ns="_test_ns", prefer_cheap=True)
        self.assertEqual(result, "oc-output")
        run_oc.assert_called_once()
        run_ds.assert_not_called()

    def test_prefer_cheap_falls_back_to_orchestrator_when_no_cheap_available(self):
        from larkhelm import memory as mem
        orch_spec = MagicMock(id="claude", provider="openai_compat_api",
                              tags=["tools"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_api.run_openai_compat",
                   return_value=("orch-output", [])) as run_orch, \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = None  # no cheap backend healthy
            reg.get_orchestrator.return_value = orch_spec
            result = mem._run_one_shot("PROMPT", ns="_test_ns", prefer_cheap=True)
        self.assertEqual(result, "orch-output")
        reg.get_by_tag.assert_called_once_with(["cheap"])
        reg.get_orchestrator.assert_called_once()
        run_orch.assert_called_once()

    def test_cheap_runtime_failure_falls_back_to_orchestrator(self):
        """If the cheap backend is selected but raises at run-time (quota
        exhausted, network error, model 5xx), we must retry once via the
        orchestrator. Without this, a flaky DeepSeek would silently drop
        memory updates for the entire chat."""
        from larkhelm import memory as mem
        cheap_spec = MagicMock(id="deepseek", provider="deepseek_api",
                               tags=["cheap"], healthy=True, enabled=True)
        orch_spec  = MagicMock(id="claude", provider="openai_compat_api",
                               tags=["tools"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_cli.run_deepseek",
                   side_effect=RuntimeError("simulated 429")) as run_ds, \
             patch("larkhelm.backend_api.run_openai_compat",
                   return_value=("orch-rescue", [])) as run_orch, \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = cheap_spec
            reg.get_orchestrator.return_value = orch_spec
            result = mem._run_one_shot("PROMPT", ns="_test_ns", prefer_cheap=True)
        self.assertEqual(result, "orch-rescue",
            "cheap failure must transparently fall back to orchestrator")
        run_ds.assert_called_once()
        run_orch.assert_called_once()

    def test_cheap_failure_with_no_orchestrator_raises(self):
        """No fallback target → original cheap exception propagates so the
        cascade caller can log and skip."""
        from larkhelm import memory as mem
        cheap_spec = MagicMock(id="deepseek", provider="deepseek_api",
                               tags=["cheap"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_cli.run_deepseek",
                   side_effect=RuntimeError("simulated 500")), \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_by_tag.return_value = cheap_spec
            reg.get_orchestrator.return_value = None  # no fallback target
            with self.assertRaises(RuntimeError):
                mem._run_one_shot("PROMPT", ns="_test_ns", prefer_cheap=True)

    def test_orchestrator_only_failure_does_not_loop(self):
        """When prefer_cheap=False, an orchestrator failure must propagate
        directly — no implicit retry. Implicit retry would mask backend
        health issues from callers that rely on the exception to mark
        orchestrator unhealthy."""
        from larkhelm import memory as mem
        orch_spec = MagicMock(id="claude", provider="openai_compat_api",
                              tags=["tools"], healthy=True, enabled=True)
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY") as reg, \
             patch("larkhelm.backend_api.run_openai_compat",
                   side_effect=RuntimeError("orch fail")) as run_orch, \
             patch("larkhelm.chat_state._clear_sid"):
            reg.get_orchestrator.return_value = orch_spec
            with self.assertRaises(RuntimeError):
                mem._run_one_shot("PROMPT", ns="_test_ns")  # prefer_cheap=False
        self.assertEqual(run_orch.call_count, 1,
            "non-cheap path must not retry — caller relies on first failure")

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


# ════════════════════════════════════════════════════════════════════════
#  #5 — Cascade UNCHANGED short-circuit (S53) + cascade coordinator (S43)
# ════════════════════════════════════════════════════════════════════════


class CascadeShortCircuitTests(unittest.TestCase):
    """``_should_skip_extract_by_hash`` + ``_try_extract_*`` integration."""

    def test_hash_match_with_close_len_returns_true(self):
        from larkhelm import memory as mem
        session = "x" * 500
        cur_hash = mem._session_hash(session)
        prev_fm = {
            "last_extracted_session_hash": cur_hash,
            "last_extracted_session_len": "500",
        }
        self.assertTrue(mem._should_skip_extract_by_hash(prev_fm, session))

    def test_hash_match_far_len_returns_false(self):
        from larkhelm import memory as mem
        session = "x" * 500
        cur_hash = mem._session_hash(session)
        prev_fm = {
            "last_extracted_session_hash": cur_hash,
            "last_extracted_session_len": "100",
        }
        self.assertFalse(mem._should_skip_extract_by_hash(prev_fm, session))

    def test_hash_mismatch_returns_false(self):
        from larkhelm import memory as mem
        prev_fm = {"last_extracted_session_hash": "deadbeef" + "0" * 8,
                   "last_extracted_session_len": "500"}
        self.assertFalse(mem._should_skip_extract_by_hash(prev_fm, "x" * 500))

    def test_missing_fields_returns_false(self):
        from larkhelm import memory as mem
        self.assertFalse(mem._should_skip_extract_by_hash({}, "x" * 100))
        self.assertFalse(mem._should_skip_extract_by_hash(
            {"last_extracted_session_hash": ""}, "x" * 100))

    def test_short_circuit_skips_one_shot(self):
        """When the hash matches, ``_try_extract_project`` must return WITHOUT
        invoking ``_run_one_shot``. This is the central token-saving path."""
        from larkhelm import memory as mem
        session = "## Work Context\n" + ("body line\n" * 20)
        cur_hash = mem._session_hash(session)
        with patch.object(mem, "_load_md_frontmatter", return_value={
                "last_extracted_session_hash": cur_hash,
                "last_extracted_session_len": str(len(session)),
             }), \
             patch.object(mem, "_run_one_shot") as run_one_shot:
            mem._try_extract_project(session, "/tmp/x")
        run_one_shot.assert_not_called()

    def test_short_circuit_can_be_disabled_by_config(self):
        from larkhelm import memory as mem
        session = "## Work Context\nbody"
        cur_hash = mem._session_hash(session)
        with patch.dict(_cfg.config, {"memory_cascade_shortcircuit": False}, clear=False):
            self.assertFalse(mem._should_skip_extract_by_hash({
                "last_extracted_session_hash": cur_hash,
                "last_extracted_session_len": str(len(session)),
            }, session))


class CascadeCoordinatorTests(unittest.TestCase):
    """``_cascade_extract`` cancel-event + BoundedSemaphore semantics."""

    def setUp(self):
        from larkhelm import memory as mem
        # Reset coordinator state between tests.
        with mem._active_cancels_lock:
            mem._active_cascade_cancels.clear()
        with mem._CASCADE_SEM_LOCK:
            mem._CASCADE_SEM = None

    def test_new_cascade_cancels_prior(self):
        """Submitting a new cascade for the same chat MUST set the prior
        cascade's cancel event."""
        from larkhelm import memory as mem
        prior_ev = __import__("threading").Event()
        with mem._active_cancels_lock:
            mem._active_cascade_cancels["chat_x"] = prior_ev
        with patch("larkhelm.chat_state._get_cwd", return_value=None), \
             patch.object(mem, "_run_one_shot", return_value="UNCHANGED"):
            mem._cascade_extract("session content", "chat_x")
        # Give the spawned cascade a moment to wire up the new event.
        import time as _t
        _t.sleep(0.05)
        self.assertTrue(prior_ev.is_set(),
                        "prior cancel event must be set when new cascade arrives")

    def test_sem_busy_skips_extract(self):
        """When the semaphore is exhausted, the worker MUST exit without
        calling ``_run_one_shot`` (otherwise we'd burst past the configured
        concurrency cap)."""
        import threading as _t
        from larkhelm import memory as mem
        # Squeeze the sem to 1 for this test.
        with mem._CASCADE_SEM_LOCK:
            mem._CASCADE_SEM = _t.BoundedSemaphore(1)
        # Hold the sole permit ourselves.
        mem._CASCADE_SEM.acquire()
        try:
            with patch("larkhelm.chat_state._get_cwd", return_value=None), \
                 patch.object(mem, "_load_md_frontmatter", return_value={}), \
                 patch.object(mem, "_run_one_shot") as run_one_shot:
                mem._cascade_extract("payload", "chat_y")
                # Wait long enough for the worker to time out on sem.acquire.
                import time as _time
                _time.sleep(1.0)
            run_one_shot.assert_not_called()
        finally:
            try:
                mem._CASCADE_SEM.release()
            except ValueError:
                pass


if __name__ == "__main__":
    unittest.main()
