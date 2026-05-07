"""
larkhelm Phase 1 module test suite.

Coverage:
  - api_session: truncate_history, save/load/clear_history
  - backend_registry: BackendRegistry.load (tags parsing), get_by_tag, get_orchestrator, all_enabled, health_check
  - router: resolve_backend (all 5 routing rules)
  - memory: maybe_auto_update trigger conditions
"""
import atexit
import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Config init ─────────────────────────────────────────────────────────────
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_phase1_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_DUMMY_CONFIG = {
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg_mod
_cfg_mod._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

import larkhelm.api_session as api_session
from larkhelm.backend_registry import BackendRegistry, BackendSpec, _resolve_env_vars


# ═══════════════════════════════════════════════════════════════════════════
#  api_session.py
# ═══════════════════════════════════════════════════════════════════════════
class TestTruncateHistory(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(api_session.truncate_history([]), [])

    def test_short_list_unchanged(self):
        history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]
        result = api_session.truncate_history(history)
        self.assertEqual(result, history)

    def test_over_max_without_system_keeps_latest(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(api_session._MAX_HISTORY + 10)]
        result = api_session.truncate_history(history)
        self.assertEqual(len(result), api_session._MAX_HISTORY)
        # Should keep the tail (latest messages)
        self.assertEqual(result[-1]["content"], f"msg {api_session._MAX_HISTORY + 9}")

    def test_over_max_with_system_preserves_system(self):
        system_msg = {"role": "system", "content": "You are a helpful assistant."}
        rest = [{"role": "user", "content": f"msg {i}"} for i in range(api_session._MAX_HISTORY + 5)]
        history = [system_msg] + rest
        result = api_session.truncate_history(history)
        self.assertEqual(len(result), api_session._MAX_HISTORY)
        self.assertEqual(result[0]["role"], "system")
        self.assertEqual(result[0]["content"], system_msg["content"])

    def test_exactly_max_length_unchanged(self):
        history = [{"role": "user", "content": f"msg {i}"} for i in range(api_session._MAX_HISTORY)]
        result = api_session.truncate_history(history)
        self.assertEqual(len(result), api_session._MAX_HISTORY)

    def test_system_only_no_crash(self):
        history = [{"role": "system", "content": "only system"}]
        result = api_session.truncate_history(history)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], "system")


class TestApiSessionFileOps(unittest.TestCase):
    """Tests for save_history / load_history / clear_history."""

    def setUp(self):
        sessions_dir = _cfg_mod.DATA_DIR / "api_sessions"
        if sessions_dir.exists():
            shutil.rmtree(sessions_dir)

    def test_save_and_load_roundtrip(self):
        history = [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "world"}]
        api_session.save_history("anthropic_api", "chat_save", history)
        loaded = api_session.load_history("anthropic_api", "chat_save")
        self.assertEqual(loaded, history)

    def test_load_missing_returns_empty(self):
        loaded = api_session.load_history("anthropic_api", "no_such_chat_xyz")
        self.assertEqual(loaded, [])

    def test_clear_removes_file(self):
        api_session.save_history("google_api", "chat_clear", [{"role": "user", "content": "x"}])
        api_session.clear_history("google_api", "chat_clear")
        loaded = api_session.load_history("google_api", "chat_clear")
        self.assertEqual(loaded, [])

    def test_clear_nonexistent_no_error(self):
        try:
            api_session.clear_history("anthropic_api", "nonexistent_chat")
        except Exception as e:
            self.fail(f"clear_history raised: {e}")

    def test_save_truncates_over_max(self):
        big_history = [{"role": "user", "content": f"msg {i}"} for i in range(api_session._MAX_HISTORY + 20)]
        api_session.save_history("openai_compat_api", "chat_big", big_history)
        loaded = api_session.load_history("openai_compat_api", "chat_big")
        self.assertLessEqual(len(loaded), api_session._MAX_HISTORY)

    def test_different_providers_independent(self):
        api_session.save_history("anthropic_api", "shared_chat", [{"role": "user", "content": "A"}])
        api_session.save_history("google_api", "shared_chat", [{"role": "user", "content": "G"}])
        a = api_session.load_history("anthropic_api", "shared_chat")
        g = api_session.load_history("google_api", "shared_chat")
        self.assertEqual(a[0]["content"], "A")
        self.assertEqual(g[0]["content"], "G")


# ═══════════════════════════════════════════════════════════════════════════
#  backend_registry.py
# ═══════════════════════════════════════════════════════════════════════════
class TestBackendRegistryLoad(unittest.TestCase):
    def setUp(self):
        self.reg = BackendRegistry()

    def _load(self, specs: list[dict]) -> None:
        self.reg.load(specs)

    def test_load_basic(self):
        self._load([{
            "id": "claude", "provider": "claude_cli",
            "display_name": "Claude", "role": "orchestrator",
            "tags": ["vision", "tools"], "command": "claude",
        }])
        spec = self.reg.get("claude")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.id, "claude")
        self.assertEqual(spec.role, "orchestrator")
        self.assertEqual(spec.tags, ["vision", "tools"])

    def test_tags_as_list(self):
        self._load([{"id": "a", "provider": "claude_cli", "tags": ["vision", "tools"]}])
        self.assertIn("vision", self.reg.get("a").tags)
        self.assertIn("tools", self.reg.get("a").tags)

    def test_tags_as_comma_string(self):
        self._load([{"id": "b", "provider": "claude_cli", "tags": "vision, tools, cheap"}])
        tags = self.reg.get("b").tags
        self.assertIn("vision", tags)
        self.assertIn("tools", tags)
        self.assertIn("cheap", tags)
        self.assertEqual(len(tags), 3)

    def test_tags_default_empty(self):
        self._load([{"id": "c", "provider": "claude_cli"}])
        self.assertEqual(self.reg.get("c").tags, [])

    def test_load_clears_previous(self):
        self._load([{"id": "old", "provider": "claude_cli"}])
        self._load([{"id": "new", "provider": "claude_cli"}])
        self.assertIsNone(self.reg.get("old"))
        self.assertIsNotNone(self.reg.get("new"))

    def test_enabled_defaults_true(self):
        self._load([{"id": "d", "provider": "claude_cli"}])
        self.assertTrue(self.reg.get("d").enabled)

    def test_enabled_false_from_config(self):
        self._load([{"id": "e", "provider": "claude_cli", "enabled": False}])
        self.assertFalse(self.reg.get("e").enabled)

    def test_api_key_env_expansion(self):
        os.environ["_TEST_API_KEY_XYZ"] = "secret123"
        try:
            self._load([{"id": "f", "provider": "anthropic_api", "api_key": "${_TEST_API_KEY_XYZ}"}])
            self.assertEqual(self.reg.get("f").api_key, "secret123")
        finally:
            del os.environ["_TEST_API_KEY_XYZ"]

    def test_api_key_missing_env_stays_placeholder(self):
        # Phase 4: unresolved ${} placeholder → api_key cleared to "", enabled=False
        os.environ.pop("_MISSING_KEY_ZZZZ", None)
        self._load([{"id": "g", "provider": "anthropic_api", "api_key": "${_MISSING_KEY_ZZZZ}"}])
        self.assertEqual(self.reg.get("g").api_key, "")
        self.assertFalse(self.reg.get("g").enabled)


class TestBackendRegistryQueries(unittest.TestCase):
    def setUp(self):
        self.reg = BackendRegistry()
        self.reg.load([
            {"id": "vision_backend", "provider": "claude_cli", "role": "orchestrator",
             "tags": ["vision", "tools"], "command": "claude"},
            {"id": "cheap_backend", "provider": "gemini_cli", "role": "cheap",
             "tags": ["cheap", "fast"], "command": "gemini"},
            {"id": "disabled_backend", "provider": "claude_cli", "role": "worker",
             "tags": ["vision"], "enabled": False, "command": "claude"},
        ])

    def test_get_by_tag_single(self):
        spec = self.reg.get_by_tag(["vision"])
        self.assertIsNotNone(spec)
        self.assertIn("vision", spec.tags)
        self.assertTrue(spec.enabled)

    def test_get_by_tag_multiple_requires_all(self):
        spec = self.reg.get_by_tag(["cheap", "fast"])
        self.assertIsNotNone(spec)
        self.assertEqual(spec.id, "cheap_backend")

    def test_get_by_tag_no_match(self):
        spec = self.reg.get_by_tag(["nonexistent_tag_xyz"])
        self.assertIsNone(spec)

    def test_get_by_tag_excludes_disabled(self):
        # disabled_backend has "vision" but is disabled
        spec = self.reg.get_by_tag(["vision"])
        self.assertNotEqual(spec.id, "disabled_backend")

    def test_get_orchestrator(self):
        spec = self.reg.get_orchestrator()
        self.assertIsNotNone(spec)
        self.assertEqual(spec.role, "orchestrator")

    def test_all_enabled_excludes_disabled(self):
        enabled = self.reg.all_enabled()
        ids = [s.id for s in enabled]
        self.assertIn("vision_backend", ids)
        self.assertIn("cheap_backend", ids)
        self.assertNotIn("disabled_backend", ids)

    def test_get_by_tag_excludes_unhealthy(self):
        spec = self.reg.get("vision_backend")
        spec.healthy = False
        result = self.reg.get_by_tag(["vision"])
        # disabled_backend also has vision but is disabled — so no match
        self.assertIsNone(result)


class TestResolveEnvVars(unittest.TestCase):
    def test_no_placeholders(self):
        self.assertEqual(_resolve_env_vars("plain string"), "plain string")

    def test_known_env_var(self):
        os.environ["_TEST_RESOLVE_VAR"] = "resolved"
        try:
            self.assertEqual(_resolve_env_vars("prefix_${_TEST_RESOLVE_VAR}_suffix"), "prefix_resolved_suffix")
        finally:
            del os.environ["_TEST_RESOLVE_VAR"]

    def test_missing_env_var_stays(self):
        os.environ.pop("_MISSING_RESOLVE_VAR", None)
        result = _resolve_env_vars("${_MISSING_RESOLVE_VAR}")
        self.assertEqual(result, "${_MISSING_RESOLVE_VAR}")

    def test_empty_string(self):
        self.assertEqual(_resolve_env_vars(""), "")


# ═══════════════════════════════════════════════════════════════════════════
#  router.py — resolve_backend (all 5 routing rules)
# ═══════════════════════════════════════════════════════════════════════════

def _make_spec(id: str, role: str = "worker", tags: list = None,
               healthy: bool = True, enabled: bool = True) -> BackendSpec:
    return BackendSpec(
        id=id, provider="claude_cli", display_name=id,
        role=role, tags=tags or [], command="claude",
        healthy=healthy, enabled=enabled,
    )


class TestResolveBackend(unittest.TestCase):
    """Tests for router.resolve_backend — each routing rule in isolation."""

    def _patch_registry(self, reg: BackendRegistry):
        return patch("larkhelm.router.BACKEND_REGISTRY", reg)

    def _patch_state(self, state: dict):
        return patch("larkhelm.router._get_chat_state", return_value=state)

    def _patch_cheap(self, enabled: bool):
        return patch("larkhelm.router._cfg.config", {"enable_cheap_routing": enabled})

    def test_rule1_images_route_to_vision(self):
        """has_images → get_by_tag(["vision"])"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "vision", "provider": "claude_cli", "role": "orchestrator",
             "tags": ["vision", "tools"], "command": "claude"},
        ])
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(False):
            result = resolve_backend("chat1", "hello", has_images=True)
        self.assertEqual(result.id, "vision")

    def test_rule2_doc_urls_route_to_tools(self):
        """has_doc_urls → get_by_tag(["tools"])"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "tools_backend", "provider": "claude_cli", "role": "orchestrator",
             "tags": ["tools"], "command": "claude"},
        ])
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(False):
            result = resolve_backend("chat1", "hello", has_doc_urls=True)
        self.assertEqual(result.id, "tools_backend")

    def test_rule3_cheap_routing_short_message(self):
        """enable_cheap + short message → get_by_tag(["cheap", "fast"])"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "orchestrator", "provider": "claude_cli", "role": "orchestrator",
             "tags": ["tools"], "command": "claude"},
            {"id": "cheap", "provider": "gemini_cli", "role": "cheap",
             "tags": ["cheap", "fast"], "command": "gemini"},
        ])
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(True):
            result = resolve_backend("chat1", "hi", has_images=False, has_doc_urls=False)
        self.assertEqual(result.id, "cheap")

    def test_rule3_cheap_not_triggered_for_long_message(self):
        """enable_cheap enabled but long message → skip cheap route"""
        from larkhelm.router import resolve_backend, _SHORT_MSG_THRESHOLD
        reg = BackendRegistry()
        reg.load([
            {"id": "orchestrator", "provider": "claude_cli", "role": "orchestrator",
             "tags": ["tools"], "command": "claude"},
            {"id": "cheap", "provider": "gemini_cli", "role": "cheap",
             "tags": ["cheap", "fast"], "command": "gemini"},
        ])
        long_msg = "x" * (_SHORT_MSG_THRESHOLD + 1)
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(True):
            result = resolve_backend("chat1", long_msg)
        self.assertEqual(result.id, "orchestrator")

    def test_rule4_user_preference(self):
        """chat_state backend_id → that backend (if healthy+enabled)"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "orchestrator", "provider": "claude_cli", "role": "orchestrator",
             "tags": [], "command": "claude"},
            {"id": "preferred", "provider": "gemini_cli", "role": "worker",
             "tags": [], "command": "gemini"},
        ])
        with self._patch_registry(reg), self._patch_state({"backend_id": "preferred"}), self._patch_cheap(False):
            result = resolve_backend("chat1", "x" * 200)
        self.assertEqual(result.id, "preferred")

    def test_rule4_unhealthy_preferred_falls_through(self):
        """Preferred backend unhealthy → fall through to orchestrator"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "orchestrator", "provider": "claude_cli", "role": "orchestrator",
             "tags": [], "command": "claude"},
            {"id": "sick", "provider": "gemini_cli", "role": "worker", "tags": [], "command": "gemini"},
        ])
        reg.get("sick").healthy = False
        with self._patch_registry(reg), self._patch_state({"backend_id": "sick"}), self._patch_cheap(False):
            result = resolve_backend("chat1", "x" * 200)
        self.assertEqual(result.id, "orchestrator")

    def test_rule5_orchestrator_fallback(self):
        """No other rule matches → get_orchestrator()"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "orch", "provider": "claude_cli", "role": "orchestrator",
             "tags": [], "command": "claude"},
        ])
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(False):
            result = resolve_backend("chat1", "x" * 200)
        self.assertEqual(result.id, "orch")

    def test_rule5_first_enabled_healthy_fallback(self):
        """No orchestrator → first healthy+enabled backend"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "unhealthy", "provider": "claude_cli", "role": "worker",
             "tags": [], "command": "claude"},
            {"id": "worker2", "provider": "gemini_cli", "role": "worker",
             "tags": [], "command": "gemini"},
        ])
        reg.get("unhealthy").healthy = False
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(False):
            result = resolve_backend("chat1", "x" * 200)
        self.assertEqual(result.id, "worker2")

    def test_no_backends_raises(self):
        """All backends down → RuntimeError"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "only", "provider": "claude_cli", "role": "worker", "tags": [], "command": "claude"},
        ])
        reg.get("only").healthy = False
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(False):
            with self.assertRaises(RuntimeError):
                resolve_backend("chat1", "hi")

    def test_rule1_no_vision_falls_through(self):
        """has_images=True but no vision backend → fall through to orchestrator"""
        from larkhelm.router import resolve_backend
        reg = BackendRegistry()
        reg.load([
            {"id": "orch", "provider": "claude_cli", "role": "orchestrator",
             "tags": ["tools"], "command": "claude"},
        ])
        with self._patch_registry(reg), self._patch_state({}), self._patch_cheap(False):
            result = resolve_backend("chat1", "hi", has_images=True)
        self.assertEqual(result.id, "orch")


# ═══════════════════════════════════════════════════════════════════════════
#  memory.py — maybe_auto_update trigger conditions
# ═══════════════════════════════════════════════════════════════════════════
class TestMaybeAutoUpdate(unittest.TestCase):

    @patch("larkhelm.memory._get_turn_count", return_value=0)
    def test_turn_zero_no_trigger(self, mock_turns):
        """Turn count = 0 → no background thread spawned"""
        from larkhelm.memory import maybe_auto_update
        with patch("larkhelm.memory.generate_memory") as mock_gen:
            maybe_auto_update("chat_zero")
        mock_gen.assert_not_called()

    @patch("larkhelm.memory._get_turn_count", return_value=5)
    def test_non_multiple_no_trigger(self, mock_turns):
        """Turn count = 5 (not a multiple of AUTO_UPDATE_EVERY) → no trigger"""
        from larkhelm.memory import maybe_auto_update, AUTO_UPDATE_EVERY
        self.assertNotEqual(5, 0)
        self.assertNotEqual(5 % AUTO_UPDATE_EVERY, 0)
        with patch("larkhelm.memory.generate_memory") as mock_gen:
            maybe_auto_update("chat_5")
        mock_gen.assert_not_called()

    @patch("larkhelm.memory._get_turn_count", return_value=20)
    def test_multiple_triggers_update(self, mock_turns):
        """Turn count = 20 (multiple of AUTO_UPDATE_EVERY=20) → trigger"""
        from larkhelm.memory import maybe_auto_update, AUTO_UPDATE_EVERY
        self.assertEqual(20 % AUTO_UPDATE_EVERY, 0)
        started = threading.Event()
        original_start = threading.Thread.start

        def _start(self_thread):
            started.set()
            original_start(self_thread)

        with patch("larkhelm.memory._read_logs", return_value=[]):
            with patch.object(threading.Thread, "start", _start):
                maybe_auto_update("chat_20")
        self.assertTrue(started.wait(timeout=1.0), "Expected background thread to start")

    @patch("larkhelm.memory._get_turn_count", return_value=3)
    def test_force_triggers_regardless(self, mock_turns):
        """force=True overrides turn count check"""
        from larkhelm.memory import maybe_auto_update
        started = threading.Event()
        original_start = threading.Thread.start

        def _start(self_thread):
            started.set()
            original_start(self_thread)

        with patch("larkhelm.memory._read_logs", return_value=[]):
            with patch.object(threading.Thread, "start", _start):
                maybe_auto_update("chat_force", force=True)
        self.assertTrue(started.wait(timeout=1.0), "Expected background thread to start with force=True")

    @patch("larkhelm.memory._get_turn_count", return_value=20)
    def test_no_logs_no_memory_written(self, mock_turns):
        """Empty logs → generate_memory never called even at trigger point"""
        from larkhelm.memory import maybe_auto_update
        thread_done = threading.Event()
        _real_start = threading.Thread.start

        def _patched_start(self_thread):
            original_fn = self_thread._target
            def _wrapped(*args, **kwargs):
                try:
                    original_fn(*args, **kwargs)
                finally:
                    thread_done.set()
            self_thread._target = _wrapped
            _real_start(self_thread)

        with patch("larkhelm.memory._read_logs", return_value=[]):
            with patch("larkhelm.memory.generate_memory") as mock_gen:
                with patch.object(threading.Thread, "start", _patched_start):
                    maybe_auto_update("chat_nologs")
                thread_done.wait(timeout=2.0)
        mock_gen.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
