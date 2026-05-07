
import unittest
import json
import os
import shutil
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch
import dataclasses

# Setup temporary environment
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_v21_test_")
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

from larkhelm.backend_registry import BackendSpec, BackendRegistry
from larkhelm.api_session import load_history, save_history, truncate_history, clear_history
from larkhelm.router import resolve_backend
from larkhelm.memory import load_memory, save_memory, inject_memory, maybe_auto_update
import larkhelm.chat_state as chat_state

class TestBackendRegistry(unittest.TestCase):
    def test_load_and_get(self):
        registry = BackendRegistry()
        specs = [
            {
                "id": "claude-cli",
                "provider": "claude_cli",
                "display_name": "Claude CLI",
                "role": "orchestrator",
                "tags": ["vision", "tools"],
                "command": "true"
            },
            {
                "id": "haiku-api",
                "provider": "anthropic_api",
                "display_name": "Haiku API",
                "role": "cheap",
                "tags": ["cheap", "fast"],
                "model": "claude-3-haiku",
                "api_key": "sk-xxx"
            }
        ]
        registry.load(specs)
        
        self.assertEqual(len(registry.all_enabled()), 2)
        self.assertEqual(registry.get("claude-cli").provider, "claude_cli")
        self.assertEqual(registry.get_orchestrator().id, "claude-cli")
        self.assertEqual(registry.get_by_tag(["cheap"]).id, "haiku-api")
        self.assertIsNone(registry.get("nonexistent"))

    def test_health_check_cli(self):
        registry = BackendRegistry()
        specs = [{
            "id": "bad-cli",
            "provider": "claude_cli",
            "display_name": "Bad",
            "role": "worker",
            "tags": [],
            "command": "/nonexistent/binary/path/that/should/fail"
        }]
        registry.load(specs)
        registry.health_check()
        self.assertFalse(registry.get("bad-cli").healthy)

    def test_health_check_api_missing_key(self):
        registry = BackendRegistry()
        specs = [{
            "id": "bad-api",
            "provider": "anthropic_api",
            "display_name": "Bad",
            "role": "worker",
            "tags": [],
            "model": "m",
            "api_key": ""
        }]
        registry.load(specs)
        registry.health_check()
        self.assertFalse(registry.get("bad-api").healthy)

class TestApiSession(unittest.TestCase):
    def test_save_load_history(self):
        chat_id = "test_chat"
        provider = "anthropic_api"
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"}
        ]
        save_history(provider, chat_id, history)
        loaded = load_history(provider, chat_id)
        self.assertEqual(loaded, history)

    def test_truncate_history(self):
        history = [{"role": "system", "content": "sys"}] + \
                  [{"role": "user", "content": str(i)} for i in range(50)]
        truncated = truncate_history(history)
        self.assertLessEqual(len(truncated), 40)
        self.assertEqual(truncated[0]["role"], "system")
        self.assertEqual(truncated[-1]["content"], "49")

    def test_clear_history(self):
        chat_id = "clear_chat"
        provider = "google_api"
        save_history(provider, chat_id, [{"role": "user", "content": "x"}])
        clear_history(provider, chat_id)
        self.assertEqual(load_history(provider, chat_id), [])

class TestRouter(unittest.TestCase):
    def setUp(self):
        self.registry = BackendRegistry()
        specs = [
            {"id": "c1", "provider": "claude_cli", "role": "orchestrator", "tags": ["vision", "tools"], "display_name": "C1"},
            {"id": "g1", "provider": "gemini_cli", "role": "worker", "tags": ["vision"], "display_name": "G1"},
            {"id": "cheap1", "provider": "openai_compat_api", "role": "cheap", "tags": ["cheap", "fast"], "display_name": "Cheap1", "model": "m", "api_key": "k"},
        ]
        self.registry.load(specs)
        # Mock global BACKEND_REGISTRY
        import larkhelm.config as _cfg
        self.old_registry = _cfg.BACKEND_REGISTRY
        _cfg.BACKEND_REGISTRY = self.registry
        import larkhelm.backend_registry as _br
        _br.BACKEND_REGISTRY = self.registry

    def tearDown(self):
        import larkhelm.config as _cfg
        _cfg.BACKEND_REGISTRY = self.old_registry
        import larkhelm.backend_registry as _br
        _br.BACKEND_REGISTRY = self.old_registry

    def test_resolve_vision(self):
        spec = resolve_backend("chat1", "look at this", has_images=True)
        self.assertIn("vision", spec.tags)

    def test_resolve_tools(self):
        spec = resolve_backend("chat1", "use a tool", has_doc_urls=True)
        self.assertIn("tools", spec.tags)

    def test_resolve_cheap(self):
        with patch("larkhelm.router.BACKEND_REGISTRY", self.registry), \
             patch.object(_cfg_module, "config", {"enable_cheap_routing": True}):
            spec = resolve_backend("chat1", "short msg", has_images=False, has_doc_urls=False)
            self.assertEqual(spec.id, "cheap1")

    def test_resolve_orchestrator_default(self):
        spec = resolve_backend("chat1", "long message " * 20)
        self.assertEqual(spec.role, "orchestrator")

class TestMemory(unittest.TestCase):
    def test_save_load_memory(self):
        chat_id = "mem_chat"
        content = "This is a project about testing."
        save_memory(chat_id, content)
        loaded = load_memory(chat_id)
        self.assertEqual(loaded, content)

    def test_inject_memory(self):
        chat_id = "inject_chat"
        content = "Remember this."
        save_memory(chat_id, content)
        msg = "What do you know?"
        enriched = inject_memory(chat_id, msg)
        self.assertIn("[PERSISTENT MEMORY]", enriched)
        self.assertIn(content, enriched)
        self.assertIn(msg, enriched)

    def test_maybe_auto_update_triggers(self):
        chat_id = "auto_chat"
        chat_state._set_chat_field(chat_id, "turn_count", 20)
        done = threading.Event()
        fake_logs = [{"ts": "t", "role": "user", "content": "hi"}]

        real_save = save_memory

        def _save_and_signal(cid, content):
            real_save(cid, content)
            done.set()

        with patch("larkhelm.memory._read_logs", return_value=fake_logs), \
             patch("larkhelm.memory.generate_memory", return_value="new mem") as mock_gen, \
             patch("larkhelm.memory.save_memory", side_effect=_save_and_signal):
            maybe_auto_update(chat_id)
            done.wait(timeout=2.0)
            mock_gen.assert_called()
        self.assertEqual(load_memory(chat_id), "new mem")

if __name__ == "__main__":
    unittest.main()
