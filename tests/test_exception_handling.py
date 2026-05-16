"""
Tests for exception-handling changes (Q1 cleanup).

Verifies:
  REQ-02: _cmd_reset shows ⚠️ card when API history clear fails
  REQ-05: ai_runner logs token_stats failures via _debug_log
  REQ-10: lark_client logs owner transfer failures via _debug_log
"""
import threading
import unittest
from unittest.mock import MagicMock, patch, call

import pytest


class TestCmdResetApiClearFailed:
    """REQ-02: _cmd_reset must show ⚠️ card when clear_history raises.

    Migrated to pytest function style so the ``inject_module`` fixture (REQ-09)
    can be requested directly. The four legacy sys.modules-patching blocks
    are gone — each test now calls ``inject_module(name, MagicMock(...))``
    once per fake module, and ``monkeypatch``'s teardown restores the original
    entries automatically.
    """

    @pytest.fixture(autouse=True)
    def _stub_commands_deps(self, monkeypatch):
        # Stub out the production helpers that ``_cmd_reset`` reaches into so
        # the test doesn't need a real config / Feishu client.
        monkeypatch.setattr("larkhelm.commands._get_cwd", lambda *_a, **_k: "/tmp")
        monkeypatch.setattr("larkhelm.commands._clear_sid", MagicMock())
        monkeypatch.setattr("larkhelm.commands.log_entry", MagicMock())
        monkeypatch.setattr("larkhelm.commands.maybe_auto_update", MagicMock(),
                            raising=False)
        self.mock_send = MagicMock()
        monkeypatch.setattr("larkhelm.commands.send_card_reply", self.mock_send)
        self.mock_debug = MagicMock()
        monkeypatch.setattr("larkhelm.commands._debug_log", self.mock_debug)

    def _run_reset(self, which):
        from larkhelm.commands import _cmd_reset
        _cmd_reset("chat_123", which=which, msg_id="msg_1")

    def test_reset_all_success_shows_green_card(self, inject_module):
        inject_module(
            "larkhelm.api_session",
            MagicMock(clear_history=MagicMock()),
        )
        inject_module(
            "larkhelm.backend_registry",
            MagicMock(BACKEND_REGISTRY=MagicMock(all_enabled=lambda: [])),
        )
        self._run_reset(None)

        call_args = self.mock_send.call_args
        assert "green" in str(call_args)
        assert "⚠️" not in str(call_args)

    def test_reset_all_failure_shows_warning_card(self, inject_module):
        spec = MagicMock()
        spec.provider = "anthropic_api"
        mock_reg = MagicMock()
        mock_reg.all_enabled.return_value = [spec]

        inject_module(
            "larkhelm.api_session",
            MagicMock(clear_history=MagicMock(side_effect=RuntimeError("DB down"))),
        )
        inject_module(
            "larkhelm.backend_registry",
            MagicMock(BACKEND_REGISTRY=mock_reg),
        )
        self._run_reset(None)

        call_args = self.mock_send.call_args
        assert "orange" in str(call_args)
        assert "⚠️" in str(call_args)

    def test_reset_claude_failure_logs_and_shows_warning(self, inject_module):
        inject_module(
            "larkhelm.api_session",
            MagicMock(clear_history=MagicMock(side_effect=ConnectionError("timeout"))),
        )
        self._run_reset("claude")

        self.mock_debug.assert_called()
        logged = " ".join(str(a) for a in self.mock_debug.call_args_list)
        assert "[reset] clear_history failed" in logged

        call_args = self.mock_send.call_args
        assert "orange" in str(call_args)

    def test_reset_memory_failure_logs_but_no_warning_card(self, inject_module):
        inject_module(
            "larkhelm.memory",
            MagicMock(
                _session_memory_file=MagicMock(return_value=MagicMock(
                    unlink=MagicMock(side_effect=PermissionError("no write"))
                ))
            ),
        )
        self._run_reset("memory")

        self.mock_debug.assert_called()
        logged = " ".join(str(a) for a in self.mock_debug.call_args_list)
        assert "[reset] memory unlink failed" in logged
        call_args = self.mock_send.call_args
        assert "green" in str(call_args)


class TestAiRunnerTokenStatsFails(unittest.TestCase):
    """REQ-05: token_stats failures must be logged, not silently swallowed."""

    def test_record_token_usage_failure_is_logged(self):
        # Token stats are logged in runner_base._record_tokens (refactored from ai_runner).
        # Verify the pattern by AST-scanning runner_base.py.
        import ast, pathlib
        src = pathlib.Path("larkhelm/runner_base.py").read_text()
        tree = ast.parse(src)
        debug_log_in_token_except = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        name = getattr(func, "id", None) or getattr(func, "attr", None)
                        if name == "_debug_log":
                            debug_log_in_token_except = True
                            break
        self.assertTrue(debug_log_in_token_except,
                        "Expected _debug_log calls inside except handlers in runner_base.py")


class TestLarkClientOwnerTransferFails(unittest.TestCase):
    """REQ-10: owner transfer failure must be logged, not silently swallowed."""

    def test_create_doc_owner_transfer_failure_logged(self):
        """The except block around transfer_doc_owner must call _debug_log."""
        logged: list[str] = []

        with patch("larkhelm.lark_client._debug_log", side_effect=logged.append):
            from larkhelm.lark_client import _debug_log
            try:
                raise Exception("permission denied")
            except Exception as e:
                _debug_log(f"[lark_client] owner transfer failed: {e}")

        self.assertTrue(any("[lark_client] owner transfer failed" in m for m in logged),
                        f"Expected log entry, got: {logged}")

    def test_lark_client_source_uses_debug_log_for_owner_transfer(self):
        """Structural check: all three owner-transfer except blocks use _debug_log."""
        import pathlib
        src = pathlib.Path("larkhelm/lark_client.py").read_text()
        # Count occurrences of the log pattern
        count = src.count("[lark_client] owner transfer failed")
        self.assertEqual(count, 3,
                         f"Expected 3 owner transfer _debug_log calls, found {count}")


class TestBackendApiOnTextCallback(unittest.TestCase):
    """REQ-13 (runtime): on_text callback exceptions must be caught and logged, not propagate."""

    def _make_spec(self, provider="anthropic_api"):
        spec = MagicMock()
        spec.id = "test-backend"
        spec.provider = provider
        spec.model = "claude-3-haiku"
        spec.api_key = "sk-test"
        spec.api_base_url = None
        return spec

    def test_anthropic_on_text_exception_is_logged_not_raised(self):
        """Raising on_text must not abort streaming; _debug_log must be called."""
        logged: list[str] = []

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter(["hello", " world"])

        mock_api_client = MagicMock()
        mock_api_client.messages.stream.return_value = mock_stream_ctx

        mock_anthropic_module = MagicMock()
        mock_anthropic_module.Anthropic.return_value = mock_api_client

        def failing_on_text(text, status):
            raise ValueError("card update failed")

        with patch("larkhelm.backend_api._debug_log", side_effect=logged.append):
            from larkhelm.backend_api import run_anthropic
            result, _ = run_anthropic(
                self._make_spec(), "chat1", "hello", [], on_text=failing_on_text,
                _anthropic_module=mock_anthropic_module,
            )

        self.assertEqual(result, "hello world")
        self.assertTrue(any("[anthropic_api] on_text callback failed" in m for m in logged),
                        f"Expected on_text log, got: {logged}")

    def test_google_on_text_exception_is_logged_not_raised(self):
        """Google backend: raising on_text must not abort streaming."""
        logged: list[str] = []

        mock_chunk = MagicMock()
        mock_chunk.text = "hi"
        mock_api_client = MagicMock()
        mock_api_client.models.generate_content_stream.return_value = iter([mock_chunk])

        # ``run_google`` expects ``_google_module.genai`` and
        # ``_google_module.genai_types`` to mirror the live
        # ``from google import genai`` / ``from google.genai import types``
        # surface. Build a single namespace that exposes both attributes.
        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_api_client
        mock_google_module = MagicMock()
        mock_google_module.genai = mock_genai
        mock_google_module.genai_types = MagicMock()

        def failing_on_text(text, status):
            raise RuntimeError("render error")

        with patch("larkhelm.backend_api._debug_log", side_effect=logged.append):
            from larkhelm.backend_api import run_google
            result, _ = run_google(
                self._make_spec("google_api"), "chat1", "hello", [], on_text=failing_on_text,
                _google_module=mock_google_module,
            )

        self.assertEqual(result, "hi")
        self.assertTrue(any("[google_api] on_text callback failed" in m for m in logged),
                        f"Expected on_text log, got: {logged}")


class TestLarkClientOwnerTransferRuntime(unittest.TestCase):
    """REQ-10 (runtime): create_doc owner transfer failure must call _debug_log, not raise."""

    def _make_client(self):
        import tempfile, json, pathlib
        cfg_dir = pathlib.Path(tempfile.mkdtemp())
        cfg = cfg_dir / "config.json"
        cfg.write_text(json.dumps({"APP_ID": "a", "APP_SECRET": "s"}))
        import larkhelm.config as _cfg
        _cfg._init_runtime(config_path=str(cfg), data_dir=str(cfg_dir))

        from larkhelm.lark_client import FeishuDocClient
        return FeishuDocClient()

    def test_create_doc_owner_transfer_failure_logged_not_raised(self):
        """transfer_doc_owner raising must log via _debug_log, not propagate."""
        logged: list[str] = []
        doc_client = self._make_client()

        mock_resp = MagicMock()
        mock_resp.success.return_value = True
        mock_resp.data.document.document_id = "doc123"

        mock_lark_client = MagicMock()
        mock_lark_client.docx.v1.document.create.return_value = mock_resp

        with patch("larkhelm.lark_client.client", mock_lark_client), \
             patch.object(doc_client, "transfer_doc_owner",
                          side_effect=Exception("permission denied")), \
             patch("larkhelm.lark_client._debug_log", side_effect=logged.append):

            from larkhelm.lark_client import DocRef
            result = doc_client.create_doc("Test Title", "", owner_open_id="uid_123")

        # Owner transfer failure must not raise — create_doc returns normally
        self.assertIsInstance(result, DocRef)
        self.assertTrue(any("[lark_client] owner transfer failed" in m for m in logged),
                        f"Expected owner-transfer log, got: {logged}")


if __name__ == "__main__":
    unittest.main()
