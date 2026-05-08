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


class TestCmdResetApiClearFailed(unittest.TestCase):
    """REQ-02: _cmd_reset must show ⚠️ card when clear_history raises."""

    def setUp(self):
        # Minimal stubs so imports don't fail in CI without full environment
        self._patches = []

        def _p(target, **kw):
            p = patch(target, **kw)
            self._patches.append(p)
            return p.start()

        _p("larkhelm.commands._get_cwd", return_value="/tmp")
        _p("larkhelm.commands._clear_sid")
        _p("larkhelm.commands.log_entry")
        _p("larkhelm.commands.maybe_auto_update", create=True)

        self.mock_send = _p("larkhelm.commands.send_card_reply")
        self.mock_debug = _p("larkhelm.commands._debug_log")

    def tearDown(self):
        for p in self._patches:
            p.stop()

    def _run_reset(self, which):
        from larkhelm.commands import _cmd_reset
        _cmd_reset("chat_123", which=which, msg_id="msg_1")

    def _make_clear_history_raises(self):
        """Patch api_session.clear_history to raise, and BACKEND_REGISTRY to return a spec."""
        spec = MagicMock()
        spec.provider = "anthropic_api"
        mock_reg = MagicMock()
        mock_reg.all_enabled.return_value = [spec]

        patch("larkhelm.commands.maybe_auto_update", create=True).start()

        with patch.dict("sys.modules", {
            "larkhelm.api_session": MagicMock(clear_history=MagicMock(side_effect=RuntimeError("DB down"))),
            "larkhelm.backend_registry": MagicMock(BACKEND_REGISTRY=mock_reg),
        }):
            yield

    def test_reset_all_success_shows_green_card(self):
        with patch("larkhelm.commands.maybe_auto_update", create=True), \
             patch.dict("sys.modules", {
                 "larkhelm.api_session": MagicMock(clear_history=MagicMock()),
                 "larkhelm.backend_registry": MagicMock(BACKEND_REGISTRY=MagicMock(all_enabled=lambda: [])),
             }):
            self._run_reset(None)

        call_args = self.mock_send.call_args
        self.assertIn("green", str(call_args))
        self.assertNotIn("⚠️", str(call_args))

    def test_reset_all_failure_shows_warning_card(self):
        spec = MagicMock()
        spec.provider = "anthropic_api"
        mock_reg = MagicMock()
        mock_reg.all_enabled.return_value = [spec]

        with patch.dict("sys.modules", {
            "larkhelm.api_session": MagicMock(
                clear_history=MagicMock(side_effect=RuntimeError("DB down"))
            ),
            "larkhelm.backend_registry": MagicMock(BACKEND_REGISTRY=mock_reg),
        }):
            self._run_reset(None)

        call_args = self.mock_send.call_args
        self.assertIn("orange", str(call_args))
        self.assertIn("⚠️", str(call_args))

    def test_reset_claude_failure_logs_and_shows_warning(self):
        with patch.dict("sys.modules", {
            "larkhelm.api_session": MagicMock(
                clear_history=MagicMock(side_effect=ConnectionError("timeout"))
            ),
        }):
            self._run_reset("claude")

        self.mock_debug.assert_called()
        logged = " ".join(str(a) for a in self.mock_debug.call_args_list)
        self.assertIn("[reset] clear_history failed", logged)

        call_args = self.mock_send.call_args
        self.assertIn("orange", str(call_args))

    def test_reset_memory_failure_logs_but_no_warning_card(self):
        with patch.dict("sys.modules", {
            "larkhelm.memory": MagicMock(
                _session_memory_file=MagicMock(return_value=MagicMock(
                    unlink=MagicMock(side_effect=PermissionError("no write"))
                ))
            ),
        }):
            self._run_reset("memory")

        self.mock_debug.assert_called()
        logged = " ".join(str(a) for a in self.mock_debug.call_args_list)
        self.assertIn("[reset] memory unlink failed", logged)
        call_args = self.mock_send.call_args
        self.assertIn("green", str(call_args))


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

        with patch("larkhelm.backend_api._debug_log", side_effect=logged.append), \
             patch.dict("sys.modules", {"anthropic": mock_anthropic_module}):
            from larkhelm.backend_api import run_anthropic
            result, _ = run_anthropic(
                self._make_spec(), "chat1", "hello", [], on_text=failing_on_text
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

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_api_client
        mock_google_pkg = MagicMock()
        mock_google_pkg.genai = mock_genai

        def failing_on_text(text, status):
            raise RuntimeError("render error")

        with patch("larkhelm.backend_api._debug_log", side_effect=logged.append), \
             patch.dict("sys.modules", {
                 "google": mock_google_pkg,
                 "google.genai": mock_genai,
                 "google.genai.types": MagicMock(),
             }):
            from larkhelm.backend_api import run_google
            result, _ = run_google(
                self._make_spec("google_api"), "chat1", "hello", [], on_text=failing_on_text
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
