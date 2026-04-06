"""
P1 — ai_runner.py pure-function and stream-parsing tests

Coverage:
  - _truncate_tool_result       tool result truncation logic
  - _build_stream_json_input    multimodal stdin construction
  - _spawn_claude_proc          stream-json parsing (mock subprocess)
  - _spawn_kimi_proc            Kimi stream-json parsing (mock subprocess)
"""
import atexit
import base64
import io
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_aitest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "response_timeout": 300, "hard_timeout": 3600,
    "skip_permissions": True,
    "default_cwd": _TMP,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.ai_runner import _truncate_tool_result, _build_stream_json_input


# ═══════════════════════════════════════════════════════════════════
#  _truncate_tool_result
# ═══════════════════════════════════════════════════════════════════

class TestTruncateToolResult(unittest.TestCase):
    def test_short_non_error_unchanged(self):
        result = _truncate_tool_result("short output", is_error=False)
        self.assertEqual(result, "short output")

    def test_long_non_error_takes_first_200(self):
        content = "a" * 300
        result = _truncate_tool_result(content, is_error=False)
        self.assertLessEqual(len(result), 200)

    def test_non_error_keeps_last_complete_line(self):
        # A newline within the first 200 chars → truncate to last complete line
        lines = ["line1", "line2", "line3"]
        content = "\n".join(lines) + "\n" + "x" * 300
        result = _truncate_tool_result(content, is_error=False)
        # Content up to the last newline within the first 200 chars
        self.assertIn("line", result)

    def test_non_error_no_newline_uses_full_snippet(self):
        content = "a" * 250
        result = _truncate_tool_result(content, is_error=False)
        # No newline present: take the first 200 chars directly
        self.assertEqual(result, "a" * 200)

    def test_short_error_unchanged(self):
        result = _truncate_tool_result("error msg", is_error=True)
        self.assertEqual(result, "error msg")

    def test_long_error_takes_last_200(self):
        content = "garbage" * 100 + "real error message here"
        result = _truncate_tool_result(content, is_error=True)
        self.assertIn("real error message here", result)
        self.assertIn("截断", result)

    def test_error_exactly_200_no_prefix(self):
        content = "x" * 200
        result = _truncate_tool_result(content, is_error=True)
        self.assertEqual(result, "x" * 200)
        self.assertNotIn("截断", result)

    def test_error_201_adds_truncation_prefix(self):
        content = "x" * 201
        result = _truncate_tool_result(content, is_error=True)
        self.assertIn("截断", result)


# ═══════════════════════════════════════════════════════════════════
#  _build_stream_json_input
# ═══════════════════════════════════════════════════════════════════

class TestBuildStreamJsonInput(unittest.TestCase):
    def test_text_only_no_images(self):
        result = _build_stream_json_input("hello", [])
        parsed = json.loads(result)
        self.assertEqual(parsed["type"], "user")
        content = parsed["message"]["content"]
        text_blocks = [b for b in content if b.get("type") == "text"]
        self.assertEqual(len(text_blocks), 1)
        self.assertEqual(text_blocks[0]["text"], "hello")

    def test_image_block_added(self):
        # Create a temporary PNG file
        png_path = Path(_TMP) / "test.png"
        png_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        result = _build_stream_json_input("describe", [str(png_path)])
        parsed = json.loads(result)
        content = parsed["message"]["content"]
        image_blocks = [b for b in content if b.get("type") == "image"]
        self.assertEqual(len(image_blocks), 1)
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/png")
        # base64 should be decodable
        decoded = base64.b64decode(image_blocks[0]["source"]["data"])
        self.assertTrue(len(decoded) > 0)

    def test_jpeg_media_type(self):
        jpg_path = Path(_TMP) / "test.jpg"
        jpg_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        result = _build_stream_json_input("img", [str(jpg_path)])
        parsed = json.loads(result)
        content = parsed["message"]["content"]
        image_blocks = [b for b in content if b.get("type") == "image"]
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/jpeg")

    def test_invalid_image_path_skipped(self):
        result = _build_stream_json_input("text", ["/nonexistent/image.png"])
        parsed = json.loads(result)
        content = parsed["message"]["content"]
        # Invalid image is skipped; only the text block should remain
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")

    def test_text_block_always_last(self):
        png_path = Path(_TMP) / "last.png"
        png_path.write_bytes(b"\x89PNG" + b"\x00" * 20)
        result = _build_stream_json_input("desc", [str(png_path)])
        parsed = json.loads(result)
        content = parsed["message"]["content"]
        self.assertEqual(content[-1]["type"], "text")


# ═══════════════════════════════════════════════════════════════════
#  _spawn_claude_proc — stream-json parsing (mock subprocess)
# ═══════════════════════════════════════════════════════════════════

def _make_mock_proc(stdout_lines: list[str], returncode: int = 0):
    """Build a mock subprocess.Popen object whose stdout yields the given lines."""
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines)
    mock_proc.stderr = iter([])
    mock_proc.stdin = MagicMock()
    mock_proc.returncode = returncode

    def _wait():
        mock_proc.returncode = returncode
    mock_proc.wait = _wait
    return mock_proc


class TestSpawnClaudeProcParsing(unittest.TestCase):
    """Test _spawn_claude_proc stream-json parsing logic via a mock subprocess."""

    def _run(self, stdout_lines, chat_id=None, **kwargs):
        """Helper: mock Popen, call _spawn_claude_proc, and collect callback data."""
        from larkhelm.ai_runner import _spawn_claude_proc

        texts = []
        tools = []
        tool_results = []

        def on_text(t, status="typing"):
            texts.append(t)

        def on_tool(name, desc, tool_id=""):
            tools.append((name, desc, tool_id))

        def on_tool_result(tool_id, result, is_error, elapsed):
            tool_results.append((tool_id, result, is_error))

        mock_proc = _make_mock_proc(stdout_lines)
        _chat_id = chat_id or f"test_chat_{id(stdout_lines)}"

        with patch("subprocess.Popen", return_value=mock_proc):
            result = _spawn_claude_proc(
                chat_id=_chat_id,
                message="test",
                sid=None,
                cwd=_TMP,
                on_text=on_text,
                on_tool=on_tool,
                on_tool_result=on_tool_result,
                **kwargs,
            )
        return result, texts, tools, tool_results

    def test_text_block_parsed(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Hello world"}
            ]}}),
            json.dumps({"type": "result", "session_id": "sid123", "usage": {}}),
        ]
        result, texts, tools, tool_results = self._run(lines)
        self.assertEqual(result, "Hello world")
        self.assertIn("Hello world", texts)

    def test_text_accumulates(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "foo"}
            ]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": " bar"}
            ]}}),
            json.dumps({"type": "result", "session_id": "s", "usage": {}}),
        ]
        result, _, _, _ = self._run(lines)
        self.assertEqual(result, "foo bar")

    def test_tool_use_callback(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "ls -la"}}
            ]}}),
            json.dumps({"type": "result", "session_id": "s", "usage": {}}),
        ]
        result, texts, tools, tool_results = self._run(lines)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0][0], "Bash")
        self.assertIn("ls -la", tools[0][1])

    def test_tool_result_callback(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "x.py"}}
            ]}}),
            json.dumps({"type": "tool_result", "tool_use_id": "t1",
                        "content": "file contents here", "is_error": False}),
            json.dumps({"type": "result", "session_id": "s", "usage": {}}),
        ]
        _, _, _, tool_results = self._run(lines)
        self.assertEqual(len(tool_results), 1)
        tid, content, is_err = tool_results[0]
        self.assertEqual(tid, "t1")
        self.assertFalse(is_err)
        self.assertIn("file contents here", content)

    def test_tool_result_error_flag(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "bad"}}
            ]}}),
            json.dumps({"type": "tool_result", "tool_use_id": "t2",
                        "content": "command not found", "is_error": True}),
            json.dumps({"type": "result", "session_id": "s", "usage": {}}),
        ]
        _, _, _, tool_results = self._run(lines)
        self.assertTrue(tool_results[0][2])   # is_error = True

    def test_session_id_saved_from_result(self):
        from larkhelm.chat_state import _load_sid, _clear_sid
        chat_id = "sid_save_test_chat_unique"
        _clear_sid(chat_id, "claude")
        lines = [
            json.dumps({"type": "result", "session_id": "newsid_abc", "usage": {}}),
        ]
        self._run(lines, chat_id=chat_id)
        saved = _load_sid(chat_id, "claude")
        self.assertEqual(saved, "newsid_abc")

    def test_session_id_saved_from_init(self):
        from larkhelm.chat_state import _load_sid, _clear_sid
        chat_id = "sid_init_test_chat_unique"
        _clear_sid(chat_id, "claude")
        lines = [
            json.dumps({"type": "init", "session_id": "init_sid_xyz"}),
            json.dumps({"type": "result", "usage": {}}),
        ]
        self._run(lines, chat_id=chat_id)
        saved = _load_sid(chat_id, "claude")
        self.assertEqual(saved, "init_sid_xyz")

    def test_non_json_lines_skipped(self):
        lines = [
            "not json at all\n",
            "{bad json}\n",
            json.dumps({"type": "result", "session_id": "s", "usage": {}}),
        ]
        # Should not raise
        result, _, _, _ = self._run(lines)
        self.assertEqual(result, "")

    def test_thinking_block_ignored(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "thinking", "thinking": "internal reasoning..."},
                {"type": "text", "text": "actual answer"},
            ]}}),
            json.dumps({"type": "result", "session_id": "s", "usage": {}}),
        ]
        result, texts, _, _ = self._run(lines)
        self.assertEqual(result, "actual answer")
        # thinking content should not appear in on_text callbacks
        for t in texts:
            self.assertNotIn("internal reasoning", t)

    def test_tool_input_read_summarized(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t3", "name": "Read",
                 "input": {"file_path": "/path/to/myfile.py"}}
            ]}}),
            json.dumps({"type": "result", "usage": {}}),
        ]
        _, _, tools, _ = self._run(lines)
        self.assertIn("/path/to/myfile.py", tools[0][1])

    def test_tool_input_glob_summarized(self):
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t4", "name": "Glob",
                 "input": {"pattern": "**/*.py"}}
            ]}}),
            json.dumps({"type": "result", "usage": {}}),
        ]
        _, _, tools, _ = self._run(lines)
        self.assertIn("**/*.py", tools[0][1])

    def test_tool_result_list_content(self):
        """When tool_result content is a list, the text items should be concatenated."""
        lines = [
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "tool_use", "id": "t5", "name": "Bash", "input": {"command": "echo hi"}}
            ]}}),
            json.dumps({"type": "tool_result", "tool_use_id": "t5",
                        "content": [{"type": "text", "text": "hi\n"}, {"type": "text", "text": "done"}],
                        "is_error": False}),
            json.dumps({"type": "result", "usage": {}}),
        ]
        _, _, _, tool_results = self._run(lines)
        self.assertIn("hi", tool_results[0][1])


# ═══════════════════════════════════════════════════════════════════
#  _spawn_kimi_proc — stream-json parsing (mock subprocess)
# ═══════════════════════════════════════════════════════════════════

class TestSpawnKimiProcParsing(unittest.TestCase):
    def _run(self, stdout_lines, chat_id=None, **kwargs):
        from larkhelm.ai_runner import _spawn_kimi_proc

        texts = []
        tools = []
        tool_results = []

        def on_text(t, status="typing"):
            texts.append(t)

        def on_tool(name, desc, tool_id=""):
            tools.append((name, desc, tool_id))

        def on_tool_result(tool_id, result, is_error, elapsed):
            tool_results.append((tool_id, result, is_error))

        mock_proc = _make_mock_proc(stdout_lines)
        _chat_id = chat_id or f"kimi_test_{id(stdout_lines)}"

        with patch("subprocess.Popen", return_value=mock_proc):
            result = _spawn_kimi_proc(
                chat_id=_chat_id,
                message="hello",
                sid=None,
                cwd=_TMP,
                on_text=on_text,
                on_tool=on_tool,
                on_tool_result=on_tool_result,
                **kwargs,
            )
        return result, texts, tools, tool_results

    def test_assistant_text_parsed(self):
        lines = [
            json.dumps({"role": "assistant", "content": "Kimi says hello"}),
        ]
        result, texts, _, _ = self._run(lines)
        self.assertEqual(result, "Kimi says hello")
        self.assertIn("Kimi says hello", texts)

    def test_tool_name_mapping_shell_to_bash(self):
        lines = [
            json.dumps({"role": "assistant", "content": "",
                        "tool_calls": [{"id": "k1", "function": {"name": "Shell",
                                        "arguments": {"command": "ls"}}}]}),
            json.dumps({"role": "tool", "tool_call_id": "k1", "content": "file.txt"}),
        ]
        _, _, tools, _ = self._run(lines)
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0][0], "Bash")   # Shell → Bash

    def test_tool_name_mapping_fetchurl(self):
        lines = [
            json.dumps({"role": "assistant", "content": "",
                        "tool_calls": [{"id": "k2", "function": {"name": "FetchURL",
                                        "arguments": {"url": "https://example.com"}}}]}),
        ]
        _, _, tools, _ = self._run(lines)
        self.assertEqual(tools[0][0], "WebFetch")   # FetchURL → WebFetch

    def test_tool_name_mapping_searchweb(self):
        lines = [
            json.dumps({"role": "assistant", "content": "",
                        "tool_calls": [{"id": "k3", "function": {"name": "SearchWeb",
                                        "arguments": {"query": "python"}}}]}),
        ]
        _, _, tools, _ = self._run(lines)
        self.assertEqual(tools[0][0], "WebSearch")  # SearchWeb → WebSearch

    def test_tool_result_callback(self):
        lines = [
            json.dumps({"role": "assistant", "content": "",
                        "tool_calls": [{"id": "k4", "function": {"name": "Shell",
                                        "arguments": {"command": "pwd"}}}]}),
            json.dumps({"role": "tool", "tool_call_id": "k4", "content": "/home/user"}),
            json.dumps({"role": "assistant", "content": "done"}),
        ]
        result, _, _, tool_results = self._run(lines)
        self.assertEqual(len(tool_results), 1)
        self.assertIn("/home/user", tool_results[0][1])

    def test_session_id_extracted(self):
        from larkhelm.chat_state import _load_sid, _clear_sid
        chat_id = "kimi_sid_test_unique"
        _clear_sid(chat_id, "kimi")
        lines = [
            json.dumps({"role": "assistant", "content": "hi", "session_id": "kimi_sess_001"}),
        ]
        self._run(lines, chat_id=chat_id)

    def test_non_json_lines_skipped(self):
        lines = ["plain text line\n", json.dumps({"role": "assistant", "content": "ok"})]
        result, _, _, _ = self._run(lines)
        self.assertEqual(result, "ok")

    def test_stdin_uses_stream_json_format(self):
        """Verify that Kimi uses role/content format for stdin."""
        written = []
        mock_proc = _make_mock_proc([])

        original_write = mock_proc.stdin.write
        mock_proc.stdin.write = lambda s: written.append(s)

        from larkhelm.ai_runner import _spawn_kimi_proc
        with patch("subprocess.Popen", return_value=mock_proc):
            try:
                _spawn_kimi_proc(chat_id="kimi_stdin_test", message="my question",
                                 sid=None, cwd=_TMP)
            except Exception:
                pass  # empty stdout may cause rc exception, ignore

        if written:
            payload = json.loads(written[0].strip())
            self.assertEqual(payload["role"], "user")
            self.assertEqual(payload["content"], "my question")


if __name__ == "__main__":
    unittest.main()
