
import json
import os
import threading
import time
import unittest
import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import tempfile
import shutil

import larkhelm.config as _cfg
from larkhelm.runner_base import BaseProcessRunner, QueryCancelledError
from larkhelm.runner_gemini import GeminiRunner

# Setup temp environment for tests
_TMP = tempfile.mkdtemp(prefix="larkhelm_runner_test_")

def setUpModule():
    _cfg_file = Path(_TMP) / "config.json"
    _cfg_file.write_text(json.dumps({
        "APP_ID": "x", "APP_SECRET": "x",
        "response_timeout": 300, "hard_timeout": 3600,
        "skip_permissions": True,
        "default_cwd": _TMP,
    }))
    _cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

def tearDownModule():
    shutil.rmtree(_TMP, ignore_errors=True)

def _make_mock_proc(stdout_lines: list[str], returncode: int = 0):
    mock_proc = MagicMock()
    mock_proc.stdout = iter(stdout_lines)
    mock_proc.stderr = iter([])
    mock_proc.stdin = MagicMock()
    mock_proc.returncode = returncode

    def _wait(timeout=None):
        mock_proc.returncode = returncode
    mock_proc.wait = _wait
    return mock_proc

class TestGeminiRunnerParsing(unittest.TestCase):
    def test_gemini_text_parsing(self):
        runner = GeminiRunner("chat1", "hello", None, _TMP)
        lines = [
            json.dumps({"type": "message", "role": "assistant", "content": "Hello from Gemini"}),
            json.dumps({"type": "result", "session_id": "gsid123", "usage": {}})
        ]
        mock_proc = _make_mock_proc(lines)
        with patch("subprocess.Popen", return_value=mock_proc):
            res = runner.run()
        self.assertEqual(res, "Hello from Gemini")
        self.assertEqual(runner._new_sid, "gsid123")

class TestBaseProcessRunnerRetry(unittest.TestCase):
    def test_retry_on_failure_no_result(self):
        class MockRunner(BaseProcessRunner):
            def build_args(self): return ["mock"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return False
            def cleanup_extra(self): pass
            def __init__(self, chat_id, message, sid, cwd, **kwargs):
                super().__init__("mock", chat_id, message, sid, cwd, **kwargs)
                self._ctor_kwargs = kwargs

        runner = MockRunner("chat1", "msg", "sid123", _TMP, allow_retry=True)
        mock_proc_fail = _make_mock_proc([], returncode=1)
        mock_proc_success = _make_mock_proc([json.dumps({"type":"result"})], returncode=0)
        
        with patch("subprocess.Popen") as mock_popen, \
             patch("larkhelm.runner_base._clear_sid") as mock_clear:
            mock_popen.side_effect = [mock_proc_fail, mock_proc_success]
            
            def smart_parse(self, ev):
                if ev.get("type") == "result":
                    self._result_text = "success after retry"
                    return True
                return False
            
            with patch.object(MockRunner, "parse_stdout_event", smart_parse):
                res = runner.run()
        
        self.assertEqual(res, "success after retry")
        mock_clear.assert_called_once()

class TestBaseProcessRunnerCancellation(unittest.TestCase):
    def test_cancellation_raises_error(self):
        class MockRunner(BaseProcessRunner):
            def build_args(self): return ["mock"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return False
            def cleanup_extra(self): pass
            def __init__(self, chat_id, message, sid, cwd, **kwargs):
                super().__init__("mock", chat_id, message, sid, cwd, **kwargs)

        cancel_ev = threading.Event()
        runner = MockRunner("chat1", "msg", None, _TMP, cancel_ev=cancel_ev)
        
        class HangingStdout:
            def __iter__(self): return self
            def __next__(self):
                while not runner._cancelled_flag.is_set() and not runner._completed.is_set():
                    time.sleep(0.01)
                raise StopIteration()

        mock_proc = MagicMock()
        mock_proc.stdout = HangingStdout()
        mock_proc.stderr = io.StringIO("")
        mock_proc.stdin = MagicMock()
        mock_proc.returncode = 0
        
        original_sleep = time.sleep
        def mocked_sleep(x):
            if x == 0.3: # _watch loop sleep
                cancel_ev.set()
                return
            original_sleep(x)

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("time.sleep", side_effect=mocked_sleep):
            
            with self.assertRaises(QueryCancelledError):
                runner.run()
        
        self.assertTrue(runner._cancelled_flag.is_set())

class TestBaseProcessRunnerCleanup(unittest.TestCase):
    def test_tmp_files_unlinked(self):
        class MockRunner(BaseProcessRunner):
            def build_args(self):
                fd, path = tempfile.mkstemp(dir=_TMP)
                os.close(fd)
                self._tmp_files.append(path)
                return ["mock"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return True
            def cleanup_extra(self): pass
            def __init__(self, chat_id, message, sid, cwd, **kwargs):
                super().__init__("mock", chat_id, message, sid, cwd, **kwargs)

        runner = MockRunner("chat1", "msg", None, _TMP)
        mock_proc = _make_mock_proc([json.dumps({"type":"result"})])
        
        with patch("subprocess.Popen", return_value=mock_proc):
            runner.run()
        
        self.assertEqual(len(runner._tmp_files), 1)
        path = runner._tmp_files[0]
        self.assertFalse(os.path.exists(path))

class TestOnTextInitException(unittest.TestCase):
    """P0 fix: on_text("", status="init") exception must not leak semaphore."""

    def test_semaphore_released_when_on_text_init_raises(self):
        from larkhelm.runner_base import _ai_proc_sem

        class MockRunner(BaseProcessRunner):
            def build_args(self): return ["mock"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return True
            def cleanup_extra(self): pass
            def __init__(self, chat_id, message, sid, cwd, **kwargs):
                super().__init__("mock", chat_id, message, sid, cwd, **kwargs)

        before = _ai_proc_sem._value
        mock_proc = _make_mock_proc([json.dumps({"type": "result"})])

        def exploding_on_text(text, status):
            raise RuntimeError("card update failed")

        runner = MockRunner("chat1", "msg", None, _TMP, on_text=exploding_on_text)
        with patch("subprocess.Popen", return_value=mock_proc):
            runner.run()

        after = _ai_proc_sem._value
        self.assertEqual(before, after, "semaphore must be fully restored after on_text init exception")

    def test_on_text_init_exception_is_logged(self):
        logged: list[str] = []

        class MockRunner(BaseProcessRunner):
            def build_args(self): return ["mock"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return True
            def cleanup_extra(self): pass
            def __init__(self, chat_id, message, sid, cwd, **kwargs):
                super().__init__("mock", chat_id, message, sid, cwd, **kwargs)

        def exploding_on_text(text, status):
            if status == "init":
                raise ValueError("boom")

        mock_proc = _make_mock_proc([json.dumps({"type": "result"})])
        runner = MockRunner("chat1", "msg", None, _TMP, on_text=exploding_on_text)

        with patch("subprocess.Popen", return_value=mock_proc), \
             patch("larkhelm.runner_base._debug_log", side_effect=logged.append):
            runner.run()

        self.assertTrue(any("on_text init callback failed" in m for m in logged),
                        f"Expected log entry, got: {logged}")


class TestGeminiRunnerCtorKwargs(unittest.TestCase):
    """P1 fix: GeminiRunner._ctor_kwargs must preserve use_session and record_under on retry."""

    def test_ctor_kwargs_set_on_gemini_runner(self):
        runner = GeminiRunner(
            "chat1", "msg", None, _TMP,
            use_session=False, record_under="crew_chat1",
        )
        self.assertIn("use_session", runner._ctor_kwargs)
        self.assertIn("record_under", runner._ctor_kwargs)
        self.assertFalse(runner._ctor_kwargs["use_session"])
        self.assertEqual(runner._ctor_kwargs["record_under"], "crew_chat1")

    def test_clone_preserves_use_session_and_record_under(self):
        runner = GeminiRunner(
            "chat1", "msg", "old_sid", _TMP,
            use_session=False, record_under="crew_chat1", command="gemini",
        )
        cloned = runner._clone(sid=None)
        self.assertIsInstance(cloned, GeminiRunner)
        self.assertFalse(cloned.use_session)
        self.assertEqual(cloned.record_under, "crew_chat1")


if __name__ == "__main__":
    unittest.main()
