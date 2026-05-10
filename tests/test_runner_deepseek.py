"""Unit tests for DeepSeekRunner — exercise the SSE parser, history persistence,
backoff, and cancel/soft-timeout paths without touching the real DeepSeek API.

The runner does HTTP via the ``requests`` package; we monkeypatch
``requests.post`` so the tests are hermetic and fast.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


# Bootstrap config before importing the runner — _load_sid / _save_sid need
# DATA_DIR + SESSION_DIR to exist on disk.
_TMP = Path(tempfile.mkdtemp(prefix="larkhelm-deepseek-test-"))
os.environ["LARKHELM_DATA_DIR"] = str(_TMP)


def _bootstrap_config() -> None:
    """Initialize the minimal subset of larkhelm.config needed for runner tests.

    We avoid calling ``_init_runtime`` (which insists on APP_ID/APP_SECRET) and
    instead set the globals the runner reads.
    """
    import larkhelm.config as _cfg
    _cfg.DATA_DIR    = _TMP
    _cfg.SESSION_DIR = _TMP / "sessions"
    _cfg.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    _cfg.LOG_DIR     = _TMP / "logs"
    _cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    _cfg.STATE_FILE  = _TMP / "state.json"
    _cfg.DEBUG_LOG   = _TMP / "larkhelm.log"
    _cfg.RESPONSE_TIMEOUT = 300
    _cfg.HARD_TIMEOUT     = 600
    _cfg.DEEPSEEK_API_KEY  = "sk-test-fake"
    _cfg.DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    _cfg.DEEPSEEK_MODEL    = "deepseek-chat"


_bootstrap_config()

from larkhelm.runner_deepseek import (    # noqa: E402  (post-bootstrap import)
    DeepSeekRunner,
    _load_history,
    _save_history,
    _clear_history,
)
from larkhelm.runner_base import QueryCancelledError  # noqa: E402


class _FakeStreamResponse:
    """Mimics the bits of requests.Response that DeepSeekRunner._consume_sse uses."""

    def __init__(self, status_code: int, lines: list[str], text: str = ""):
        self.status_code = status_code
        self._lines = lines
        self.text = text
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        for line in self._lines:
            yield line

    def close(self):
        self.closed = True


def _ds_sse(content_chunks: list[str], usage: dict | None = None) -> list[str]:
    """Build a list of raw SSE lines mimicking DeepSeek's streaming output."""
    lines: list[str] = []
    for chunk in content_chunks:
        lines.append("data: " + json.dumps({
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
        }))
        lines.append("")  # SSE event separator
    if usage:
        lines.append("data: " + json.dumps({
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }))
        lines.append("")
    lines.append(": keep-alive")
    lines.append("data: [DONE]")
    return lines


class DeepSeekRunnerTest(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(_TMP, ignore_errors=True)

    def setUp(self):
        # Re-apply config every test: another test in the suite may have called
        # _init_runtime() and cleared the bootstrap globals.
        _bootstrap_config()
        # Each test starts with a clean session file
        _clear_history("chat-test")

    # ------------------------------------------------------------------ run() happy path

    def test_streaming_text_collected_and_history_saved(self):
        captured_text: list[str] = []

        def on_text(t, status="typing"):
            captured_text.append(t)

        sse = _ds_sse(["Hello", ", ", "world!"], usage={
            "prompt_tokens": 12, "completion_tokens": 3,
            "prompt_cache_hit_tokens": 5, "prompt_cache_miss_tokens": 7,
        })
        fake_resp = _FakeStreamResponse(200, sse)

        with mock.patch("requests.post", return_value=fake_resp) as m_post:
            runner = DeepSeekRunner(
                "chat-test", "Hi", sid=None, cwd="/tmp",
                cancel_ev=threading.Event(), on_text=on_text,
            )
            output = runner.run()

        self.assertEqual(output, "Hello, world!")
        # on_text was called once for "init", then once per chunk with the
        # cumulative text so far
        self.assertGreaterEqual(len(captured_text), 4)
        self.assertEqual(captured_text[-1], "Hello, world!")

        # POST was issued with the right URL + bearer header
        args, kwargs = m_post.call_args
        self.assertIn("/chat/completions", args[0])
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer sk-test-fake")
        self.assertTrue(kwargs["json"]["stream"])
        self.assertEqual(kwargs["json"]["messages"][-1]["content"], "Hi")

        # History was persisted: user + assistant turn
        hist = _load_history("chat-test")
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0], {"role": "user", "content": "Hi"})
        self.assertEqual(hist[1], {"role": "assistant", "content": "Hello, world!"})

    def test_system_prompt_prepended_to_messages(self):
        sse = _ds_sse(["ack"])
        with mock.patch("requests.post", return_value=_FakeStreamResponse(200, sse)) as m_post:
            DeepSeekRunner(
                "chat-test", "Hi", sid=None, cwd="/tmp",
                system_prompt="You speak only French.",
            ).run()
        sent = m_post.call_args.kwargs["json"]["messages"]
        self.assertEqual(sent[0], {"role": "system", "content": "You speak only French."})
        self.assertEqual(sent[-1], {"role": "user", "content": "Hi"})

    def test_history_replayed_into_next_request(self):
        # Round 1: seed history
        with mock.patch("requests.post", return_value=_FakeStreamResponse(200, _ds_sse(["one"]))):
            DeepSeekRunner("chat-test", "first", sid=None, cwd="/tmp").run()

        # Round 2: previous turn must be replayed
        with mock.patch("requests.post", return_value=_FakeStreamResponse(200, _ds_sse(["two"]))) as m_post:
            DeepSeekRunner("chat-test", "second", sid=None, cwd="/tmp").run()
        sent_msgs = m_post.call_args.kwargs["json"]["messages"]
        roles_contents = [(m["role"], m["content"]) for m in sent_msgs]
        self.assertEqual(roles_contents, [
            ("user", "first"),
            ("assistant", "one"),
            ("user", "second"),
        ])

    def test_use_session_false_skips_history(self):
        # Pre-populate history; use_session=False must ignore it AND skip writing back
        _save_history("chat-test", [
            {"role": "user", "content": "older"},
            {"role": "assistant", "content": "older-reply"},
        ])
        with mock.patch("requests.post", return_value=_FakeStreamResponse(200, _ds_sse(["ok"]))) as m_post:
            DeepSeekRunner(
                "chat-test", "fresh", sid=None, cwd="/tmp", use_session=False,
            ).run()
        sent_msgs = m_post.call_args.kwargs["json"]["messages"]
        # No system prompt, no history → only the user turn
        self.assertEqual(sent_msgs, [{"role": "user", "content": "fresh"}])
        # And the existing history file is untouched
        self.assertEqual(len(_load_history("chat-test")), 2)

    # ------------------------------------------------------------------ token recording

    def test_tokens_recorded_on_result(self):
        sse = _ds_sse(["x"], usage={
            "prompt_tokens": 100, "completion_tokens": 50,
            "prompt_cache_hit_tokens": 10, "prompt_cache_miss_tokens": 90,
        })
        with mock.patch("requests.post", return_value=_FakeStreamResponse(200, sse)), \
             mock.patch("larkhelm.token_stats.record_token_usage") as m_rec:
            DeepSeekRunner("chat-test", "Hi", sid=None, cwd="/tmp").run()
        self.assertEqual(m_rec.call_count, 1)
        args, _ = m_rec.call_args
        self.assertEqual(args[0], "chat-test")
        self.assertEqual(args[1], "deepseek")
        self.assertEqual(args[2]["input_tokens"], 100)
        self.assertEqual(args[2]["output_tokens"], 50)
        self.assertEqual(args[2]["cache_read"], 10)
        self.assertEqual(args[2]["cache_create"], 90)

    # ------------------------------------------------------------------ failure modes

    def test_missing_api_key_raises(self):
        import larkhelm.config as _cfg
        prev = _cfg.DEEPSEEK_API_KEY
        _cfg.DEEPSEEK_API_KEY = ""
        try:
            with self.assertRaises(RuntimeError) as cm:
                DeepSeekRunner("chat-test", "Hi", sid=None, cwd="/tmp").run()
            self.assertIn("API key", str(cm.exception))
        finally:
            _cfg.DEEPSEEK_API_KEY = prev

    def test_429_triggers_retry_then_succeeds(self):
        responses = [
            _FakeStreamResponse(429, [], text="rate limited"),
            _FakeStreamResponse(200, _ds_sse(["ok"])),
        ]
        # Patch backoff to zero so the test runs in milliseconds
        with mock.patch("larkhelm.runner_deepseek._REQUEST_BACKOFF", (0.0, 0.0, 0.0)), \
             mock.patch("requests.post", side_effect=responses) as m_post:
            output = DeepSeekRunner("chat-test", "Hi", sid=None, cwd="/tmp").run()
        self.assertEqual(output, "ok")
        self.assertEqual(m_post.call_count, 2)

    def test_400_raises_immediately_without_retry(self):
        bad = _FakeStreamResponse(400, [], text="malformed prompt")
        with mock.patch("requests.post", return_value=bad) as m_post:
            with self.assertRaises(RuntimeError) as cm:
                DeepSeekRunner("chat-test", "Hi", sid=None, cwd="/tmp").run()
        self.assertIn("400", str(cm.exception))
        self.assertEqual(m_post.call_count, 1)
        # And no history written for a failed call
        self.assertEqual(_load_history("chat-test"), [])

    def test_cancel_event_aborts_stream(self):
        cancel_ev = threading.Event()
        # Build an SSE response that fires cancel halfway through
        slow_lines = []
        for i in range(5):
            slow_lines.append("data: " + json.dumps({
                "choices": [{"index": 0, "delta": {"content": f"chunk{i} "}}],
            }))
            slow_lines.append("")

        original_iter = _FakeStreamResponse.iter_lines

        def cancelling_iter(self, decode_unicode=False):
            for idx, line in enumerate(original_iter(self, decode_unicode)):
                if idx == 4:
                    cancel_ev.set()
                yield line

        fake = _FakeStreamResponse(200, slow_lines)
        with mock.patch.object(_FakeStreamResponse, "iter_lines", cancelling_iter), \
             mock.patch("requests.post", return_value=fake):
            with self.assertRaises(QueryCancelledError):
                DeepSeekRunner(
                    "chat-test", "Hi", sid=None, cwd="/tmp", cancel_ev=cancel_ev,
                ).run()
        # Stream should have been closed when cancel was observed
        self.assertTrue(fake.closed)
        # No history written on cancel
        self.assertEqual(_load_history("chat-test"), [])

    # ------------------------------------------------------------------ history utils

    def test_history_trimmed_to_cap(self):
        """_save_history must keep only the last _HISTORY_TURN_CAP * 2 messages."""
        from larkhelm.runner_deepseek import _HISTORY_TURN_CAP
        big = []
        for i in range(_HISTORY_TURN_CAP * 3):
            big.append({"role": "user", "content": f"u{i}"})
            big.append({"role": "assistant", "content": f"a{i}"})
        _save_history("chat-test", big)
        loaded = _load_history("chat-test")
        self.assertEqual(len(loaded), _HISTORY_TURN_CAP * 2)
        # Last entry must survive
        self.assertEqual(loaded[-1]["content"], f"a{_HISTORY_TURN_CAP * 3 - 1}")

    def test_corrupt_history_treated_as_empty(self):
        from larkhelm.chat_state import _save_sid
        _save_sid("chat-test", "{not valid json", "deepseek")
        self.assertEqual(_load_history("chat-test"), [])


if __name__ == "__main__":
    unittest.main()
