"""Tests for the audio branch + ``/voice`` command in ``larkhelm.handlers._message``.

Coverage map (PRD §3 / design.md §1.2):

| AC                         | Test                              |
| -------------------------- | --------------------------------- |
| REQ-02 (disabled silent)   | TestVoiceDisabledSilent           |
| REQ-04 (oversize reject)   | TestVoiceOversize                 |
| REQ-09 (no-merge dispatch) | TestVoiceTranscribeNoMerge        |
| REQ-09 (merge dispatch)    | TestVoiceTranscribeWithMerge      |
| REQ-06 (download fail)     | TestVoiceDownloadFailed           |
| REQ-08 (empty text)        | TestVoiceEmptyTextFails           |
| REQ-10 (cleanup on fail)   | TestVoiceCleanupOnFailure         |

Mocking strategy
----------------
All six external dependencies of the audio branch are imported at the top
of ``larkhelm.handlers._message`` (so ``patch.object(msg_mod, name, ...)``
intercepts them); a fresh per-case ``MagicMock`` lets us assert call
counts and inspect captured arguments without actually downloading,
transcribing, or hitting the bridge subprocess pool.

The no-merge dispatch path now off-threads ``_do_query`` (review fix:
SDK worker must not block on LLM calls); ``_AudioBranchBase`` patches
``threading.Thread`` with ``_SyncThread`` so the mock ``_do_query`` is
invoked synchronously inside ``handle_message``, keeping call-count
assertions deterministic without losing the daemon-thread structural
guarantee (verified by ``TestVoiceNoMergeUsesDaemonThread``).
"""
from __future__ import annotations

import atexit
import itertools
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Monotonic counter so each ``_make_audio_event`` call yields a distinct
# (event_id, message_id) pair — the dedup cache in ``larkhelm.dedup``
# keys on both, so reusing ``msg_001`` across tests would silently drop
# every test after the first.
_msg_seq = itertools.count(1)

# Bootstrap config in an isolated tempdir BEFORE importing the target module
# (mirrors tests/test_voice_merge.py:33-54). Using ``_init_runtime`` here
# avoids depending on the user's real ``~/.config/larkhelm/config.json``.
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_voice_msg_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_DUMMY_CONFIG = {
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
    "voice_enabled": True,
    "voice_max_duration_ms": 60000,
    "voice_merge_window_sec": 0,
    "voice_keep_audio": False,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

import larkhelm.handlers._message as msg_mod  # noqa: E402


def _make_audio_event(
    file_key: str = "fk_001",
    duration_ms: int = 5000,
    chat_id: str = "chat_voice_test",
    msg_id: str | None = None,
    event_id: str | None = None,
) -> SimpleNamespace:
    """Construct a minimal P2ImMessageReceiveV1-like object for handle_message.

    Only the attributes ``handle_message`` actually inspects are populated.
    Both ``msg_id`` and ``event_id`` default to monotonic counter values so
    the module-level dedup cache (``larkhelm.dedup``) does not silently
    drop later events with the same ids.
    """
    n = next(_msg_seq)
    msg_id = msg_id or f"msg_{n:04d}"
    event_id = event_id or f"evt_{n:04d}"
    content = json.dumps({"file_key": file_key, "duration": duration_ms})
    message = SimpleNamespace(
        message_type="audio",
        content=content,
        chat_id=chat_id,
        chat_type="p2p",
        message_id=msg_id,
        mentions=None,
        parent_id=None,
    )
    sender = SimpleNamespace(sender_id=SimpleNamespace(open_id="user_001"))
    event = SimpleNamespace(message=message, sender=sender)
    header = SimpleNamespace(event_id=event_id)
    return SimpleNamespace(event=event, header=header)


class _SyncThread:
    """Drop-in replacement for ``threading.Thread`` that runs synchronously.

    The audio branch off-threads ``_do_query`` to avoid blocking the SDK
    event worker. Tests assert on the mocked ``_do_query``'s call_count;
    a real ``Thread.start()`` introduces a race with the assertion.
    Replacing the constructor with this synchronous variant keeps the
    target invocation deterministic while preserving the daemon-thread
    structural check (recorded into ``constructed`` for inspection).
    """

    constructed: "list[dict]" = []

    def __init__(self, *args, **kwargs):
        # Capture the exact kwargs handle_message passed so a separate
        # test can verify ``daemon=True`` and ``name="voice-query-..."``.
        self._target = kwargs.get("target")
        self._args = kwargs.get("args", ())
        self._kwargs = kwargs.get("kwargs", {}) or {}
        _SyncThread.constructed.append({
            "target": self._target,
            "kwargs": self._kwargs,
            "daemon": kwargs.get("daemon"),
            "name": kwargs.get("name"),
        })

    def start(self) -> None:
        if self._target is not None:
            self._target(*self._args, **self._kwargs)


class _AudioBranchBase(unittest.TestCase):
    """Shared bootstrap: snapshot voice cfg, install the 6 standard mocks."""

    def setUp(self) -> None:
        # Snapshot mutable cfg so each test can fiddle freely.
        self._saved = {
            "VOICE_ENABLED": _cfg.VOICE_ENABLED,
            "VOICE_MAX_DURATION_MS": _cfg.VOICE_MAX_DURATION_MS,
            "VOICE_MERGE_WINDOW_SEC": _cfg.VOICE_MERGE_WINDOW_SEC,
            "VOICE_KEEP_AUDIO": _cfg.VOICE_KEEP_AUDIO,
            "ALLOWED_CHATS": list(_cfg.ALLOWED_CHATS),
        }
        _cfg.VOICE_ENABLED = True
        _cfg.VOICE_MAX_DURATION_MS = 60000
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        _cfg.VOICE_KEEP_AUDIO = False
        _cfg.ALLOWED_CHATS = []  # empty = allow all

        _SyncThread.constructed.clear()

        self._patches: list = []
        self.mock_download = self._install_mock(
            "_download_message_file", return_value="/tmp/fake_audio.opus",
        )
        self.mock_transcribe = self._install_mock(
            "transcribe_file",
            return_value={"ok": True, "text": "hello", "duration": 1.0,
                          "lang": "zh", "error": None},
        )
        self.mock_add_voice = self._install_mock("add_voice", return_value=None)
        self.mock_do_query = self._install_mock("_do_query", return_value=None)
        self.mock_send_card = self._install_mock(
            "send_card_reply", return_value="placeholder_mid",
        )
        self.mock_update_card = self._install_mock("update_card", return_value=True)
        # Patch logging + cancel-event side effects so they don't touch the
        # filesystem during tests.  The audio branch now calls these to
        # match the text path's pre-dispatch contract.
        self.mock_log_entry = self._install_mock("log_entry", return_value=None)
        self.mock_reset_cancel = self._install_mock("_reset_cancel", return_value=None)
        # Replace threading.Thread on the module's threading reference with
        # a synchronous variant so mocked _do_query call_count is reliable.
        self._thread_patch = patch.object(msg_mod.threading, "Thread", _SyncThread)
        self._thread_patch.start()
        self._patches.append(self._thread_patch)

    def _install_mock(self, attr: str, **mock_kwargs) -> MagicMock:
        mock = MagicMock(**mock_kwargs)
        p = patch.object(msg_mod, attr, mock)
        p.start()
        self._patches.append(p)
        return mock

    def tearDown(self) -> None:
        for p in self._patches:
            p.stop()
        for k, v in self._saved.items():
            setattr(_cfg, k, v)


class TestVoiceDisabledSilent(_AudioBranchBase):
    """REQ-02 — VOICE_ENABLED=False: branch returns silently, no I/O."""

    def test_disabled_silent_return(self) -> None:
        _cfg.VOICE_ENABLED = False
        msg_mod.handle_message(_make_audio_event())
        self.assertEqual(self.mock_download.call_count, 0)
        self.assertEqual(self.mock_transcribe.call_count, 0)
        self.assertEqual(self.mock_send_card.call_count, 0)
        self.assertEqual(self.mock_update_card.call_count, 0)
        self.assertEqual(self.mock_do_query.call_count, 0)
        self.assertEqual(self.mock_add_voice.call_count, 0)


class TestVoiceOversize(_AudioBranchBase):
    """REQ-04 — duration over VOICE_MAX_DURATION_MS rejected before download."""

    def test_oversize_rejected(self) -> None:
        msg_mod.handle_message(_make_audio_event(duration_ms=999999))
        self.assertEqual(self.mock_send_card.call_count, 1)
        # The "音频过长" warning card must mention "上限" so the user
        # understands why their message was rejected.
        args, kwargs = self.mock_send_card.call_args
        title = args[2] if len(args) >= 3 else kwargs.get("title", "")
        body = args[3] if len(args) >= 4 else kwargs.get("body", "")
        self.assertTrue("上限" in body or "上限" in title,
                        f"expected '上限' in title/body, got title={title!r} body={body!r}")
        self.assertEqual(self.mock_download.call_count, 0)
        self.assertEqual(self.mock_transcribe.call_count, 0)


class TestVoiceTranscribeNoMerge(_AudioBranchBase):
    """REQ-09 (no merge) — transcribe success → _do_query direct call."""

    def test_transcribe_success_no_merge(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        self.mock_transcribe.return_value = {
            "ok": True, "text": "你好", "duration": 1.2,
            "lang": "zh", "error": None,
        }
        msg_mod.handle_message(_make_audio_event())
        self.assertEqual(self.mock_do_query.call_count, 1)
        self.assertEqual(self.mock_add_voice.call_count, 0)
        # Prompt is forwarded as the ``message`` kwarg (REQ-17 — backend sees
        # the same value the text path would dispatch).
        kwargs = self.mock_do_query.call_args.kwargs
        self.assertEqual(kwargs.get("message"), "你好")


class TestVoiceTranscribeWithMerge(_AudioBranchBase):
    """REQ-09 (merge path) — VOICE_MERGE_WINDOW_SEC>0 routes through add_voice."""

    def test_transcribe_success_with_merge(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 10
        self.mock_transcribe.return_value = {
            "ok": True, "text": "你好", "duration": 1.2,
            "lang": "zh", "error": None,
        }
        msg_mod.handle_message(_make_audio_event())
        self.assertEqual(self.mock_add_voice.call_count, 1)
        self.assertEqual(self.mock_do_query.call_count, 0)
        # The first positional arg is chat_id; the second is the text.
        args, _ = self.mock_add_voice.call_args
        self.assertEqual(args[1], "你好")


class TestVoiceDownloadFailed(_AudioBranchBase):
    """REQ-06 — _download_message_file returns None: skip transcribe, update placeholder."""

    def test_download_failed(self) -> None:
        self.mock_download.return_value = None
        msg_mod.handle_message(_make_audio_event())
        self.assertEqual(self.mock_transcribe.call_count, 0)
        self.assertEqual(self.mock_update_card.call_count, 1)
        # The failure card title/body should mention "下载失败".
        args, kwargs = self.mock_update_card.call_args
        title = args[1] if len(args) >= 2 else kwargs.get("title", "")
        body = args[2] if len(args) >= 3 else kwargs.get("body", "")
        self.assertTrue("下载失败" in body or "下载失败" in title,
                        f"expected '下载失败' in title/body, got title={title!r} body={body!r}")


class TestVoiceCleanupOnFailure(_AudioBranchBase):
    """REQ-10 — transcribe failure with VOICE_KEEP_AUDIO=False unlinks the audio file."""

    def test_audio_cleanup_on_failure(self) -> None:
        self.mock_transcribe.return_value = {
            "ok": False, "text": "", "duration": 0.0,
            "lang": "zh", "error": "inference_failed:test",
        }
        _cfg.VOICE_KEEP_AUDIO = False
        with patch("os.unlink") as mock_unlink:
            msg_mod.handle_message(_make_audio_event())
        # Cleanup must run even when the transcribe step fails.
        mock_unlink.assert_called_once()
        self.assertEqual(mock_unlink.call_args.args[0], "/tmp/fake_audio.opus")


class TestVoiceEmptyTextFails(_AudioBranchBase):
    """REQ-08 — ``ok=True`` but ``text=""`` shares the failure path with ``ok=False``.

    Without this case the OR branch ``not result.get("ok") or not text_out``
    has only one of its two arms exercised; a regression that flips the
    operator could ship undetected.
    """

    def test_empty_text_treated_as_failure(self) -> None:
        self.mock_transcribe.return_value = {
            "ok": True, "text": "   ",  # only whitespace → strip()=="" path
            "duration": 0.5, "lang": "zh", "error": None,
        }
        msg_mod.handle_message(_make_audio_event())
        # Must NOT dispatch — text is empty.
        self.assertEqual(self.mock_do_query.call_count, 0)
        self.assertEqual(self.mock_add_voice.call_count, 0)
        # Must update placeholder card with a failure message.
        self.assertGreaterEqual(self.mock_update_card.call_count, 1)
        last_args, last_kwargs = self.mock_update_card.call_args
        title = last_args[1] if len(last_args) >= 2 else last_kwargs.get("title", "")
        body = last_args[2] if len(last_args) >= 3 else last_kwargs.get("body", "")
        self.assertTrue(
            "转写失败" in title or "转写失败" in body or "空文本" in body,
            f"expected failure card, got title={title!r} body={body!r}",
        )


class TestVoiceNoMergeUsesDaemonThread(_AudioBranchBase):
    """Review fix — no-merge dispatch must construct ``daemon=True`` Thread.

    Mirrors the text-path off-loading at ``_message.py:600-611`` so the
    SDK event worker is not held for the whole LLM call. Structural test
    against the captured Thread kwargs.
    """

    def test_no_merge_off_threads_do_query(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        msg_mod.handle_message(_make_audio_event())
        # Exactly one Thread for the _do_query dispatch.
        voice_threads = [
            c for c in _SyncThread.constructed
            if (c["name"] or "").startswith("voice-query-")
        ]
        self.assertEqual(len(voice_threads), 1)
        self.assertIs(voice_threads[0]["daemon"], True)
        self.assertIs(voice_threads[0]["target"], self.mock_do_query)
        # _do_query was invoked synchronously by _SyncThread.start().
        self.assertEqual(self.mock_do_query.call_count, 1)


class TestVoiceLogEntryAndResetCancel(_AudioBranchBase):
    """Review fix — audio branch parity with text path's pre-dispatch contract.

    ``log_entry`` must record the transcribed text so the memory system
    sees the turn; ``_reset_cancel`` must clear stale cancel events from
    a prior ``/cancel`` that would otherwise abort ``_do_query`` immediately.
    """

    def test_no_merge_calls_log_entry_and_reset_cancel(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        self.mock_transcribe.return_value = {
            "ok": True, "text": "你好世界", "duration": 0.8,
            "lang": "zh", "error": None,
        }
        msg_mod.handle_message(_make_audio_event())
        # log_entry called with role="user" and the transcribed text.
        self.mock_log_entry.assert_called_once()
        args, kwargs = self.mock_log_entry.call_args
        self.assertEqual(args[1], "user")
        self.assertEqual(args[2], "你好世界")
        # _reset_cancel called exactly once with chat_id.
        self.mock_reset_cancel.assert_called_once()
        self.assertEqual(self.mock_reset_cancel.call_args.args[0],
                         "chat_voice_test")

    def test_merge_path_also_logs_and_resets(self) -> None:
        _cfg.VOICE_MERGE_WINDOW_SEC = 10
        self.mock_transcribe.return_value = {
            "ok": True, "text": "你好", "duration": 0.5,
            "lang": "zh", "error": None,
        }
        msg_mod.handle_message(_make_audio_event())
        # Even merge path needs log_entry + reset_cancel for parity.
        self.mock_log_entry.assert_called_once()
        self.mock_reset_cancel.assert_called_once()


# ── /voice command tests ──────────────────────────────────────────────
# Exercises ``larkhelm.commands._cmd_voice`` directly. Mirrors the
# precedent set by ``tests/test_phase4.py`` for ``_cmd_lock`` (review
# fix #1 — `_cmd_voice` was shipped with zero direct tests).

class _CmdVoiceBase(unittest.TestCase):
    """Bootstrap: snapshot cfg, mock send_card_reply, install per-chat lang store."""

    def setUp(self) -> None:
        import larkhelm.commands as cmd_mod
        self.cmd_mod = cmd_mod
        self._saved = {
            "VOICE_ENABLED": _cfg.VOICE_ENABLED,
            "VOICE_MODEL_SIZE": _cfg.VOICE_MODEL_SIZE,
            "VOICE_COMPUTE_TYPE": _cfg.VOICE_COMPUTE_TYPE,
            "VOICE_DEFAULT_LANG": _cfg.VOICE_DEFAULT_LANG,
            "VOICE_MAX_DURATION_MS": _cfg.VOICE_MAX_DURATION_MS,
            "VOICE_MERGE_WINDOW_SEC": _cfg.VOICE_MERGE_WINDOW_SEC,
            "VOICE_MAX_MERGE": _cfg.VOICE_MAX_MERGE,
        }
        _cfg.VOICE_ENABLED = True
        _cfg.VOICE_MODEL_SIZE = "small"
        _cfg.VOICE_COMPUTE_TYPE = "int8"
        _cfg.VOICE_DEFAULT_LANG = "zh"
        _cfg.VOICE_MAX_DURATION_MS = 180000
        _cfg.VOICE_MERGE_WINDOW_SEC = 0
        _cfg.VOICE_MAX_MERGE = 5

        self.mock_send_card = MagicMock(return_value="mid_001")
        self._send_patch = patch.object(cmd_mod, "send_card_reply",
                                        self.mock_send_card)
        self._send_patch.start()

        # Use a real chat_state lang store but isolated to a fresh chat_id.
        self.chat_id = f"chat_voice_cmd_{next(_msg_seq)}"

    def tearDown(self) -> None:
        self._send_patch.stop()
        for k, v in self._saved.items():
            setattr(_cfg, k, v)


class TestCmdVoiceStatus(_CmdVoiceBase):
    """``/voice`` and ``/voice status`` render a status card."""

    def test_default_args_renders_status(self) -> None:
        with patch("larkhelm.voice.transcribe.is_model_loaded",
                   return_value=False):
            self.cmd_mod._cmd_voice(self.chat_id, "", "msg_001")
        self.mock_send_card.assert_called_once()
        args, _ = self.mock_send_card.call_args
        # send_card_reply(chat_id, msg_id, title, body, color=...)
        title = args[2]
        body = args[3]
        self.assertIn("Voice", title)
        self.assertIn("总开关", body)
        self.assertIn("当前语种", body)

    def test_status_renders_loaded_true(self) -> None:
        with patch("larkhelm.voice.transcribe.is_model_loaded",
                   return_value=True):
            self.cmd_mod._cmd_voice(self.chat_id, "status", "msg_001")
        self.mock_send_card.assert_called_once()
        body = self.mock_send_card.call_args.args[3]
        self.assertIn("已加载", body)

    def test_status_falls_back_when_probe_raises(self) -> None:
        # is_model_loaded probe failure must not bubble up — _cmd_voice
        # treats it as "not loaded" and continues to render the card.
        with patch("larkhelm.voice.transcribe.is_model_loaded",
                   side_effect=RuntimeError("simulated import failure")):
            self.cmd_mod._cmd_voice(self.chat_id, "status", "msg_001")
        self.mock_send_card.assert_called_once()
        body = self.mock_send_card.call_args.args[3]
        self.assertIn("未加载", body)


class TestCmdVoiceLang(_CmdVoiceBase):
    """``/voice lang <x>`` validates against the whitelist."""

    def test_lang_valid_zh(self) -> None:
        from larkhelm.chat_state import _get_voice_lang
        self.cmd_mod._cmd_voice(self.chat_id, "lang en", "msg_001")
        self.mock_send_card.assert_called_once()
        title = self.mock_send_card.call_args.args[2]
        self.assertIn("已切换", title)
        self.assertEqual(_get_voice_lang(self.chat_id), "en")

    def test_lang_uppercase_normalized(self) -> None:
        # _cmd_voice lowercases before whitelist check.
        from larkhelm.chat_state import _get_voice_lang
        self.cmd_mod._cmd_voice(self.chat_id, "lang ZH", "msg_001")
        title = self.mock_send_card.call_args.args[2]
        self.assertIn("已切换", title)
        self.assertEqual(_get_voice_lang(self.chat_id), "zh")

    def test_lang_invalid_rejected(self) -> None:
        self.cmd_mod._cmd_voice(self.chat_id, "lang ja", "msg_001")
        self.mock_send_card.assert_called_once()
        title = self.mock_send_card.call_args.args[2]
        body = self.mock_send_card.call_args.args[3]
        self.assertIn("无效", title)
        self.assertTrue("zh" in body and "en" in body and "auto" in body)


class TestCmdVoiceUsage(_CmdVoiceBase):
    """Unrecognized args render the usage card."""

    def test_unknown_args_renders_usage(self) -> None:
        self.cmd_mod._cmd_voice(self.chat_id, "garbage args here", "msg_001")
        self.mock_send_card.assert_called_once()
        title = self.mock_send_card.call_args.args[2]
        self.assertIn("用法", title)


if __name__ == "__main__":
    unittest.main(verbosity=2)
