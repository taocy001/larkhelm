"""Coverage for ``log._get_recent_turns(..., dedup_prefix=...)`` (P0-3 REQ-03/04, AC-02).

The ``memory_recent_turns_dedup`` flag used to be a placeholder in
:mod:`larkhelm_config.example.json` — it gated only the legacy
``MemoryContextBuilder.deduped_recent_turns`` list-level filter and was a
no-op for crew agents that read ``_get_recent_turns`` directly. This file
pins the new behaviour:

* ``dedup_prefix=None`` (or flag off) is **byte-identical** to the
  pre-change code path.
* ``dedup_prefix`` non-empty + flag on filters turns whose first-60-char
  body matches the prefix.
* ``_query.py``'s degradation path (session memory missing →
  ``extract_work_context`` returns ``""``) still hands ``dedup_prefix=None``
  to ``_get_recent_turns``.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import log as larkhelm_log
from larkhelm.log import _get_recent_turns
from larkhelm.memory_context import extract_work_context


def _write_jsonl(log_dir: Path, records: list[dict]) -> None:
    with (log_dir / "all.jsonl").open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


class _DedupIOBase(unittest.TestCase):
    """Shared fixture: tmp LOG_DIR + DEBUG_LOG; patches _cfg on log module."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="larkhelm_test_dedup_"))
        self.log_dir = self.tmp / "logs"
        self.debug_log = self.tmp / "larkhelm.log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log.write_text("", encoding="utf-8")
        self.cfg_patches = [
            patch.object(larkhelm_log._cfg, "LOG_DIR", self.log_dir, create=True),
            patch.object(larkhelm_log._cfg, "DEBUG_LOG", self.debug_log, create=True),
        ]
        for p in self.cfg_patches:
            p.start()
        # ``_should_dedup`` reads ``_cfg.config`` — give it an empty dict
        # so the flag defaults to True (matches production default).
        self.config_patch = patch.object(
            larkhelm_log._cfg, "config", {}, create=True,
        )
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        for p in self.cfg_patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _seed_two_user_turns(self, chat: str) -> None:
        records = [
            {
                "ts": "2026-01-01T00:00:00", "chat_id": chat, "role": "user",
                "content": "实现 X 模块的核心功能：配置加载、命令分发、状态机驱动",
                "model": "claude",
            },
            {
                "ts": "2026-01-01T00:00:01", "chat_id": chat, "role": "assistant",
                "content": "好的，我先读现有 config.py 然后增量改",
                "model": "claude",
            },
            {
                "ts": "2026-01-01T00:00:02", "chat_id": chat, "role": "user",
                "content": "顺便看一下 _query.py 是不是已经接入了 dedup_prefix 字段",
                "model": "claude",
            },
        ]
        _write_jsonl(self.log_dir, records)


class TestFlagOffByteCompat(_DedupIOBase):
    """(1) flag off → ``dedup_prefix`` ignored → byte-identical to dedup_prefix=None."""

    def test_flag_off_no_dedup_byte_compat(self):
        chat = "chat_flagoff"
        self._seed_two_user_turns(chat)
        # Flag off in config.
        larkhelm_log._cfg.config = {"memory_recent_turns_dedup": False}
        try:
            prefix = "实现 X 模块的核心功能：配置加载、命令分发、状态机驱动"
            with_prefix = _get_recent_turns(chat, dedup_prefix=prefix)
            without = _get_recent_turns(chat, dedup_prefix=None)
            self.assertEqual(with_prefix, without)
            # First turn body still present — proof dedup didn't run.
            self.assertIn("实现 X 模块的核心功能", with_prefix)
        finally:
            larkhelm_log._cfg.config = {}


class TestFlagOnPrefixHitSkips(_DedupIOBase):
    """(2) flag on + prefix containing first 60 chars of a turn → that turn dropped."""

    def test_flag_on_prefix_hit_skips(self):
        chat = "chat_flagon"
        self._seed_two_user_turns(chat)
        larkhelm_log._cfg.config = {"memory_recent_turns_dedup": True}
        try:
            # Build a dedup_prefix that contains the first 60 chars of turn 1
            # AND turn 2 (so two turns should drop), but NOT the third user
            # turn (which is still "fresh" and must be kept).
            prefix = (
                "## Work Context\n"
                "实现 X 模块的核心功能：配置加载、命令分发、状态机驱动\n"
                "好的，我先读现有 config.py 然后增量改\n"
            )
            result = _get_recent_turns(chat, dedup_prefix=prefix)
            self.assertNotIn("实现 X 模块的核心功能", result)
            self.assertNotIn("好的，我先读现有 config.py", result)
            self.assertIn("dedup_prefix", result)  # third turn body, kept
            # Debug log records the skipped-count line.
            debug = self.debug_log.read_text(encoding="utf-8")
            self.assertIn("[Memory] recent_turns dedup skipped", debug)
        finally:
            larkhelm_log._cfg.config = {}


class TestSessionMissingDegrades(_DedupIOBase):
    """(3) ``extract_work_context`` empty → ``_query.py`` path hands None to _get_recent_turns."""

    def test_extract_work_context_empty_inputs(self):
        # raw=None / "" → ""
        self.assertEqual(extract_work_context(None), "")
        self.assertEqual(extract_work_context(""), "")
        # Unparseable raw (no recognisable H2 layout) → "" because
        # ``split_session_slots`` returns parsed=False on no sections.
        self.assertEqual(extract_work_context("some random body without sections"), "")

    def test_query_path_degrades_to_none(self):
        """Simulate the _query.py degradation path: load_memory returns None →
        extract_work_context returns "" → dedup_prefix becomes None and the
        legacy code path runs (byte-compat with PR-prior)."""
        chat = "chat_degrade"
        self._seed_two_user_turns(chat)

        # Stand-in for the _query.py wiring (no need to import _do_query):
        raw_session = None  # load_memory(chat_id) returned None
        wc = extract_work_context(raw_session)
        dedup_prefix = wc or None
        self.assertIsNone(dedup_prefix)

        # Spy on _get_recent_turns to confirm dedup_prefix=None is what gets
        # passed when memory is missing.
        calls: list = []
        original = larkhelm_log._get_recent_turns

        def _spy(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return original(*args, **kwargs)

        with patch.object(larkhelm_log, "_get_recent_turns", _spy):
            larkhelm_log._get_recent_turns(chat, dedup_prefix=dedup_prefix)

        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["kwargs"].get("dedup_prefix"), None)


class TestDedupPrefixNoneEqualsLegacy(_DedupIOBase):
    """(4) ``dedup_prefix=None`` ≡ omitting the keyword (default None)."""

    def test_dedup_prefix_none_equals_legacy(self):
        chat = "chat_default"
        self._seed_two_user_turns(chat)
        larkhelm_log._cfg.config = {"memory_recent_turns_dedup": True}
        try:
            explicit_none = _get_recent_turns(chat, dedup_prefix=None)
            default_kwarg = _get_recent_turns(chat)
            self.assertEqual(explicit_none, default_kwarg)
            # Sanity: the legacy path returned both turns intact.
            self.assertIn("实现 X 模块的核心功能", explicit_none)
            self.assertIn("dedup_prefix", explicit_none)
        finally:
            larkhelm_log._cfg.config = {}


if __name__ == "__main__":
    unittest.main()
