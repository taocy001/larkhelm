"""Tests for the B3 extended ``workspace_meta.json`` schema and the
``_handle_stale_workspace`` decision helper.

The schema grew from ``{task_hash, completed}`` to:

    {task_hash, completed, commit_sha, finalized_at, chat_id, plan_id}

Old metas with only the first two fields must still parse via
``dict.get(...)`` with safe defaults — checked here.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Minimal config bootstrap so larkhelm.log / chat_state work
_TMP = tempfile.mkdtemp(prefix="larkhelm_ws_meta_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.crew._commands import (
    _read_workspace_meta,
    _write_workspace_meta,
    _handle_stale_workspace,
)


def _fresh_ws() -> Path:
    """Return a fresh empty ``.crew_workspace`` directory in a tempdir."""
    td = Path(tempfile.mkdtemp(prefix="larkhelm_ws_"))
    ws = td / ".crew_workspace"
    ws.mkdir()
    return ws


class ReadWriteRoundtripTests(unittest.TestCase):

    def test_write_full_schema_roundtrips(self):
        ws = _fresh_ws()
        _write_workspace_meta(
            ws, task_hash="abc", completed=True,
            commit_sha="deadbeef", finalized_at=12345.6,
            chat_id="oc_x", plan_id="p1",
        )
        m = _read_workspace_meta(ws)
        self.assertEqual(m["task_hash"], "abc")
        self.assertTrue(m["completed"])
        self.assertEqual(m["commit_sha"], "deadbeef")
        self.assertEqual(m["finalized_at"], 12345.6)
        self.assertEqual(m["chat_id"], "oc_x")
        self.assertEqual(m["plan_id"], "p1")

    def test_legacy_2field_meta_still_parses(self):
        """Old write call signature ``_write_workspace_meta(ws, task_hash)``
        still produces a fresh meta. Old on-disk metas (just task_hash +
        completed, no other keys) parse to dict cleanly — caller uses
        ``dict.get(...)`` for the new fields."""
        ws = _fresh_ws()
        # Simulate an on-disk legacy meta written by an older version.
        (ws / "workspace_meta.json").write_text(
            json.dumps({"task_hash": "old_abc", "completed": False})
        )
        m = _read_workspace_meta(ws)
        # Fields present
        self.assertEqual(m["task_hash"], "old_abc")
        self.assertFalse(m["completed"])
        # Missing fields default cleanly
        self.assertEqual(m.get("commit_sha", ""), "")
        self.assertEqual(m.get("finalized_at", 0.0), 0.0)
        self.assertEqual(m.get("chat_id", ""), "")
        self.assertEqual(m.get("plan_id", ""), "")

    def test_legacy_write_call_zero_defaults(self):
        """``_write_workspace_meta(ws, task_hash, completed=False)`` from
        a pre-B3 call site must produce a complete schema with safe
        defaults for the new fields."""
        ws = _fresh_ws()
        _write_workspace_meta(ws, task_hash="legacy", completed=False)
        m = _read_workspace_meta(ws)
        self.assertEqual(m["task_hash"], "legacy")
        self.assertFalse(m["completed"])
        self.assertEqual(m["commit_sha"], "")
        self.assertEqual(m["finalized_at"], 0.0)
        self.assertEqual(m["chat_id"], "")
        self.assertEqual(m["plan_id"], "")

    def test_missing_file_returns_empty_dict(self):
        ws = _fresh_ws()
        # No file present
        self.assertEqual(_read_workspace_meta(ws), {})

    def test_garbage_json_returns_empty_dict(self):
        ws = _fresh_ws()
        (ws / "workspace_meta.json").write_text("{not valid json")
        self.assertEqual(_read_workspace_meta(ws), {})


class HandleStaleWorkspaceTests(unittest.TestCase):

    def _write_meta(self, ws: Path, **overrides) -> None:
        payload = {
            "task_hash":    "old_hash",
            "completed":    False,
            "commit_sha":   "",
            "finalized_at": 0.0,
            "chat_id":      "oc_chatA",
            "plan_id":      "",
        }
        payload.update(overrides)
        (ws / "workspace_meta.json").write_text(json.dumps(payload))

    def test_empty_meta_returns_no_reuse_no_notice(self):
        ws = _fresh_ws()
        reuse, notice = _handle_stale_workspace(
            ws, chat_id="oc_chatA",
            new_task_hash="new_hash", meta={},
        )
        self.assertFalse(reuse)
        self.assertEqual(notice, "")

    def test_completed_meta_no_notice(self):
        """``completed=True`` means the user already wrapped this task —
        starting a different one is just normal flow, no notice needed."""
        ws = _fresh_ws()
        meta = {"task_hash": "old_hash", "completed": True,
                "chat_id": "oc_chatA"}
        self._write_meta(ws, completed=True)
        reuse, notice = _handle_stale_workspace(
            ws, chat_id="oc_chatA",
            new_task_hash="new_hash", meta=meta,
        )
        self.assertFalse(reuse)
        self.assertEqual(notice, "")

    def test_same_task_hash_reuses(self):
        ws = _fresh_ws()
        meta = {"task_hash": "shared_hash", "completed": False,
                "chat_id": "oc_chatA"}
        self._write_meta(ws, task_hash="shared_hash")
        reuse, notice = _handle_stale_workspace(
            ws, chat_id="oc_chatA",
            new_task_hash="shared_hash", meta=meta,
        )
        self.assertTrue(reuse)
        self.assertEqual(notice, "")

    def test_different_chat_silent(self):
        """Multi-chat collision — reviewer guidance is silent hand-off,
        no notice card."""
        ws = _fresh_ws()
        meta = {"task_hash": "old_hash", "completed": False,
                "chat_id": "oc_chatB"}
        self._write_meta(ws, chat_id="oc_chatB")
        reuse, notice = _handle_stale_workspace(
            ws, chat_id="oc_chatA",   # different chat
            new_task_hash="new_hash", meta=meta,
        )
        self.assertFalse(reuse)
        self.assertEqual(notice, "")

    def test_same_chat_recent_different_task_emits_notice(self):
        """Same chat + different task + age < cutoff = the user just kicked
        off something new while old was pending. Notice card warranted."""
        ws = _fresh_ws()
        self._write_meta(ws, chat_id="oc_chatA")
        meta = _read_workspace_meta(ws)
        # Force the configured threshold to a big value so the meta we
        # just wrote is well within window.
        with patch.object(_cfg, "WORKSPACE_FINALIZE_PROMPT_AGE_SEC", 99999.0):
            reuse, notice = _handle_stale_workspace(
                ws, chat_id="oc_chatA",
                new_task_hash="new_hash", meta=meta,
            )
        self.assertFalse(reuse)
        self.assertTrue(notice)
        self.assertIn("旧任务", notice)

    def test_same_chat_aged_out_silent(self):
        """Different task in same chat but older than the cutoff = the
        notice is silenced (it's just an abandoned scratch space)."""
        ws = _fresh_ws()
        self._write_meta(ws, chat_id="oc_chatA")
        meta = _read_workspace_meta(ws)
        # 0-second cutoff means anything is "aged out".
        with patch.object(_cfg, "WORKSPACE_FINALIZE_PROMPT_AGE_SEC", 0.0):
            reuse, notice = _handle_stale_workspace(
                ws, chat_id="oc_chatA",
                new_task_hash="new_hash", meta=meta,
            )
        self.assertFalse(reuse)
        self.assertEqual(notice, "")

    def test_legacy_meta_no_chat_id_treated_as_current(self):
        """Old metas have empty chat_id. They must be treated as belonging
        to the current chat for byte-compat — otherwise existing single-chat
        users would see notice cards on every workspace transition."""
        ws = _fresh_ws()
        legacy = {"task_hash": "old_hash", "completed": False}
        (ws / "workspace_meta.json").write_text(json.dumps(legacy))
        meta = _read_workspace_meta(ws)
        with patch.object(_cfg, "WORKSPACE_FINALIZE_PROMPT_AGE_SEC", 99999.0):
            reuse, notice = _handle_stale_workspace(
                ws, chat_id="oc_chatA",
                new_task_hash="new_hash", meta=meta,
            )
        # Different task + same-chat (legacy compat) + within window = notice
        self.assertFalse(reuse)
        self.assertTrue(notice)


class ConfigDefaultTests(unittest.TestCase):

    def test_workspace_finalize_prompt_age_sec_default(self):
        """Config default per B3 spec: 3600s (1h)."""
        self.assertEqual(getattr(_cfg, "WORKSPACE_FINALIZE_PROMPT_AGE_SEC",
                                 None), 3600.0)


if __name__ == "__main__":
    unittest.main()
