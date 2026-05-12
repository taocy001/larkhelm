"""Coverage for ``log._read_logs_tail`` and its use in ``memory.maybe_auto_update``.

These tests guard the I/O upper-bound fix: prior to the fix, the auto-update
path called ``_read_logs`` which scanned the full ``all.jsonl`` (potentially
100 MB). ``_read_logs_tail`` reads at most ``max_bytes`` from the file tail
and skips the legacy ``.jsonl.1`` rotation backup entirely.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import log as larkhelm_log


class TestTailScan(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="larkhelm_test_tail_"))
        self.log_dir = self.tmp / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log = self.tmp / "larkhelm.log"
        self.cfg_patches = [
            patch.object(larkhelm_log._cfg, "LOG_DIR", self.log_dir, create=True),
            patch.object(larkhelm_log._cfg, "DEBUG_LOG", self.debug_log, create=True),
        ]
        for p in self.cfg_patches:
            p.start()

    def tearDown(self):
        for p in self.cfg_patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_jsonl(self, records: list[dict]) -> None:
        path = self.log_dir / "all.jsonl"
        with path.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── AC-04: scan is bounded by max_bytes ──────────────────────────────────

    def test_scan_bytes_bounded(self):
        """File >> max_bytes: cumulative read must be ≤ max_bytes."""
        records = []
        for i in range(50000):
            cid = "C_A" if i % 2 == 0 else "C_B"
            records.append({
                "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}",
                "chat_id": cid, "role": "user",
                "content": f"line-{i}-payload-padding-payload",
                "model": "claude",
            })
        self._write_jsonl(records)
        file_size = (self.log_dir / "all.jsonl").stat().st_size
        max_bytes = 1024 * 1024
        self.assertGreater(file_size, max_bytes,
                           "test fixture must exceed scan window")

        PathCls = Path
        original_open = PathCls.open
        read_total = [0]

        def counting_open(self, mode="r", *args, **kwargs):
            f = original_open(self, mode, *args, **kwargs)
            if "b" in mode and str(self).endswith("all.jsonl"):
                real_read = f.read

                def _read(*a, **kw):
                    data = real_read(*a, **kw)
                    if data is not None:
                        read_total[0] += len(data)
                    return data
                f.read = _read
            return f

        with patch.object(PathCls, "open", counting_open):
            result = larkhelm_log._read_logs_tail("C_A", max_bytes=max_bytes)

        self.assertLessEqual(read_total[0], max_bytes,
                             f"tail-scan read {read_total[0]} > max_bytes {max_bytes}")
        self.assertGreater(len(result), 0,
                           "tail-scan must return at least some C_A records")
        for r in result:
            self.assertEqual(r["chat_id"], "C_A")

    # ── AC-05: maybe_auto_update's role/model filter still works on tail ─────

    def test_role_model_filter_preserved(self):
        """Tail records should pass through the same whitelist/blacklist that
        ``maybe_auto_update`` applies, so tool/crew/shell rows are dropped."""
        records = [
            {"ts": "T1", "chat_id": "CX", "role": "user",
             "content": "user-msg", "model": "claude"},
            {"ts": "T2", "chat_id": "CX", "role": "assistant",
             "content": "assistant-msg", "model": "claude"},
            {"ts": "T3", "chat_id": "CX", "role": "tool",
             "content": "tool-event", "model": "claude"},
            {"ts": "T4", "chat_id": "CX", "role": "user",
             "content": "crew-sub-msg", "model": "crew"},
            {"ts": "T5", "chat_id": "CX", "role": "user",
             "content": "shell-output", "model": "shell"},
            {"ts": "T6", "chat_id": "CX", "role": "milestone",
             "content": "milestone-event", "model": "milestone"},
        ]
        self._write_jsonl(records)
        logs = larkhelm_log._read_logs_tail("CX")
        # mirror the filter used in memory.maybe_auto_update
        filtered = [
            r for r in logs
            if r["role"] in ("user", "assistant", "milestone")
            and r.get("model") not in ("crew", "shell")
        ]
        contents = {r["content"] for r in filtered}
        self.assertIn("user-msg", contents)
        self.assertIn("assistant-msg", contents)
        self.assertIn("milestone-event", contents)
        self.assertNotIn("tool-event", contents)
        self.assertNotIn("crew-sub-msg", contents)
        self.assertNotIn("shell-output", contents)

    def test_tail_truncated_emits_debug(self):
        """When the file exceeds ``max_bytes`` the debug log must record it
        so operators can correlate a missing pre-window record with the bound.
        """
        records = []
        for i in range(20000):
            records.append({
                "ts": f"T{i}", "chat_id": "C1", "role": "user",
                "content": "x" * 200, "model": "claude",
            })
        self._write_jsonl(records)
        size = (self.log_dir / "all.jsonl").stat().st_size
        self.assertGreater(size, 1024 * 1024,
                           "fixture must be > default 1 MiB window")

        larkhelm_log._read_logs_tail("C1")
        dbg = self.debug_log.read_text(encoding="utf-8")
        self.assertIn("[Memory] tail-scan truncated", dbg)

    def test_no_truncation_when_under_window(self):
        """File smaller than max_bytes should not emit a truncation line."""
        records = [
            {"ts": f"T{i}", "chat_id": "C2", "role": "user",
             "content": "hi", "model": "claude"}
            for i in range(5)
        ]
        self._write_jsonl(records)
        # Clear any prior debug content
        self.debug_log.write_text("", encoding="utf-8")
        larkhelm_log._read_logs_tail("C2", max_bytes=1024 * 1024)
        dbg = self.debug_log.read_text(encoding="utf-8")
        self.assertNotIn("tail-scan truncated", dbg)

    def test_missing_file_returns_empty(self):
        self.assertEqual(larkhelm_log._read_logs_tail("nobody"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
