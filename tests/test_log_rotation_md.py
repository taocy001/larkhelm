"""Coverage for the Markdown shard rotation in ``log._log_file`` / ``log_entry``.

Prior to the fix a single ``{date}.md`` could grow unboundedly when a chat
generated heavy ``/dev`` traffic. The fix rolls to ``{date}-1.md`` once the
active shard reaches ``_MAX_LOG_MD_BYTES`` (10 MiB), keeping the function
signature unchanged so all callers transparently pick up rotation.
"""
from __future__ import annotations

import shutil
import tempfile
import unittest
import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from larkhelm import log as larkhelm_log
from larkhelm import memory_io as _mio


class TestMdRotation(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="larkhelm_test_md_rot_"))
        self.log_dir = self.tmp / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log = self.tmp / "larkhelm.log"
        self.chat_id = "chat_md_rotation"
        self.cfg_patches = [
            patch.object(larkhelm_log._cfg, "LOG_DIR", self.log_dir, create=True),
            patch.object(larkhelm_log._cfg, "DEBUG_LOG", self.debug_log, create=True),
        ]
        for p in self.cfg_patches:
            p.start()
        self.date_str = datetime.now().strftime("%Y-%m-%d")
        self.chat_dir = self.log_dir / self.chat_id
        self.chat_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        for p in self.cfg_patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── AC-06: shard rolls over when the active file exceeds 10 MiB ─────────

    def test_shard_on_size_exceeded(self):
        first = self.chat_dir / f"{self.date_str}.md"
        first.write_bytes(b"x" * (11 * 1024 * 1024))
        first_size_before = first.stat().st_size

        larkhelm_log.log_entry(self.chat_id, "user", "shard-2-entry",
                               model="claude")

        second = self.chat_dir / f"{self.date_str}-1.md"
        self.assertTrue(second.exists(),
                        "shard {date}-1.md must be created when first ≥ 10 MiB")
        self.assertEqual(first.stat().st_size, first_size_before,
                         "first shard must NOT be modified after rollover")
        body = second.read_text(encoding="utf-8")
        self.assertIn("shard-2-entry", body)

    # ── shard header announces "(part 2)" ───────────────────────────────────

    def test_shard_header_present(self):
        first = self.chat_dir / f"{self.date_str}.md"
        first.write_bytes(b"x" * (11 * 1024 * 1024))
        larkhelm_log.log_entry(self.chat_id, "user", "hi", model="claude")
        second = self.chat_dir / f"{self.date_str}-1.md"
        first_line = second.read_text(encoding="utf-8").splitlines()[0]
        self.assertTrue(first_line.startswith("# "),
                        f"header must start with '# ', got {first_line!r}")
        self.assertIn(self.chat_id, first_line)
        self.assertIn("(part 2)", first_line,
                      f"shard header should mark (part 2), got {first_line!r}")

    # ── subsequent writes stay in -1.md (no premature -2.md creation) ───────

    def test_second_write_stays_in_same_shard(self):
        first = self.chat_dir / f"{self.date_str}.md"
        first.write_bytes(b"x" * (11 * 1024 * 1024))
        larkhelm_log.log_entry(self.chat_id, "user", "entry-1", model="claude")
        second = self.chat_dir / f"{self.date_str}-1.md"
        size_after_first = second.stat().st_size

        larkhelm_log.log_entry(self.chat_id, "user", "entry-2", model="claude")
        third = self.chat_dir / f"{self.date_str}-2.md"
        self.assertFalse(third.exists(),
                         "{date}-2.md must not appear while {date}-1.md still < 10 MiB")
        self.assertGreater(second.stat().st_size, size_after_first,
                           "second write should append to existing -1 shard")

    # ── AC-07a: _read_logs still returns rotated-day content ────────────────

    def test_read_logs_unaffected(self):
        first = self.chat_dir / f"{self.date_str}.md"
        first.write_bytes(b"x" * (11 * 1024 * 1024))
        larkhelm_log.log_entry(self.chat_id, "user", "entry-A", model="claude")
        larkhelm_log.log_entry(self.chat_id, "assistant", "entry-B", model="claude")
        records = larkhelm_log._read_logs(self.chat_id)
        contents = [r["content"] for r in records]
        self.assertIn("entry-A", contents)
        self.assertIn("entry-B", contents)

    # ── AC-07b: export_memory packs all shards ──────────────────────────────

    def test_export_includes_all_shards(self):
        first = self.chat_dir / f"{self.date_str}.md"
        first.write_bytes(b"x" * (11 * 1024 * 1024))
        larkhelm_log.log_entry(self.chat_id, "user", "shard-2-entry",
                               model="claude")
        second = self.chat_dir / f"{self.date_str}-1.md"
        self.assertTrue(second.exists())

        # export_memory expects DATA_DIR (tmp); logs/ is under it.
        out = self.tmp / "out.zip"
        _mio.export_memory(out, data_dir=self.tmp)
        with zipfile.ZipFile(out, "r") as zf:
            names = zf.namelist()
        self.assertIn(f"data/logs/{self.chat_id}/{self.date_str}.md", names)
        self.assertIn(f"data/logs/{self.chat_id}/{self.date_str}-1.md", names)

    # ── deadlock-fix buffer: pending rotation msgs are queued then drained ──

    def test_pending_rotation_msg_buffered_and_drained(self):
        """The fix queues an info() notice instead of calling info() while holding
        ``_log_lock`` (which is non-reentrant). Verify both halves of the contract:
        ``_log_file`` enqueues exactly one message on rollover, and ``log_entry``
        drains the buffer before returning so it cannot accumulate across writes.
        """
        first = self.chat_dir / f"{self.date_str}.md"
        first.write_bytes(b"x" * (11 * 1024 * 1024))

        # Capture buffer state at the moment _log_file returns (still inside
        # _log_lock). Patching the bound function lets us snapshot before
        # log_entry's drain step runs.
        captured: list[list[str]] = []
        original = larkhelm_log._log_file

        def spy(chat_id):
            result = original(chat_id)
            captured.append(list(larkhelm_log._pending_md_rotation_msgs))
            return result

        # Make sure we start from a clean buffer (other tests in this file
        # also trigger rotation, so leftover state would muddle the assertion).
        larkhelm_log._pending_md_rotation_msgs.clear()

        with patch.object(larkhelm_log, "_log_file", side_effect=spy):
            larkhelm_log.log_entry(self.chat_id, "user", "rolls-over",
                                   model="claude")

        self.assertEqual(len(captured), 1,
                         "spy must have observed exactly one _log_file call")
        self.assertEqual(len(captured[0]), 1,
                         "exactly one rotation msg should have been buffered "
                         f"during the rollover write, got {captured[0]!r}")
        self.assertIn(f"{self.date_str}-1.md", captured[0][0])
        # After log_entry returns the buffer is drained so subsequent writes
        # don't replay stale notices.
        self.assertEqual(larkhelm_log._pending_md_rotation_msgs, [],
                         "buffer must be drained after log_entry returns")


if __name__ == "__main__":
    unittest.main(verbosity=2)
