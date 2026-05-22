"""Integration tests for SEC-H2: log_entry Markdown-path redaction.

Prior to this fix ``log_entry`` applied ``redact_error`` only to the JSONL
``record["content"]`` field; the Markdown shard received the raw content.
These tests assert that both write paths produce redacted output and that
normal (non-secret) content is preserved unchanged.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from larkhelm import log as larkhelm_log
from larkhelm.log import log_entry, _pending_md_rotation_msgs


class TestLogRedact(unittest.TestCase):
    """Integration: log_entry writes redacted content to both Markdown and JSONL."""

    # ── helpers ──────────────────────────────────────────────────────────────

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="larkhelm_test_redact_"))
        self.log_dir = self.tmp / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log = self.tmp / "larkhelm.log"
        self.chat_id = "chat_redact_test"
        self.cfg_patches = [
            patch.object(larkhelm_log._cfg, "LOG_DIR", self.log_dir, create=True),
            patch.object(larkhelm_log._cfg, "DEBUG_LOG", self.debug_log, create=True),
        ]
        for p in self.cfg_patches:
            p.start()
        self.date_str = datetime.now().strftime("%Y-%m-%d")

    def tearDown(self):
        for p in self.cfg_patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read_md(self) -> str:
        chat_dir = self.log_dir / self.chat_id
        # There may be multiple shards; read all of them.
        parts: list[str] = []
        for p in sorted(chat_dir.glob("*.md")):
            parts.append(p.read_text(encoding="utf-8"))
        return "\n".join(parts)

    def _read_jsonl(self) -> list[dict]:
        jsonl_path = self.log_dir / "all.jsonl"
        if not jsonl_path.exists():
            return []
        lines = []
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                continue
        return [r for r in lines if r.get("chat_id") == self.chat_id]

    # ── AC-01: api_key= pattern ───────────────────────────────────────────────

    def test_md_redacts_api_key_eq(self):
        """Markdown shard must not contain the raw api_key value."""
        secret = "api_key=sk-abc1234567890abcdef1234"
        log_entry(self.chat_id, "user", secret, model="claude")

        md = self._read_md()
        self.assertNotIn("sk-abc1234567890abcdef1234", md,
                         "raw api_key value must be redacted in Markdown")
        self.assertIn("api_key=***", md,
                      "redacted placeholder must appear in Markdown")

    # ── AC-02: Authorization Bearer pattern ──────────────────────────────────

    def test_md_redacts_authorization_bearer(self):
        """Markdown shard must not contain the raw Bearer token."""
        secret = "Authorization: Bearer mysupersecrettoken12345"
        log_entry(self.chat_id, "assistant", secret, model="claude")

        md = self._read_md()
        self.assertNotIn("mysupersecrettoken12345", md,
                         "Bearer token must be redacted in Markdown")
        self.assertIn("Bearer ***", md,
                      "redacted placeholder must appear in Markdown")

    # ── AC-03: sk- prefix pattern ─────────────────────────────────────────────

    def test_md_redacts_sk_prefix(self):
        """Markdown shard must not contain a raw sk-... key (20+ chars)."""
        secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
        log_entry(self.chat_id, "user", secret, model="gemini")

        md = self._read_md()
        self.assertNotIn("sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890", md,
                         "raw sk- key must be redacted in Markdown")
        self.assertIn("sk-***", md,
                      "redacted placeholder must appear in Markdown")

    # ── AC-04: JSONL path continues to redact ────────────────────────────────

    def test_jsonl_still_redacts(self):
        """JSONL record["content"] must also be redacted (regression guard)."""
        secret = "api_key=sk-abc1234567890abcdef1234"
        log_entry(self.chat_id, "user", secret, model="claude")

        records = self._read_jsonl()
        self.assertTrue(records, "at least one JSONL record must be written")
        for r in records:
            self.assertNotIn("sk-abc1234567890abcdef1234", r.get("content", ""),
                             "raw secret must not appear in JSONL content field")

    # ── AC-05: no false positive on ordinary text ─────────────────────────────

    def test_md_no_false_positive(self):
        """Normal text must not be altered by the redaction pass."""
        normal = "Hello world, this is a regular message with no secrets."
        log_entry(self.chat_id, "user", normal, model="claude")

        md = self._read_md()
        self.assertIn(normal, md,
                      "non-secret content must be preserved verbatim in Markdown")

    # ── AC-05b: known conservative false positive on code-assignment ─────────

    def test_md_conservative_false_positive_code_assignment(self):
        """Documents the known conservative false positive: ``api_key = func()``
        is redacted because the regex cannot distinguish a function name from a
        credential value.  The closing ``)`` is not consumed (it is excluded from
        the character class), so ``api_key = compute_func()`` becomes
        ``api_key = ***)``.  This test pins the behavior so a future regex
        change does not silently loosen it.
        """
        code_snippet = "api_key = compute_func()"
        log_entry(self.chat_id, "assistant", code_snippet, model="claude")

        md = self._read_md()
        # The function name is consumed by the credential-value character class.
        self.assertNotIn("compute_func", md)
        self.assertIn("api_key", md)
        self.assertIn("***", md)

    # ── Both paths use the same redacted string (consistency check) ──────────

    def test_md_and_jsonl_content_consistent(self):
        """The redacted value in JSONL and the Markdown body must agree."""
        secret = "Authorization: Bearer topsecrettoken99"
        log_entry(self.chat_id, "assistant", secret, model="claude")

        md = self._read_md()
        records = self._read_jsonl()
        self.assertTrue(records)
        jsonl_content = records[-1]["content"]

        # Both must not contain the raw token.
        self.assertNotIn("topsecrettoken99", md)
        self.assertNotIn("topsecrettoken99", jsonl_content)
        # Both must contain the placeholder.
        self.assertIn("Bearer ***", md)
        self.assertIn("Bearer ***", jsonl_content)
