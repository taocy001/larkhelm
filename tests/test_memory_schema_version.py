"""Unit tests for memory schema version handling (S47).

`_save_md` writes ``schema_version: "1"`` into the frontmatter on every
save so future loaders can detect old / forward-incompatible files.
``_check_schema_version`` reads it back and emits a one-shot warning when
a file is newer than this binary supports.

These tests pin the contract so the version-stamping path can't quietly
regress (which would block future migrations).
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import memory as mem


class SchemaVersionStampTests(unittest.TestCase):
    """Every save round-trips the schema_version into frontmatter."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = Path(self._tmpdir.name) / "project_test.md"

    def test_save_writes_schema_version(self):
        mem._save_md(self.path, "body content", max_chars=100)
        text = self.path.read_text(encoding="utf-8")
        self.assertIn(f'schema_version: "{mem.MEMORY_SCHEMA_VERSION}"', text)

    def test_load_returns_schema_version(self):
        mem._save_md(self.path, "body", max_chars=100)
        fm = mem._load_md_frontmatter(self.path)
        self.assertEqual(fm.get("schema_version"), mem.MEMORY_SCHEMA_VERSION)

    def test_extra_fm_pairs_can_override_schema_version(self):
        # Migration tooling can re-stamp an existing file at its
        # declared version. The caller-passed value must win.
        mem._save_md(
            self.path, "body", max_chars=100,
            extra_fm_pairs={"schema_version": "0"},
        )
        fm = mem._load_md_frontmatter(self.path)
        self.assertEqual(fm.get("schema_version"), "0")
        # Only one schema_version line in the file (no duplicate)
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(text.count("schema_version:"), 1)

    def test_extra_fm_string_can_carry_schema_version(self):
        # Legacy callers that pass extra_fm as a raw string can include
        # schema_version: ... and the auto-stamp must not duplicate.
        mem._save_md(
            self.path, "body", max_chars=100,
            extra_fm='schema_version: "2"\n',
        )
        text = self.path.read_text(encoding="utf-8")
        self.assertEqual(text.count("schema_version:"), 1)

    def test_extra_fm_substring_does_not_suppress_stamp(self):
        # Regression for round-2 review must-fix #1: a legitimate key
        # containing the substring "schema_version" (e.g. a hypothetical
        # last_schema_version_check audit field) must NOT suppress the
        # auto-stamp. The presence check uses an anchored regex now,
        # so the stamp survives.
        mem._save_md(
            self.path, "body", max_chars=100,
            extra_fm='last_schema_version_check: "2026-01-01T00:00:00"\n',
        )
        text = self.path.read_text(encoding="utf-8")
        # Both the real schema_version line and the audit field must be
        # present. The previous substring-based check incorrectly matched
        # "schema_version" inside "last_schema_version_check" and
        # suppressed the auto-stamp; the anchored regex no longer does.
        self.assertIn(f'schema_version: "{mem.MEMORY_SCHEMA_VERSION}"', text)
        self.assertIn('last_schema_version_check:', text)
        # And critically: only ONE real ``schema_version:`` line (the
        # auto-stamp), not zero (suppressed) and not duplicated.
        anchored = sum(1 for ln in text.splitlines()
                       if ln.lstrip().startswith("schema_version:"))
        self.assertEqual(anchored, 1,
                         f"expected exactly 1 schema_version line, got {anchored}:\n{text}")

    def test_extra_fm_pairs_substring_does_not_suppress_stamp(self):
        # Same regression but via extra_fm_pairs. The dict check uses
        # ``"schema_version" in pairs`` which is an exact-key match, so
        # a legitimate key like ``last_schema_version_check`` here also
        # must not suppress the auto-stamp.
        mem._save_md(
            self.path, "body", max_chars=100,
            extra_fm_pairs={"last_schema_version_check": "2026-01-01"},
        )
        text = self.path.read_text(encoding="utf-8")
        self.assertIn(f'schema_version: "{mem.MEMORY_SCHEMA_VERSION}"', text)
        self.assertIn('last_schema_version_check:', text)


class CheckSchemaVersionTests(unittest.TestCase):
    """The version check emits a one-shot warning for newer files."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.path = Path(self._tmpdir.name) / "session_x.md"
        # Reset the seen cache so test order doesn't matter
        mem._SCHEMA_WARN_SEEN.clear()

    def test_current_version_no_warning(self):
        with patch("larkhelm.memory._debug_log") as mock_log:
            mem._check_schema_version(self.path, {"schema_version": mem.MEMORY_SCHEMA_VERSION})
            self.assertFalse(
                any("schema_version" in str(c.args[0]) for c in mock_log.call_args_list),
                "should not warn on current version",
            )

    def test_missing_field_defaults_to_v1(self):
        v = mem._check_schema_version(self.path, {})
        self.assertEqual(v, "1")

    def test_older_version_no_warning(self):
        # Older files should NOT warn — they implicitly upgrade on next save.
        with patch("larkhelm.memory._debug_log") as mock_log:
            mem._check_schema_version(self.path, {"schema_version": "0"})
            self.assertFalse(
                any("newer" in str(c.args[0]) for c in mock_log.call_args_list),
                "should not warn on older version",
            )

    def test_newer_version_warns_once(self):
        future = str(int(mem.MEMORY_SCHEMA_VERSION) + 5)
        with patch("larkhelm.memory._debug_log") as mock_log:
            mem._check_schema_version(self.path, {"schema_version": future})
            mem._check_schema_version(self.path, {"schema_version": future})
            mem._check_schema_version(self.path, {"schema_version": future})
            warns = [c for c in mock_log.call_args_list
                     if "newer than" in str(c.args[0])]
            self.assertEqual(len(warns), 1,
                             f"expected exactly one warning, got {len(warns)}: {warns!r}")

    def test_non_numeric_version_silently_accepted(self):
        # Future-proofing for semver. Today's loader treats any non-int
        # version as "unknown, just read it" without crashing.
        with patch("larkhelm.memory._debug_log") as mock_log:
            v = mem._check_schema_version(self.path, {"schema_version": "1.2.3"})
            self.assertEqual(v, "1.2.3")
            self.assertFalse(
                any("schema_version" in str(c.args[0]) for c in mock_log.call_args_list),
                "non-numeric versions should be silent",
            )


if __name__ == "__main__":
    unittest.main()
