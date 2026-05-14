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
