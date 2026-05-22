"""
larkhelm · secure_io unit tests

Coverage:
  - secure_open: creates new file with 0o600, appends correctly, existing files not chmod'd
  - secure_atomic_write: creates tmp with 0o600, atomically replaces target, content correct
  - fallback path: os.open raises TypeError → fchmod fallback, no exception raised
  - fchmod-unavailable fallback: no exception raised even if fchmod missing

Permission assertions run on macOS / Linux only (st_mode reliable); on Windows
the assertion is skipped and only content / no-raise behaviour is checked.
"""
import os
import platform
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure the package root is importable when running pytest directly
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from larkhelm.secure_io import (
    SENSITIVE_FILE_MODE,
    secure_open,
    secure_atomic_write,
)

_ON_POSIX = platform.system() != "Windows"


def _mode(p: Path) -> int:
    """Return the permission bits of *p* (e.g. 0o600)."""
    return p.stat().st_mode & 0o777


class TestSecureOpen(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="test_secure_io_")
        self.tmp_dir = Path(self._td)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def test_creates_new_file_with_0o600(self):
        p = self.tmp_dir / "new.log"
        self.assertFalse(p.exists())
        with secure_open(p, "a", "utf-8") as f:
            f.write("hello\n")
        self.assertTrue(p.exists())
        if _ON_POSIX:
            self.assertEqual(_mode(p), 0o600, f"expected 0o600 got {oct(_mode(p))}")

    def test_content_written_correctly(self):
        p = self.tmp_dir / "content.log"
        with secure_open(p, "a", "utf-8") as f:
            f.write("line1\n")
        with secure_open(p, "a", "utf-8") as f:
            f.write("line2\n")
        self.assertEqual(p.read_text(), "line1\nline2\n")

    def test_existing_file_permissions_not_changed(self):
        p = self.tmp_dir / "existing.log"
        p.write_text("seed\n")
        if _ON_POSIX:
            os.chmod(p, 0o644)
        with secure_open(p, "a", "utf-8") as f:
            f.write("appended\n")
        # secure_open must NOT tighten an existing 0o644 file
        if _ON_POSIX:
            self.assertEqual(_mode(p), 0o644)
        self.assertIn("appended", p.read_text())

    def test_unicode_content(self):
        p = self.tmp_dir / "unicode.log"
        text = "你好世界\n"
        with secure_open(p, "a", "utf-8") as f:
            f.write(text)
        self.assertEqual(p.read_text(encoding="utf-8"), text)

    def test_fallback_on_type_error(self):
        """Simulate os.open raising TypeError (Windows CPython); must not raise."""
        p = self.tmp_dir / "fallback.log"
        _real_open = os.open

        def _mock_open(path, flags, *args, **kwargs):
            if args or kwargs:
                raise TypeError("mode not supported")
            return _real_open(path, flags)

        with patch("larkhelm.secure_io.os.open", side_effect=_mock_open):
            with secure_open(p, "a", "utf-8") as f:
                f.write("fallback\n")
        self.assertEqual(p.read_text(), "fallback\n")

    def test_fallback_fchmod_unavailable(self):
        """Simulate os.open TypeError + os.fchmod missing; must not raise."""
        p = self.tmp_dir / "nofchmod.log"
        _real_open = os.open

        def _mock_open(path, flags, *args, **kwargs):
            if args or kwargs:
                raise TypeError("mode not supported")
            return _real_open(path, flags)

        with patch("larkhelm.secure_io.os.open", side_effect=_mock_open), \
             patch("larkhelm.secure_io.os.fchmod", side_effect=AttributeError("no fchmod")):
            with secure_open(p, "a", "utf-8") as f:
                f.write("no fchmod\n")
        self.assertEqual(p.read_text(), "no fchmod\n")


class TestSecureAtomicWrite(unittest.TestCase):

    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="test_secure_io_aw_")
        self.tmp_dir = Path(self._td)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def test_creates_target_with_0o600(self):
        p = self.tmp_dir / "config.json"
        secure_atomic_write(p, '{"key": "value"}')
        self.assertTrue(p.exists())
        if _ON_POSIX:
            self.assertEqual(_mode(p), 0o600, f"expected 0o600 got {oct(_mode(p))}")

    def test_content_written_correctly_str(self):
        p = self.tmp_dir / "state.json"
        secure_atomic_write(p, '{"chat_id": "abc"}')
        self.assertEqual(p.read_text(encoding="utf-8"), '{"chat_id": "abc"}')

    def test_content_written_correctly_bytes(self):
        p = self.tmp_dir / "bytes.bin"
        data = b"\x00\x01\x02"
        secure_atomic_write(p, data)
        self.assertEqual(p.read_bytes(), data)

    def test_atomic_no_tmp_left_on_success(self):
        p = self.tmp_dir / "config.json"
        secure_atomic_write(p, "{}")
        tmp = Path(str(p) + ".tmp")
        self.assertFalse(tmp.exists(), "tmp file should be gone after successful write")

    def test_overwrites_existing_content(self):
        p = self.tmp_dir / "config.json"
        p.write_text('{"old": true}')
        secure_atomic_write(p, '{"new": true}')
        self.assertEqual(p.read_text(encoding="utf-8"), '{"new": true}')

    def test_unicode_content(self):
        p = self.tmp_dir / "unicode.json"
        content = '{"msg": "你好"}'
        secure_atomic_write(p, content)
        self.assertEqual(p.read_text(encoding="utf-8"), content)

    def test_fallback_on_type_error(self):
        """Simulate os.open TypeError; must not raise and content must be correct."""
        p = self.tmp_dir / "fallback.json"
        _real_open = os.open

        def _mock_open(path, flags, *args, **kwargs):
            if args or kwargs:
                raise TypeError("mode not supported")
            return _real_open(path, flags)

        with patch("larkhelm.secure_io.os.open", side_effect=_mock_open):
            secure_atomic_write(p, '{"fallback": true}')
        self.assertEqual(p.read_text(encoding="utf-8"), '{"fallback": true}')

    def test_fallback_fchmod_unavailable(self):
        """Simulate os.open TypeError + os.fchmod missing; must not raise."""
        p = self.tmp_dir / "nofchmod.json"
        _real_open = os.open

        def _mock_open(path, flags, *args, **kwargs):
            if args or kwargs:
                raise TypeError("mode not supported")
            return _real_open(path, flags)

        with patch("larkhelm.secure_io.os.open", side_effect=_mock_open), \
             patch("larkhelm.secure_io.os.fchmod", side_effect=AttributeError("no fchmod")):
            secure_atomic_write(p, '{"x": 1}')
        self.assertEqual(p.read_text(encoding="utf-8"), '{"x": 1}')


if __name__ == "__main__":
    unittest.main()
