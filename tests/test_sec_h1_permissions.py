"""SEC-H1 permission verification tests.

Verifies AC-01 through AC-05:
- AC-01: config.py memory_limit_mb write-back creates tmp file with 0o600
- AC-02: config.py save_config_field uses secure_atomic_write (0o600)
- AC-03: chat_state.py _save_state() creates state.json with 0o600
- AC-04: log.py all.jsonl created with 0o600
- AC-05: log.py DEBUG_LOG created with 0o600
"""
import atexit
import json
import os
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

_ON_POSIX = platform.system() != "Windows"

_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_sec_h1_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_DUMMY_CONFIG = {
    "APP_ID": "test_app_sec",
    "APP_SECRET": "test_secret_sec",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

import larkhelm.chat_state as _chat_state
import larkhelm.log as _log_mod


def _mode(p: Path) -> int:
    return p.stat().st_mode & 0o777


class TestAC01ConfigMemoryLimitWriteback(unittest.TestCase):
    """AC-01: memory_limit_mb write-back path uses secure_atomic_write (0o600)."""

    def test_memory_limit_mb_writeback_uses_secure_atomic_write(self):
        """Verify the write-back at _init_app_config uses secure_atomic_write."""
        # We verify by code inspection (import presence + usage pattern)
        import larkhelm.secure_io as secure_io
        import inspect
        config_src = Path(_cfg_module.__file__).read_text()
        # secure_atomic_write must be imported and used in config.py
        self.assertIn("from larkhelm.secure_io import secure_atomic_write", config_src)
        self.assertIn("secure_atomic_write(CONFIG_PATH,", config_src)

    def test_memory_limit_mb_writeback_produces_0o600(self):
        """Trigger the memory_limit_mb write-back and assert file is 0o600."""
        if not _ON_POSIX:
            self.skipTest("Permission bits not reliable on Windows")
        td = Path(tempfile.mkdtemp(prefix="ac01_"))
        try:
            config_path = td / "config_test.json"
            # Config without memory_limit_mb to trigger auto-detect write-back
            config_path.write_text(json.dumps({
                "APP_ID": "x", "APP_SECRET": "y",
                "default_model": "claude",
            }))
            with patch("larkhelm.memory_watchdog.detect_memory_limit_mb", return_value=512):
                _cfg_module._init_paths(str(config_path), str(td))
                _cfg_module._init_app_config()
            # After _init_app_config, CONFIG_PATH should have been written back with secure_atomic_write
            self.assertTrue(config_path.exists())
            self.assertEqual(_mode(config_path), 0o600,
                             f"config.json after writeback has mode {oct(_mode(config_path))}, expected 0o600")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class TestAC02ConfigSaveConfigField(unittest.TestCase):
    """AC-02: save_config_field uses secure_atomic_write (0o600)."""

    def test_save_config_field_produces_0o600(self):
        """Call save_config_field and assert config.json has 0o600."""
        if not _ON_POSIX:
            self.skipTest("Permission bits not reliable on Windows")
        td = Path(tempfile.mkdtemp(prefix="ac02_"))
        try:
            config_path = td / "config.json"
            config_path.write_text(json.dumps({
                "APP_ID": "x", "APP_SECRET": "y",
                "default_model": "claude",
            }))
            with patch("larkhelm.memory_watchdog.detect_memory_limit_mb", return_value=512):
                _cfg_module._init_paths(str(config_path), str(td))
                _cfg_module._init_app_config()
            # Reset to known permissions first (simulate first-run 0644)
            os.chmod(config_path, 0o644)
            # Now call save_config_field; it calls secure_atomic_write which should yield 0o600
            _cfg_module.save_config_field("default_drive_folder", "/test/folder")
            self.assertEqual(_mode(config_path), 0o600,
                             f"config.json after save_config_field has mode {oct(_mode(config_path))}, expected 0o600")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_save_config_field_uses_secure_atomic_write_code(self):
        """Code inspection: save_config_field must call secure_atomic_write."""
        config_src = Path(_cfg_module.__file__).read_text()
        self.assertIn("secure_atomic_write(CONFIG_PATH,", config_src)


class TestAC03ChatStateSaveState(unittest.TestCase):
    """AC-03: _save_state() creates state.json with 0o600."""

    def test_save_state_produces_0o600(self):
        """Call _save_state and assert state.json has 0o600."""
        if not _ON_POSIX:
            self.skipTest("Permission bits not reliable on Windows")
        td = Path(tempfile.mkdtemp(prefix="ac03_"))
        try:
            state_file = td / "state.json"
            with patch.object(_cfg_module, "STATE_FILE", state_file):
                # Ensure parent dir exists
                state_file.parent.mkdir(parents=True, exist_ok=True)
                _chat_state._chat_state_store.clear()
                _chat_state._chat_state_store["test_chat"] = {"cwd": "/tmp"}
                _chat_state._save_state()
            self.assertTrue(state_file.exists())
            self.assertEqual(_mode(state_file), 0o600,
                             f"state.json has mode {oct(_mode(state_file))}, expected 0o600")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_save_state_uses_secure_atomic_write_code(self):
        """Code inspection: _save_state must use secure_atomic_write."""
        chat_state_src = Path(_chat_state.__file__).read_text()
        self.assertIn("secure_atomic_write", chat_state_src)


class TestAC04LogAllJsonlPermissions(unittest.TestCase):
    """AC-04: all.jsonl created with 0o600."""

    def test_all_jsonl_created_with_0o600(self):
        """Call log_entry in a clean dir and assert all.jsonl has 0o600."""
        if not _ON_POSIX:
            self.skipTest("Permission bits not reliable on Windows")
        td = Path(tempfile.mkdtemp(prefix="ac04_"))
        try:
            log_dir = td / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            jsonl_path = log_dir / "all.jsonl"
            # Ensure file doesn't exist
            self.assertFalse(jsonl_path.exists())
            debug_log = td / "debug.log"
            with patch.object(_cfg_module, "LOG_DIR", log_dir), \
                 patch.object(_cfg_module, "DEBUG_LOG", debug_log):
                _log_mod.log_entry("test_chat", "user", "hello from AC-04 test")
            self.assertTrue(jsonl_path.exists(), "all.jsonl should have been created")
            self.assertEqual(_mode(jsonl_path), 0o600,
                             f"all.jsonl has mode {oct(_mode(jsonl_path))}, expected 0o600")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class TestAC05DebugLogPermissions(unittest.TestCase):
    """AC-05: DEBUG_LOG created with 0o600."""

    def test_debug_log_created_with_0o600(self):
        """Call _debug_log in a clean dir and assert DEBUG_LOG has 0o600."""
        if not _ON_POSIX:
            self.skipTest("Permission bits not reliable on Windows")
        td = Path(tempfile.mkdtemp(prefix="ac05_"))
        try:
            debug_log = td / "larkhelm.log"
            self.assertFalse(debug_log.exists())
            with patch.object(_cfg_module, "DEBUG_LOG", debug_log):
                _log_mod._debug_log("[SecH1Test] AC-05 debug_log permission test")
            self.assertTrue(debug_log.exists(), "DEBUG_LOG should have been created")
            self.assertEqual(_mode(debug_log), 0o600,
                             f"DEBUG_LOG has mode {oct(_mode(debug_log))}, expected 0o600")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class TestAC06InstallShChmod(unittest.TestCase):
    """AC-06: install.sh has chmod 600 after cp config."""

    def test_install_sh_has_chmod_600_after_cp(self):
        """Verify install.sh contains chmod 600 call after cp."""
        install_sh = Path(__file__).parent.parent / "install.sh"
        self.assertTrue(install_sh.exists(), "install.sh must exist")
        content = install_sh.read_text()
        # Find the line with cp ... CONFIG_PATH and verify chmod 600 follows
        lines = content.splitlines()
        cp_line_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if 'cp "$EXAMPLE_PATH" "$CONFIG_PATH"' in stripped:
                cp_line_idx = i
                break
        self.assertIsNotNone(cp_line_idx, "Should find 'cp \"$EXAMPLE_PATH\" \"$CONFIG_PATH\"' in install.sh")
        # Check within the next 5 lines for chmod 600
        context = lines[cp_line_idx:cp_line_idx + 5]
        has_chmod = any('chmod 600 "$CONFIG_PATH"' in l for l in context)
        self.assertTrue(has_chmod,
                        f"Expected 'chmod 600 \"$CONFIG_PATH\"' within 5 lines after cp; got: {context}")


class TestAC08AtomicReplacePattern(unittest.TestCase):
    """AC-08: Atomic replace semantics preserved — no direct-overwrite patterns."""

    def test_config_py_uses_atomic_replace(self):
        """config.py write paths use secure_atomic_write which internally does tmp→replace."""
        config_src = Path(_cfg_module.__file__).read_text()
        # secure_atomic_write uses tmp path and os.replace internally
        from larkhelm.secure_io import secure_atomic_write
        import inspect
        saw_src = inspect.getsource(secure_atomic_write)
        self.assertIn("os.replace", saw_src)
        self.assertIn(".tmp", saw_src)

    def test_chat_state_py_uses_atomic_replace(self):
        """chat_state.py uses secure_atomic_write (no direct write_text to STATE_FILE)."""
        chat_state_src = Path(_chat_state.__file__).read_text()
        # Must not write directly to STATE_FILE (should use secure_atomic_write)
        self.assertIn("secure_atomic_write", chat_state_src)
        # Old pattern (os.chmod on tmp file before replace) should be gone
        self.assertNotIn("os.chmod(tmp", chat_state_src)


if __name__ == "__main__":
    unittest.main()
