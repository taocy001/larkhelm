"""
AC-03 — _bash_needs_approval() resolves symlinks via realpath to prevent path traversal

The original test relied on a hardcoded absolute path; it has been rewritten to use
tempfile for a portable, environment-independent version.
"""
import atexit
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_ac03_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.perm import _bash_needs_approval, _is_dangerous_cmd, get_safe_prefixes


class TestBashNeedsApprovalSymlink(unittest.TestCase):
    def setUp(self):
        # Create two subdirectories inside _TMP: safe (whitelisted cwd) and outside (sensitive dir)
        self.safe_dir    = Path(_TMP) / "ac03_safe"
        self.outside_dir = Path(_TMP) / "ac03_outside"
        self.safe_dir.mkdir(exist_ok=True)
        self.outside_dir.mkdir(exist_ok=True)

        # Place a "sensitive file" under outside
        (self.outside_dir / "secret.txt").write_text("sensitive data")

        # Create a symlink inside safe pointing to outside
        self.link = self.safe_dir / "link_to_outside"
        try:
            os.symlink(self.outside_dir, self.link)
        except FileExistsError:
            pass

    def test_symlink_access_outside_cwd_flagged(self):
        """Accessing a file outside cwd via symlink should trigger approval."""
        cmd = f"cat {self.link}/secret.txt"
        # Only allow safe_dir (not the _TMP root, since outside is also under _TMP)
        # safe_dir itself is the whitelist; outside_dir is not whitelisted
        # Reconstruct: place outside directory outside of _TMP
        with tempfile.TemporaryDirectory(prefix="larkhelm_outside_") as outside:
            outside_path = Path(outside)
            (outside_path / "secret.txt").write_text("sensitive data")
            link = self.safe_dir / "ext_link"
            try:
                os.symlink(outside_path, link)
            except FileExistsError:
                link.unlink()
                os.symlink(outside_path, link)

            cmd = f"cat {link}/secret.txt"
            result = _bash_needs_approval(cmd, str(self.safe_dir))
            self.assertTrue(result, f"Should detect symlink bypass of safe_dir: cmd={cmd}")

    def test_safe_path_no_approval_needed(self):
        """Accessing a regular file inside safe_dir should not require approval.
        Note: on macOS /var/folders resolves to /private/var/folders, so both the
        cwd and the file path must use their resolved forms.
        """
        safe_file = (self.safe_dir / "normal.txt").resolve()
        safe_file.write_text("hello")
        resolved_cwd = str(self.safe_dir.resolve())
        cmd = f"cat {safe_file}"
        result = _bash_needs_approval(cmd, resolved_cwd)
        self.assertFalse(result, "Files inside safe_dir should not trigger approval")

    def test_sibling_dir_outside_cwd_flagged(self):
        """Accessing a sibling directory of cwd (also under _TMP but not safe_dir) should trigger approval."""
        sibling = Path(_TMP) / "ac03_sibling"
        sibling.mkdir(exist_ok=True)
        sibling_file = sibling / "data.txt"
        sibling_file.write_text("data")
        cmd = f"cat {sibling_file.resolve()}"
        resolved_cwd = str(self.safe_dir.resolve())
        result = _bash_needs_approval(cmd, resolved_cwd)
        self.assertTrue(result, "Paths outside safe_dir should trigger approval")

    def test_dangerous_cmd_always_needs_approval(self):
        """Dangerous commands (e.g. rm -rf) should always require approval regardless of path."""
        cmd = f"rm -rf {self.safe_dir}"
        result = _bash_needs_approval(cmd, str(self.safe_dir))
        self.assertTrue(result, "Dangerous commands should always trigger approval")


class TestGetSafePrefixes(unittest.TestCase):
    def test_always_includes_tmp(self):
        prefixes = get_safe_prefixes("/some/cwd")
        has_tmp = any("/tmp" in p for p in prefixes)
        self.assertTrue(has_tmp, "Whitelist should include /tmp")

    def test_includes_cwd(self):
        cwd = "/home/user/project"
        prefixes = get_safe_prefixes(cwd)
        has_cwd = any(cwd in p for p in prefixes)
        self.assertTrue(has_cwd, "Whitelist should include cwd itself")

    def test_returns_list(self):
        result = get_safe_prefixes("/tmp")
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestIsDangerousCmd(unittest.TestCase):
    """Directly test _is_dangerous_cmd (without path checks)."""

    def _check(self, cmd):
        return _is_dangerous_cmd(cmd, "/tmp")

    def test_safe_command(self):
        self.assertFalse(self._check("ls -la /tmp"))

    def test_rm_detected(self):
        self.assertTrue(self._check("rm -rf /tmp/test"))

    def test_sudo_detected(self):
        self.assertTrue(self._check("sudo apt install vim"))

    def test_redirect_to_etc(self):
        self.assertTrue(self._check("echo x > /etc/hosts"))

    def test_redirect_to_tmp_safe(self):
        self.assertFalse(self._check("echo x > /tmp/file.txt"))


if __name__ == "__main__":
    unittest.main()
