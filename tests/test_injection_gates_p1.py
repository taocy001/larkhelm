"""Tests for P1 on-demand injection gates (P1c).

P1c — project_guide_enabled: guide content injected for API backends, skipped for claude_cli.
"""
from __future__ import annotations

import atexit
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# ── Bootstrap shared config (same pattern as test_query_backend_skip_recent) ──
_TMP = tempfile.mkdtemp(prefix="larkhelm_igates_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg  # noqa: E402
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)


# ══════════════════════════════════════════════════════════════════════════════
#  P1c — project_guide_enabled
# ══════════════════════════════════════════════════════════════════════════════

_GUIDE_CONTENT = "# Project Guide\n\nUse snake_case for variables."


class ProjectGuideInjectionTests(unittest.TestCase):

    def setUp(self):
        self._guide_file = Path(_TMP) / "project_guide.md"
        self._guide_file.write_text(_GUIDE_CONTENT, encoding="utf-8")
        _cfg.config["project_guide_enabled"] = True
        _cfg.config["project_guide_path"] = str(self._guide_file)

    def tearDown(self):
        _cfg.config["project_guide_enabled"] = False
        _cfg.config["project_guide_path"] = ""

    def _run_guide_gate(self, provider: str, existing_memory: str = "") -> str:
        """Replicate the project guide gate logic from _do_query."""
        memory_ctx = existing_memory
        if bool(_cfg.config.get("project_guide_enabled")) and _cfg.config.get("project_guide_path"):
            _is_cli_claude = provider == "claude_cli"
            if not _is_cli_claude:
                try:
                    _guide_path = Path(_cfg.config["project_guide_path"]).expanduser()
                    _guide_content = _guide_path.read_text(encoding="utf-8")
                    if len(_guide_content) > 4000:
                        _guide_content = _guide_content[:4000]
                    memory_ctx = (
                        f"[Project Guide]\n{_guide_content}\n[/Project Guide]\n\n"
                        + memory_ctx
                    )
                except Exception:
                    pass
        return memory_ctx

    def test_api_backend_injects_guide(self):
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertIn("[Project Guide]", result)
        self.assertIn(_GUIDE_CONTENT, result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_google_api_backend_injects_guide(self):
        result = self._run_guide_gate("google_api")
        self.assertIn("[Project Guide]", result)

    def test_claude_cli_skips_guide(self):
        result = self._run_guide_gate("claude_cli", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertNotIn("[Project Guide]", result)
        self.assertIn("[SESSION MEMORY]", result)

    def test_guide_prepended_before_memory(self):
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        guide_pos = result.index("[Project Guide]")
        session_pos = result.index("[SESSION MEMORY]")
        self.assertLess(guide_pos, session_pos)

    def test_gate_disabled_no_injection(self):
        _cfg.config["project_guide_enabled"] = False
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertNotIn("[Project Guide]", result)

    def test_no_path_no_injection(self):
        _cfg.config["project_guide_path"] = ""
        result = self._run_guide_gate("anthropic_api")
        self.assertNotIn("[Project Guide]", result)

    def test_guide_truncated_at_4000_chars(self):
        long_guide = "x" * 5000
        long_file = Path(_TMP) / "long_guide.md"
        long_file.write_text(long_guide, encoding="utf-8")
        _cfg.config["project_guide_path"] = str(long_file)

        result = self._run_guide_gate("anthropic_api")
        guide_match = re.search(
            r"\[Project Guide\]\n(.*?)\n\[/Project Guide\]", result, re.DOTALL
        )
        self.assertIsNotNone(guide_match)
        self.assertLessEqual(len(guide_match.group(1)), 4000)

    def test_missing_guide_file_no_crash(self):
        _cfg.config["project_guide_path"] = "/nonexistent/path/guide.md"
        result = self._run_guide_gate("anthropic_api", "[SESSION MEMORY]\nfoo[/SESSION MEMORY]")
        self.assertNotIn("[Project Guide]", result)
        self.assertIn("[SESSION MEMORY]", result)


if __name__ == "__main__":
    unittest.main()
