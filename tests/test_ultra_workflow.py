"""Claude Code Workflow 集成（/ultra + claude_workflow_enabled）测试。

Coverage:
  - _summarize_workflow_input   Workflow tool_use 卡片摘要
  - workflow_supported          claude CLI 版本探测与缓存
  - _ultra_precheck             /ultra 参数 / flag / 版本三道闸
  - build_args / build_env      Workflow(*) 放行与 CLAUDE_CODE_DISABLE_WORKFLOWS
"""
import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ── Initialize config ─────────────────────────────────────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_ultratest_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
    "skip_permissions": True,
    "default_cwd": _TMP,
}))

import larkhelm.config as _cfg_module
_cfg_module._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.runner_base import _summarize_workflow_input
from larkhelm import runner_claude
from larkhelm.runner_claude import ClaudeRunner, workflow_supported


_SCRIPT = """export const meta = {
  name: 'review-changes',
  description: 'Review changed files across dimensions',
  phases: [
    { title: 'Review', detail: 'fan out reviewers' },
    { title: 'Verify', detail: 'adversarial check' },
  ],
}
phase('Review')
const r = await agent('do it', {label: "ignore: 'me'"})
"""


class TestSummarizeWorkflowInput(unittest.TestCase):
    def test_inline_script_meta_extracted(self):
        out = _summarize_workflow_input({"script": _SCRIPT})
        self.assertIn("review-changes", out)
        self.assertIn("Review changed files", out)
        self.assertIn("2 phases", out)
        self.assertIn("Review → Verify", out)

    def test_script_path_fallback(self):
        out = _summarize_workflow_input({"scriptPath": "/tmp/x/find-bugs.js"})
        self.assertIn("find-bugs", out)

    def test_named_workflow(self):
        out = _summarize_workflow_input({"name": "my-audit"})
        self.assertIn("my-audit", out)

    def test_empty_input(self):
        self.assertEqual(_summarize_workflow_input({}), "(inline script)")

    def test_dispatched_from_summarize_tool_input(self):
        from larkhelm.runner_base import BaseProcessRunner
        out = BaseProcessRunner._summarize_tool_input("Workflow", {"script": _SCRIPT})
        self.assertIn("review-changes", out)

    def test_length_capped(self):
        long_meta = ("export const meta = {name: '" + "n" * 300 +
                     "', description: '" + "d" * 300 + "'}")
        self.assertLessEqual(len(_summarize_workflow_input({"script": long_meta})), 160)


class TestWorkflowSupported(unittest.TestCase):
    def setUp(self):
        runner_claude._workflow_probe_cache.clear()

    def tearDown(self):
        runner_claude._workflow_probe_cache.clear()

    def _probe_with_version_output(self, stdout: str):
        proc = MagicMock()
        proc.stdout = stdout
        with patch("subprocess.run", return_value=proc):
            return workflow_supported("claude-test")

    def test_new_version_supported(self):
        ok, ver = self._probe_with_version_output("2.1.173 (Claude Code)")
        self.assertTrue(ok)
        self.assertEqual(ver, "2.1.173")

    def test_old_version_unsupported(self):
        ok, ver = self._probe_with_version_output("2.1.100 (Claude Code)")
        self.assertFalse(ok)
        self.assertEqual(ver, "2.1.100")

    def test_boundary_version(self):
        ok, _ = self._probe_with_version_output("2.1.154")
        self.assertTrue(ok)

    def test_unparseable_output(self):
        ok, ver = self._probe_with_version_output("not a version")
        self.assertFalse(ok)
        self.assertEqual(ver, "")

    def test_probe_failure(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            ok, ver = workflow_supported("claude-missing")
        self.assertFalse(ok)
        self.assertEqual(ver, "")

    def test_result_cached(self):
        proc = MagicMock()
        proc.stdout = "2.1.173"
        with patch("subprocess.run", return_value=proc) as run_mock:
            workflow_supported("claude-test")
            workflow_supported("claude-test")
        self.assertEqual(run_mock.call_count, 1)


class TestUltraPrecheck(unittest.TestCase):
    def setUp(self):
        self._saved = _cfg_module.config.get("claude_workflow_enabled", False)

    def tearDown(self):
        _cfg_module.config["claude_workflow_enabled"] = self._saved

    def _run(self, text, flag, supported=(True, "2.1.173")):
        _cfg_module.config["claude_workflow_enabled"] = flag
        from larkhelm import commands
        with patch.object(commands, "send_card_reply") as card_mock, \
             patch.object(runner_claude, "workflow_supported", return_value=supported):
            result = commands._ultra_precheck("oc_test", text, "msg_1")
        return result, card_mock

    def test_empty_args_usage_card(self):
        result, card = self._run("/ultra", flag=True)
        self.assertIsNone(result)
        self.assertEqual(card.call_count, 1)
        self.assertIn("用法", card.call_args[0][2])

    def test_flag_disabled_card(self):
        result, card = self._run("/ultra audit the repo", flag=False)
        self.assertIsNone(result)
        self.assertIn("未启用", card.call_args[0][2])

    def test_cli_too_old_card(self):
        result, card = self._run("/ultra audit the repo", flag=True,
                                 supported=(False, "2.1.100"))
        self.assertIsNone(result)
        self.assertIn("过旧", card.call_args[0][2])

    def test_happy_path_rewrites_prompt(self):
        result, card = self._run("/ultra audit the repo", flag=True)
        self.assertEqual(result, "ultracode: audit the repo")
        card.assert_not_called()


class TestBuildArgsEnvGating(unittest.TestCase):
    def setUp(self):
        self._saved = _cfg_module.config.get("claude_workflow_enabled", False)
        self._saved_skip = _cfg_module.SKIP_PERMISSIONS

    def tearDown(self):
        _cfg_module.config["claude_workflow_enabled"] = self._saved
        _cfg_module.SKIP_PERMISSIONS = self._saved_skip

    def _settings_allow_list(self, flag: bool) -> list:
        _cfg_module.config["claude_workflow_enabled"] = flag
        _cfg_module.SKIP_PERMISSIONS = False
        r = ClaudeRunner("oc_test", "hi", None, _TMP)
        args = r.build_args()
        settings_file = args[args.index("--settings") + 1]
        try:
            return json.loads(Path(settings_file).read_text())["permissions"]["allow"]
        finally:
            for p in r._tmp_files:
                Path(p).unlink(missing_ok=True)

    def test_allow_list_includes_workflow_when_enabled(self):
        self.assertIn("Workflow(*)", self._settings_allow_list(True))

    def test_allow_list_excludes_workflow_when_disabled(self):
        self.assertNotIn("Workflow(*)", self._settings_allow_list(False))

    def test_env_disables_workflows_when_flag_off(self):
        _cfg_module.config["claude_workflow_enabled"] = False
        env = ClaudeRunner("oc_test", "hi", None, _TMP).build_env()
        self.assertEqual(env.get("CLAUDE_CODE_DISABLE_WORKFLOWS"), "1")

    def test_env_allows_workflows_when_flag_on(self):
        _cfg_module.config["claude_workflow_enabled"] = True
        env = ClaudeRunner("oc_test", "hi", None, _TMP).build_env()
        self.assertNotIn("CLAUDE_CODE_DISABLE_WORKFLOWS", env)


if __name__ == "__main__":
    unittest.main()
