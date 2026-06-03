"""Tests for _preflight_env_check (AC-01, AC-02)."""
import unittest
from unittest.mock import patch, MagicMock
import subprocess


class TestPreflightArch(unittest.TestCase):
    def _make_spec(self, require_arch="", require_docker_image=""):
        from larkhelm.crew_types import AgentSpec
        return AgentSpec(
            id="test", role="test", model="", system="", prompt="",
            depends_on=[], timeout=60,
            require_arch=require_arch,
            require_docker_image=require_docker_image,
        )

    def test_arch_pass_linux_amd64(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec(require_arch="linux/amd64")
        with patch("sys.platform", "linux"), patch("platform.machine", return_value="x86_64"):
            ok, reason = _preflight_env_check(spec, {})
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_arch_mismatch_darwin_arm64(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec(require_arch="linux/amd64")
        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"):
            ok, reason = _preflight_env_check(spec, {})
        self.assertFalse(ok)
        self.assertIn("linux", reason)

    def test_no_arch_requirement_passes(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec()
        ok, reason = _preflight_env_check(spec, {})
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_arch_aarch64_matches_arm64(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec(require_arch="linux/arm64")
        with patch("sys.platform", "linux"), patch("platform.machine", return_value="aarch64"):
            ok, reason = _preflight_env_check(spec, {})
        self.assertTrue(ok)


class TestPreflightDocker(unittest.TestCase):
    def _make_spec(self, require_docker_image=""):
        from larkhelm.crew_types import AgentSpec
        return AgentSpec(
            id="test", role="test", model="", system="", prompt="",
            depends_on=[], timeout=60,
            require_docker_image=require_docker_image,
        )

    def test_docker_present(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec(require_docker_image="python:3.11-slim")
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        with patch("subprocess.run", return_value=mock_proc) as mock_run:
            ok, reason = _preflight_env_check(spec, {})
        self.assertTrue(ok)
        mock_run.assert_called_once_with(
            ["docker", "image", "inspect", "python:3.11-slim"],
            shell=False, timeout=5, capture_output=True,
        )

    def test_docker_missing(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec(require_docker_image="nonexistent:latest")
        mock_proc = MagicMock()
        mock_proc.returncode = 1
        with patch("subprocess.run", return_value=mock_proc):
            ok, reason = _preflight_env_check(spec, {})
        self.assertFalse(ok)
        self.assertIn("nonexistent:latest", reason)

    def test_docker_inspect_exception(self):
        from larkhelm.crew._runner import _preflight_env_check
        spec = self._make_spec(require_docker_image="python:3.11-slim")
        with patch("subprocess.run", side_effect=FileNotFoundError("docker not found")):
            ok, reason = _preflight_env_check(spec, {})
        self.assertFalse(ok)
        self.assertIn("docker inspect failed", reason)


class TestPreflightAgentWrapper(unittest.TestCase):
    def test_wrapper_sets_skipped_on_arch_fail(self):
        from larkhelm.crew._runner import _run_agent_wrapper
        from larkhelm.crew_types import (
            AgentSpec, AgentState, AgentStatus, CrewState, CrewPlan,
        )
        import threading

        spec = AgentSpec(
            id="eng", role="工程师", model="", system="", prompt="",
            depends_on=[], timeout=60, require_arch="linux/amd64",
        )
        plan = CrewPlan(title="test", agents=[spec])
        agent_state = AgentState(spec=spec)
        state = CrewState(
            crew_id="c1", chat_id="ch1", plan=plan,
            agents={"eng": agent_state},
        )

        with patch("sys.platform", "darwin"), patch("platform.machine", return_value="arm64"), \
             patch("larkhelm.metrics.inc_crew_preflight"):
            _run_agent_wrapper(state, "eng")

        self.assertEqual(state.agents["eng"].status, AgentStatus.SKIPPED)
        self.assertIn("linux", state.agents["eng"].skip_reason)


class TestConfigSchema(unittest.TestCase):
    def test_empty_app_id_is_error(self):
        from larkhelm.config import validate_config
        errors = validate_config({"APP_ID": "", "APP_SECRET": "valid_secret"})
        self.assertTrue(any("APP_ID" in e for e in errors))

    def test_negative_response_timeout(self):
        from larkhelm.config import validate_config
        errors = validate_config({
            "APP_ID": "x", "APP_SECRET": "y",
            "response_timeout": -1, "hard_timeout": 21600,
        })
        self.assertTrue(any("response_timeout" in e for e in errors))

    def test_hard_timeout_not_greater_than_response_timeout(self):
        from larkhelm.config import validate_config
        errors = validate_config({
            "APP_ID": "x", "APP_SECRET": "y",
            "response_timeout": 300, "hard_timeout": 300,
        })
        self.assertTrue(any("hard_timeout" in e for e in errors))

    def test_max_card_len_out_of_range(self):
        from larkhelm.config import validate_config
        errors = validate_config({
            "APP_ID": "x", "APP_SECRET": "y", "max_card_len": 50000,
        })
        self.assertTrue(any("max_card_len" in e for e in errors))

    def test_embedding_traffic_above_one(self):
        from larkhelm.config import validate_config
        errors = validate_config({
            "APP_ID": "x", "APP_SECRET": "y", "embedding_traffic": 1.5,
        })
        self.assertTrue(any("embedding_traffic" in e for e in errors))

    def test_valid_config_returns_empty(self):
        from larkhelm.config import validate_config
        errors = validate_config({
            "APP_ID": "valid_app_id",
            "APP_SECRET": "valid_secret",
            "response_timeout": 300,
            "hard_timeout": 21600,
            "max_card_len": 3000,
            "embedding_traffic": 0.1,
        })
        self.assertEqual(errors, [])

    def test_missing_app_secret_is_error(self):
        from larkhelm.config import validate_config
        errors = validate_config({"APP_ID": "valid_id", "APP_SECRET": ""})
        self.assertTrue(any("APP_SECRET" in e for e in errors))


if __name__ == "__main__":
    unittest.main()
