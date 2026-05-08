
import unittest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup dummy environment
_TMP_DIR = tempfile.mkdtemp()
os.environ["LARKHELM_DATA_DIR"] = _TMP_DIR

import larkhelm.config as cfg
_DUMMY_CONFIG = {
    "APP_ID": "test",
    "APP_SECRET": "test",
    "claude_command": "false", # CLI exists but is 'false'
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))
cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

from larkhelm.backend_registry import BACKEND_REGISTRY
from larkhelm.crew._runner import _run_agent_step

class TestCrewApiBackendBug(unittest.TestCase):
    def test_crew_agent_step_with_api_orchestrator(self):
        # 1. Setup API orchestrator
        api_spec = {
            "id": "sonnet-api",
            "provider": "anthropic_api",
            "role": "orchestrator",
            "enabled": True,
            "healthy": True,
            "api_key": "sk-test"
        }
        BACKEND_REGISTRY.load([api_spec])
        
        # 2. Mock state and spec
        mock_state = MagicMock()
        agent_spec = MagicMock()
        agent_spec.model = "claude-3-sonnet" # Not raw_ or hermes_
        
        # 3. Patch run_claude to see what it's called with
        # and also patch backend_api.run_anthropic to see if IT should have been called
        with patch("larkhelm.backend_cli.run_claude") as mock_run_claude, \
             patch("larkhelm.backend_api.run_anthropic") as mock_run_anthropic:
            
            mock_run_claude.return_value = "CLI output"
            mock_run_anthropic.return_value = ("API output", [])
            
            # This should ideally call run_anthropic since orchestrator is API
            # but currently it hardcodes run_claude
            try:
                _run_agent_step(mock_state, "agent1", agent_spec, None, None)
            except Exception as e:
                print(f"Caught expected or unexpected error: {e}")
            
            # Verify bug: it called run_claude instead of run_anthropic
            self.assertTrue(mock_run_claude.called, "Bug: run_claude was called even though orchestrator is API")
            self.assertFalse(mock_run_anthropic.called, "Bug: run_anthropic was NOT called for API orchestrator")

if __name__ == "__main__":
    unittest.main()
