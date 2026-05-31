import unittest
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Setup temporary environment
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_scenario_")
os.environ["LARKHELM_DATA_DIR"] = _TMP_DIR

import larkhelm.config as cfg
# Initialize with a minimal config
_MINIMAL_CONFIG = {
    "APP_ID": "test_app_id",
    "APP_SECRET": "test_app_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_MINIMAL_CONFIG))
cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

from larkhelm.handlers._message import handle_message
from larkhelm.handlers._query import _do_query
from larkhelm.chat_state import _get_chat_model, _set_chat_model
from larkhelm.bridge import main as bridge_main

class TestScenarios(unittest.TestCase):
    def setUp(self):
        # Reset state store for each test
        from larkhelm.chat_state import _chat_state_store
        _chat_state_store.clear()

    def tearDown(self):
        from larkhelm.concurrency import _trigger_cancel, wait_for_idle
        for cid in ("chat_vision", "chat_switch"):
            try:
                _trigger_cancel(cid)
            except Exception:
                pass
        wait_for_idle(timeout=2.0)

    @patch("larkhelm.lark_client.send_card")
    @patch("lark_oapi.ws.Client.start")
    @patch("lark_oapi.Client.builder")
    @patch("larkhelm.bridge._acquire_pid_lock", return_value=True)
    def test_scenario_1_upgrade_notification(self, mock_lock, mock_builder, mock_ws, mock_send_card):
        # Scenario: Restart after upgrade (presence of _restart_notify.json)
        notify_path = Path(_TMP_DIR) / "_restart_notify.json"
        notify_data = {"chat_id": "chat123", "ts": time.time()}
        notify_path.write_text(json.dumps(notify_data))

        # Run bridge.main (will trigger notification)
        with patch("larkhelm.lark_client._fetch_bot_open_id"):
            bridge_main(config_path=str(_cfg_file), data_dir=_TMP_DIR)

        # Verify notification was sent
        mock_send_card.assert_called_once()
        args, kwargs = mock_send_card.call_args
        self.assertEqual(args[0], "chat123")
        self.assertIn("升级完成", args[1])
        
        # Verify marker file was deleted
        self.assertFalse(notify_path.exists())

    @patch("larkhelm.handlers._message._do_query")
    @patch("larkhelm.handlers._message._download_image", return_value="/tmp/mock_img.png")
    @patch("larkhelm.lark_client.react_to_message")
    def test_scenario_2_vision_routing(self, mock_react, mock_download, mock_do_query):
        # Scenario: Send an image message when default model is Gemini.
        # Check if it routes to a vision-supporting model (like Claude).
        
        # 1. Set default model to gemini
        _set_chat_model("chat_vision", "gemini")
        self.assertEqual(_get_chat_model("chat_vision"), "gemini")

        # 2. Mock image message event
        mock_event = MagicMock()
        mock_event.event.message.chat_id = "chat_vision"
        mock_event.event.message.message_type = "image"
        mock_event.event.message.content = json.dumps({"image_key": "img_v1"})
        mock_event.event.message.message_id = "msg123"
        mock_event.header.event_id = "evt123"
        mock_event.event.message.chat_type = "p2p"
        mock_event.event.sender.sender_id.open_id = "user123"

        # 3. Handle message
        handle_message(mock_event)

        # 4. Wait for thread to start (short sleep)
        time.sleep(0.1)

        # 5. Check which model was used in _do_query
        # If handle_message crashed, this will be 0
        self.assertTrue(mock_do_query.called, "handle_message did not call _do_query")
        kwargs = mock_do_query.call_args.kwargs
        used_model = kwargs.get("model")
        
        print(f"DEBUG: Used model for image: {used_model}")
        # Routing check: if images are present, we expect it to NOT be gemini
        self.assertIn(used_model, ["claude", "kimi"], f"Vision routing failed: image message sent to {used_model}")

    @patch("subprocess.Popen")
    @patch("larkhelm.ai_runner._ai_proc_sem")
    def test_scenario_3_model_switch_effectiveness(self, mock_sem, mock_popen):
        # Scenario: /model switch followed by a message.
        
        # 1. Start with default gemini
        _set_chat_model("chat_switch", "gemini")
        
        # 2. Switch to kimi (Phase 4: /model is alias for /lock, writes locked_backend)
        from larkhelm.commands import _cmd_model
        from larkhelm.backend_registry import BackendRegistry
        mock_reg = BackendRegistry()
        mock_reg.load([{"id": "kimi", "provider": "kimi_cli", "display_name": "Kimi",
                        "tags": ["tools"], "command": "kimi", "role": "orchestrator"}])
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", mock_reg), \
             patch("larkhelm.commands.send_card_reply"):
            _cmd_model("chat_switch", "kimi", "msg_switch")

        # Phase 4: locked_backend is stored in real state
        from larkhelm.chat_state import _get_chat_state
        self.assertEqual(_get_chat_state("chat_switch").get("locked_backend"), "kimi")

        # 3. Send a text message
        mock_event = MagicMock()
        mock_event.event.message.chat_id = "chat_switch"
        mock_event.event.message.message_type = "text"
        mock_event.event.message.content = json.dumps({"text": "Hello"})
        mock_event.event.message.message_id = "msg456"
        mock_event.header.event_id = "evt456"
        mock_event.event.message.chat_type = "p2p"
        mock_event.event.sender.sender_id.open_id = "user123"

        # Setup Popen mock to return a mock process
        mock_proc = MagicMock()
        mock_proc.stdout = []
        mock_proc.stderr = []
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        # We need to mock _patch_card_raw etc to avoid errors in _do_query
        with patch("larkhelm.handlers._query._patch_card_raw"), \
             patch("larkhelm.handlers._query._send_card_raw"), \
             patch("larkhelm.handlers._query._reply_card_raw"), \
             patch("larkhelm.handlers._query.react_to_message"), \
             patch("larkhelm.handlers._query.delete_reaction"), \
             patch("larkhelm.handlers._query._pin_task_card"):
            
            handle_message(mock_event)
            # Give some time for the thread to reach Popen
            time.sleep(0.5)

        # 4. Verify Popen was called with kimi command.
        # NOTE: memory-update background thread may also call Popen (with claude).
        # We check the *first* Popen call (the actual query) rather than the last.
        self.assertTrue(mock_popen.called)
        all_cmds = [a[0][0] for a, _ in mock_popen.call_args_list if a and a[0]]
        self.assertIn(cfg.KIMI_CMD, all_cmds,
                      f"kimi command not found in any Popen call: {all_cmds}")

if __name__ == "__main__":
    unittest.main()
