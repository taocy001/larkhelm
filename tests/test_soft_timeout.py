import unittest
import json
import os
import tempfile
import time
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

# Setup temporary environment
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_soft_timeout_")
os.environ["LARKHELM_DATA_DIR"] = _TMP_DIR

import larkhelm.config as cfg
_MINIMAL_CONFIG = {
    "APP_ID": "test_app_id",
    "APP_SECRET": "test_app_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_MINIMAL_CONFIG))
cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

# Import after config init
from larkhelm.handlers._query import _do_query

class TestSoftTimeout(unittest.TestCase):
    def setUp(self):
        from larkhelm.chat_state import _chat_state_store
        _chat_state_store.clear()
        # Ensure we don't have residual locks
        from larkhelm.concurrency import _chat_locks
        _chat_locks.clear()

    @patch("larkhelm.backend_cli.run_claude")
    @patch("larkhelm.handlers._query._patch_card_raw")
    @patch("larkhelm.handlers._query._send_card_raw")
    @patch("larkhelm.handlers._query._reply_card_raw")
    @patch("larkhelm.handlers._query._pin_task_card")
    @patch("larkhelm.handlers._query.react_to_message")
    @patch("larkhelm.handlers._query.delete_reaction")
    def test_cancel_button_hidden_after_soft_timeout(self, mock_del, mock_react, mock_pin, mock_reply, mock_send, mock_patch, mock_query):
        # We want to verify that after on_soft_timeout is called, 
        # the card updates (via heartbeat or direct push) have no cancel button.
        
        captured_cards = []
        def capture_card(card_json):
            if isinstance(card_json, str):
                captured_cards.append(json.loads(card_json))
            else:
                # If it's already a dict or something else, handle it
                captured_cards.append(card_json)

        def patch_side_effect(mid, card_json):
            capture_card(card_json)
            return True
        mock_patch.side_effect = patch_side_effect
        
        def send_side_effect(chat_id, card_json, **kwargs):
            capture_card(card_json)
            return "mid_123"
        mock_send.side_effect = send_side_effect
        
        def reply_side_effect(msg_id, card_json, **kwargs):
            capture_card(card_json)
            return "mid_123"
        mock_reply.side_effect = reply_side_effect

        # Mock run_claude (new routing path) to trigger soft timeout
        def mock_query_impl(spec, chat_id, message, sid, cwd,
                            cancel_ev=None, on_text=None, on_tool=None,
                            on_tool_result=None, on_soft_timeout=None,
                            on_start=None, images=None, session_namespace=None,
                            allow_retry=False, system_prompt=None):
            # 1. Simulate some progress
            if on_text:
                on_text("Starting...")
            time.sleep(0.2)

            # 2. Trigger soft timeout
            if on_soft_timeout:
                on_soft_timeout()

            # 3. Simulate more progress after soft timeout
            if on_text:
                on_text("Continuing in background...")
            time.sleep(0.5)  # Wait for heartbeat to pick it up or direct push
            return "Final output"

        mock_query.side_effect = mock_query_impl

        # Run _do_query in a separate thread because it's blocking
        t = threading.Thread(target=_do_query, args=("chat1", "Hello", "claude", "msg1"))
        t.start()
        t.join(timeout=5.0)

        # Analysis of captured cards
        # Some cards should have buttons, and at least one card after soft timeout should NOT have buttons.
        
        found_card_with_cancel = False
        found_card_without_cancel_after_soft_timeout = False
        
        def _has_cancel_button(card: dict) -> bool:
            """Walk JSON 2.0 ``body.elements`` (post-migration) for a button
            whose callback cmd starts with ``cancel:``. Buttons may be either
            bare ``tag:"button"`` elements (1-button rows) or nested inside
            a ``column_set`` → ``column`` (multi-button rows)."""
            buttons: list = []
            elements = (card.get("body") or {}).get("elements", [])
            for el in elements:
                if el.get("tag") == "button":
                    buttons.append(el)
                elif el.get("tag") == "column_set":
                    for col in el.get("columns", []):
                        for child in col.get("elements", []):
                            if child.get("tag") == "button":
                                buttons.append(child)
            for btn in buttons:
                for behavior in btn.get("behaviors", []):
                    cmd = (behavior.get("value") or {}).get("cmd", "")
                    if "cancel:" in cmd:
                        return True
            return False

        in_background_seen = False
        for card in captured_cards:
            # Check title to see if it's "后台" (indicates _in_background was True)
            title = card.get("header", {}).get("title", {}).get("content", "")
            has_cancel = _has_cancel_button(card)

            if "后台" in title:
                in_background_seen = True
                if not has_cancel:
                    found_card_without_cancel_after_soft_timeout = True
            else:
                if has_cancel:
                    found_card_with_cancel = True

        self.assertTrue(in_background_seen, "Should have reached background state")
        self.assertTrue(found_card_with_cancel, "Should have shown cancel button initially")
        self.assertTrue(found_card_without_cancel_after_soft_timeout, "Should have hidden cancel button after soft timeout")

if __name__ == "__main__":
    unittest.main()
