"""AC-09: intent feedback JSONL format + LRU pending registry."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm.agent_hub.intent_feedback import (
    record_feedback, register_pending, resolve_pending,
)
from larkhelm.agent_hub.intent_types import AgentContext, IntentResult


class TestRecordFeedback(unittest.TestCase):

    def test_writes_jsonl_with_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intent_feedback.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path):
                pred = IntentResult(
                    agent_type="dev", confidence=0.92, layer="L2",
                    raw_text="帮我写个登录模块",
                )
                fid = record_feedback(pred, corrected="chat", chat_id="oc_xxx",
                                      text="帮我写个登录模块")
            self.assertTrue(path.exists())
            mode = os.stat(path).st_mode & 0o777
            self.assertEqual(mode, 0o600)
            line = path.read_text().strip()
            rec = json.loads(line)
            self.assertEqual(rec["chat_id"], "oc_xxx")
            self.assertEqual(rec["predicted_intent"], "dev")
            self.assertEqual(rec["corrected_intent"], "chat")
            self.assertEqual(rec["confidence"], 0.92)
            self.assertEqual(rec["layer"], "L2")
            self.assertEqual(rec["feedback_id"], fid)
            self.assertIn("ts", rec)

    def test_record_feedback_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intent_feedback.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path):
                for i in range(3):
                    record_feedback(
                        IntentResult(agent_type="dev", raw_text=f"t{i}"),
                        corrected="chat", chat_id="c", text=f"t{i}",
                    )
            self.assertEqual(len(path.read_text().splitlines()), 3)


class TestPendingLRU(unittest.TestCase):

    def _ctx(self) -> AgentContext:
        return AgentContext(
            chat_id="c", user_msg_id="m", text="hi", images=None,
            parent_id=None, cancel_ev=threading.Event(), cwd="/tmp",
        )

    def test_register_and_resolve(self):
        intent = IntentResult(agent_type="dev", raw_text="hi")
        ctx = self._ctx()
        register_pending("fb_a", intent, ctx, text="hi")
        entry = resolve_pending("fb_a")
        self.assertIsNotNone(entry)
        self.assertIs(entry.intent, intent)
        self.assertIs(entry.ctx, ctx)
        self.assertEqual(entry.text, "hi")
        # Resolved entry is removed (resolve == pop).
        self.assertIsNone(resolve_pending("fb_a"))

    def test_lru_capacity(self):
        from larkhelm.agent_hub import intent_feedback as fb
        # Register 260 entries; cap is 256, oldest 4 should evict.
        for i in range(260):
            register_pending(
                f"fb_{i:03}",
                IntentResult(agent_type="chat", raw_text=str(i)),
                self._ctx(), text=str(i),
            )
        self.assertIsNone(resolve_pending("fb_000"))
        self.assertIsNone(resolve_pending("fb_003"))
        self.assertIsNotNone(resolve_pending("fb_259"))
        # Cleanup remaining entries to keep tests independent.
        for i in range(260):
            resolve_pending(f"fb_{i:03}")


if __name__ == "__main__":
    unittest.main()
