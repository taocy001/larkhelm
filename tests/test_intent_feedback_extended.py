"""Phase-D follow-up: extended-signal collection for intent_feedback.jsonl.

Covers:

  1. ``record_feedback`` writes ``signal_type="force_chat"`` by default
     and ``text`` is NOT truncated (byte-compat with pre-Phase-D rows).
  2. ``record_signal`` writes the requested ``signal_type`` and respects
     ``intent_feedback_signal_text_max`` for non-force-chat rows.
  3. Master switch ``intent_feedback_extended_signals=false`` turns
     ``record_signal`` into a no-op (returns None, no JSONL line).
  4. ``track_dispatch`` / ``consume_dispatch`` round-trip is single-shot
     (consume pops) and respects ``max_age_sec``.
  5. ``_maybe_record_l1_gray_zone`` fires only inside the configured
     band and skips L2 / fallback layers.
  6. ``_maybe_record_l2_dispatched`` skips pure chat L2 results (the
     fallback class would drown the signal).
  7. ``AgentDispatcher._fallback_to_chat`` emits ``dispatch_failed`` and
     pops the dispatch-history entry so a subsequent /cancel can't
     double-bill the same prediction.
  8. Schema-compat: extended fields don't break the pre-Phase-D trainer
     loader (``corrected_intent="" `` rows are skipped silently because
     it requires a non-empty label).
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm.agent_hub.intent_feedback import (
    clear_dispatch_history_for_tests,
    consume_dispatch,
    peek_dispatch,
    record_feedback,
    record_signal,
    track_dispatch,
)
from larkhelm.agent_hub.intent_router import (
    _maybe_record_l1_gray_zone,
    _maybe_record_l2_dispatched,
)
from larkhelm.agent_hub.intent_types import (
    AgentContext, AgentResult, IntentResult,
)


def _intent(agent_type="dev", confidence=0.9, layer="L1", text="x") -> IntentResult:
    return IntentResult(
        agent_type=agent_type, confidence=confidence, layer=layer, raw_text=text,
    )


def _ctx(chat_id="oc_test", text="hello") -> AgentContext:
    return AgentContext(
        chat_id=chat_id, user_msg_id="m1", text=text, images=None,
        parent_id=None, cancel_ev=threading.Event(), cwd="/tmp",
    )


class _CfgPatch:
    """Patch ``larkhelm.config`` attributes used by intent_feedback.

    Easier to write than chasing through ``_init_runtime``: we just
    monkey-patch the module-level constants the helpers read.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.saved: dict = {}

    def __enter__(self):
        import larkhelm.config as _cfg
        for k, v in self.kwargs.items():
            self.saved[k] = getattr(_cfg, k, None)
            setattr(_cfg, k, v)
        return self

    def __exit__(self, *a):
        import larkhelm.config as _cfg
        for k, v in self.saved.items():
            setattr(_cfg, k, v)


class TestRecordFeedbackSchema(unittest.TestCase):

    def test_force_chat_writes_signal_type_force_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path):
                fid = record_feedback(
                    _intent(agent_type="dev"), corrected="chat",
                    chat_id="oc_x", text="a" * 5000,
                )
            self.assertTrue(path.exists())
            rec = json.loads(path.read_text().strip())
            self.assertEqual(rec["signal_type"], "force_chat")
            self.assertEqual(rec["feedback_id"], fid)
            # force_chat preserves full text (byte-compat with old rows).
            self.assertEqual(len(rec["text"]), 5000)

    def test_record_signal_truncates_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True,
                           INTENT_FEEDBACK_SIGNAL_TEXT_MAX=50):
                fid = record_signal(
                    "l2_dispatched", _intent(layer="L2"),
                    chat_id="oc_x", text="b" * 5000,
                )
            self.assertIsNotNone(fid)
            rec = json.loads(path.read_text().strip())
            self.assertEqual(rec["signal_type"], "l2_dispatched")
            self.assertEqual(rec["corrected_intent"], "")
            # 50-char cap + ellipsis.
            self.assertEqual(len(rec["text"]), 51)
            self.assertTrue(rec["text"].endswith("…"))

    def test_record_signal_respects_master_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=False):
                fid = record_signal(
                    "l2_dispatched", _intent(layer="L2"),
                    chat_id="oc_x", text="x",
                )
            self.assertIsNone(fid)
            self.assertFalse(path.exists())

    def test_record_signal_refuses_force_chat_alias(self):
        # The dedicated record_feedback path must own the force_chat
        # write contract (full-text retention). Funneling it through
        # record_signal would silently truncate per the signal cap.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
                fid = record_signal(
                    "force_chat", _intent(),
                    chat_id="oc_x", text="x",
                )
            self.assertIsNone(fid)
            self.assertFalse(path.exists())

    def test_jsonl_0600_perms(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
                record_signal(
                    "l1_gray_zone", _intent(),
                    chat_id="oc_x", text="x",
                )
            mode = os.stat(path).st_mode & 0o777
            self.assertEqual(mode, 0o600)


class TestDispatchHistory(unittest.TestCase):

    def setUp(self):
        clear_dispatch_history_for_tests()

    def tearDown(self):
        clear_dispatch_history_for_tests()

    def test_track_then_consume_pops(self):
        with _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
            track_dispatch("oc_A", _intent("dev"), text="t1")
            hit = consume_dispatch("oc_A", max_age_sec=10)
            self.assertIsNotNone(hit)
            self.assertEqual(hit[0].agent_type, "dev")
            self.assertEqual(hit[1], "t1")
            # Second consume returns None — single-shot.
            self.assertIsNone(consume_dispatch("oc_A", max_age_sec=10))

    def test_peek_does_not_pop(self):
        with _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
            track_dispatch("oc_A", _intent("dev"))
            p1 = peek_dispatch("oc_A", max_age_sec=10)
            p2 = peek_dispatch("oc_A", max_age_sec=10)
            self.assertIsNotNone(p1)
            self.assertIsNotNone(p2)
            self.assertEqual(p1[0].agent_type, "dev")

    def test_window_expiry(self):
        with _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
            track_dispatch("oc_A", _intent("dev"))
            # Force "age" past the window by patching time.monotonic.
            future = time.monotonic() + 1000
            with patch("larkhelm.agent_hub.intent_feedback.time.monotonic",
                       return_value=future):
                self.assertIsNone(consume_dispatch("oc_A", max_age_sec=60))

    def test_master_switch_makes_track_noop(self):
        with _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=False):
            track_dispatch("oc_A", _intent("dev"))
            self.assertIsNone(consume_dispatch("oc_A", max_age_sec=10))


class TestL1GrayZone(unittest.TestCase):

    def test_in_band_fires(self):
        # Default threshold 0.70, band 0.10 → fires for [0.70, 0.80).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(
                     INTENT_FEEDBACK_EXTENDED_SIGNALS=True,
                     INTENT_L1_PROMOTION_THRESHOLD=0.70,
                     INTENT_FEEDBACK_L1_GRAY_BAND=0.10,
                 ):
                _maybe_record_l1_gray_zone(
                    _intent(confidence=0.72, layer="L1"),
                    "oc_x", "test",
                )
            self.assertTrue(path.exists())
            rec = json.loads(path.read_text().strip())
            self.assertEqual(rec["signal_type"], "l1_gray_zone")
            self.assertEqual(rec["corrected_intent"], "")

    def test_above_band_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(
                     INTENT_FEEDBACK_EXTENDED_SIGNALS=True,
                     INTENT_L1_PROMOTION_THRESHOLD=0.70,
                     INTENT_FEEDBACK_L1_GRAY_BAND=0.10,
                 ):
                _maybe_record_l1_gray_zone(
                    _intent(confidence=0.95, layer="L1"),
                    "oc_x", "test",
                )
            self.assertFalse(path.exists())

    def test_l2_layer_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(
                     INTENT_FEEDBACK_EXTENDED_SIGNALS=True,
                     INTENT_L1_PROMOTION_THRESHOLD=0.70,
                     INTENT_FEEDBACK_L1_GRAY_BAND=0.10,
                 ):
                _maybe_record_l1_gray_zone(
                    _intent(confidence=0.72, layer="L2"),
                    "oc_x", "test",
                )
            self.assertFalse(path.exists())


class TestL2Dispatched(unittest.TestCase):

    def test_l2_non_chat_fires(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
                _maybe_record_l2_dispatched(
                    _intent(agent_type="dev", layer="L2"),
                    "oc_x", "test",
                )
            self.assertTrue(path.exists())
            rec = json.loads(path.read_text().strip())
            self.assertEqual(rec["signal_type"], "l2_dispatched")
            self.assertEqual(rec["predicted_intent"], "dev")

    def test_l2_chat_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
                _maybe_record_l2_dispatched(
                    _intent(agent_type="chat", layer="L2"),
                    "oc_x", "test",
                )
            self.assertFalse(path.exists())

    def test_l1_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
                _maybe_record_l2_dispatched(
                    _intent(agent_type="dev", layer="L1"),
                    "oc_x", "test",
                )
            self.assertFalse(path.exists())


class TestDispatcherDispatchFailed(unittest.TestCase):

    def setUp(self):
        clear_dispatch_history_for_tests()

    def tearDown(self):
        clear_dispatch_history_for_tests()

    def test_fallback_records_dispatch_failed_and_pops_history(self):
        # Simulate: AgentDispatcher.dispatch tracks intent, executor
        # raises, _fallback_to_chat fires record_signal("dispatch_failed")
        # AND consume_dispatch pops the history entry so a subsequent
        # /cancel can't double-bill it.
        from larkhelm.agent_hub.agent_base import AgentExecutor, AgentRegistry
        from larkhelm.agent_hub.agent_dispatcher import AgentDispatcher

        class _BoomExecutor(AgentExecutor):
            agent_type = "dev"
            description = "boom"

            def execute(self, intent, ctx):
                raise RuntimeError("simulated dev crash")

        class _ChatStub(AgentExecutor):
            agent_type = "chat"
            description = "chat"

            def execute(self, intent, ctx):
                return AgentResult(success=True, output="fallback ok",
                                   backend_id="chat_stub")

        reg = AgentRegistry()
        reg.register(_BoomExecutor())
        reg.register(_ChatStub())
        dispatcher = AgentDispatcher(registry=reg, acl={})

        intent = IntentResult(
            agent_type="dev", layer="L1", confidence=0.9,
            is_explicit_command=True, raw_text="测试 dev 崩",
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path), \
                 patch("larkhelm.agent_hub.agent_dispatcher.write_audit"), \
                 patch("larkhelm.lark_client.send_card", return_value="m"), \
                 _CfgPatch(INTENT_FEEDBACK_EXTENDED_SIGNALS=True):
                result = dispatcher.dispatch(intent, _ctx(chat_id="oc_bd"))
            # Fallback chat ran.
            self.assertTrue(result.success)
            # Exactly one dispatch_failed row (no double-write).
            lines = path.read_text().strip().splitlines()
            failed = [json.loads(l) for l in lines
                      if json.loads(l)["signal_type"] == "dispatch_failed"]
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["predicted_intent"], "dev")
            self.assertEqual(failed[0]["corrected_intent"], "chat")
            # History entry must have been popped so a subsequent
            # consume returns nothing.
            self.assertIsNone(consume_dispatch("oc_bd", max_age_sec=600))


class TestBackwardCompat(unittest.TestCase):

    def test_force_chat_record_has_all_legacy_fields(self):
        # The pre-Phase-D trainer (scripts/train_intent_classifier.py)
        # reads `text`, `corrected_intent`. Those keys must remain
        # present and shaped identically; only adds are allowed.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path):
                record_feedback(
                    _intent(agent_type="dev", confidence=0.85),
                    corrected="chat", chat_id="oc_x", text="hello",
                )
            rec = json.loads(path.read_text().strip())
            for key in ("ts", "chat_id", "text", "predicted_intent",
                        "corrected_intent", "confidence", "layer",
                        "feedback_id"):
                self.assertIn(key, rec)
            self.assertEqual(rec["text"], "hello")
            self.assertEqual(rec["corrected_intent"], "chat")


if __name__ == "__main__":
    unittest.main()
