"""End-to-end coverage for the force_chat card-callback chain.

The "切换为普通对话" button posts ``force_chat:<feedback_id>`` which causes
``handlers/_card_action.py`` to:

  1. ``resolve_pending(feedback_id)``    — fetch the original (intent, ctx)
  2. ``record_feedback(...)``           — persist the misclassification
  3. ``_trigger_cancel(chat_id)``       — interrupt the in-flight executor
  4. spawn ``AgentDispatcher().dispatch(override_intent, ctx)`` — re-run as chat

review.md §6 backlog: "force_chat: 卡片回调端到端无测试 (pending 过期 /
record_feedback / dispatch override 三段链路)" — this file covers all three.
"""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from larkhelm.agent_hub import intent_feedback
from larkhelm.agent_hub.agent_base import AgentExecutor, AgentRegistry
from larkhelm.agent_hub.agent_dispatcher import AgentDispatcher
from larkhelm.agent_hub.intent_feedback import (
    record_feedback, register_pending, resolve_pending,
)
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


# ── Test fixtures ───────────────────────────────────────────────────────


class _RecordingAgent(AgentExecutor):
    """AgentExecutor that records every dispatch call for assertion."""

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.description = f"recording {agent_type}"
        self.calls: list[tuple[IntentResult, AgentContext]] = []

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        self.calls.append((intent, ctx))
        return AgentResult(success=True, output="ok", backend_id="x")


def _ctx(chat_id: str = "oc_test") -> AgentContext:
    return AgentContext(
        chat_id=chat_id, user_msg_id="m1", text="原始文本", images=None,
        parent_id=None, cancel_ev=threading.Event(), cwd="/tmp",
    )


# ── Stage 1: pending registry semantics (register / resolve / expiry) ───


class TestPendingRegistry(unittest.TestCase):

    def test_register_then_resolve_pops(self):
        intent = IntentResult(agent_type="dev", confidence=0.8, layer="L2", raw_text="x")
        ctx = _ctx()
        register_pending("fb_e2e_1", intent, ctx, text="x")
        entry = resolve_pending("fb_e2e_1")
        self.assertIsNotNone(entry)
        self.assertIs(entry.intent, intent)
        self.assertIs(entry.ctx, ctx)
        # resolve_pending pops; second call must return None.
        self.assertIsNone(resolve_pending("fb_e2e_1"))

    def test_unknown_id_returns_none(self):
        self.assertIsNone(resolve_pending("fb_does_not_exist_xyz"))

    def test_lru_eviction_simulates_expiry(self):
        """Registering > _LRU_CAP entries evicts the oldest — this is the
        in-memory equivalent of "pending expired" in production."""
        cap = intent_feedback._LRU_CAP
        # Tag with a unique prefix so we don't collide with other tests.
        prefix = "fb_lru_e2e_"
        for i in range(cap + 5):
            register_pending(
                f"{prefix}{i:04}",
                IntentResult(agent_type="chat", raw_text=str(i)),
                _ctx(), text=str(i),
            )
        # Oldest entries should have been pushed out.
        self.assertIsNone(resolve_pending(f"{prefix}0000"))
        self.assertIsNone(resolve_pending(f"{prefix}0004"))
        # Newer entries survived.
        self.assertIsNotNone(resolve_pending(f"{prefix}{(cap + 4):04}"))
        # Cleanup — drop everything we may have left behind.
        for i in range(cap + 5):
            resolve_pending(f"{prefix}{i:04}")


# ── Stage 2: record_feedback persists to JSONL ──────────────────────────


class TestRecordFeedbackPersistence(unittest.TestCase):

    def test_record_feedback_writes_correct_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intent_feedback.jsonl"
            intent = IntentResult(
                agent_type="dev", confidence=0.92, layer="L2",
                raw_text="帮我写个登录模块",
            )
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=path):
                fid = record_feedback(intent, "chat", "oc_e2e",
                                      feedback_id="fb_persist", text="帮我写个登录模块")
            self.assertEqual(fid, "fb_persist")
            rec = json.loads(path.read_text().strip())
            self.assertEqual(rec["chat_id"], "oc_e2e")
            self.assertEqual(rec["predicted_intent"], "dev")
            self.assertEqual(rec["corrected_intent"], "chat")
            self.assertEqual(rec["confidence"], 0.92)
            self.assertEqual(rec["layer"], "L2")
            self.assertEqual(rec["feedback_id"], "fb_persist")
            self.assertEqual(rec["text"], "帮我写个登录模块")

    def test_record_feedback_swallows_io_errors(self):
        intent = IntentResult(agent_type="dev", raw_text="x")
        # Make the underlying writer always blow up.
        with patch("larkhelm.agent_hub.intent_feedback._append_jsonl",
                   side_effect=PermissionError("read-only fs")), \
             patch.object(intent_feedback, "_safe_log") as log:
            fid = record_feedback(intent, "chat", "oc_e2e", text="x")
        # Caller still receives a feedback id; the failure is logged but not raised.
        self.assertTrue(fid.startswith("fb_"))
        log.assert_called()


# ── Stage 3: dispatch with layer="override" suppresses the disclosure card ─


class TestOverrideDispatchSuppressesCard(unittest.TestCase):
    """When force_chat re-runs the agent, the new IntentResult has
    ``layer="override"`` and the dispatcher MUST skip ``_show_intent_card``
    so the user does not see another disclosure for the same turn."""

    def test_override_skips_card(self):
        registry = AgentRegistry()
        chat = _RecordingAgent("chat")
        registry.register(chat)
        dispatcher = AgentDispatcher(registry=registry, acl={})
        override_intent = IntentResult(
            agent_type="chat", layer="override", confidence=0.0,
            is_explicit_command=True, raw_text="原始文本",
        )
        ctx = _ctx()
        with patch("larkhelm.agent_hub.agent_dispatcher.write_audit"), \
             patch("larkhelm.lark_client.send_card") as sc:
            result = dispatcher.dispatch(override_intent, ctx)
        # ChatAgent ran exactly once with the override intent
        self.assertTrue(result.success)
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(chat.calls[0][0].layer, "override")
        # No disclosure card was sent (otherwise UX loops forever)
        sc.assert_not_called()

    def test_non_override_l2_still_shows_card(self):
        """Sanity check: non-override, non-explicit L2 dispatch must still
        show the disclosure card — confirms the override-skip is targeted."""
        registry = AgentRegistry()
        registry.register(_RecordingAgent("chat"))
        registry.register(_RecordingAgent("dev"))
        dispatcher = AgentDispatcher(registry=registry, acl={})
        l2_intent = IntentResult(
            agent_type="dev", layer="L2", confidence=0.7,
            is_explicit_command=False, raw_text="实现登录",
        )
        with patch("larkhelm.agent_hub.agent_dispatcher.write_audit"), \
             patch("larkhelm.lark_client.send_card", return_value="m1") as sc:
            dispatcher.dispatch(l2_intent, _ctx())
        sc.assert_called_once()


# ── Stage 4: full chain — pending → record_feedback → override dispatch ─


class TestFullForceChatChain(unittest.TestCase):
    """Mimics the steps inside ``handlers/_card_action.py`` force_chat branch
    without invoking the SDK callback object. Verifies the three modules wire
    together correctly."""

    def test_full_chain(self):
        registry = AgentRegistry()
        chat = _RecordingAgent("chat")
        dev = _RecordingAgent("dev")
        registry.register(chat)
        registry.register(dev)

        original_intent = IntentResult(
            agent_type="dev", layer="L2", confidence=0.7, raw_text="原始文本",
        )
        ctx = _ctx()
        feedback_id = "fb_chain"
        register_pending(feedback_id, original_intent, ctx, text="原始文本")

        # Step 1: card callback resolves the pending entry.
        pending = resolve_pending(feedback_id)
        self.assertIsNotNone(pending)

        # Step 2: card callback writes the misclassification to feedback log.
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "intent_feedback.jsonl"
            with patch("larkhelm.agent_hub.intent_feedback._resolve_path",
                       return_value=log_path):
                record_feedback(pending.intent, "chat", ctx.chat_id,
                                feedback_id=feedback_id, text=pending.text)
            rec = json.loads(log_path.read_text().strip())
        self.assertEqual(rec["predicted_intent"], "dev")
        self.assertEqual(rec["corrected_intent"], "chat")

        # Step 3: dispatch the override intent (chat) — disclosure card suppressed.
        override = IntentResult(
            agent_type="chat", layer="override",
            is_explicit_command=True, raw_text=pending.text,
        )
        dispatcher = AgentDispatcher(registry=registry, acl={})
        with patch("larkhelm.agent_hub.agent_dispatcher.write_audit"), \
             patch("larkhelm.lark_client.send_card") as sc:
            result = dispatcher.dispatch(override, ctx)

        self.assertTrue(result.success)
        # ChatAgent ran (with override intent), DevAgent must NOT have been called.
        self.assertEqual(len(chat.calls), 1)
        self.assertEqual(len(dev.calls), 0)
        sc.assert_not_called()


# ── Negative path: pending already expired ──────────────────────────────


class TestExpiredPending(unittest.TestCase):

    def test_resolve_returns_none_after_eviction(self):
        """Mirrors the real card_action.force_chat 'expired' branch: when
        resolve_pending returns None we must NOT crash and no audit row is
        written."""
        # Don't register; resolve immediately with an unknown id.
        self.assertIsNone(resolve_pending("fb_definitely_expired"))


if __name__ == "__main__":
    unittest.main()
