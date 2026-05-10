"""AC-04 / AC-09: AgentDispatcher card display, ACL, fallback, audit."""
import threading
import unittest
from unittest.mock import patch

from larkhelm.agent_hub.agent_base import AgentExecutor, AgentRegistry
from larkhelm.agent_hub.agent_dispatcher import AgentDispatcher
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class _MockExecutor(AgentExecutor):
    def __init__(self, agent_type: str, success: bool = True, output: str = ""):
        self.agent_type = agent_type
        self.description = f"mock {agent_type}"
        self._success = success
        self._output = output
        self.calls: list[tuple[IntentResult, AgentContext]] = []

    def execute(self, intent, ctx):
        self.calls.append((intent, ctx))
        return AgentResult(success=self._success, output=self._output, backend_id="x")


def _ctx(chat_id="oc_test") -> AgentContext:
    return AgentContext(
        chat_id=chat_id, user_msg_id="m", text="hi", images=None,
        parent_id=None, cancel_ev=threading.Event(), cwd="/tmp",
    )


class TestDispatcherHappyPath(unittest.TestCase):

    def test_dispatch_success(self):
        reg = AgentRegistry()
        dev = _MockExecutor("dev")
        reg.register(dev)
        reg.register(_MockExecutor("chat"))

        dispatcher = AgentDispatcher(registry=reg, acl={})
        intent = IntentResult(
            agent_type="dev", layer="L1", confidence=0.9,
            is_explicit_command=True, raw_text="hi",
        )
        with patch("larkhelm.agent_hub.agent_dispatcher.write_audit") as wa, \
             patch("larkhelm.lark_client.send_card", return_value="m1"):
            result = dispatcher.dispatch(intent, _ctx())
        self.assertTrue(result.success)
        self.assertEqual(len(dev.calls), 1)
        wa.assert_called_once()

    def test_intent_card_for_non_explicit(self):
        reg = AgentRegistry()
        reg.register(_MockExecutor("dev"))
        reg.register(_MockExecutor("chat"))
        dispatcher = AgentDispatcher(registry=reg, acl={})
        intent = IntentResult(agent_type="dev", layer="L2", confidence=0.7,
                              is_explicit_command=False, raw_text="实现登录")
        with patch("larkhelm.lark_client.send_card", return_value="mid") as sc, \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            dispatcher.dispatch(intent, _ctx())
        sc.assert_called_once()
        kwargs = sc.call_args
        # Buttons must contain a "force_chat" entry pointing at a feedback_id.
        buttons = kwargs.kwargs.get("buttons") or (kwargs.args[5] if len(kwargs.args) > 5 else [])
        joined = str(buttons)
        self.assertIn("force_chat:", joined)


class TestDispatcherACL(unittest.TestCase):

    def test_acl_denied(self):
        reg = AgentRegistry()
        reg.register(_MockExecutor("dev"))
        reg.register(_MockExecutor("chat"))
        dispatcher = AgentDispatcher(registry=reg, acl={"dev": ["oc_admin*"]})
        intent = IntentResult(agent_type="dev", is_explicit_command=True, raw_text="x")
        with patch("larkhelm.lark_client.send_card") as sc, \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            result = dispatcher.dispatch(intent, _ctx(chat_id="oc_random"))
        self.assertFalse(result.success)
        self.assertIn("ACL", result.error)
        sc.assert_called()

    def test_acl_glob_matches(self):
        reg = AgentRegistry()
        dev = _MockExecutor("dev")
        reg.register(dev)
        reg.register(_MockExecutor("chat"))
        dispatcher = AgentDispatcher(registry=reg, acl={"dev": ["oc_admin*"]})
        intent = IntentResult(agent_type="dev", is_explicit_command=True, raw_text="x")
        with patch("larkhelm.lark_client.send_card", return_value="m"), \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            result = dispatcher.dispatch(intent, _ctx(chat_id="oc_admin42"))
        self.assertTrue(result.success)
        self.assertEqual(len(dev.calls), 1)


class TestDispatcherFallback(unittest.TestCase):

    def test_executor_exception_falls_back_to_chat(self):
        reg = AgentRegistry()

        class _BadAgent(AgentExecutor):
            agent_type = "dev"

            def execute(self, intent, ctx):
                raise RuntimeError("boom")

        reg.register(_BadAgent())
        chat = _MockExecutor("chat", success=True, output="fallback")
        reg.register(chat)
        dispatcher = AgentDispatcher(registry=reg, acl={})
        intent = IntentResult(agent_type="dev", is_explicit_command=True, raw_text="x")
        with patch("larkhelm.lark_client.send_card", return_value="m"), \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            result = dispatcher.dispatch(intent, _ctx())
        self.assertEqual(len(chat.calls), 1)
        self.assertTrue(result.success)


class TestForceChatFeedback(unittest.TestCase):

    def test_force_chat_records_feedback(self):
        from larkhelm.agent_hub.intent_feedback import (
            register_pending, resolve_pending, record_feedback,
        )

        intent = IntentResult(agent_type="dev", confidence=0.8, layer="L2", raw_text="t")
        ctx = _ctx()
        register_pending("fb_test", intent, ctx, text="t")
        entry = resolve_pending("fb_test")
        self.assertIsNotNone(entry)

        # The user clicked "switch to plain chat" — record the misclassification.
        with patch("larkhelm.agent_hub.intent_feedback._append_jsonl") as ap:
            fid = record_feedback(intent, "chat", "oc_test",
                                  feedback_id="fb_test", text="t")
        self.assertEqual(fid, "fb_test")
        ap.assert_called_once()
        record = ap.call_args.args[1]
        self.assertEqual(record["predicted_intent"], "dev")
        self.assertEqual(record["corrected_intent"], "chat")


if __name__ == "__main__":
    unittest.main()
