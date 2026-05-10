"""AC-05: explicit slash commands bypass intent_router and call the matching path directly."""
import unittest

from larkhelm.agent_hub.intent_router import resolve_intent


class TestExplicitCommandBypass(unittest.TestCase):

    def test_dev_command_marked_explicit(self):
        intent = resolve_intent("/dev 实现登录模块")
        self.assertEqual(intent.agent_type, "dev")
        self.assertTrue(intent.is_explicit_command)
        self.assertEqual(intent.layer, "L1")
        self.assertEqual(intent.confidence, 1.0)

    def test_plan_command_marked_explicit(self):
        intent = resolve_intent("/plan 多步开发")
        self.assertTrue(intent.is_explicit_command)

    def test_intent_router_active_helper(self):
        # Helper used by handlers/_message.py: must respect flag + traffic.
        from larkhelm.handlers._message import _intent_router_active
        import larkhelm.config as _cfg

        original_cfg = getattr(_cfg, "config", {})
        _cfg.config = {"intent_router_enabled": False, "intent_router_traffic": 1.0}
        try:
            self.assertFalse(_intent_router_active("oc_x"))
        finally:
            _cfg.config = original_cfg

        _cfg.config = {"intent_router_enabled": True, "intent_router_traffic": 0.0}
        try:
            self.assertFalse(_intent_router_active("oc_x"))
        finally:
            _cfg.config = original_cfg

        _cfg.config = {"intent_router_enabled": True, "intent_router_traffic": 1.0}
        try:
            self.assertTrue(_intent_router_active("oc_x"))
        finally:
            _cfg.config = original_cfg

    def test_dev_explicit_command_dispatches_through_dev_executor(self):
        from larkhelm.agent_hub.agent_base import AgentRegistry
        from larkhelm.agent_hub.agent_dispatcher import AgentDispatcher
        from larkhelm.agent_hub.intent_types import (
            AgentContext, AgentResult, IntentResult,
        )
        from larkhelm.agent_hub.agent_base import AgentExecutor
        import threading
        from unittest.mock import patch

        class _DevSpy(AgentExecutor):
            agent_type = "dev"

            def __init__(self):
                self.calls = []

            def execute(self, intent, ctx):
                self.calls.append((intent, ctx))
                return AgentResult(success=True)

        class _ChatSpy(AgentExecutor):
            agent_type = "chat"

            def execute(self, intent, ctx):
                return AgentResult(success=True)

        dev = _DevSpy()
        reg = AgentRegistry()
        reg.register(dev)
        reg.register(_ChatSpy())
        dispatcher = AgentDispatcher(registry=reg, acl={})

        intent = resolve_intent("/dev 实现登录")
        ctx = AgentContext(
            chat_id="oc_x", user_msg_id="m", text="/dev 实现登录",
            images=None, parent_id=None,
            cancel_ev=threading.Event(), cwd="/tmp",
        )
        with patch("larkhelm.lark_client.send_card", return_value="m"), \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            result = dispatcher.dispatch(intent, ctx)
        self.assertEqual(len(dev.calls), 1)
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
