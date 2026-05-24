"""P1-5b: integration tests for the dispatch card layer badge.

Mocks ``larkhelm.lark_client.send_card`` and ``write_audit`` so the
dispatcher runs end-to-end without touching the Feishu SDK. The
assertions pin the *rendered* card body text (with badge appended) so a
regression in ``format_dispatch_badge`` OR a drift in the dispatcher's
body assembly would trip the test.
"""
import threading
import unittest
from unittest.mock import patch

from larkhelm.agent_hub.agent_base import AgentExecutor, AgentRegistry
from larkhelm.agent_hub.agent_dispatcher import AgentDispatcher
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


class _MockExecutor(AgentExecutor):
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.description = f"mock {agent_type}"

    def execute(self, intent, ctx):
        return AgentResult(success=True, output="ok", backend_id="x")


def _ctx(chat_id: str = "oc_test") -> AgentContext:
    return AgentContext(
        chat_id=chat_id, user_msg_id="m", text="实现登录", images=None,
        parent_id=None, cancel_ev=threading.Event(), cwd="/tmp",
    )


def _build_dispatcher() -> AgentDispatcher:
    reg = AgentRegistry()
    reg.register(_MockExecutor("dev"))
    reg.register(_MockExecutor("chat"))
    return AgentDispatcher(registry=reg, acl={})


def _card_body_and_title(send_card_mock) -> tuple[str, str]:
    """Extract (title, body) from the first send_card call.

    The dispatcher uses positional args: ``send_card(chat_id, title,
    body, color=..., buttons=...)``. Falls back to kwargs in case the
    call site ever migrates.
    """
    call = send_card_mock.call_args
    args, kwargs = call.args, call.kwargs
    title = args[1] if len(args) > 1 else kwargs.get("title", "")
    body = args[2] if len(args) > 2 else kwargs.get("content", "")
    return title, body


class TestDispatchCardBadge(unittest.TestCase):

    def test_l2_body_contains_badge(self):
        dispatcher = _build_dispatcher()
        intent = IntentResult(
            agent_type="dev", layer="L2", confidence=0.7,
            is_explicit_command=False, raw_text="实现登录",
        )
        with patch("larkhelm.lark_client.send_card", return_value="mid") as sc, \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            dispatcher.dispatch(intent, _ctx())
        title, body = _card_body_and_title(sc)
        self.assertEqual(title, "🛠 Dev Agent")
        self.assertIn("层级：L2 (L2)", body)

    def test_fallback_body_contains_arrow_badge(self):
        dispatcher = _build_dispatcher()
        intent = IntentResult(
            agent_type="chat", layer="fallback", confidence=0.0,
            is_explicit_command=False, raw_text="…",
        )
        with patch("larkhelm.lark_client.send_card", return_value="mid") as sc, \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            dispatcher.dispatch(intent, _ctx())
        _title, body = _card_body_and_title(sc)
        self.assertIn("层级：fallback (L2→chat)", body)

    def test_l1_body_omits_badge(self):
        dispatcher = _build_dispatcher()
        intent = IntentResult(
            agent_type="dev", layer="L1", confidence=0.85,
            is_explicit_command=False, raw_text="帮我实现登录",
        )
        with patch("larkhelm.lark_client.send_card", return_value="mid") as sc, \
             patch("larkhelm.agent_hub.agent_dispatcher.write_audit"):
            dispatcher.dispatch(intent, _ctx())
        _title, body = _card_body_and_title(sc)
        self.assertIn("层级：L1", body)
        self.assertNotIn("(L2)", body)
        self.assertNotIn("(L2→chat)", body)


if __name__ == "__main__":
    unittest.main()
