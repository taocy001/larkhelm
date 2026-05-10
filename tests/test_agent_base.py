"""AC-02 / AC-03: AgentExecutor ABC enforcement + AgentRegistry CRUD."""
import unittest

from larkhelm.agent_hub.agent_base import (
    AGENT_REGISTRY, AgentExecutor, AgentRegistry,
)
from larkhelm.agent_hub.intent_types import IntentResult


class TestAgentExecutorAbstract(unittest.TestCase):

    def test_cannot_instantiate_without_execute(self):
        class BadAgent(AgentExecutor):
            agent_type = "bad"
        with self.assertRaises(TypeError):
            BadAgent()

    def test_concrete_subclass_instantiates(self):
        class _OkAgent(AgentExecutor):
            agent_type = "ok"

            def execute(self, intent, ctx):
                return None

        agent = _OkAgent()
        self.assertEqual(agent.agent_type, "ok")
        self.assertEqual(
            agent.can_handle(IntentResult(agent_type="ok")),
            1.0,
        )
        self.assertEqual(
            agent.can_handle(IntentResult(agent_type="other")),
            0.0,
        )


class _Stub(AgentExecutor):
    agent_type = "stub"

    def execute(self, intent, ctx):
        return None


class _Stub2(AgentExecutor):
    agent_type = "stub2"

    def execute(self, intent, ctx):
        return None


class TestAgentRegistry(unittest.TestCase):

    def test_register_and_get(self):
        reg = AgentRegistry()
        s = _Stub()
        reg.register(s)
        self.assertIs(reg.get("stub"), s)

    def test_register_replaces_same_type(self):
        reg = AgentRegistry()
        a, b = _Stub(), _Stub()
        reg.register(a)
        reg.register(b)
        self.assertIs(reg.get("stub"), b)

    def test_register_requires_executor(self):
        reg = AgentRegistry()
        with self.assertRaises(TypeError):
            reg.register("not-an-executor")  # type: ignore[arg-type]

    def test_register_requires_agent_type(self):
        class _NoType(AgentExecutor):
            def execute(self, intent, ctx):
                return None
        reg = AgentRegistry()
        with self.assertRaises(ValueError):
            reg.register(_NoType())

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register(_Stub())
        self.assertTrue(reg.unregister("stub"))
        self.assertFalse(reg.unregister("stub"))
        self.assertIsNone(reg.get("stub"))

    def test_list_types_sorted(self):
        reg = AgentRegistry()
        reg.register(_Stub2())
        reg.register(_Stub())
        self.assertEqual(reg.list_types(), ["stub", "stub2"])

    def test_match_returns_best_score(self):
        reg = AgentRegistry()
        reg.register(_Stub())
        self.assertIs(reg.match(IntentResult(agent_type="stub")), reg.get("stub"))
        self.assertIsNone(reg.match(IntentResult(agent_type="missing")))


class TestBuiltinRegistration(unittest.TestCase):
    def test_5_builtins_registered(self):
        # Importing agent_hub triggers builtin registration as a side effect.
        import larkhelm.agent_hub  # noqa: F401
        types = AGENT_REGISTRY.list_types()
        for required in ("chat", "dev", "crew", "plan", "doc"):
            self.assertIn(required, types)


if __name__ == "__main__":
    unittest.main()
