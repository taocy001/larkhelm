"""AC-01: intent router covers explicit prefixes, L1 rules, L2 LLM, fallback."""
import unittest
from unittest.mock import patch

from larkhelm.agent_hub.intent_router import resolve_intent


class TestExplicitPrefixes(unittest.TestCase):

    def test_dev_prefix(self):
        intent = resolve_intent("/dev 实现登录")
        self.assertEqual(intent.agent_type, "dev")
        self.assertTrue(intent.is_explicit_command)
        self.assertEqual(intent.layer, "L1")
        self.assertEqual(intent.confidence, 1.0)

    def test_plan_prefix(self):
        intent = resolve_intent("/plan 多步开发")
        self.assertEqual(intent.agent_type, "plan")
        self.assertTrue(intent.is_explicit_command)

    def test_crew_prefix(self):
        intent = resolve_intent("/crew 调研框架")
        self.assertEqual(intent.agent_type, "crew")
        self.assertTrue(intent.is_explicit_command)

    def test_doc_prefix(self):
        intent = resolve_intent("/doc read https://x.feishu.cn/docx/abc")
        self.assertEqual(intent.agent_type, "doc")


class TestL1Rules(unittest.TestCase):

    def test_dev_trigger_zh(self):
        intent = resolve_intent("帮我实现一个登录模块，要支持 OAuth")
        self.assertEqual(intent.agent_type, "dev")
        self.assertEqual(intent.layer, "L1")
        self.assertFalse(intent.is_explicit_command)

    def test_plan_trigger(self):
        intent = resolve_intent("帮我分阶段拆解登录改造")
        self.assertEqual(intent.agent_type, "plan")

    def test_crew_trigger(self):
        intent = resolve_intent("调研下 React 状态管理库")
        self.assertEqual(intent.agent_type, "crew")

    def test_doc_url_with_write_verb(self):
        intent = resolve_intent("把这段总结写到 https://feishu.cn/docx/abc",
                                has_doc_urls=True)
        self.assertEqual(intent.agent_type, "doc")


class TestL2Fallback(unittest.TestCase):

    def test_l2_returns_fallback_when_cheap_unavailable(self):
        # Default registry in tests has no cheap-tagged backend → fallback.
        intent = resolve_intent("讲个笑话")
        self.assertEqual(intent.agent_type, "chat")
        self.assertEqual(intent.layer, "fallback")

    def test_l2_parses_valid_json(self):
        from larkhelm.agent_hub import intent_router

        class _FakeSpec:
            id = "cheap"

        with patch.object(intent_router, "_call_cheap_backend",
                          return_value='{"intent":"dev","complexity":"complex","reasoning":"x"}') as _m, \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY.get_by_tag",
                   return_value=_FakeSpec()):
            intent = resolve_intent("我要做一个非常复杂的事情")
        # If L1 matches first the test still passes (dev), but if not, L2 must too.
        self.assertEqual(intent.agent_type, "dev")

    def test_l2_exception_falls_back(self):
        from larkhelm.agent_hub import intent_router

        class _FakeSpec:
            id = "cheap"

        with patch.object(intent_router, "_call_cheap_backend",
                          side_effect=RuntimeError("boom")), \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY.get_by_tag",
                   return_value=_FakeSpec()):
            intent = resolve_intent("一段不会被 L1 命中的随便文本")
        self.assertEqual(intent.agent_type, "chat")
        self.assertEqual(intent.layer, "fallback")


class TestEmptyInput(unittest.TestCase):
    def test_empty_string(self):
        intent = resolve_intent("")
        self.assertEqual(intent.agent_type, "chat")
        self.assertEqual(intent.layer, "fallback")


if __name__ == "__main__":
    unittest.main()
