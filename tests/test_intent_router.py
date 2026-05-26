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

    def test_doc_prefix_retired(self):
        # 方案B (commit 7c9845c) retired /doc as a user-facing slash command.
        # The router must NOT recognise it as an explicit prefix anymore —
        # falls through to L2 / fallback like any free-form text.
        intent = resolve_intent("/doc read https://x.feishu.cn/docx/abc")
        self.assertFalse(intent.is_explicit_command)
        self.assertNotEqual(intent.layer, "L1")


class TestDynamicRuleEdgeCases(unittest.TestCase):
    """AC-02, AC-06, AC-07: edge cases for _score_dynamic_rules and skill defs."""

    def test_empty_re_pattern_skipped(self):
        """AC-02: a dynamic rule with 're:' prefix but empty pattern must not raise
        and must not match anything (previously caused re.error or wrong match)."""
        from larkhelm.agent_hub import intent_router
        from larkhelm.agent_hub.skill_registry import SkillRegistry

        fake_registry = SkillRegistry()
        fake_registry._rules = [("re:", "some_agent", 0.99, "empty re")]

        original_get = None
        try:
            from larkhelm.agent_hub import skill_registry as _sr
            original_get = _sr.SKILL_REGISTRY.get_l1_rules
            _sr.SKILL_REGISTRY.get_l1_rules = lambda: [("re:", "some_agent", 0.99, "empty re")]
            result = resolve_intent("帮我翻译 hello world")
        finally:
            if original_get is not None:
                from larkhelm.agent_hub import skill_registry as _sr
                _sr.SKILL_REGISTRY.get_l1_rules = original_get

        # The key requirement: no exception raised. The empty 're:' must have been skipped.
        self.assertIsNotNone(result)
        self.assertNotEqual(result.agent_type, "some_agent",
                            "empty re: pattern must not match anything")

    def test_re_pattern_compiled_once(self):
        """AC-06: the same regex pattern must be compiled only once and cached in
        _re_pattern_cache regardless of how many messages trigger it."""
        from larkhelm.agent_hub import intent_router

        # Reset the cache to get a clean baseline
        intent_router._re_pattern_cache.clear()

        pat = r"test_unique_pattern_\d+"
        from larkhelm.agent_hub import skill_registry as _sr
        original_get = _sr.SKILL_REGISTRY.get_l1_rules
        _sr.SKILL_REGISTRY.get_l1_rules = lambda: [(f"re:{pat}", "some_agent", 0.5, "")]
        try:
            resolve_intent("no match here 123")
            resolve_intent("another message 456")
        finally:
            _sr.SKILL_REGISTRY.get_l1_rules = original_get

        # The pattern should now be in the cache
        self.assertIn(pat, intent_router._re_pattern_cache,
                      "regex pattern was not added to _re_pattern_cache after use")
        import re
        cached = intent_router._re_pattern_cache[pat]
        self.assertIsInstance(cached, re.Pattern,
                              "_re_pattern_cache value should be a compiled re.Pattern")

    def test_reviewer_skill_strip_pattern(self):
        """AC-07: reviewer skill must have a non-empty, valid strip_trigger_pattern."""
        import re
        from larkhelm.agent_hub.builtin.skills._defs import _BUILTIN_SKILL_DICTS
        reviewer = next(
            (s for s in _BUILTIN_SKILL_DICTS if s["id"] == "reviewer"), None
        )
        self.assertIsNotNone(reviewer, "reviewer skill not found in _BUILTIN_SKILL_DICTS")
        pattern = reviewer.get("strip_trigger_pattern", "")
        self.assertTrue(bool(pattern),
                        "reviewer strip_trigger_pattern must be non-empty")
        # Must compile without error
        compiled = re.compile(pattern)
        # Must strip known trigger phrases
        sample = "帮我review 以下代码"
        stripped = re.sub(compiled, "", sample).strip()
        self.assertEqual(stripped, "以下代码",
                         f"strip_trigger_pattern did not strip trigger; got {stripped!r}")


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

    def test_doc_url_with_new_doc_object_defers_to_l2(self):
        # Regression for the wiki-link case where the user wants to *read*
        # the URL'd doc and *write a brand-new document* (verb's object is
        # the new doc, not the URL'd one). Must NOT land on DocAgent's L1
        # rule — fall through to L2 LLM (or fallback in unit tests, since
        # no cheap backend is registered).
        text = (
            "https://my.feishu.cn/wiki/Uu7Rwvcn 这是一篇超节点场景交换机OS中SUE功能的文档，"
            "里面有很多观点是错误的，找出所有错误观点重新写一份正确的文档。"
        )
        intent = resolve_intent(text, has_doc_urls=True)
        self.assertNotEqual(intent.agent_type, "doc")
        self.assertNotEqual(intent.layer, "L1")

    def test_doc_url_new_doc_short_form(self):
        # Shorter variant: "写一份" + 文档 keyword in the same sentence.
        intent = resolve_intent(
            "看 https://feishu.cn/docx/abc 然后写一份新的总结",
            has_doc_urls=True,
        )
        self.assertNotEqual(intent.agent_type, "doc")


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
