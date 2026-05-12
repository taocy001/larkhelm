"""Unit tests for the three-level fallback in :func:`larkhelm.cmd_plan._parse_plan`.

Covers PRD AC-01 ~ AC-06: explicit ``[type]`` syntax, keyword mapping,
batched LLM fallback, mixed input, and fail-soft on LLM errors. All LLM paths
are mocked via :mod:`unittest.mock` — no real Claude/subprocess invocation.
"""
from __future__ import annotations

import unittest
from unittest import mock

from larkhelm.cmd_plan import _parse_plan, PlanStep


class TestKeywordMapping(unittest.TestCase):
    """AC-01, AC-02, P1, P2: keyword table behavior with LLM mocked out."""

    def test_chinese_keywords_each_type(self):
        # AC-01: 4 Chinese keyword lines → 4 steps with the right type each.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            _, steps = _parse_plan("开发登录\n审查安全\n修复 bug\n回归测试")
        m_llm.assert_not_called()
        self.assertEqual(len(steps), 4)
        self.assertEqual([s.type for s in steps], ["dev", "review", "fix", "test"])
        # idx is 0..n-1 in order.
        self.assertEqual([s.idx for s in steps], [0, 1, 2, 3])

    def test_english_keywords_case_insensitive(self):
        # AC-02: English tokens match regardless of case via \b...\b.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            _, steps = _parse_plan("Review API\nFIX bug\ntest all")
        m_llm.assert_not_called()
        self.assertEqual([s.type for s in steps], ["review", "fix", "test"])

    def test_desc_strips_keyword_prefix(self):
        # PRD §3 P1: keyword prefix is stripped from desc.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            _, steps = _parse_plan("开发登录\nfix 登录 bug\n审查 数据安全")
        m_llm.assert_not_called()
        descs = [s.desc for s in steps]
        self.assertEqual(descs, ["登录", "登录 bug", "数据安全"])

    def test_keyword_conflict_priority(self):
        # PRD §3 P2: dev > fix > review > test on multi-hit.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            _, steps = _parse_plan("开发并修复登录")
        m_llm.assert_not_called()
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].type, "dev")
        # Prefix '开发' is stripped; the rest is preserved as desc.
        self.assertEqual(steps[0].desc, "并修复登录")


class TestLLMFallback(unittest.TestCase):
    """AC-03, AC-06: LLM fallback called once for ambiguous lines + fail-soft."""

    def test_ambiguous_lines_invoke_llm_once(self):
        # AC-03: two ambiguous lines → exactly one LLM call with both lines.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            m_llm.return_value = [("dev", "用户管理"), ("review", "权限校验")]
            _, steps = _parse_plan("处理用户管理\n搞定权限校验")

        self.assertEqual(m_llm.call_count, 1)
        called_lines = m_llm.call_args.args[0]
        self.assertEqual(called_lines, ["处理用户管理", "搞定权限校验"])
        self.assertEqual([s.type for s in steps], ["dev", "review"])
        self.assertEqual([s.desc for s in steps], ["用户管理", "权限校验"])

    def test_llm_exception_failsoft(self):
        # AC-06: LLM raises → ambiguous batch dropped, explicit step preserved,
        # _debug_log gets a "[Plan]"-prefixed message.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps",
                        side_effect=RuntimeError("classifier blew up")), \
             mock.patch("larkhelm.cmd_plan._debug_log") as m_log:
            title, steps = _parse_plan("[dev] A\n模糊一下")

        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].type, "dev")
        self.assertEqual(steps[0].desc, "A")
        # _debug_log was called with a message that begins with "[Plan]".
        self.assertTrue(m_log.called)
        joined = " ".join(call.args[0] for call in m_log.call_args_list if call.args)
        self.assertIn("[Plan]", joined)


class TestMixedInput(unittest.TestCase):
    """AC-04: explicit + keyword + LLM-classified lines coexist in one /plan."""

    def test_explicit_and_keyword_and_fallback(self):
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            m_llm.return_value = [("fix", "C")]
            title, steps = _parse_plan("总标题\n[dev] A\n审查 B\n搞定 C")

        # LLM only saw the genuinely ambiguous line.
        self.assertEqual(m_llm.call_count, 1)
        self.assertEqual(m_llm.call_args.args[0], ["搞定 C"])

        self.assertEqual(title, "总标题")
        self.assertEqual(len(steps), 3)
        self.assertEqual([s.type for s in steps], ["dev", "review", "fix"])
        self.assertEqual([s.desc for s in steps], ["A", "B", "C"])
        self.assertEqual([s.idx for s in steps], [0, 1, 2])


class TestBackwardCompat(unittest.TestCase):
    """AC-05 + other no-regression guarantees."""

    def test_explicit_only_no_llm_call(self):
        # AC-05: pure [type] input must not touch the LLM fallback at all.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            _, steps = _parse_plan("[dev] A\n[review] B\n[fix] C\n[test] D")
        m_llm.assert_not_called()
        self.assertEqual(len(steps), 4)
        self.assertEqual([s.type for s in steps], ["dev", "review", "fix", "test"])
        self.assertEqual([s.desc for s in steps], ["A", "B", "C", "D"])

    def test_title_extraction_unchanged(self):
        # Legacy behavior: first non-step line that matches neither [type] nor
        # any keyword becomes the title.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            title, steps = _parse_plan("项目计划 v1\n[dev] A\n[review] B")
        m_llm.assert_not_called()
        self.assertEqual(title, "项目计划 v1")
        self.assertEqual(len(steps), 2)

    def test_all_keyword_lines_skip_llm(self):
        # PRD §3 P2 cost optimization: ambiguous_lines empty → no LLM call.
        with mock.patch("larkhelm.cmd_plan._llm_classify_steps") as m_llm:
            _, steps = _parse_plan("开发 A\n审查 B\n测试 C")
        m_llm.assert_not_called()
        self.assertEqual([s.type for s in steps], ["dev", "review", "test"])


if __name__ == "__main__":
    unittest.main()
