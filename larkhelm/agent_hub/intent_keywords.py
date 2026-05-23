"""larkhelm · agent_hub · L1 keyword rules with confidence tiers.

Three problems with the prior single-flat-tuple design:

  1. **Substring matching has no morpheme awareness.** ``"实现"`` matched
     both the verb (``"实现一个登录功能"`` — really dev) and the noun
     (``"从代码实现的角度看 bug"`` — really chat / crew). The classifier
     could not tell the two apart.
  2. **Single trigger = full classification.** L1 fired with confidence
     0.85+ on any keyword hit, so the cheaper LLM L2 (configured for 80%
     traffic) never ran on prompts that grazed a bad keyword.
  3. **No negative patterns for non-doc agents.** ``_NEW_DOC_OBJECT_RE``
     guarded the doc path, but dev/crew/plan had no equivalent.

This module re-encodes triggers as ``KeywordRule`` records with a
``strength`` weight 0..1. The router (``intent_router._resolve_l1``)
picks the highest-scoring agent across all matched rules; if that score
falls below ``L1_PROMOTION_THRESHOLD`` the L1 abstains and the prompt is
deferred to L2.

Mined from real conversation history (``logs/all.jsonl`` user prompts
joined to ``model`` field as label) + ``intent_feedback.jsonl`` user
corrections. Numbers in comments are observed precision when known.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# ── Confidence promotion threshold ─────────────────────────────────────
# Hit-strength below this collapses L1 to "abstain" so the prompt is
# routed to L2 (cheap LLM JSON or embedding classifier) for a real
# classification. Operators can override via ``config.json``:
# ``intent_l1_promotion_threshold`` (NEW knob).
L1_PROMOTION_THRESHOLD: float = 0.70


# ── Keyword rule ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class KeywordRule:
    """One scoring rule for the L1 keyword classifier.

    ``pattern`` is a *literal* substring (case-insensitive against the
    lowercased message) OR a compiled :class:`re.Pattern`. Literal
    strings are cheaper; regex is reserved for shapes the literal
    matcher can't capture (e.g. variable separators).

    ``strength`` is the confidence the L1 should assign when this rule
    fires. Strong verb-with-object phrases get 0.85; ambiguous single
    nouns get 0.40 so they need help from another rule to cross the
    promotion threshold.
    """

    pattern: "str | re.Pattern[str]"
    agent: str
    strength: float
    note: str = ""

    def match(self, text: str, text_l: str) -> bool:
        p = self.pattern
        if isinstance(p, re.Pattern):
            return bool(p.search(text))
        # Literal: substring match on the lowercased text so ASCII rules
        # are case-insensitive; CJK rules pass through unchanged because
        # ``str.lower()`` is identity for CJK characters.
        return p in text or p in text_l


@dataclass(frozen=True)
class NegativeRule:
    """Pattern that *blocks* one or more agents when present.

    Used to suppress false-positive dev matches like 「代码实现」 or
    「review the implementation」 where the keyword appears as a noun
    rather than the actionable verb.
    """

    pattern: "str | re.Pattern[str]"
    blocks: frozenset = field(default_factory=frozenset)
    note: str = ""

    def match(self, text: str, text_l: str) -> bool:
        p = self.pattern
        if isinstance(p, re.Pattern):
            return bool(p.search(text))
        return p in text or p in text_l


# ── Strong DEV rules (verb-with-object; high precision) ────────────────
# Mined: "新建" 9/9, "需求开发" 5/5, generic "开发" 49/82 (ambiguous).
# Tightened to require an object marker so noun usage of 实现 / 开发
# stays at L2.
_DEV_RULES: tuple[KeywordRule, ...] = (
    # Strong: verb + object marker
    KeywordRule("实现一个", "dev", 0.88, "verb + 一个 → dev imperative"),
    KeywordRule("实现这个", "dev", 0.85, "verb + 这个"),
    KeywordRule("实现该", "dev", 0.85),
    KeywordRule("实现新", "dev", 0.85),
    KeywordRule("实现一套", "dev", 0.85),
    KeywordRule("实现一下", "dev", 0.78),
    KeywordRule("开发一个", "dev", 0.85),
    KeywordRule("开发这个", "dev", 0.85),
    KeywordRule("开发一套", "dev", 0.85),
    KeywordRule("开发新的", "dev", 0.85),
    KeywordRule("开发个", "dev", 0.80),
    KeywordRule("写一个", "dev", 0.85),
    KeywordRule("写个", "dev", 0.80),
    KeywordRule("写一份", "dev", 0.55, "could be doc — defer"),
    KeywordRule("搭一个", "dev", 0.85),
    KeywordRule("搭个", "dev", 0.80),
    KeywordRule("新建项目", "dev", 0.90),
    KeywordRule("新建一个", "dev", 0.78),
    KeywordRule("增加.{0,6}功能", "dev", 0.75),
    KeywordRule("加一个.{0,4}功能", "dev", 0.80),
    KeywordRule("加个.{0,4}功能", "dev", 0.78),
    # English verb-context
    KeywordRule(re.compile(r"\bimplement\s+(?:a|an|the|this|that|new)\b", re.IGNORECASE), "dev", 0.85),
    KeywordRule(re.compile(r"\bbuild\s+me\b", re.IGNORECASE), "dev", 0.85),
    KeywordRule(re.compile(r"\bscaffold\b", re.IGNORECASE), "dev", 0.85),
    KeywordRule(re.compile(r"\bcreate\s+a\s+(?:project|repo|module)", re.IGNORECASE), "dev", 0.85),
    KeywordRule(re.compile(r"\bwrite\s+a\s+(?:function|class|module|script)", re.IGNORECASE), "dev", 0.85),
    # Ambiguous (weak): single token that often appears as noun
    KeywordRule("开发", "dev", 0.40, "noun-heavy — needs more signal"),
    KeywordRule("写代码", "dev", 0.55),
)


# ── Strong CREW rules (multi-perspective / research / review) ──────────
# Mined: "代码检视/检视" appears as crew/plan trigger (7/9 prec 0.78 in
# plan corpus, but semantically belongs to crew when no plan structure).
# "调研/研究/策划" are canonical crew triggers.
_CREW_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("调研", "crew", 0.85),
    KeywordRule("研究下", "crew", 0.80),
    KeywordRule("研究一下", "crew", 0.80),
    KeywordRule("研究研究", "crew", 0.85),
    KeywordRule("策划", "crew", 0.85),
    KeywordRule("分析整理", "crew", 0.85),
    KeywordRule("多角色协作", "crew", 0.90),
    KeywordRule("多个角度", "crew", 0.78),
    KeywordRule("多视角", "crew", 0.78),
    KeywordRule("从.{0,4}角度.{0,12}从.{0,4}角度", "crew", 0.85, "two-perspective"),
    KeywordRule("讨论方案", "crew", 0.78),
    KeywordRule("头脑风暴", "crew", 0.90),
    KeywordRule("竞品分析", "crew", 0.85),
    KeywordRule("市场分析", "crew", 0.85),
    KeywordRule("技术分析", "crew", 0.78),
    # Code review (the user's reported false-negative case)
    KeywordRule("代码检视", "crew", 0.88, "verb + object"),
    KeywordRule("代码审查", "crew", 0.88),
    KeywordRule("代码评审", "crew", 0.88),
    KeywordRule("组织一次", "crew", 0.78),
    KeywordRule("做一次代码", "crew", 0.85),
    KeywordRule("评审一下", "crew", 0.78),
    KeywordRule("审查一下", "crew", 0.78),
    KeywordRule("复盘", "crew", 0.78),
    # Mined from production crew dispatches
    KeywordRule("修复这个", "crew", 0.78, "follow-up after analysis"),
    KeywordRule("需求开发", "crew", 0.85),
    KeywordRule("分析结果", "crew", 0.78),
    KeywordRule("规划的", "crew", 0.78),
    # English
    KeywordRule(re.compile(r"\bbrainstorm\b", re.IGNORECASE), "crew", 0.88),
    KeywordRule(re.compile(r"\bresearch\s+and\s+summari[sz]e\b", re.IGNORECASE), "crew", 0.85),
    KeywordRule(re.compile(r"\bcode[\s\-]review\b", re.IGNORECASE), "crew", 0.85),
    KeywordRule(re.compile(r"\bdo\s+a\s+code\s+review\b", re.IGNORECASE), "crew", 0.88),
    # Ambiguous
    KeywordRule("研究", "crew", 0.50, "could be casual"),
    KeywordRule("检视", "crew", 0.55, "could appear as noun"),
)


# ── Strong PLAN rules (multi-stage / phased / sequential) ──────────────
# Plan corpus is small (23); patterns are derived from observed
# multi-step prompts.
_PLAN_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("分阶段", "plan", 0.88),
    KeywordRule("多阶段", "plan", 0.88),
    KeywordRule("拆分计划", "plan", 0.90),
    KeywordRule("拆分成", "plan", 0.65, "could be code refactor"),
    KeywordRule("多步执行", "plan", 0.85),
    KeywordRule("依次完成", "plan", 0.85),
    KeywordRule("一步一步", "plan", 0.80),
    KeywordRule("step by step", "plan", 0.85),
    KeywordRule(re.compile(r"\bphased\b", re.IGNORECASE), "plan", 0.85),
    KeywordRule(re.compile(r"\bmulti[\s\-]?stage\s+plan\b", re.IGNORECASE), "plan", 0.90),
    # Mined sequential markers
    KeywordRule("完成之后.{0,8}进入", "plan", 0.85),
    KeywordRule("完成后再进入", "plan", 0.85),
    KeywordRule("读取计划", "plan", 0.88),
    KeywordRule("文档读取计划", "plan", 0.90),
    KeywordRule("做完.{0,4}再做", "plan", 0.80),
    KeywordRule("先.{1,12}再.{1,12}最后", "plan", 0.82, "first-then-finally chain"),
)


# ── Strong DOC rules (Feishu doc write intent) ─────────────────────────
_DOC_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("写到飞书文档", "doc", 0.92),
    KeywordRule("写入飞书文档", "doc", 0.92),
    KeywordRule("保存到 wiki", "doc", 0.90),
    KeywordRule("保存到wiki", "doc", 0.90),
    KeywordRule("保存到飞书", "doc", 0.88),
    KeywordRule("更新这份文档", "doc", 0.88),
    KeywordRule("整理成文档", "doc", 0.82),
    KeywordRule("追加到文档", "doc", 0.85),
    KeywordRule("覆盖文档", "doc", 0.78),
    KeywordRule(re.compile(r"\bwrite\s+to\s+feishu\b", re.IGNORECASE), "doc", 0.92),
    KeywordRule(re.compile(r"\bappend\s+to\s+wiki\b", re.IGNORECASE), "doc", 0.88),
    KeywordRule(re.compile(r"\bsave\s+to\s+(?:feishu|wiki)\b", re.IGNORECASE), "doc", 0.85),
)


# ── Negative rules ─────────────────────────────────────────────────────
# These block specific agents when the keyword usage is *non-imperative*
# (noun, attribute, possessive). Without these the L1 confidence tier
# scheme still helps, but they make false-positive prevention explicit.
_NEGATIVE_RULES: tuple[NegativeRule, ...] = (
    # "代码实现 / 实现细节 / 具体实现 / 现有实现 / 老实现 / 旧实现" — the
    # word 实现 is a noun ("the implementation") not a verb. Don't
    # classify these as dev even if other weak dev triggers fire.
    NegativeRule(
        re.compile(r"(?:代码|具体|现有|老|旧|这份|这个|当前|原|原来的|目前的)\s*实现"),
        blocks=frozenset({"dev"}),
        note="实现 used as noun → block dev",
    ),
    NegativeRule(
        re.compile(r"实现\s*(?:细节|方式|方案|思路|逻辑|过程)"),
        blocks=frozenset({"dev"}),
        note="实现 + abstract suffix → block dev",
    ),
    NegativeRule(
        re.compile(r"\b(?:review|look\s+at|inspect)\s+the\s+implementation\b", re.IGNORECASE),
        blocks=frozenset({"dev"}),
        note="english noun usage",
    ),
    # "一份/一篇/一个 (新/正确/完整/另) 文档/笔记/文章/wiki" — the user
    # wants to *create* a brand-new doc, not write back to the URL'd doc
    # they pasted. Defer to L2 / dev / crew instead of slamming into
    # DocAgent (which would write to the linked URL).
    NegativeRule(
        re.compile(
            r"一?\s*[份篇个]\s*(?:新的?|正确的?|完整的?|另一?|另起的?)?\s*"
            r"(?:文档|文稿|笔记|总结|稿|文章|wiki|doc|document|note|article|page)"
            r"|a\s+new\s+(?:doc|document|note|article|page|wiki)",
            re.IGNORECASE,
        ),
        blocks=frozenset({"doc"}),
        note="user wants a NEW doc, not write-back to URL'd doc",
    ),
)


# ── All rule sets in one place ─────────────────────────────────────────


def all_rules() -> tuple[KeywordRule, ...]:
    return _DEV_RULES + _CREW_RULES + _PLAN_RULES + _DOC_RULES


def all_negative_rules() -> tuple[NegativeRule, ...]:
    return _NEGATIVE_RULES


# ── L2 few-shot examples ───────────────────────────────────────────────
# Anchor each agent description in the L2 prompt with 2 positive + 1
# negative example. DeepSeek/Kimi cheap classifiers jump in accuracy
# vs. abstract descriptions alone. Examples are short, distinct, and
# drawn from real (anonymized) historical prompts where possible.
AGENT_FEW_SHOTS: dict[str, list[tuple[str, str]]] = {
    "dev": [
        ("+", "实现一个 jwt 鉴权中间件，要求支持 refresh token"),
        ("+", "scaffold a fastapi project with postgres + alembic"),
        ("-", "从代码实现的角度看这个 bug 该怎么修"),
    ],
    "crew": [
        ("+", "组织一次代码检视，从功能、流程、文档三个角度评估"),
        ("+", "做个竞品分析、市场分析、技术分析，5 分钟以内的播客 app"),
        ("-", "实现一个登录接口"),
    ],
    "plan": [
        ("+", "分阶段把这个项目从单体拆成微服务，每阶段一个验收点"),
        ("+", "step by step 把环境搭起来，先安装依赖，再起服务，最后跑测试"),
        ("-", "step by step 教我怎么用这个 cli"),
    ],
    "doc": [
        ("+", "把刚才的总结整理成文档保存到 wiki"),
        ("+", "把这份分析追加到这个飞书文档 https://x.feishu.cn/docx/abc"),
        ("-", "看下这个飞书文档 https://x.feishu.cn/docx/abc 找出错误观点重新写一份新文档"),
    ],
    "chat": [
        ("+", "刚才的报错是什么意思？"),
        ("+", "你状态怎样"),
        ("-", "实现一个登录中间件"),
    ],
}


__all__ = [
    "L1_PROMOTION_THRESHOLD",
    "KeywordRule",
    "NegativeRule",
    "all_rules",
    "all_negative_rules",
    "AGENT_FEW_SHOTS",
]
