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


# ── REVIEWER rules (quick single-pass code review) ────────────────────
# Distinct from CREW "代码检视/审查" (0.88) which triggers a full structured
# multi-perspective review.  These lighter phrases route to ReviewAgent for a
# quick single-turn code check with a reviewer-profile backend.
_REVIEWER_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("帮我review", "reviewer", 0.85),
    KeywordRule("review一下", "reviewer", 0.82),
    KeywordRule("review下", "reviewer", 0.82),
    KeywordRule("帮我看看代码", "reviewer", 0.80),
    KeywordRule("看看这段代码", "reviewer", 0.80),
    KeywordRule("看看这个代码", "reviewer", 0.80),
    KeywordRule("快速review", "reviewer", 0.85),
    KeywordRule("检查代码", "reviewer", 0.72, "ambiguous — needs another signal"),
    KeywordRule("代码有问题", "reviewer", 0.65),
    KeywordRule(re.compile(r"\breview\s+(?:my|this|the|these)\s+code\b", re.IGNORECASE), "reviewer", 0.85),
    KeywordRule(re.compile(r"\bcheck\s+(?:my|this|the)\s+code\b", re.IGNORECASE), "reviewer", 0.80),
    KeywordRule(re.compile(r"\bquick\s+review\b", re.IGNORECASE), "reviewer", 0.85),
    KeywordRule(re.compile(r"\bcode\s+review\b", re.IGNORECASE), "reviewer", 0.72, "standalone → might be crew too"),
)


# ── SEARCH rules (web search augmentation) ─────────────────────────────
_SEARCH_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("帮我搜", "search", 0.88),
    KeywordRule("搜索一下", "search", 0.85),
    KeywordRule("搜索下", "search", 0.85),
    KeywordRule("网上搜", "search", 0.90),
    KeywordRule("上网查", "search", 0.90),
    KeywordRule("查一下最新", "search", 0.85),
    KeywordRule("最新版本是什么", "search", 0.88),
    KeywordRule("最新的版本", "search", 0.82),
    KeywordRule("最新消息", "search", 0.80),
    KeywordRule(re.compile(r"\bsearch\s+(?:for|the\s+web|online)\b", re.IGNORECASE), "search", 0.88),
    KeywordRule(re.compile(r"\blook\s+(?:it\s+)?up\s+online\b", re.IGNORECASE), "search", 0.88),
    KeywordRule(re.compile(r"\bwhat'?s?\s+the\s+latest\b", re.IGNORECASE), "search", 0.82),
)


# ── SHELL rules (execute shell command + AI interpretation) ────────────
_SHELL_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("帮我运行", "shell", 0.88),
    KeywordRule("帮我执行", "shell", 0.88),
    KeywordRule("运行这个脚本", "shell", 0.90),
    KeywordRule("执行这个脚本", "shell", 0.90),
    KeywordRule("跑一下这段", "shell", 0.85),
    KeywordRule("跑一下这个", "shell", 0.80),
    KeywordRule("执行命令", "shell", 0.82),
    KeywordRule("运行命令", "shell", 0.82),
    KeywordRule(re.compile(r"\brun\s+(?:this\s+)?(?:script|command|cmd)\b", re.IGNORECASE), "shell", 0.88),
    KeywordRule(re.compile(r"\bexecute\s+(?:this\s+)?(?:script|command)\b", re.IGNORECASE), "shell", 0.88),
    KeywordRule(re.compile(r"^[\$>]\s*\w", re.MULTILINE), "shell", 0.80, "message starts with shell prompt"),
)


# ── CALENDAR rules (Feishu calendar operations) ───────────────────────
_CALENDAR_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("查看日程", "calendar", 0.90),
    KeywordRule("今天的日程", "calendar", 0.92),
    KeywordRule("本周日程", "calendar", 0.90),
    KeywordRule("明天日程", "calendar", 0.90),
    KeywordRule("日历安排", "calendar", 0.85),
    KeywordRule("日程安排", "calendar", 0.85),
    KeywordRule("创建日程", "calendar", 0.92),
    KeywordRule("新建日程", "calendar", 0.92),
    KeywordRule("安排会议", "calendar", 0.88),
    KeywordRule("查询空闲", "calendar", 0.88),
    KeywordRule("有什么安排", "calendar", 0.82),
    KeywordRule("飞书日历", "calendar", 0.92),
    KeywordRule(re.compile(r"\b(calendar|schedule|meeting|event)\s+(?:today|tomorrow|this\s+week)\b", re.IGNORECASE), "calendar", 0.88),
    KeywordRule(re.compile(r"\bcreate\s+(?:a\s+)?(?:meeting|event|reminder)\b", re.IGNORECASE), "calendar", 0.88),
    KeywordRule(re.compile(r"\bwhat'?s?\s+(?:on\s+)?(?:my\s+)?(?:calendar|schedule)\b", re.IGNORECASE), "calendar", 0.88),
)


# ── HISTORY_SEARCH rules (search past conversations) ──────────────────
_HISTORY_SEARCH_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("上次讨论", "history_search", 0.88),
    KeywordRule("之前讨论", "history_search", 0.88),
    KeywordRule("之前说过", "history_search", 0.85),
    KeywordRule("之前写的", "history_search", 0.85),
    KeywordRule("上次你写的", "history_search", 0.90),
    KeywordRule("历史记录", "history_search", 0.80),
    KeywordRule("之前的对话", "history_search", 0.88),
    KeywordRule("搜索历史", "history_search", 0.88),
    KeywordRule("找找之前", "history_search", 0.85),
    KeywordRule(re.compile(r"\b(search|find)\s+(?:in\s+)?(?:past|previous|earlier|old)\s+(?:conversation|history|chat)\b", re.IGNORECASE), "history_search", 0.88),
    KeywordRule(re.compile(r"\blast\s+time\s+(?:we|you)\b", re.IGNORECASE), "history_search", 0.85),
)


# ── GITHUB rules (PR / Issue / CI operations) ─────────────────────────
_GITHUB_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("查看pr", "github", 0.90),
    KeywordRule("列出pr", "github", 0.90),
    KeywordRule("pr列表", "github", 0.90),
    KeywordRule("open pr", "github", 0.88),
    KeywordRule("pull request", "github", 0.82),
    KeywordRule("查看issue", "github", 0.90),
    KeywordRule("issue列表", "github", 0.90),
    KeywordRule("提交issue", "github", 0.88),
    KeywordRule("创建issue", "github", 0.88),
    KeywordRule("ci状态", "github", 0.88),
    KeywordRule("查看ci", "github", 0.88),
    KeywordRule("workflow状态", "github", 0.88),
    KeywordRule("最近提交", "github", 0.78),
    KeywordRule(re.compile(r"\bgh\s+(?:pr|issue|run|repo)\b", re.IGNORECASE), "github", 0.90),
    KeywordRule(re.compile(r"\bgithub\.com/\S+\b", re.IGNORECASE), "github", 0.78),
    KeywordRule(re.compile(r"\bpr\s*#\d+\b", re.IGNORECASE), "github", 0.90),
    KeywordRule(re.compile(r"\bcheck\s+(?:ci|workflow|build|pipeline)\b", re.IGNORECASE), "github", 0.85),
    KeywordRule(re.compile(r"\blist\s+(?:open\s+)?(?:pr|pull\s*request|issue)s?\b", re.IGNORECASE), "github", 0.88),
)


# ── TRANSLATE rules (中英互译 / multi-language) ────────────────────────
_TRANSLATE_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("翻译成中文", "translate", 0.92),
    KeywordRule("翻译成英文", "translate", 0.92),
    KeywordRule("翻译成英语", "translate", 0.92),
    KeywordRule("翻译一下", "translate", 0.88),
    KeywordRule("帮我翻译", "translate", 0.88),
    KeywordRule("帮翻译", "translate", 0.88),
    KeywordRule("中文翻译", "translate", 0.85),
    KeywordRule("英文翻译", "translate", 0.85),
    KeywordRule("英文怎么说", "translate", 0.90),
    KeywordRule("中文怎么说", "translate", 0.90),
    KeywordRule("用英文说", "translate", 0.88),
    KeywordRule("用中文说", "translate", 0.88),
    KeywordRule(re.compile(r"\btranslate\s+(?:this|to|into|from)\b", re.IGNORECASE), "translate", 0.90),
    KeywordRule(re.compile(r"\btranslate\s+(?:it\s+)?(?:to\s+)?(?:chinese|english|japanese|korean)\b", re.IGNORECASE), "translate", 0.92),
)


# ── HOTFIX rules (fix + code-review pipeline, 2 stages) ───────────────────
# Distinct from ``quick-fix`` SkillDef (single _do_query) — hotfix spawns a
# full implementer→reviewer DAG and writes review.md.  Also reachable via
# the explicit ``/hotfix`` slash command.
_HOTFIX_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("紧急修复",       "hotfix", 0.90, "直接触发 hotfix pipeline"),
    KeywordRule("修复并审查",     "hotfix", 0.90),
    KeywordRule("修完帮我review", "hotfix", 0.90, "强于 reviewer skill 的 0.88"),
    KeywordRule("修完审查一下",   "hotfix", 0.88),
    KeywordRule(re.compile(r"\bhotfix\b", re.IGNORECASE), "hotfix", 0.90, "英文 hotfix"),
)


# ── REVIEW-ONLY rules (standalone structured review pipeline) ──────────────
# Distinct from ``reviewer`` SkillDef (quick single-call review) — review-only
# writes a persistent review.md.  Also reachable via ``/review`` slash command.
_REVIEW_ONLY_RULES: tuple[KeywordRule, ...] = (
    KeywordRule("写一份review报告",    "review-only", 0.88),
    KeywordRule("生成review报告",      "review-only", 0.88),
    KeywordRule("做一次完整的代码审查", "review-only", 0.85),
    KeywordRule(re.compile(r"\bwrite\s+a\s+(?:code\s+)?review\s+report\b", re.IGNORECASE), "review-only", 0.85),
    KeywordRule(re.compile(r"\bfull\s+code\s+review\b", re.IGNORECASE), "review-only", 0.82),
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
    # "运行原理 / 运行机制 / 运行时" — "运行" used as noun/adjective, not a
    # shell execution request.
    NegativeRule(
        re.compile(r"运行\s*(?:原理|机制|时|逻辑|过程|流程)"),
        blocks=frozenset({"shell"}),
        note="运行 used as noun → block shell",
    ),
    # "跑步 / 跑路 / 跑分" — 跑 unrelated to shell execution.
    NegativeRule(
        re.compile(r"跑\s*(?:步|路|分|通|题)"),
        blocks=frozenset({"shell"}),
        note="跑 in non-shell context",
    ),
    # "翻译文档 / 翻译器 / 翻译软件" — user is asking about a translation
    # tool, not requesting a translation service.
    NegativeRule(
        re.compile(r"翻译\s*(?:软件|工具|插件|器|API|接口|服务|引擎|模型)"),
        blocks=frozenset({"translate"}),
        note="翻译 + tool suffix → not a translate request",
    ),
    # "测试原理 / 测试框架 / 测试覆盖率是什么" — user is asking *about* testing
    # concepts, not requesting test generation.
    NegativeRule(
        re.compile(r"测试\s*(?:原理|框架|覆盖率是|工具|策略|方法论|体系|理论)"),
        blocks=frozenset({"test-gen"}),
        note="测试 + concept suffix → explanation, not generation",
    ),
    # "补充说明 / 补充一下背景" — 补充 used as general "add more info",
    # not as "add docstring/comment".
    NegativeRule(
        re.compile(r"补充\s*(?:说明|一下|背景|细节|内容|信息|描述)(?!docstring|注释|文档)"),
        blocks=frozenset({"tech-doc"}),
        note="补充 used generically → not a doc-generation request",
    ),
)


# ── All rule sets in one place ─────────────────────────────────────────


def all_rules() -> tuple[KeywordRule, ...]:
    return (
        _DEV_RULES + _CREW_RULES + _PLAN_RULES + _DOC_RULES
        + _CALENDAR_RULES + _HISTORY_SEARCH_RULES
        + _REVIEWER_RULES + _GITHUB_RULES + _SEARCH_RULES + _SHELL_RULES + _TRANSLATE_RULES
        + _HOTFIX_RULES + _REVIEW_ONLY_RULES
    )


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
    "calendar": [
        ("+", "查看今天的日程安排"),
        ("+", "help me schedule a meeting tomorrow at 3pm"),
        ("-", "把会议纪要整理成文档"),
    ],
    "history_search": [
        ("+", "上次我们讨论的那个 API 设计方案在哪里？"),
        ("+", "find the script you wrote for me last time"),
        ("-", "搜索这个飞书文档里的内容"),
    ],
    "github": [
        ("+", "列出所有 open PR，告诉我哪个最需要 review"),
        ("+", "check the CI status for the latest commit"),
        ("-", "实现一个 github oauth 登录功能"),
    ],
    "reviewer": [
        ("+", "帮我 review 一下这段代码，有没有明显问题"),
        ("+", "quick review: check this python function for bugs"),
        ("-", "组织一次代码检视，从功能和安全两个角度评估"),
    ],
    "search": [
        ("+", "帮我搜一下 Python 3.13 有什么新特性"),
        ("+", "search for the latest version of fastapi"),
        ("-", "搜索这份飞书文档里的内容"),
    ],
    "shell": [
        ("+", "帮我运行 git status 并告诉我哪些文件需要提交"),
        ("+", "run this script: python manage.py migrate"),
        ("-", "解释这段脚本的运行原理"),
    ],
    "translate": [
        ("+", "把这段话翻译成英文"),
        ("+", "translate this paragraph to Chinese"),
        ("-", "推荐一个好用的翻译软件"),
    ],
    "quick-fix": [
        ("+", "帮我修复这个 bug，认证中间件抛出 KeyError"),
        ("+", "fix this bug: null pointer in user_service.py line 42"),
        ("-", "分析这个性能问题，制定完整优化方案"),
    ],
    "explain-code": [
        ("+", "解释这段代码是做什么的，特别是这个闭包"),
        ("+", "explain what this async function does and why it uses a semaphore"),
        ("-", "如何给团队解释这段代码的设计取舍"),
    ],
    "test-gen": [
        ("+", "帮我写单元测试，覆盖 UserService.create_user 的边界条件"),
        ("+", "generate unit tests for this parser function"),
        ("-", "测试这个需求是否符合用户预期，从头验收"),
    ],
    "tech-doc": [
        ("+", "帮我写 README，这是个命令行工具项目"),
        ("+", "generate docstrings for all public methods in this file"),
        ("-", "把文档保存到飞书文档里"),
    ],
    "hotfix": [
        ("+", "紧急修复：登录接口 500 错误，不需要规划直接改"),
        ("+", "hotfix this null-pointer in payment_service.py, then review the fix"),
        ("-", "分析这个 bug 背后的架构问题，制定系统性修复方案"),
    ],
    "review-only": [
        ("+", "帮我写一份 review 报告，对 src/auth/ 目录做完整审查"),
        ("+", "full code review of this PR diff, write a review.md"),
        ("-", "快速看一下这段代码有没有问题"),
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
