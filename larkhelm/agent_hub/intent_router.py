"""larkhelm · agent_hub · intent router (L1 rules + L2 cheap-LLM JSON classifier).

Public entry: :func:`resolve_intent`. The router never executes the agent;
it only labels the user's text. ``orchestration._detect_agent_protocol`` is
a separate downstream parser.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from larkhelm.agent_hub.intent_types import IntentResult


# ── Explicit slash-command prefixes mapped to agent_type ────────────────
_EXPLICIT_PREFIXES: list[tuple[tuple[str, ...], str]] = [
    (("/dev",), "dev"),
    (("/crew",), "crew"),
    (("/plan",), "plan"),
    (("/doc",), "doc"),
]


# ── L1 rule heuristics ─────────────────────────────────────────────────
_DEV_TRIGGERS = (
    "实现", "写一个", "写个", "开发", "写代码", "新建项目", "搭一个",
    "implement", "build me", "scaffold", "create a project", "write a function",
)
_CREW_TRIGGERS = (
    "调研", "研究", "策划", "分析整理", "多角色协作", "讨论方案",
    "brainstorm", "research and summarize",
)
_PLAN_TRIGGERS = (
    "分阶段", "step by step", "拆分计划", "多步执行", "依次完成",
    "phased", "multi-stage plan",
)
_DOC_TRIGGERS = (
    "写到飞书文档", "保存到 wiki", "保存到wiki", "更新这份文档", "整理成文档",
    "write to feishu doc", "append to wiki",
)


def _match_prefix(text_l: str) -> str | None:
    for prefixes, agent in _EXPLICIT_PREFIXES:
        for p in prefixes:
            if text_l == p or text_l.startswith(p + " ") or text_l.startswith(p + "\n"):
                return agent
    return None


def _resolve_l1(text: str, images: list | None, has_doc_urls: bool) -> IntentResult | None:
    t = text.lower()

    if has_doc_urls and any(k in text for k in ("写", "更新", "保存", "覆盖", "追加", "append", "write", "update")):
        return IntentResult(
            agent_type="doc", layer="L1", confidence=0.85,
            reasoning="contains feishu doc URL + write verb", raw_text=text,
        )

    for kw in _DOC_TRIGGERS:
        if kw in text or kw in t:
            return IntentResult(agent_type="doc", layer="L1", confidence=0.9,
                                reasoning=f"trigger: {kw}", raw_text=text)
    for kw in _PLAN_TRIGGERS:
        if kw in text or kw in t:
            return IntentResult(agent_type="plan", layer="L1", confidence=0.9,
                                reasoning=f"trigger: {kw}", raw_text=text)
    for kw in _CREW_TRIGGERS:
        if kw in text or kw in t:
            return IntentResult(agent_type="crew", layer="L1", confidence=0.85,
                                reasoning=f"trigger: {kw}", raw_text=text)
    for kw in _DEV_TRIGGERS:
        if kw in text or kw in t:
            return IntentResult(agent_type="dev", layer="L1", confidence=0.9,
                                reasoning=f"trigger: {kw}",
                                complexity="complex" if len(text) > 60 else "medium",
                                raw_text=text)
    return None


def _build_l2_prompt(agent_descriptions: list[tuple[str, str]]) -> str:
    desc_lines = "\n".join(
        f"- {atype}: {desc[:120]}" for atype, desc in agent_descriptions if desc
    ) or "- chat: 普通对话与简单问答\n- dev: 完整软件开发流水线\n- crew: 多角色调研协作\n- plan: 多阶段开发计划\n- doc: 飞书文档读写"
    return (
        "You are an intent classifier. Read the user's message and respond with ONLY a JSON object. "
        "Use this exact schema:\n"
        '{"intent":"chat|dev|crew|plan|doc","complexity":"simple|medium|complex","reasoning":"…"}\n\n'
        f"Available agent_types and descriptions:\n{desc_lines}\n\n"
        "Rules:\n"
        "- 'chat' is the default for casual conversation, factual questions, code reading.\n"
        "- 'dev' for non-trivial code writing tasks.\n"
        "- 'crew' for research / multi-agent / brainstorming tasks.\n"
        "- 'plan' when the user explicitly asks for a multi-stage/phased plan.\n"
        "- 'doc' when the user wants to read/write Feishu doc or wiki content.\n"
        "Output JSON only, no markdown fences."
    )


_JSON_DECODER = json.JSONDecoder()


def _extract_first_json_object(text: str) -> dict | None:
    """Find the first balanced JSON object in ``text`` and decode it.

    Uses :meth:`json.JSONDecoder.raw_decode` rather than a non-greedy regex
    (the previous ``_JSON_RE = r'\\{.*?\\}'``) so that nested JSON like
    ``{"a": {"b": 1}}`` is captured fully instead of being truncated at the
    first ``}``. Scans every ``{`` position and returns the first one that
    parses as a dict; bytes after that object are ignored.
    """
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = _JSON_DECODER.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _parse_l2_json(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r'^```(?:json)?\s*', '', s)
        s = re.sub(r'\s*```$', '', s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return _extract_first_json_object(raw)


def _resolve_l2(text: str) -> IntentResult:
    try:
        from larkhelm.backend_registry import BACKEND_REGISTRY
        from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
    except Exception:
        return _fallback(text)

    cheap = BACKEND_REGISTRY.get_by_tag(["cheap"])
    if cheap is None:
        return _fallback(text)

    descriptions: list[tuple[str, str]] = []
    for atype in AGENT_REGISTRY.list_types():
        ag = AGENT_REGISTRY.get(atype)
        if ag is not None:
            descriptions.append((atype, ag.description or ""))

    system_prompt = _build_l2_prompt(descriptions)

    raw = _call_cheap_backend(cheap, system_prompt, text)
    if raw is None:
        return _fallback(text)

    parsed = _parse_l2_json(raw)
    if not isinstance(parsed, dict):
        return _fallback(text)

    intent_name = str(parsed.get("intent", "")).strip().lower() or "chat"
    if intent_name not in {"chat", "dev", "crew", "plan", "doc", "search"}:
        intent_name = "chat"
    complexity = str(parsed.get("complexity", "medium")).strip().lower()
    if complexity not in {"simple", "medium", "complex"}:
        complexity = "medium"
    reasoning = str(parsed.get("reasoning", ""))[:200]

    return IntentResult(
        agent_type=intent_name, layer="L2", confidence=0.7,
        complexity=complexity, reasoning=reasoning, raw_text=text,
    )


def _call_cheap_backend(
    spec, system_prompt: str, text: str,
    *, _backend_call: "Callable[..., Any] | None" = None,
) -> str | None:
    """Invoke the cheap backend via :mod:`backend_api` if available.

    ``_backend_call`` is a test hook: production callers leave it ``None``
    so ``call_backend_oneshot`` is resolved by import; tests pass a fake
    callable to short-circuit ``sys.modules`` patching. When the import
    itself fails (and no hook is supplied) the function returns ``None`` —
    same as the legacy behaviour.
    """
    if _backend_call is None:
        try:
            from larkhelm.backend_api import call_backend_oneshot as _live_backend_call
            _backend_call = _live_backend_call
        except Exception:
            return None

    try:
        return _backend_call(spec, system_prompt, text, max_tokens=256, timeout=15.0)
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[IntentRouter] cheap backend call failed: {e}")
        return None


def _fallback(text: str) -> IntentResult:
    return IntentResult(agent_type="chat", layer="fallback", confidence=0.0, raw_text=text)


def resolve_intent(
    text: str,
    images: list | None = None,
    has_doc_urls: bool = False,
    chat_id: str | None = None,
) -> IntentResult:
    """Classify the user's text into an :class:`IntentResult`.

    Order:
      1. Explicit slash command prefix → ``is_explicit_command=True``.
      2. L1 keyword/heuristic rules.
      3. L2 cheap-backend JSON classifier.
      4. Fallback ``chat``.
    Any exception inside L2 collapses to fallback (NFR-SEC-02).
    """
    if not isinstance(text, str):
        return _fallback("")
    stripped = text.strip()
    if not stripped:
        return _fallback(stripped)

    text_l = stripped.lower()

    explicit = _match_prefix(text_l)
    if explicit:
        return IntentResult(
            agent_type=explicit,
            is_explicit_command=True,
            layer="L1",
            confidence=1.0,
            raw_text=stripped,
        )

    try:
        l1 = _resolve_l1(stripped, images, has_doc_urls)
    except Exception:
        l1 = None
    if l1 is not None:
        return l1

    try:
        return _resolve_l2(stripped)
    except Exception:
        return _fallback(stripped)


__all__ = ["resolve_intent"]
