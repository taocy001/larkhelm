"""larkhelm · agent_hub · intent router (L1 rules + L2 cheap-LLM JSON classifier).

Public entry: :func:`resolve_intent`. The router never executes the agent;
it only labels the user's text. ``orchestration._detect_agent_protocol`` is
a separate downstream parser.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Any, Callable

from larkhelm.agent_hub.intent_types import IntentResult


# REQ-03: module-level cache for the embedding classifier. Building the
# classifier itself is cheap, but ``precompute(descriptions)`` invokes the
# embedding backend once per agent description on every call to
# ``_try_embedding_l2``. With ``intent_layer2_strategy="embedding"`` that
# means N redundant backend round-trips per user message. The signature
# changes whenever the registered (agent_type, description) pairs or the
# similarity threshold change, so AgentRegistry plugins that load later
# still re-precompute correctly.
_EMB_CACHE_LOCK: threading.Lock = threading.Lock()
_EMB_CLASSIFIER: Any = None
_EMB_SIGNATURE: str = ""
_re_pattern_cache: dict[str, re.Pattern] = {}


def _emb_signature(descriptions: list[tuple[str, str]], threshold: float) -> str:
    """Deterministic signature over sorted (agent_type, description) pairs."""
    serial = "\n".join(f"{a}\x00{d}" for a, d in sorted(descriptions))
    return hashlib.sha1(f"{threshold:.4f}\n{serial}".encode("utf-8")).hexdigest()


def _resolve_embedding_classifier(
    backend: Any,
    descriptions: list[tuple[str, str]],
    threshold: float,
) -> Any:
    """Return a module-cached :class:`EmbeddingIntentClassifier`.

    Rebuilds (and re-precomputes) when the (descriptions, threshold)
    signature changes, otherwise hands back the cached instance so only
    the per-query embed call remains.
    """
    global _EMB_CLASSIFIER, _EMB_SIGNATURE
    sig = _emb_signature(descriptions, threshold)
    with _EMB_CACHE_LOCK:
        if _EMB_CLASSIFIER is not None and _EMB_SIGNATURE == sig:
            return _EMB_CLASSIFIER
        from larkhelm.agent_hub.intent_embedding import EmbeddingIntentClassifier
        clf = EmbeddingIntentClassifier(backend, threshold=threshold)
        clf.precompute(descriptions)
        _EMB_CLASSIFIER = clf
        _EMB_SIGNATURE = sig
        return clf


def _reset_embedding_cache_for_tests() -> None:
    """Test-only hook: clear the module-level classifier cache."""
    global _EMB_CLASSIFIER, _EMB_SIGNATURE
    with _EMB_CACHE_LOCK:
        _EMB_CLASSIFIER = None
        _EMB_SIGNATURE = ""


# ── Explicit slash-command prefixes mapped to agent_type ────────────────
# Note: ``/doc`` was retired as a user-facing slash command (方案B). DocAgent
# is still reachable via L1 trigger heuristics + L2 LLM intent classification.
_EXPLICIT_PREFIXES: list[tuple[tuple[str, ...], str]] = [
    (("/dev",), "dev"),
    (("/crew",), "crew"),
    (("/plan",), "plan"),
    (("/hotfix",), "hotfix"),
    (("/review",), "review-only"),
]


# ── P1-5b: dispatch-card layer badge map ────────────────────────────────
# Only ``"L2"`` and ``"fallback"`` show a badge; every other layer value
# (including ``"L1"`` whether or not the user typed an explicit slash
# command, ``"microlearn"``, ``"override"``, and any future / unknown
# value) falls through ``dict.get(layer, "")`` and renders as plain text.
_LAYER_BADGE: dict[str, str] = {
    "L2":       "(L2)",
    "fallback": "(L2→chat)",
}


def format_dispatch_badge(intent: IntentResult) -> str:
    """Render the layer badge appended to the dispatch card body.

    Returns one of:
      * ``""``           — layer ∈ {"L1", "explicit", "microlearn", "override", <unknown>}
      * ``"(L2)"``       — layer == "L2"
      * ``"(L2→chat)"``  — layer == "fallback"

    Pure / O(1) / no side effects / never raises.
    """
    return _LAYER_BADGE.get(intent.layer, "")


# ── L1 rule heuristics ─────────────────────────────────────────────────
# Keyword rules + negative patterns + few-shot examples live in
# ``intent_keywords`` for testability. The router scores each agent by
# the max strength of any matched rule (minus any rule blocked by a
# negative pattern), then promotes to L1 only when the winner exceeds
# ``L1_PROMOTION_THRESHOLD``. Below threshold → abstain → defer to L2.
from larkhelm.agent_hub.intent_keywords import (  # noqa: E402
    AGENT_FEW_SHOTS,
    L1_PROMOTION_THRESHOLD as _DEFAULT_L1_PROMOTION_THRESHOLD,
    all_negative_rules,
    all_rules,
)


def _match_prefix(text_l: str) -> str | None:
    """Return the agent name when ``text_l`` opens with an explicit
    slash-command prefix. The router treats this as the highest-priority
    decision (confidence 1.0, ``is_explicit_command=True``).
    """
    for prefixes, agent in _EXPLICIT_PREFIXES:
        for p in prefixes:
            if text_l == p or text_l.startswith(p + " ") or text_l.startswith(p + "\n"):
                return agent
    return None


def _l1_threshold() -> float:
    """Hot-read the promotion threshold from config.

    Tests / operators can override via ``intent_l1_promotion_threshold``
    in ``config.json``. Floor at 0.05 so a bad value can't silently
    disable L1 entirely (set the flag below to ``false`` for that).
    """
    try:
        import larkhelm.config as _cfg
        raw = getattr(_cfg, "INTENT_L1_PROMOTION_THRESHOLD", None)
        if raw is None:
            return _DEFAULT_L1_PROMOTION_THRESHOLD
        return max(0.05, float(raw))
    except Exception:
        return _DEFAULT_L1_PROMOTION_THRESHOLD


def _l1_disabled() -> bool:
    """``intent_l1_enabled=false`` → skip the keyword tier entirely and
    let every prompt fall through to L2. Useful when an operator wants
    to bisect a misclassification without restarting.
    """
    try:
        import larkhelm.config as _cfg
        return not bool(getattr(_cfg, "INTENT_L1_ENABLED", True))
    except Exception:
        return False


def _resolve_l1(text: str, images: list | None, has_doc_urls: bool) -> IntentResult | None:
    if _l1_disabled():
        return None
    text_l = text.lower()

    # Step 1: negative-pattern blocklist.
    blocked: set[str] = set()
    for nr in all_negative_rules():
        try:
            if nr.match(text, text_l):
                blocked.update(nr.blocks)
        except Exception:
            continue

    # Step 2: score each agent by max-strength matched rule.
    scores: dict[str, float] = {}
    evidence: dict[str, list[str]] = {}
    for rule in all_rules():
        if rule.agent in blocked:
            continue
        try:
            if rule.match(text, text_l):
                if rule.strength > scores.get(rule.agent, 0.0):
                    scores[rule.agent] = rule.strength
                    evidence.setdefault(rule.agent, []).append(
                        str(rule.pattern)[:30]
                    )
        except Exception:
            continue

    # Step 2b: dynamic keyword rules from SkillRegistry and PipelineRegistry.
    # These extend the static rules so user-created Skills/Pipelines are
    # immediately routable without a restart.
    def _score_dynamic_rules(rules: "list[tuple[str, str, float, str]]") -> None:
        for (raw_pattern, agent_id, strength, note) in rules:
            if agent_id in blocked:
                continue
            try:
                if raw_pattern.startswith("re:"):
                    pat = raw_pattern[3:]
                    if not pat:
                        continue
                    compiled = _re_pattern_cache.get(pat)
                    if compiled is None:
                        compiled = re.compile(pat, re.IGNORECASE)
                        _re_pattern_cache[pat] = compiled
                    matched = bool(compiled.search(text))
                else:
                    matched = raw_pattern.lower() in text_l
                if matched and strength > scores.get(agent_id, 0.0):
                    scores[agent_id] = strength
                    evidence.setdefault(agent_id, []).append(raw_pattern[:30])
            except Exception:
                continue

    try:
        from larkhelm.agent_hub.skill_registry import SKILL_REGISTRY
        _score_dynamic_rules(SKILL_REGISTRY.get_l1_rules())
    except Exception:
        pass

    try:
        from larkhelm.agent_hub.pipeline_registry import PIPELINE_REGISTRY
        _score_dynamic_rules(PIPELINE_REGISTRY.get_l1_rules())
    except Exception:
        pass

    # Step 3: doc-URL + write-verb still gets its own promotion (matches
    # the prior _NEW_DOC_OBJECT_RE guarded behaviour) since the URL is a
    # stronger signal than any keyword. Negative rule above already
    # blocks doc when the user's writing object is a NEW doc.
    if has_doc_urls and "doc" not in blocked:
        if any(k in text for k in ("写", "更新", "保存", "覆盖", "追加",
                                    "append", "write", "update")):
            if scores.get("doc", 0.0) < 0.82:
                scores["doc"] = 0.82
                evidence.setdefault("doc", []).append("doc URL + write verb")

    if not scores:
        return None

    # Step 4: pick the highest-scoring agent. On ties, prefer the more
    # specialised agent (doc > plan > crew > dev > chat) — these are the
    # agents that incur the most cost when mis-routed *for* them (e.g.
    # dispatching dev when the user wanted crew burns a pipeline).
    PREF_ORDER = ("doc", "plan", "crew", "dev")
    best_agent = max(
        scores,
        key=lambda a: (scores[a], -PREF_ORDER.index(a) if a in PREF_ORDER else -99),
    )
    best_score = scores[best_agent]

    if best_score < _l1_threshold():
        # Abstain: let L2 disambiguate. Same fall-through as a true miss.
        return None

    complexity = "complex" if len(text) > 60 else "medium"
    return IntentResult(
        agent_type=best_agent,
        layer="L1",
        confidence=best_score,
        complexity=complexity,
        reasoning=f"keywords: {', '.join(evidence[best_agent][:3])}",
        raw_text=text,
    )


def _build_l2_prompt(agent_descriptions: list[tuple[str, str]]) -> str:
    """Few-shot L2 prompt — description + 2 positive + 1 negative example
    per agent. Cheap LLMs (DeepSeek / Kimi) classify on examples
    materially more accurately than on abstract descriptions alone.
    """
    if not agent_descriptions:
        agent_descriptions = [
            ("chat", "普通对话与简单问答"),
            ("dev",  "完整软件开发流水线"),
            ("crew", "多角色调研协作"),
            ("plan", "多阶段开发计划"),
            ("doc",  "飞书文档读写"),
        ]
    blocks: list[str] = []
    for atype, desc in agent_descriptions:
        if not desc:
            continue
        block = [f"- {atype}: {desc[:120]}"]
        shots = AGENT_FEW_SHOTS.get(atype) or ()
        for sign, ex in shots:
            mark = "✓" if sign == "+" else "✗"
            block.append(f"    {mark} {ex[:120]}")
        blocks.append("\n".join(block))
    desc_lines = "\n".join(blocks)
    return (
        "You are an intent classifier. Read the user's message and respond with ONLY a JSON object. "
        "Use this exact schema:\n"
        '{"intent":"chat|dev|crew|plan|doc","complexity":"simple|medium|complex","reasoning":"…"}\n\n'
        f"Available agent_types with examples (✓ matches, ✗ does NOT match):\n{desc_lines}\n\n"
        "Rules:\n"
        "- 'chat' is the default for casual conversation, factual questions, code reading, debugging Q&A.\n"
        "- 'dev' for non-trivial code-writing tasks the user wants the bot to actually do.\n"
        "- 'crew' for research / multi-perspective / brainstorming / code review tasks.\n"
        "- 'plan' when the user explicitly asks for a multi-stage / phased plan with sequential steps.\n"
        "- 'doc' when the user wants to read/write Feishu doc or wiki content (writes back to an URL).\n"
        "- Nouns like '代码实现' / '实现细节' do NOT make a prompt dev — only verb-with-object phrasings do.\n"
        "- A doc URL alone is NOT enough for 'doc' — the user must also want to write back to that URL.\n"
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


def _try_embedding_l2(
    text: str,
    descriptions: list[tuple[str, str]],
) -> IntentResult | None:
    """REQ-03: cosine-based L2 classifier. ``None`` → caller falls back.

    The classifier itself is cheap to instantiate (just a dict copy),
    but the embedding backend may need to load an ONNX model on first
    call, so we resolve it lazily. Subsequent calls with the same
    (descriptions, threshold) re-use the cached classifier and only pay
    the per-query embed cost (see :func:`_resolve_embedding_classifier`).
    """
    try:
        from larkhelm.memory_embedding import get_embedding_backend
        import larkhelm.config as _cfg
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[IntentRouter] embedding strategy import failed: {e}")
        return None
    backend = None
    try:
        backend = get_embedding_backend()
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[IntentRouter] get_embedding_backend failed: {e}")
        return None
    if backend is None:
        return None
    threshold = float(getattr(_cfg, "INTENT_EMBEDDING_THRESHOLD", 0.30) or 0.30)
    classifier = _resolve_embedding_classifier(backend, descriptions, threshold)
    return classifier.classify(text)


def _resolve_l2(text: str) -> IntentResult:
    try:
        from larkhelm.backend_registry import BACKEND_REGISTRY
        from larkhelm.agent_hub.agent_base import AGENT_REGISTRY
        import larkhelm.config as _cfg
    except Exception:
        return _fallback(text)

    descriptions: list[tuple[str, str]] = []
    for atype in AGENT_REGISTRY.list_types():
        ag = AGENT_REGISTRY.get(atype)
        if ag is not None:
            descriptions.append((atype, ag.description or ""))

    strategy = str((getattr(_cfg, "config", {}) or {}).get("intent_layer2_strategy", "llm") or "llm").lower()
    if strategy == "embedding":
        embedded = _try_embedding_l2(text, descriptions)
        if embedded is not None:
            return embedded
        # Fall through to LLM path if embedding declined or unavailable.

    cheap = BACKEND_REGISTRY.get_by_tag(["cheap"])
    if cheap is None:
        return _fallback(text)

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


def _bump_intent_layer(layer: str, outcome: str) -> None:
    """P1-5a: best-effort layer counter bump. Never raises; never blocks
    the classifier. Centralised so unit tests can monkey-patch this single
    symbol rather than the whole metrics module."""
    try:
        from larkhelm.metrics import inc_intent_layer as _inc
        _inc(layer, outcome)
    except Exception:
        pass


def _resolve_microlearn(text: str) -> "IntentResult | None":
    """Phase D-D: feedback-driven LR classifier vote.

    Runs only when ``intent_microlearn_enabled=true`` AND a checkpoint
    exists AND the predicted confidence ≥
    ``intent_microlearn_min_confidence``. Otherwise returns None and the
    caller falls through to L2.

    Wrapped in try/except so any internal failure (no numpy, stale
    sklearn, ONNX glitch) silently collapses to L2 — never breaks
    classification.
    """
    try:
        from larkhelm.agent_hub.intent_microlearn import predict as _ml_predict
        import larkhelm.config as _cfg
    except Exception:
        return None

    if not bool(getattr(_cfg, "INTENT_MICROLEARN_ENABLED", False)):
        return None

    try:
        result = _ml_predict(text)
    except Exception:
        return None
    if result is None:
        return None

    agent, confidence = result
    min_conf = float(getattr(_cfg, "INTENT_MICROLEARN_MIN_CONFIDENCE", 0.65) or 0.65)
    if confidence < min_conf:
        return None

    if agent not in {"chat", "dev", "crew", "plan", "doc", "search"}:
        return None

    complexity = "complex" if len(text) > 60 else "medium"
    return IntentResult(
        agent_type=agent,
        layer="microlearn",
        confidence=float(confidence),
        complexity=complexity,
        reasoning=f"microlearn LR (p={confidence:.2f}, ≥{min_conf:.2f})",
        raw_text=text,
    )


def _maybe_record_l1_gray_zone(
    intent: IntentResult,
    chat_id: str | None,
    text: str,
) -> None:
    """Phase D follow-up: persist an observational sample when L1 fires
    inside the configurable gray band around its promotion threshold.

    Band default: ``[threshold, threshold + 0.10)``. These are the
    keyword-tier decisions that are mathematically "barely confident
    enough" — exactly the sample shape the L1 keyword tuner needs to
    spot weakly-discriminating rules. ``corrected_intent=""`` keeps the
    trainer's _load_feedback skipping these (no gold label).
    """
    if not chat_id:
        return
    if intent.layer != "L1" or intent.confidence <= 0.0:
        return
    try:
        import larkhelm.config as _cfg
        threshold = float(getattr(_cfg, "INTENT_L1_PROMOTION_THRESHOLD", 0.70) or 0.70)
        band = float(getattr(_cfg, "INTENT_FEEDBACK_L1_GRAY_BAND", 0.10) or 0.10)
    except Exception:
        threshold, band = 0.70, 0.10
    if not (threshold <= intent.confidence < threshold + band):
        return
    try:
        from larkhelm.agent_hub.intent_feedback import record_signal
        record_signal(
            "l1_gray_zone", intent, chat_id, text=text,
            metadata={"threshold": threshold, "band": band},
        )
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[IntentRouter] l1_gray_zone signal failed: {e}")


def _maybe_record_l2_dispatched(
    intent: IntentResult,
    chat_id: str | None,
    text: str,
) -> None:
    """Phase D follow-up: every L2 hit is by construction an L1 miss —
    capture them as the training corpus the L1 tuner needs to grow new
    keyword rules. Skip pure ``chat`` fallbacks (those are L2's default
    answer for ambiguous prompts, not informative for L1) so the
    fallback class doesn't drown the signal.
    """
    if not chat_id:
        return
    if intent.layer != "L2":
        return
    if intent.agent_type == "chat":
        return
    try:
        from larkhelm.agent_hub.intent_feedback import record_signal
        record_signal(
            "l2_dispatched", intent, chat_id, text=text,
            metadata={"reasoning": (intent.reasoning or "")[:120]},
        )
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[IntentRouter] l2_dispatched signal failed: {e}")


def resolve_intent(
    text: str,
    images: list | None = None,
    has_doc_urls: bool = False,
    chat_id: str | None = None,
) -> IntentResult:
    """Classify the user's text into an :class:`IntentResult`.

    Order:
      1. Explicit slash command prefix → ``is_explicit_command=True``.
      2. L1 keyword tier with confidence scoring; abstains below
         ``INTENT_L1_PROMOTION_THRESHOLD`` and falls through.
      3. Micro-learn LR classifier (Phase D-D, opt-in via
         ``intent_microlearn_enabled``). Returns only on high confidence;
         otherwise abstains.
      4. L2 cheap-backend classifier (embedding by default, LLM JSON
         fallback). Configured via ``intent_layer2_strategy``.
      5. Fallback ``chat``.
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
        _bump_intent_layer("explicit", "hit")
        return IntentResult(
            agent_type=explicit,
            is_explicit_command=True,
            layer="L1",
            confidence=1.0,
            raw_text=stripped,
        )

    l1_raised = False
    try:
        l1 = _resolve_l1(stripped, images, has_doc_urls)
    except Exception:
        l1 = None
        l1_raised = True
        _bump_intent_layer("l1", "error")
    if l1 is not None:
        _bump_intent_layer("l1", "hit")
        # Fire-and-forget gray-zone capture — never blocks the L1 return.
        _maybe_record_l1_gray_zone(l1, chat_id, stripped)
        return l1
    elif not l1_raised:
        # Distinguish "tried but abstained" (clean) from "raised" (above):
        # error and abstain are mutually exclusive per call.
        _bump_intent_layer("l1", "abstain")

    ml_raised = False
    try:
        ml = _resolve_microlearn(stripped)
    except Exception:
        ml = None
        ml_raised = True
        _bump_intent_layer("microlearn", "error")
    if ml is not None:
        _bump_intent_layer("microlearn", "hit")
        return ml
    elif not ml_raised:
        # Symmetric with L1: count clean abstains so dashboards can see
        # how often microlearn declines vs raises.
        _bump_intent_layer("microlearn", "abstain")

    try:
        l2 = _resolve_l2(stripped)
    except Exception:
        l2 = _fallback(stripped)
        _bump_intent_layer("l2", "error")
    # L2 returns layer="L2" on success, layer="fallback" when none of the
    # cheap-backend / embedding paths produced a usable result. Counting at
    # the call site keeps the metric definition co-located with the
    # decision (rather than spread across 4 `return _fallback(...)` sites
    # inside _resolve_l2).
    if l2.layer == "fallback":
        _bump_intent_layer("fallback", "hit")
        try:
            from larkhelm.metrics import inc_intent_l2_fallback as _inc_fb
            _inc_fb()
        except Exception:
            pass
    else:
        _bump_intent_layer("l2", "hit")
    _maybe_record_l2_dispatched(l2, chat_id, stripped)
    return l2


__all__ = ["resolve_intent", "format_dispatch_badge"]
