"""larkhelm · memory context builder

Phase B (S49–S52) consolidates memory context assembly behind one builder
class. The legacy ``memory.get_memory_context(chat_id, cwd)`` signature is
preserved as a thin forward; new call sites pass keyword arguments
(``query``, ``recent_turns``, ``has_doc_urls``, ``force_*``) so the builder
can apply the lazy-global / project-conditional / session-layered /
recent-turns-dedup optimisations.

Design contract: when ``query==""`` and no ``recent_turns`` / ``force_*``
are provided, the builder MUST emit byte-identical output to the legacy
``get_memory_context`` so existing callers see no behaviour change. All
gating helpers fail-open (return True when in doubt) for the same reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import larkhelm.config as _cfg
from larkhelm.log import _debug_log

if TYPE_CHECKING:
    from larkhelm.memory_slice import RetrievalRequest  # noqa: F401


# ── Per-slot caps for the layered session view (S49) ───────────────────────
WORK_CONTEXT_MAX = 400
HISTORY_MAX = 1200
DECISIONS_MAX = 400


# ── Smart truncation (S42) ─────────────────────────────────────────────────
#
# When a memory layer must be trimmed to fit the combined budget, prefer
# cutting at a semantic boundary (paragraph > sentence > line > word) instead
# of a raw character cut that can mid-chop a Chinese character, a markdown
# fence, or a numbered list item.
#
# Strategy: scan backwards from ``budget`` looking for the latest acceptable
# boundary that's still within a "slack window" (so we don't trim away half
# the budget chasing a paragraph break). Slack defaults to 15% of budget.

_PARAGRAPH_SEP = "\n\n"
_LINE_SEP = "\n"
_SENTENCE_ENDERS = ("。", "！", "？", "；", ". ", "! ", "? ", "; ")


def smart_truncate(text: str, budget: int, *, slack_pct: float = 0.15) -> str:
    """Truncate ``text`` to at most ``budget`` chars, preferring a semantic
    boundary within the last ``slack_pct`` of the budget.

    Returns the input unchanged when ``len(text) <= budget``. Always appends
    ``"…"`` when a trim actually occurs. Falls back to a raw character cut
    only if no boundary exists within the slack window.

    ``budget`` must be > 0; a budget of 0 or negative returns the empty
    string (matches the legacy behaviour of ``text[:0] + "…"`` collapsed).
    """
    if budget <= 0:
        return ""
    if len(text) <= budget:
        return text
    # The truncation slack: only look back this far for a boundary. Without
    # a lower bound a paragraph far below ``budget`` could shrink the output
    # way below target.
    slack = max(1, int(budget * slack_pct))
    floor = max(0, budget - slack)

    # Convention for the ellipsis suffix:
    #   - "\n…"  — used after paragraph / line breaks; the ellipsis renders
    #              on its own line as a "more content" marker.
    #   - "…"    — used after sentence-ending punctuation, word breaks, and
    #              the raw char-cut fallback; inline because the prior char
    #              already provides the visual break (punctuation or
    #              whitespace) or because there's no break at all.
    # Every boundary path .rstrip()s before appending so trailing
    # whitespace from the boundary itself doesn't show up before "…".

    # 1. Paragraph break — strongest semantic boundary.
    idx = text.rfind(_PARAGRAPH_SEP, floor, budget)
    if idx >= floor:
        return text[:idx].rstrip() + "\n…"

    # 2. Sentence-ending punctuation (Chinese full-width + ASCII with
    #    trailing space; bare ASCII "." is too ambiguous — version numbers,
    #    abbreviations — so we require the trailing space).
    best = -1
    for ender in _SENTENCE_ENDERS:
        i = text.rfind(ender, floor, budget)
        if i > best:
            best = i + len(ender)
    if best > floor:
        return text[:best].rstrip() + "…"

    # 3. Line break.
    idx = text.rfind(_LINE_SEP, floor, budget)
    if idx >= floor:
        return text[:idx].rstrip() + "\n…"

    # 4. Word break (ASCII): last whitespace within slack. CJK doesn't use
    #    whitespace word separators so this primarily helps English text.
    idx = max(text.rfind(" ", floor, budget),
              text.rfind("\t", floor, budget))
    if idx >= floor:
        return text[:idx].rstrip() + "…"

    # 5. Fallback: raw character cut. This may split a multi-byte UTF-8
    #    sequence mid-codepoint, but since Python strings are Unicode
    #    (not bytes) the cut is always on a codepoint boundary; the
    #    cosmetic hazard is splitting a grapheme cluster (e.g. emoji
    #    + skin-tone modifier). Accept this — the alternative is
    #    chasing far below the budget. No rstrip here: a raw cut may
    #    legitimately land inside a word.
    return text[:budget] + "…"


# ── Lazy-global / project-conditional gating (S50 / S51) ────────────────────

_GLOBAL_INJECT_KEYWORDS: tuple[str, ...] = (
    "偏好", "语言", "回复", "格式", "风格", "记忆", "安全", "你应该", "我希望",
    "/memory", "memory", "preference", "language", "style", "behaviour",
    "behavior",
)

_PROJECT_INJECT_KEYWORDS: tuple[str, ...] = (
    "代码", "测试", "实现", "重构", "修复", "项目", "架构",
    "bug", "fix", "function", "class", "import", "test",
    "build", "deploy", "config", "module", "refactor",
    "/dev", "/crew", "/plan",
)

_PROJECT_FORCE_INJECT_PREFIX: tuple[str, ...] = ("/dev", "/crew", "/plan")

# Path-like fragments: ``foo/bar.py`` / ``some_file.json`` / ``/etc/x``.
# Used to spot "code-flavoured" queries that lack the explicit keywords above.
_PATH_RE = re.compile(
    r"(?:[\w\-./]+\.(?:py|md|json|toml|sh|ts|tsx|js|yml|yaml|go|rs|java|cpp|c|h)\b"
    r"|/[\w\-/.]+)"
)

# Triple-backtick code fence is a strong signal that the query is about code.
_CODE_FENCE_RE = re.compile(r"```")


# ── Session slot data class ────────────────────────────────────────────────

@dataclass
class SessionSlots:
    """Three-section split of a session memory body.

    ``parsed=False`` means the H2 layout was not detected (LLM output drift,
    legacy summary, manual edit) — in that case ``raw`` is the full body and
    callers should fall back to the legacy single-block behaviour.
    """
    work_context: str = ""
    history: str = ""
    decisions: str = ""
    raw: str = ""
    parsed: bool = False

    def total_chars(self) -> int:
        return len(self.work_context) + len(self.history) + len(self.decisions)


# ── Pure helpers (testable independently) ──────────────────────────────────


def _config_flag(key: str, default: bool = True) -> bool:
    cfg = getattr(_cfg, "config", None) or {}
    return bool(cfg.get(key, default))


def should_include_global(query: str, *, force: bool = False) -> bool:
    """Return True iff the global memory layer should be injected.

    fail-open contract: empty query / disabled flag / lookup failure → True.
    The keyword set is intentionally broad so users rarely lose preferences
    silently; the rare false-positive injection costs ~800 tokens whereas a
    missed preference can cost user trust.
    """
    if force:
        return True
    if not _config_flag("memory_lazy_global", True):
        return True
    if not query or not query.strip():
        return True
    q = query.lower()
    for kw in _GLOBAL_INJECT_KEYWORDS:
        if kw in q:
            return True
    return False


def should_include_project(
    query: str,
    cwd: str | None,
    has_doc_urls: bool,
    *,
    force: bool = False,
) -> bool:
    """Return True iff the project memory layer should be injected.

    Mandatory pre-condition: ``cwd`` non-empty (project memory is keyed by
    cwd). Then: explicit force, /dev|/crew|/plan prefix, doc-URL injection,
    keyword match, path-like fragment, or code-fence presence — any one
    triggers inclusion. fail-open on empty query.
    """
    if force:
        return True
    if not cwd:
        return False
    if not _config_flag("memory_project_conditional", True):
        return True
    if not query:
        return True
    q = query.strip()
    if not q:
        return True
    for prefix in _PROJECT_FORCE_INJECT_PREFIX:
        if q.startswith(prefix):
            return True
    if has_doc_urls:
        return True
    ql = q.lower()
    for kw in _PROJECT_INJECT_KEYWORDS:
        if kw in ql:
            return True
    if _PATH_RE.search(q):
        return True
    if _CODE_FENCE_RE.search(q):
        return True
    return False


_SECTION_RE = re.compile(
    r"^##\s+(.+?)\s*$",
    re.MULTILINE,
)


def split_session_slots(raw: str) -> SessionSlots:
    """Split a session-memory body into its three canonical H2 sections.

    Recognises ``## Work Context`` / ``## Key Decisions & Facts`` /
    ``## Next Steps`` (case-insensitive, trailing punctuation tolerated).
    On any parse miss returns ``parsed=False`` with the full body in
    ``work_context`` so the caller's downstream slicing degrades to the
    legacy "use everything" behaviour.
    """
    if not raw:
        return SessionSlots(raw="", parsed=False)

    lines = raw.splitlines()
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    current_buf: list[str] = []

    def _classify(header: str) -> str | None:
        h = header.strip().lower().rstrip(":：")
        # The summariser prompt (memory.py:139-146) tells the LLM to emit
        # `## Work Context` / `## Key Decisions & Facts` / `## Next Steps`
        # "in the SAME LANGUAGE as the conversation", so Chinese sessions
        # naturally produce `## 工作上下文` / `## 关键决策` / `## 后续步骤`
        # (or `## 下一步`). Each branch lists both English and Chinese
        # idiomatic variants. The "context" / "上下文" check is wrapped in
        # an explicit AND group because Python parses `A or B and C` as
        # `A or (B and C)` — the original v1 fix missed this.
        if "work context" in h or ("工作" in h and ("context" in h or "上下文" in h)):
            return "work_context"
        if "decision" in h or "fact" in h or "决策" in h or "事实" in h:
            return "decisions"
        # "Next Steps" in Chinese: 后续 / 下一步 / 步骤 / 进展。 Latin variants
        # cover "next", "next step", and "history" (legacy summariser label).
        if ("history" in h or "next step" in h or "next" in h
                or "后续" in h or "下一步" in h or "步骤" in h or "进展" in h):
            return "history"
        return None

    for line in lines:
        m = _SECTION_RE.match(line)
        if m:
            if current_key is not None:
                sections[current_key] = current_buf
            current_key = _classify(m.group(1))
            current_buf = []
        else:
            if current_key is not None:
                current_buf.append(line)
    if current_key is not None:
        sections[current_key] = current_buf

    if not sections:
        return SessionSlots(work_context=raw.strip(), raw=raw, parsed=False)

    def _join(key: str, cap: int) -> str:
        body = "\n".join(sections.get(key, [])).strip()
        if len(body) > cap:
            body = body[:cap].rstrip() + "…"
        return body

    return SessionSlots(
        work_context=_join("work_context", WORK_CONTEXT_MAX),
        history=_join("history", HISTORY_MAX),
        decisions=_join("decisions", DECISIONS_MAX),
        raw=raw,
        parsed=True,
    )


def extract_work_context(raw: str | None) -> str:
    """Return ``slots.work_context`` from a parsed session body.

    Returns the empty string when ``raw`` is empty, when ``split_session_slots``
    failed to parse the H2 layout (``parsed=False``), or when the
    work_context slot itself is blank. Never raises — the caller treats the
    empty string as "no dedup prefix available, fall back to legacy path".
    """
    if not raw:
        return ""
    try:
        slots = split_session_slots(raw)
    except Exception:
        return ""
    if not slots.parsed:
        return ""
    return slots.work_context or ""


def dedup_recent_turns(recent: list[str], session_body: str) -> list[str]:
    """Drop ``recent`` entries already represented in ``session_body``.

    Heuristic: split each recent line into its message body (after the
    timestamp + role prefix) and check whether a non-trivial prefix of that
    body appears verbatim in ``session_body``. Conservative — only drops
    entries with a clear textual overlap so we don't accidentally hide
    new turns that merely *mention* something the session covers.
    """
    if not recent or not session_body:
        return list(recent or [])
    if not _config_flag("memory_recent_turns_dedup", True):
        return list(recent)
    out: list[str] = []
    sb = session_body
    for line in recent:
        # Recent-turn lines historically look like ``[12:00] user: hello there``
        # — strip the prefix down to the message body before comparing.
        body = line
        if "]" in line:
            body = line.split("]", 1)[1].strip()
        if ":" in body:
            body = body.split(":", 1)[1].strip()
        # Compare a reasonably long prefix to avoid false positives on tiny strings.
        probe = body[:60].strip()
        if probe and len(probe) >= 10 and probe in sb:
            continue
        out.append(line)
    return out


# ── Builder ────────────────────────────────────────────────────────────────


class MemoryContextBuilder:
    """Compose the memory context string for an outgoing query.

    Layer order (when included): global → project → session. Each layer is
    gated by a per-layer ``should_include_*`` predicate; the session layer
    is always loaded but may be sliced into Work-Context-only when the
    layered config flag is on AND the body parsed cleanly into sections.
    """

    def __init__(
        self,
        chat_id: str,
        cwd: str | None = None,
        *,
        query: str = "",
        recent_turns: list[str] | None = None,
        has_doc_urls: bool = False,
        force_project: bool = False,
        force_global: bool = False,
        # Phase D: intent-aware injection. Defaults keep legacy callers byte-
        # identical because ``build()`` only enters the retriever path when
        # ``memory_retriever_enabled`` AND an upstream caller signalled intent.
        agent_type: str = "chat",
        sub_intent: str = "",
        complexity: str = "medium",
        confidence: float = 0.0,
        sender_open_id: str | None = None,
        # Week-2: optional backend hint for backend-aware context budget.
        backend_spec=None,
    ):
        self.chat_id = chat_id
        self.cwd = cwd
        self.query = query or ""
        self.recent_turns = list(recent_turns or [])
        self.has_doc_urls = has_doc_urls
        self.force_project = force_project
        self.force_global = force_global
        self.agent_type = agent_type or "chat"
        self.sub_intent = sub_intent or ""
        self.complexity = complexity or "medium"
        self.confidence = float(confidence or 0.0)
        self.sender_open_id: str | None = sender_open_id or None
        self.backend_spec = backend_spec

    # ── public entry points ────────────────────────────────────────────

    def build(self) -> str:
        """Top-level dispatch between legacy v2 and Phase D retriever path.

        - When the retriever flag is off OR traffic-split says skip OR the
          retriever raises → fall back to ``_build_legacy_v2`` (byte-identical
          to the master code path so existing tests stay green).
        - Otherwise consult ``memory_retriever`` for an intent-shaped context."""
        try:
            from larkhelm.memory_retriever import _retriever_active
        except Exception as e:
            _debug_log(f"[MemoryRetriever] import failed, using v2: {e}")
            return self._build_legacy_v2()

        try:
            active = _retriever_active(self.chat_id)
        except Exception as e:
            _debug_log(f"[MemoryRetriever] _retriever_active failed: {e}")
            active = False
        if not active:
            return self._build_legacy_v2()

        try:
            return self._build_with_retriever()
        except Exception as e:
            _debug_log(f"[MemoryRetriever] fail-open to v2: {e}")
            return self._build_legacy_v2()

    def _build_legacy_v2(self) -> str:
        """Full context: global (if relevant) + project (if relevant) + session."""
        from larkhelm.memory import (
            GLOBAL_MAX_CHARS, PROJECT_MAX_CHARS, SESSION_MAX_CHARS,
            TOTAL_MEMORY_BUDGET, _TAG_OVERHEAD_PER_LAYER,
        )

        parts: list[tuple[str, str, str, int]] = []  # (open, content, close, max_chars)

        if self._should_include_global():
            g = self._layer_global()
            if g:
                parts.append(("[GLOBAL MEMORY]", g, "[/GLOBAL MEMORY]", GLOBAL_MAX_CHARS))

        if self._should_include_project():
            p = self._layer_project()
            if p:
                parts.append(
                    (f"[PROJECT MEMORY — {self.cwd}]", p, "[/PROJECT MEMORY]", PROJECT_MAX_CHARS)
                )

        s = self._layer_session()
        if s:
            parts.append(("[SESSION MEMORY]", s, "[/SESSION MEMORY]", SESSION_MAX_CHARS))

        if not parts:
            return ""

        total = 0
        content_total = 0
        for _, c, _, _ in parts:
            total += len(c) + _TAG_OVERHEAD_PER_LAYER
            content_total += len(c)
        if total > TOTAL_MEMORY_BUDGET:
            available = max(0, TOTAL_MEMORY_BUDGET - _TAG_OVERHEAD_PER_LAYER * len(parts))
            if content_total > 0:
                _debug_log(
                    f"[Memory] budget trim: total={total} > {TOTAL_MEMORY_BUDGET}, "
                    f"available={available}"
                )
                # Session layer gets a minimum floor so recency signal survives heavy
                # global/project layers. We compute session's budget first, then give
                # non-session layers the remainder — this keeps total ≤ available.
                _SESSION_MIN = 800
                session_len = next(
                    (len(c) for o, c, _, _ in parts if o == "[SESSION MEMORY]"), 0
                )
                if session_len > 0:
                    session_proportional = int(available * session_len / content_total)
                    session_budget = max(_SESSION_MIN, session_proportional)
                else:
                    session_budget = 0
                non_session_available = max(0, available - session_budget)
                non_session_total = content_total - session_len
                trimmed: list[tuple[str, str, str, int]] = []
                for open_tag, content, close_tag, max_c in parts:
                    if open_tag == "[SESSION MEMORY]":
                        budget_i = session_budget
                    elif non_session_total > 0:
                        budget_i = int(non_session_available * len(content) / non_session_total)
                    else:
                        budget_i = 0
                    # S42: prefer a semantic boundary (paragraph > sentence >
                    # line > word) over a raw char-cut that can split a
                    # Chinese sentence mid-character or chop a markdown fence.
                    content = smart_truncate(content, budget_i)
                    trimmed.append((open_tag, content, close_tag, max_c))
                parts = trimmed

        # P5-OPT1: meter line removed from injection path — it sat on the
        # second line of every layer and rotated whenever session_n changed by
        # a single char, busting the Anthropic prompt-cache prefix for the
        # entire system prompt. The meter is still exposed by
        # ``_layer_meter_line`` for the ``/memory observe`` card path.
        return "\n\n".join(
            f"{o}\n{c}\n{cl}"
            for o, c, cl, _ in parts
        )

    def _build_request(self) -> "RetrievalRequest":
        """Assemble a :class:`RetrievalRequest` from this builder's state."""
        from larkhelm.memory_slice import RetrievalRequest
        has_code_fence = bool(_CODE_FENCE_RE.search(self.query))
        return RetrievalRequest(
            chat_id=self.chat_id,
            cwd=self.cwd,
            query=self.query,
            recent_turns=tuple(self.recent_turns),
            agent_type=self.agent_type,
            sub_intent=self.sub_intent,
            complexity=self.complexity,
            confidence=self.confidence,
            is_explicit_cmd=False,
            has_doc_urls=self.has_doc_urls,
            has_images=False,
            has_code_fence=has_code_fence,
        )

    def _build_with_retriever(self) -> str:
        """Phase D retriever path: load slices → score → compose.

        Phase 2: the retriever is selected by :func:`resolve_actual_mode`
        and the audit record uses v2 schema. Any failure inside the
        embedding/hybrid path falls open to :class:`KeywordRetriever` with
        ``audit.fail_open=True``.
        """
        import time as _time
        from larkhelm.memory_retriever import (
            KeywordRetriever,
            _audit_decision,
            build_audit_record_v2,
            compose_slices_to_context,
            get_policy,
            get_retriever,
            load_slices,
            resolve_actual_mode,
        )
        from larkhelm.memory_embedding import get_embedding_backend

        t0 = _time.perf_counter()
        request = self._build_request()
        policy = get_policy(self.agent_type, backend_spec=self.backend_spec)
        cfg = getattr(_cfg, "config", {}) or {}

        # Mode resolution: physical mode that will actually dispatch.
        declared_mode = policy.retrieval_mode
        try:
            actual_mode = resolve_actual_mode(policy, self.chat_id, cfg)
        except Exception as e:
            _debug_log(f"[MemoryRetriever] resolve_actual_mode failed: {e}")
            actual_mode = "keyword"

        backend = None
        if actual_mode in ("embedding", "hybrid"):
            try:
                backend = get_embedding_backend(cfg)
            except Exception as e:
                _debug_log(f"[MemoryRetriever] get_embedding_backend failed: {e}")
                backend = None
            if backend is None:
                # Couldn't materialise a backend — degrade to keyword.
                actual_mode = "keyword"

        slices = load_slices(self.chat_id, self.cwd)
        fail_open = False
        llm_router_diag = None
        # Phase 3: optionally wrap the underlying retriever with the
        # LLM router. ``_should_wrap_with_llm_router`` enforces the
        # per-chat / per-agent / per-complexity gating; when it returns
        # True we construct an ``LLMRouterRetriever`` over the resolved
        # underlying retriever. The decorator pattern keeps the audit
        # ``mode`` field reporting the *data source* (keyword/hybrid),
        # while the new ``llm_router_*`` audit fields record the
        # decoration's behaviour (invoked / cache_hit / skipped_reason).
        try:
            from larkhelm.memory_retriever import _should_wrap_with_llm_router
            wrap_with_llm = _should_wrap_with_llm_router(
                request, policy, self.chat_id, cfg,
            )
        except Exception as e:
            _debug_log(
                f"[MemoryRetriever] _should_wrap_with_llm_router failed (treating as off): {e}"
            )
            wrap_with_llm = False

        try:
            underlying = get_retriever(actual_mode, backend=backend)
            if wrap_with_llm:
                from larkhelm.memory_llm_router import LLMRouterRetriever
                router = LLMRouterRetriever(underlying)
                scored = router.retrieve(request, policy, slices)
                llm_router_diag = router.diagnostics
            else:
                scored = underlying.retrieve(request, policy, slices)
        except Exception as e:
            _debug_log(
                f"[MemoryRetriever] {actual_mode} retriever failed (fail-open to keyword): {e}"
            )
            fail_open = True
            actual_mode = "keyword"
            scored = KeywordRetriever().retrieve(request, policy, slices)
            # Preserve the gate-fired signal even when the underlying
            # retriever raised inside the LLMRouter wrap — otherwise the
            # audit summary undercounts Stage C activity (review SF-01
            # round-2). We synthesise a diag that records "the gate did
            # fire, but the underlying blew up".
            if wrap_with_llm:
                from larkhelm.memory_llm_router import RouterDiagnostics
                llm_router_diag = RouterDiagnostics(
                    invoked=False, cache_hit=False,
                    skipped_reason="underlying_failure",
                    elapsed_ms=0, selected_by_llm=0,
                )
            else:
                llm_router_diag = None

        composed = compose_slices_to_context(scored, policy, cwd=self.cwd)
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        try:
            _audit_decision(build_audit_record_v2(
                request=request,
                policy=policy,
                scored=scored,
                candidate_count=len(slices),
                elapsed_ms=elapsed_ms,
                selected_chars=len(composed),
                fail_open=fail_open,
                actual_mode=actual_mode,
                declared_mode=declared_mode,
                llm_router_diag=llm_router_diag,
            ))
        except Exception as e:
            _debug_log(f"[MemoryRetriever] audit enqueue failed: {e}")
        return composed

    def build_for_crew(self) -> str:
        """Project + session only; mirrors the legacy
        ``get_project_memory_context`` shape (no per-layer meter, no global).
        """
        from larkhelm.memory import load_memory, load_project_memory

        chunks: list[str] = []
        if self.cwd:
            try:
                p = load_project_memory(self.cwd)
            except Exception as e:
                _debug_log(f"[Memory] build_for_crew load_project failed: {e}")
                p = None
            if p:
                chunks.append(f"[PROJECT MEMORY — {self.cwd}]\n{p}\n[/PROJECT MEMORY]")
        try:
            s = load_memory(self.chat_id)
        except Exception as e:
            _debug_log(f"[Memory] build_for_crew load_session failed: {e}")
            s = None
        if s:
            chunks.append(f"[SESSION MEMORY]\n{s}\n[/SESSION MEMORY]")
        return "\n\n".join(chunks) if chunks else ""

    def build_for_prompt_cache(self) -> tuple[str, str]:
        """Split memory context into (stable, volatile) for layered Anthropic caching.

        stable  = global + project memory layers (tagged) — changes only when LLM updates memory
        volatile = session memory layer (tagged) — changes every ~10 turns

        Returns (stable, volatile). Either element may be empty string.
        Reuses _layer_global() / _layer_project() / _layer_session() LRU cache paths.
        Never triggers LLM calls or new I/O beyond what build() already does.
        """
        stable_parts: list[str] = []
        volatile_parts: list[str] = []

        if self._should_include_global():
            g = self._layer_global()
            if g:
                stable_parts.append(f"[GLOBAL MEMORY]\n{g}\n[/GLOBAL MEMORY]")

        if self._should_include_project():
            p = self._layer_project()
            if p:
                stable_parts.append(
                    f"[PROJECT MEMORY — {self.cwd}]\n{p}\n[/PROJECT MEMORY]"
                )

        s = self._layer_session()
        if s:
            volatile_parts.append(f"[SESSION MEMORY]\n{s}\n[/SESSION MEMORY]")

        stable = "\n\n".join(stable_parts)
        volatile = "\n\n".join(volatile_parts)
        return stable, volatile

    def deduped_recent_turns(self, session_body: str | None = None) -> list[str]:
        if session_body is None:
            try:
                from larkhelm.memory import load_memory
                session_body = load_memory(self.chat_id) or ""
            except Exception:
                session_body = ""
        return dedup_recent_turns(self.recent_turns, session_body)

    # ── internals ──────────────────────────────────────────────────────

    def _should_include_global(self) -> bool:
        try:
            return should_include_global(self.query, force=self.force_global)
        except Exception as e:
            _debug_log(f"[Memory] should_include_global failed (fail-open): {e}")
            return True

    def _should_include_project(self) -> bool:
        try:
            return should_include_project(
                self.query, self.cwd, self.has_doc_urls,
                force=self.force_project,
            )
        except Exception as e:
            _debug_log(f"[Memory] should_include_project failed (fail-open): {e}")
            return True

    # ── Layer entry points ──────────────────────────────────────────────
    # Each ``_layer_*`` is a thin cache-aware shell over the matching
    # ``_layer_*_uncached`` body. When ``MEMORY_LEGACY_CACHE_ENABLED`` is
    # false (or the cache module fails to import), the shell drops straight
    # through to the uncached function — preserving PR-prior behaviour.

    def _legacy_cache_on(self) -> bool:
        return bool(getattr(_cfg, "MEMORY_LEGACY_CACHE_ENABLED", True))

    def _layer_global(self) -> str | None:
        if not self._legacy_cache_on():
            return self._layer_global_uncached()
        try:
            from larkhelm._context_cache import cached_memory_layer
            from larkhelm.memory import _global_memory_file
            path = _global_memory_file(self.chat_id, sender_open_id=self.sender_open_id)
        except Exception:
            return self._layer_global_uncached()
        return cached_memory_layer(
            "global", path,
            loader=lambda: self._layer_global_uncached(),
        )

    def _layer_project(self) -> str | None:
        if not self.cwd:
            return None
        if not self._legacy_cache_on():
            return self._layer_project_uncached()
        try:
            from larkhelm._context_cache import cached_memory_layer
            from larkhelm.memory import _project_memory_file
            # ``self.cwd`` already proven non-empty by the early-return
            # above; the previous ``if self.cwd else None`` was a dead
            # branch (reviewer round-1 nit #1).
            path = _project_memory_file(self.cwd)
        except Exception:
            return self._layer_project_uncached()
        return cached_memory_layer(
            "project", path,
            loader=lambda: self._layer_project_uncached(),
        )

    def _layer_session(self) -> str | None:
        if not self._legacy_cache_on():
            return self._layer_session_uncached()
        try:
            from larkhelm._context_cache import cached_memory_layer
            from larkhelm.memory import _session_memory_file
            path = _session_memory_file(self.chat_id)
        except Exception:
            return self._layer_session_uncached()
        return cached_memory_layer(
            "session", path,
            loader=lambda: self._layer_session_uncached(),
        )

    # ── Layer bodies (cache-free implementations) ───────────────────────

    def _layer_global_uncached(self) -> str | None:
        # P2 REQ-05.1: when the slot flag is on AND the file parses into
        # one or more non-empty slots, render via memory_global_slots so
        # the LLM prompt sees structured headings. Empty-slot dict + flag
        # on still falls back to the legacy free-form body so a not-yet-
        # migrated chat doesn't suddenly inject empty memory.
        try:
            from larkhelm import memory_global_slots as _mgs
            if _mgs.is_enabled():
                rendered = self._cached_global_slots_rendered()
                if rendered:
                    return rendered
        except Exception as e:
            _debug_log(f"[Memory] _layer_global slot path failed (falling back): {e}")
        try:
            from larkhelm.memory import load_global_memory
            return load_global_memory(self.chat_id, sender_open_id=self.sender_open_id)
        except Exception as e:
            _debug_log(f"[Memory] _layer_global failed: {e}")
            return None

    def _cached_global_slots_rendered(self) -> str:
        """Slot parsing + render result, cached by file mtime.

        The global_slots ``.md`` file is keyed on the SAME path used by
        the free-form body, so an independent cache layer would collide.
        We disambiguate by passing layer name ``"global_slots"``; the
        cache key embeds layer + path so the two views coexist cleanly.
        """
        from larkhelm import memory_global_slots as _mgs
        if not self._legacy_cache_on():
            slots = _mgs.load_global_slots(self.chat_id, sender_open_id=self.sender_open_id)
            return _mgs.render_for_context(slots)
        try:
            from larkhelm._context_cache import cached_memory_layer
            from larkhelm.memory import _global_memory_file
            path = _global_memory_file(self.chat_id, sender_open_id=self.sender_open_id)
        except Exception:
            slots = _mgs.load_global_slots(self.chat_id, sender_open_id=self.sender_open_id)
            return _mgs.render_for_context(slots)
        rendered = cached_memory_layer(
            "global_slots", path,
            loader=lambda: _mgs.render_for_context(
                _mgs.load_global_slots(self.chat_id, sender_open_id=self.sender_open_id)
            ),
        )
        return rendered or ""

    def _layer_project_uncached(self) -> str | None:
        if not self.cwd:
            return None
        # P2 REQ-05.2: section-rendered body when the flag is on AND at
        # least one section is populated; otherwise fall through to the
        # legacy load_project_memory (which preserves cwd-mismatch
        # checks).
        try:
            from larkhelm import memory_project_sections as _mps
            if _mps.is_enabled():
                rendered = self._cached_project_sections_rendered()
                if rendered:
                    return rendered
        except Exception as e:
            _debug_log(f"[Memory] _layer_project section path failed (falling back): {e}")
        try:
            from larkhelm.memory import load_project_memory
            return load_project_memory(self.cwd)
        except Exception as e:
            _debug_log(f"[Memory] _layer_project failed: {e}")
            return None

    def _cached_project_sections_rendered(self) -> str:
        from larkhelm import memory_project_sections as _mps
        if not self.cwd:
            return ""
        if not self._legacy_cache_on():
            sections = _mps.load_project_sections(self.cwd)
            return _mps.render_for_context(sections)
        try:
            from larkhelm._context_cache import cached_memory_layer
            from larkhelm.memory import _project_memory_file
            path = _project_memory_file(self.cwd)
        except Exception:
            sections = _mps.load_project_sections(self.cwd)
            return _mps.render_for_context(sections)
        rendered = cached_memory_layer(
            "project_sections", path,
            loader=lambda: _mps.render_for_context(
                _mps.load_project_sections(self.cwd)
            ),
        )
        return rendered or ""

    def _layer_session_uncached(self) -> str | None:
        try:
            from larkhelm.memory import load_memory, load_session_anchor
            raw = load_memory(self.chat_id)
        except Exception as e:
            _debug_log(f"[Memory] _layer_session failed: {e}")
            return None
        try:
            anchor = load_session_anchor(self.chat_id)
            if anchor:
                raw = f"[Session Anchor: {anchor}]\n\n{raw or ''}"
        except Exception as e:
            _debug_log(f"[Memory] load_session_anchor failed: {e}")
        if not raw:
            return None
        if not _config_flag("memory_session_layered", True):
            return raw
        slots = split_session_slots(raw)
        if not slots.parsed:
            return raw
        # P1-6: when memory_session_layer_smart=True (default), each
        # parsed section is independently truncated by smart_truncate
        # according to SESSION_LAYER_BUDGETS. The legacy fixed-cap
        # behaviour is preserved when the flag is off.
        if _config_flag("memory_session_layer_smart", True):
            return _layer_session_smart(
                slots, self.query,
                force_project=self.force_project,
                force_global=self.force_global,
            )
        # Lean view: Work Context always; Decisions only when query mentions
        # decision-flavoured words (or when forced). History is kept when the
        # body fits well under the cap.
        sections: list[str] = []
        if slots.work_context:
            sections.append("## Work Context\n" + slots.work_context)
        ql = self.query.lower()
        decision_kw = ("decision", "decid", "决定", "决策", "选择", "应该", "should",
                       "why", "为什么")
        if slots.decisions and (any(k in ql for k in decision_kw)
                                or self.force_project or self.force_global):
            sections.append("## Key Decisions & Facts\n" + slots.decisions)
        if slots.history:
            sections.append("## Next Steps\n" + slots.history)
        if not sections:
            return raw
        return "\n\n".join(sections)


# ── P1-6 section-wise smart truncation ────────────────────────────────────

# Priority list used when the total budget is blown — sections lower in the
# list are dropped first so work_context survives longest.
_SESSION_LAYER_PRIORITY: tuple[str, ...] = ("work_context", "decisions", "history")


def _resolve_session_budgets(budgets: "dict[str, int] | None" = None) -> dict[str, int]:
    """Pick the budgets to use: explicit > _cfg.SESSION_LAYER_BUDGETS > defaults."""
    if budgets:
        return {
            "work_context": int(budgets.get("work_context", 1200)),
            "decisions":    int(budgets.get("decisions", 800)),
            "history":      int(budgets.get("history", 600)),
        }
    cfg = getattr(_cfg, "SESSION_LAYER_BUDGETS", None)
    if isinstance(cfg, dict):
        return {
            "work_context": int(cfg.get("work_context", 1200)),
            "decisions":    int(cfg.get("decisions", 800)),
            "history":      int(cfg.get("history", 600)),
        }
    return {"work_context": 1200, "decisions": 800, "history": 600}


def _layer_session_smart(
    slots: "SessionSlots",
    query: str,
    force_project: bool = False,
    force_global: bool = False,
    budgets: "dict[str, int] | None" = None,
) -> str:
    """Independent smart_truncate per section, with priority degradation.

    Each parsed section is truncated to its own budget via
    :func:`smart_truncate` so the cut hits a paragraph / sentence boundary
    instead of mid-character. The decision section is opt-in (mirrors the
    legacy behaviour) — included only when the query has decision-flavoured
    words or the caller forces it.

    Priority degradation: if the combined truncated output overflows the
    sum of all three budgets (rare, but possible when one section is short
    and another consumes its slack), drop sections in reverse priority
    (``history`` → ``decisions`` → ``work_context``) until the total fits.
    """
    if not slots or not slots.parsed:
        return slots.raw if slots else ""

    # P2 REQ-07: when the smart-compress flag is on, pre-trim each section
    # via smart_compress(query=...) BEFORE the existing smart_truncate
    # priority degradation. smart_compress is deterministic and LLM-free;
    # smart_truncate then enforces the final byte budget. Flag off → both
    # paths fall through to the legacy truncation only.
    try:
        from larkhelm import memory_session_compress as _msc
        if _msc.is_enabled():
            new_work = _msc.smart_compress(slots.work_context or "",
                                           _resolve_session_budgets(budgets)["work_context"],
                                           query)
            new_dec  = _msc.smart_compress(slots.decisions or "",
                                           _resolve_session_budgets(budgets)["decisions"],
                                           query)
            new_hist = _msc.smart_compress(slots.history or "",
                                           _resolve_session_budgets(budgets)["history"],
                                           query)
            # Re-pack into a fresh SessionSlots so downstream code keeps
            # the same dataclass shape.
            slots = SessionSlots(
                raw=slots.raw, parsed=True,
                work_context=new_work,
                decisions=new_dec,
                history=new_hist,
            )
    except Exception as e:
        _debug_log(f"[Memory] smart_compress pre-trim failed (falling back): {e}")

    b = _resolve_session_budgets(budgets)
    # +3 per section to allow for the "…" / "\n…" ellipsis appended by
    # smart_truncate when a section is at or just over its budget.
    total_budget = b["work_context"] + b["decisions"] + b["history"] + 9

    ql = (query or "").lower()
    decision_kw = (
        "decision", "decid", "决定", "决策", "选择", "应该", "should",
        "why", "为什么",
    )
    include_decisions = bool(slots.decisions) and (
        any(k in ql for k in decision_kw) or force_project or force_global
    )

    trimmed: dict[str, str] = {}
    if slots.work_context:
        trimmed["work_context"] = smart_truncate(slots.work_context, b["work_context"])
    if include_decisions:
        trimmed["decisions"] = smart_truncate(slots.decisions, b["decisions"])
    if slots.history:
        trimmed["history"] = smart_truncate(slots.history, b["history"])

    # Priority degradation: drop low-priority sections until we fit total_budget.
    # work_context first, decisions second, history last in _SESSION_LAYER_PRIORITY;
    # drop in reverse.
    def _current_total() -> int:
        return sum(len(v) for v in trimmed.values())

    for key in reversed(_SESSION_LAYER_PRIORITY):
        if _current_total() <= total_budget:
            break
        if key in trimmed:
            trimmed.pop(key, None)

    sections: list[str] = []
    if "work_context" in trimmed and trimmed["work_context"]:
        sections.append("## Work Context\n" + trimmed["work_context"])
    if "decisions" in trimmed and trimmed["decisions"]:
        sections.append("## Key Decisions & Facts\n" + trimmed["decisions"])
    if "history" in trimmed and trimmed["history"]:
        sections.append("## Next Steps\n" + trimmed["history"])

    if not sections:
        return slots.raw
    return "\n\n".join(sections)
