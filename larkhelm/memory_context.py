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
    pass


# ── Per-slot caps for the layered session view (S49) ───────────────────────
WORK_CONTEXT_MAX = 400
HISTORY_MAX = 1200
DECISIONS_MAX = 400


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
        # Parenthesise the OR/AND group: previously
        # `"work context" in h or "工作" in h and "context" in h`
        # parsed as `A or (B and C)`, so a Chinese header like "## 工作上下文"
        # (no English "context") fell through to None and lost the section.
        if "work context" in h or ("工作" in h and ("context" in h or "上下文" in h)):
            return "work_context"
        if "decision" in h or "fact" in h or "决策" in h:
            return "decisions"
        if "history" in h or "next step" in h or "进展" in h or "next" in h:
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
    ):
        self.chat_id = chat_id
        self.cwd = cwd
        self.query = query or ""
        self.recent_turns = list(recent_turns or [])
        self.has_doc_urls = has_doc_urls
        self.force_project = force_project
        self.force_global = force_global

    # ── public entry points ────────────────────────────────────────────

    def build(self) -> str:
        """Full context: global (if relevant) + project (if relevant) + session."""
        from larkhelm.memory import (
            GLOBAL_MAX_CHARS, PROJECT_MAX_CHARS, SESSION_MAX_CHARS,
            TOTAL_MEMORY_BUDGET, _TAG_OVERHEAD_PER_LAYER, _layer_meter_line,
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

        total = sum(len(c) + _TAG_OVERHEAD_PER_LAYER for _, c, _, _ in parts)
        if total > TOTAL_MEMORY_BUDGET:
            available = max(0, TOTAL_MEMORY_BUDGET - _TAG_OVERHEAD_PER_LAYER * len(parts))
            content_total = sum(len(c) for _, c, _, _ in parts)
            if content_total > 0:
                _debug_log(
                    f"[Memory] budget trim: total={total} > {TOTAL_MEMORY_BUDGET}, "
                    f"available={available}"
                )
                trimmed: list[tuple[str, str, str, int]] = []
                for open_tag, content, close_tag, max_c in parts:
                    budget_i = int(available * len(content) / content_total)
                    if len(content) > budget_i:
                        content = content[:budget_i] + "…"
                    trimmed.append((open_tag, content, close_tag, max_c))
                parts = trimmed

        return "\n\n".join(
            f"{o}\n{_layer_meter_line(len(c), max_c)}\n{c}\n{cl}"
            for o, c, cl, max_c in parts
        )

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

    def _layer_global(self) -> str | None:
        try:
            from larkhelm.memory import load_global_memory
            return load_global_memory(self.chat_id)
        except Exception as e:
            _debug_log(f"[Memory] _layer_global failed: {e}")
            return None

    def _layer_project(self) -> str | None:
        if not self.cwd:
            return None
        try:
            from larkhelm.memory import load_project_memory
            return load_project_memory(self.cwd)
        except Exception as e:
            _debug_log(f"[Memory] _layer_project failed: {e}")
            return None

    def _layer_session(self) -> str | None:
        try:
            from larkhelm.memory import load_memory
            raw = load_memory(self.chat_id)
        except Exception as e:
            _debug_log(f"[Memory] _layer_session failed: {e}")
            return None
        if not raw:
            return None
        if not _config_flag("memory_session_layered", True):
            return raw
        slots = split_session_slots(raw)
        if not slots.parsed:
            return raw
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
