"""larkhelm · memory retriever — Phase D (Phase 1 keyword + Phase 2 hybrid).

Reads ``~/.larkhelm/memory/{global,project,session}_*.md`` files, slices
them on H2 boundaries, applies a per-agent_type ``InjectionPolicy``, and
composes a memory context string byte-shape compatible with the v2 path.

Phase 1 public API (zero-change, byte-compatible):
    load_slices(chat_id, cwd) -> list[MemorySlice]
    get_policy(agent_type) -> InjectionPolicy
    compose_slices_to_context(scored, policy, *, cwd=None) -> str
    KeywordRetriever().retrieve(req, policy, slices) -> list[ScoredSlice]
    POLICY_TABLE  -- 6 entries: chat / btw / dev / crew / plan / doc
    _retriever_active(chat_id) -> bool

Phase 2 additions:
    EmbeddingRetriever / HybridRetriever
    resolve_actual_mode(policy, chat_id, config) -> ActualRetrievalMode
    get_retriever(mode, *, backend) -> MemoryRetriever
    build_audit_record_v2(...) -> dict   (schema_version="2")
    rotate_audit_files()                  (daily + 32MiB rollover + 30d unlink)
    iter_audit_records(window, chat_id=None) -> Iterator[dict]

Log prefix: ``[MemoryRetriever]`` (NFR-OBS-1)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

import larkhelm.config as _cfg
from larkhelm._gating import hash_traffic_active
from larkhelm.log import _debug_log
from larkhelm.memory_slice import (
    ActualRetrievalMode,
    EmbeddingBackend,
    InjectionPolicy,
    MemoryRetriever,
    MemorySlice,
    RetrievalMode,
    RetrievalRequest,
    ScoredSlice,
    SliceKind,
    SliceLayer,
)


# ── Policy table ───────────────────────────────────────────────────────────

POLICY_TABLE: dict[str, InjectionPolicy] = {
    "chat": InjectionPolicy(
        agent_type="chat",
        token_budget=1200,
        layer_weights={"session": 0.5, "global": 0.4, "project": 0.1},
        kind_priority=("preference", "context_summary", "convention", "fact"),
        require_kinds=("preference",),
        alpha_recency=0.3, alpha_importance=0.4, alpha_relevance=0.3,
    ),
    "btw": InjectionPolicy(
        agent_type="btw",
        token_budget=800,
        layer_weights={"session": 0.5, "global": 0.4, "project": 0.1},
        kind_priority=("preference", "context_summary", "fact"),
        require_kinds=("preference",),
        alpha_recency=0.4, alpha_importance=0.3, alpha_relevance=0.3,
    ),
    "dev": InjectionPolicy(
        agent_type="dev",
        token_budget=3000,
        layer_weights={"project": 0.6, "session": 0.3, "global": 0.1},
        kind_priority=("convention", "incident", "decision", "task_history",
                       "context_summary", "fact"),
        require_kinds=("convention",),
        retrieval_mode="hybrid",  # Phase 2 REQ-33 — keyword fallback at runtime
        alpha_recency=0.3, alpha_importance=0.3, alpha_relevance=0.4,
    ),
    "crew": InjectionPolicy(
        agent_type="crew",
        token_budget=2400,
        layer_weights={"project": 0.5, "session": 0.4, "global": 0.1},
        kind_priority=("decision", "task_history", "context_summary",
                       "convention", "fact"),
        require_kinds=("context_summary",),
        retrieval_mode="hybrid",  # Phase 2 REQ-33 — keyword fallback at runtime
        alpha_recency=0.2, alpha_importance=0.4, alpha_relevance=0.4,
    ),
    "plan": InjectionPolicy(
        agent_type="plan",
        token_budget=2000,
        layer_weights={"project": 0.4, "session": 0.4, "global": 0.2},
        kind_priority=("decision", "context_summary", "task_history", "convention"),
        require_kinds=("context_summary",),
        retrieval_mode="hybrid",  # Phase 2 REQ-33 — keyword fallback at runtime
        alpha_recency=0.2, alpha_importance=0.5, alpha_relevance=0.3,
    ),
    "doc": InjectionPolicy(
        agent_type="doc",
        token_budget=800,
        layer_weights={"session": 0.6, "global": 0.3, "project": 0.1},
        kind_priority=("preference", "fact", "context_summary"),
        require_kinds=("preference",),
        exclude_kinds=("incident", "task_history"),
        alpha_recency=0.2, alpha_importance=0.3, alpha_relevance=0.5,
    ),
}


def get_policy(agent_type: str) -> InjectionPolicy:
    """Lookup ``POLICY_TABLE``; unknown agent_type falls back to ``chat``."""
    return POLICY_TABLE.get(agent_type, POLICY_TABLE["chat"])


# ── Slice loading ──────────────────────────────────────────────────────────

_H2_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Layer → default kind for monolith files (no H2 split).
_LAYER_DEFAULT_KIND: dict[str, SliceKind] = {
    "global": "preference",
    "project": "fact",
    "session": "context_summary",
}

# Layer → recency decay τ (days).
_LAYER_TAU_DAYS: dict[str, float] = {
    "session": 14.0,
    "project": 30.0,
    "global": 90.0,
}


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    """Strip an optional YAML-ish ``---``-fenced header and return ``(fm, body)``.

    Minimal hand-rolled parser — supports ``key: value`` lines, ``key:`` blocks
    with simple list values like ``[a, b, c]`` or ``- item`` per line. We avoid
    pulling PyYAML to keep the import graph stdlib-only."""
    if not raw or not raw.startswith("---"):
        return {}, raw

    m = _FRONTMATTER_RE.match(raw)
    if not m:
        return {}, raw

    block = m.group(1)
    body = raw[m.end():]
    fm: dict[str, object] = {}
    current_list_key: str | None = None
    current_list: list[str] = []

    for line in block.splitlines():
        if not line.strip():
            current_list_key = None
            continue
        if line.startswith("- ") and current_list_key is not None:
            current_list.append(line[2:].strip().strip('"').strip("'"))
            continue
        if current_list_key is not None:
            fm[current_list_key] = tuple(current_list)
            current_list_key = None
            current_list = []
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            current_list_key = key
            current_list = []
            continue
        # Inline list: [a, b, c]
        if val.startswith("[") and val.endswith("]"):
            items = [
                t.strip().strip('"').strip("'")
                for t in val[1:-1].split(",")
                if t.strip()
            ]
            fm[key] = tuple(items)
        else:
            fm[key] = val.strip('"').strip("'")
    if current_list_key is not None:
        fm[current_list_key] = tuple(current_list)
    return fm, body


def _split_h2_sections(body: str) -> list[tuple[str, str]]:
    """Return ``[(title, body), ...]`` for each H2 section.

    Empty list when the body has no H2 marker. ``### Hx`` and deeper headings
    stay inside the parent section's body."""
    if not body:
        return []
    matches = list(_H2_RE.finditer(body))
    if not matches:
        return []
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        sec_body = body[start:end].strip()
        sections.append((title, sec_body))
    return sections


def _heuristic_kind_for_section(layer: str, title: str, body: str) -> SliceKind:
    """Best-effort kind guess from H2 title + layer default."""
    t = (title or "").lower()
    b = (body or "").lower()
    if any(k in t for k in ("preference", "偏好", "language", "语言", "style", "风格")):
        return "preference"
    if any(k in t for k in ("convention", "约定", "规范", "rule")):
        return "convention"
    if any(k in t for k in ("decision", "决策", "选型")):
        return "decision"
    if any(k in t for k in ("incident", "故障", "bug", "事故", "oom")):
        return "incident"
    if any(k in t for k in ("history", "task", "进展", "next step", "下一步", "后续")):
        return "task_history"
    if any(k in t for k in ("context", "summary", "上下文", "摘要", "work")):
        return "context_summary"
    if "incident" in b or "故障" in b or "事故" in b:
        return "incident"
    return _LAYER_DEFAULT_KIND.get(layer, "fact")


def _slice_id(layer: str, scope: str, title: str, slice_idx: int) -> str:
    src = f"{layer}::{scope}::{title}::{slice_idx}"
    return hashlib.md5(src.encode("utf-8")).hexdigest()[:12]


_PATH_OR_MODULE_RE = re.compile(
    r"[\w]+(?:[\./][\w]+){1,}\.[a-z0-9]{1,5}\b"
    r"|larkhelm\.[\w\.]+",
    re.IGNORECASE,
)


def _auto_extract_entities(body: str) -> tuple[str, ...]:
    """Pull file paths / python module names out of body text. Capped at 12.

    Used when frontmatter doesn't declare entities — Phase 1 best-effort.
    """
    if not body:
        return ()
    found = []
    seen = set()
    for m in _PATH_OR_MODULE_RE.finditer(body):
        tok = m.group(0)
        if tok in seen:
            continue
        seen.add(tok)
        found.append(tok)
        if len(found) >= 12:
            break
    return tuple(found)


def _ensure_tuple(val: Any) -> tuple[str, ...]:
    if val is None or val == "":
        return ()
    if isinstance(val, tuple):
        return tuple(str(x) for x in val)
    if isinstance(val, list):
        return tuple(str(x) for x in val)
    if isinstance(val, str):
        return (val,) if val else ()
    return ()


def _parse_iso_ts(s: str) -> str:
    """Return a normalised ISO8601 string, or "" if unparseable."""
    if not s:
        return ""
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).isoformat()
    except (TypeError, ValueError):
        return s.strip()


def _slices_from_file(
    layer: SliceLayer,
    scope: str,
    path: Path,
    raw: str,
) -> list[MemorySlice]:
    """Materialise slices from one memory file's full text."""
    fm, body = _parse_frontmatter(raw)
    body = (body or "").strip()
    if not body:
        return []

    try:
        importance_default = float(fm.get("importance", 0.5))
    except (TypeError, ValueError):
        importance_default = 0.5

    created = _parse_iso_ts(str(fm.get("created_at", "")))
    updated = _parse_iso_ts(str(fm.get("updated_at", "")))
    if not updated:
        try:
            updated = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            updated = ""

    fm_keywords = _ensure_tuple(fm.get("keywords"))
    fm_entities = _ensure_tuple(fm.get("entities"))
    fm_tags = _ensure_tuple(fm.get("tags"))

    sections = _split_h2_sections(body)
    if not sections:
        kind = _heuristic_kind_for_section(layer, "", body)
        slice_id = _slice_id(layer, scope, "", 0)
        entities = fm_entities or _auto_extract_entities(body)
        return [MemorySlice(
            id=slice_id,
            layer=layer,
            kind=kind,
            title="",
            body=body,
            importance=importance_default,
            created_at=created,
            updated_at=updated,
            keywords=fm_keywords,
            entities=entities,
            tags=fm_tags,
            char_len=len(body),
        )]

    out: list[MemorySlice] = []
    for idx, (title, sec_body) in enumerate(sections):
        if not sec_body:
            continue
        kind = _heuristic_kind_for_section(layer, title, sec_body)
        slice_id = _slice_id(layer, scope, title, idx)
        entities = fm_entities or _auto_extract_entities(f"{title}\n{sec_body}")
        out.append(MemorySlice(
            id=slice_id,
            layer=layer,
            kind=kind,
            title=title,
            body=sec_body,
            importance=importance_default,
            created_at=created,
            updated_at=updated,
            keywords=fm_keywords,
            entities=entities,
            tags=fm_tags,
            char_len=len(sec_body),
        ))
    return out


def _memory_home() -> Path:
    """Resolve the ``~/.larkhelm/memory`` root; created lazily."""
    home = Path.home() / ".larkhelm" / "memory"
    try:
        home.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return home


def _resolve_layer_files(chat_id: str, cwd: str | None) -> list[tuple[SliceLayer, str, Path]]:
    """Return ``[(layer, scope, path)]`` for the three layer files to scan.

    Layer scope is the same id used by ``_slice_id`` (open_id for global,
    cwd for project, chat_id for session). Files that don't exist are
    filtered out by the caller."""
    home = _memory_home()
    triples: list[tuple[SliceLayer, str, Path]] = []

    open_id = ""
    try:
        from larkhelm.chat_state import _get_chat_state
        state = _get_chat_state(chat_id) if chat_id else {}
        open_id = str(state.get("sender_open_id", "") or "")
    except Exception:
        open_id = ""
    if open_id:
        triples.append(("global", open_id, home / f"global_{open_id}.md"))

    if cwd:
        try:
            canonical = str(Path(cwd).resolve())
            ph = hashlib.md5(canonical.encode()).hexdigest()[:16]
            triples.append(("project", canonical, home / f"project_{ph}.md"))
        except Exception:
            pass

    if chat_id:
        triples.append(("session", chat_id, home / f"session_{chat_id}.md"))

    return triples


@lru_cache(maxsize=64)
def _slices_for_path_cached(
    path_str: str,
    layer: str,
    scope: str,
    mtime_ns: int,
    size: int,
) -> tuple[MemorySlice, ...]:
    """LRU-cached file → slice list. Cache key includes mtime+size so a file
    overwrite invalidates the entry on the next call."""
    path = Path(path_str)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ()
    return tuple(_slices_from_file(layer, scope, path, raw))  # type: ignore[arg-type]


def load_slices(chat_id: str, cwd: str | None = None) -> list[MemorySlice]:
    """Read global / project / session memory files into ``MemorySlice`` list.

    H2 sections become individual slices; monolith files become a single
    slice with ``title=""``. LRU-cached by (path, mtime, size) — capacity
    64 (NFR-PERF-2).

    Phase 2 hook: after the slices are materialised, :mod:`memory_lifecycle`
    overlays ``.stale=True`` for any id present in the layer's sidecar
    ``.meta.json``. The lifecycle module is imported locally to avoid the
    `memory_retriever ↔ memory_lifecycle ↔ memory` triangle on bootstrap.
    """
    triples = _resolve_layer_files(chat_id, cwd)
    out: list[MemorySlice] = []
    for layer, scope, path in triples:
        try:
            st = path.stat()
        except OSError:
            continue
        try:
            cached = _slices_for_path_cached(
                str(path), layer, scope, st.st_mtime_ns, st.st_size,
            )
        except Exception as e:
            _debug_log(f"[MemoryRetriever] slice load failed for {path.name}: {e}")
            continue
        out.extend(cached)

    # Phase 2 — overlay stale marks via the lifecycle sidecar.
    if out:
        try:
            from larkhelm.memory_lifecycle import inject_stale_marks
            out = inject_stale_marks(out, triples)
        except Exception as e:
            _debug_log(f"[MemoryRetriever] inject_stale_marks failed (continuing): {e}")
    return out


# ── BM25-lite scoring ──────────────────────────────────────────────────────

# Simple alphanumeric + CJK token regex. CJK is split per-character so a query
# like "OOM 防护" tokenises to ["oom", "防", "护"], giving recall on short
# Chinese terms without depending on a segmenter.
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[一-鿿]", re.UNICODE)
_CJK_RE = re.compile(r"[一-鿿]+", re.UNICODE)


def _cjk_ngrams(text: str) -> list[str]:
    """Phase 2 (REQ-34): append bi- and tri-gram sliding windows over each
    contiguous CJK run, so queries like "OOM 防护" recall a slice whose body
    only contains the bi-gram "防护" once.

    English / digit tokens are unchanged — they don't participate. We bound
    n-gram extraction to runs of ≤ 64 chars to keep _tokenise O(L)."""
    out: list[str] = []
    for m in _CJK_RE.finditer(text or ""):
        run = m.group(0)
        if len(run) < 2:
            continue
        capped = run[:64]
        for i in range(len(capped) - 1):
            out.append(capped[i:i + 2])
        for i in range(len(capped) - 2):
            out.append(capped[i:i + 3])
    return out


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    base = [t.lower() for t in _TOKEN_RE.findall(text)]
    base.extend(_cjk_ngrams(text))
    return base


_BM25_K1 = 1.5
_BM25_B = 0.75


def _bm25_lite_score(
    query_terms: list[str],
    doc_terms: list[str],
    idf: dict[str, float],
    avgdl: float,
) -> float:
    if not query_terms or not doc_terms:
        return 0.0
    dl = len(doc_terms)
    norm = 1.0 - _BM25_B + _BM25_B * (dl / avgdl if avgdl > 0 else 1.0)
    # Frequency of each unique query term in doc_terms — O(unique × dl).
    seen_q = set(query_terms)
    score = 0.0
    if not seen_q:
        return 0.0
    # Build a small term-frequency view for the doc only over seen_q.
    tf: dict[str, int] = {}
    for t in doc_terms:
        if t in seen_q:
            tf[t] = tf.get(t, 0) + 1
    for q in query_terms:
        f = tf.get(q, 0)
        if f == 0:
            continue
        denom = f + _BM25_K1 * norm
        score += idf.get(q, 0.0) * (f * (_BM25_K1 + 1)) / denom if denom else 0.0
    return score


def _compute_idf(slices: list[MemorySlice]) -> tuple[dict[str, float], float]:
    """Return ``(idf, avgdl)`` over the candidate slice pool."""
    n = len(slices)
    if n == 0:
        return {}, 0.0
    df: dict[str, int] = {}
    total_len = 0
    for s in slices:
        terms = set(_tokenise(s.title)) | set(_tokenise(s.body))
        terms |= set(_tokenise(" ".join(s.keywords)))
        total_len += sum(1 for _ in _TOKEN_RE.findall(s.title + " " + s.body))
        for t in terms:
            df[t] = df.get(t, 0) + 1
    avgdl = total_len / n if n else 0.0
    idf = {
        t: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5))
        for t, dfi in df.items()
    }
    return idf, avgdl


def _recency_score(slice_obj: MemorySlice, now: datetime) -> float:
    """Exponential decay with per-layer τ; absent timestamps give 0.5 (neutral)."""
    ts = slice_obj.updated_at or slice_obj.created_at
    if not ts:
        return 0.5
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return 0.5
    # Make both sides naive in UTC to subtract cleanly.
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    n = now
    if n.tzinfo is not None:
        n = n.astimezone(timezone.utc).replace(tzinfo=None)
    delta_days = max(0.0, (n - dt).total_seconds() / 86400.0)
    tau = _LAYER_TAU_DAYS.get(slice_obj.layer, 30.0)
    return math.exp(-delta_days / tau)


def _entity_boost(slice_obj: MemorySlice, request: RetrievalRequest) -> float:
    """Return 1.5 if any slice.entity appears in the query (or force_files);
    else 1.0. Match is case-insensitive on the raw substring."""
    if not slice_obj.entities:
        return 1.0
    q = (request.query or "").lower()
    hay = q + " " + " ".join(str(f).lower() for f in request.force_files)
    for ent in slice_obj.entities:
        e = str(ent).lower()
        if not e:
            continue
        if e in hay:
            return 1.5
    return 1.0


def _stale_decay_factor() -> float:
    """Return the configured stale relevance multiplier; clamped to [0, 1]."""
    cfg = getattr(_cfg, "config", {}) or {}
    try:
        v = float(cfg.get("memory_stale_decay", 0.5))
    except (TypeError, ValueError):
        return 0.5
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


class KeywordRetriever:
    """Phase 1 keyword retriever: BM25-lite + entity boost + 3-factor score.

    Score: ``alpha_r·recency + alpha_i·importance + alpha_v·relevance``
    where ``relevance = BM25_lite(query, slice) × entity_boost``.

    Phase 2: ``_apply_stale_decay`` multiplies ``relevance_score`` and the
    composed total ``score`` by ``memory_stale_decay`` (default 0.5) for
    slices where ``slice.stale == True``. The reason string is extended
    with ``stale`` so /memory diagnose can surface the demotion.
    """

    def _apply_stale_decay(self, item: ScoredSlice) -> ScoredSlice:
        if not item.slice.stale:
            return item
        decay = _stale_decay_factor()
        # Multiply the *composed* score, not just the relevance factor —
        # otherwise a stale slice with extremely high recency would still
        # rank above a fresh slice. Matches design.md §3.4 "soft delete".
        new_relevance = item.relevance_score * decay
        new_score = item.score * decay
        reason = item.reason + (",stale" if item.reason else "stale")
        return ScoredSlice(
            slice=item.slice,
            score=new_score,
            recency_score=item.recency_score,
            importance_score=item.importance_score,
            relevance_score=new_relevance,
            reason=reason,
        )

    def retrieve(
        self,
        request: RetrievalRequest,
        policy: InjectionPolicy,
        candidate_slices: list[MemorySlice],
    ) -> list[ScoredSlice]:
        if not candidate_slices:
            return []

        exclude = set(policy.exclude_kinds or ())
        pool = [s for s in candidate_slices if s.kind not in exclude]
        if not pool:
            return []

        require = set(policy.require_kinds or ())
        idf, avgdl = _compute_idf(pool)
        # Normalise BM25 scores to [0, 1] by max so the three factors are
        # comparable — pure BM25 can be unbounded.
        query_terms = _tokenise(request.query)
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        raw_scored: list[tuple[MemorySlice, float, float, float, float]] = []
        max_rel = 0.0
        for sl in pool:
            doc_terms = _tokenise(sl.title) + _tokenise(sl.body)
            doc_terms += _tokenise(" ".join(sl.keywords))
            rel = _bm25_lite_score(query_terms, doc_terms, idf, avgdl)
            rel *= _entity_boost(sl, request)
            if rel > max_rel:
                max_rel = rel
            rec = _recency_score(sl, now)
            imp = max(0.0, min(1.0, float(sl.importance or 0.0)))
            raw_scored.append((sl, rel, rec, imp, 0.0))

        a_r = float(policy.alpha_recency)
        a_i = float(policy.alpha_importance)
        a_v = float(policy.alpha_relevance)

        scored: list[ScoredSlice] = []
        require_threshold = 0.3
        for sl, rel, rec, imp, _ in raw_scored:
            rel_norm = (rel / max_rel) if max_rel > 0 else 0.0
            score = a_r * rec + a_i * imp + a_v * rel_norm
            if sl.kind in require:
                score = max(score, require_threshold)
            reason_bits = []
            if rel_norm > 0:
                reason_bits.append(f"rel={rel_norm:.2f}")
            reason_bits.append(f"rec={rec:.2f}")
            reason_bits.append(f"imp={imp:.2f}")
            if sl.kind in require:
                reason_bits.append("require")
            scored.append(self._apply_stale_decay(ScoredSlice(
                slice=sl,
                score=score,
                recency_score=rec,
                importance_score=imp,
                relevance_score=rel_norm,
                reason=",".join(reason_bits),
            )))

        scored.sort(key=lambda s: s.score, reverse=True)
        top_k = int(policy.top_k or 6)
        return scored[:top_k]


# ── Embedding / Hybrid retrievers (Phase 2) ────────────────────────────────


class EmbeddingRetriever:
    """Pure cosine-similarity retriever.

    Computes a single query vector once, then fetches per-slice vectors
    from the LRU-backed :class:`EmbeddingCache`. The composed score uses
    ``alpha_recency / alpha_importance / alpha_relevance`` exactly like
    :class:`KeywordRetriever`, with cosine-sim taking the role of
    ``relevance_score``.

    Stale slices are demoted via the same :func:`_stale_decay_factor`
    multiplier — kept consistent with keyword path so toggling modes
    doesn't change ordering semantics for stale items.

    On any backend failure raises :class:`EmbeddingError` — caller
    (HybridRetriever / _build_with_retriever) decides whether to fail-open.
    """

    def __init__(self, backend: EmbeddingBackend, cache: Any = None) -> None:
        from larkhelm.memory_embedding import EmbeddingCache
        self.backend = backend
        self.cache = cache if cache is not None else EmbeddingCache()

    def _compute_query_vector(self, query: str) -> Any:
        return self.backend.embed([query or ""])[0]

    def retrieve(
        self,
        request: RetrievalRequest,
        policy: InjectionPolicy,
        candidate_slices: list[MemorySlice],
    ) -> list[ScoredSlice]:
        if not candidate_slices:
            return []
        from larkhelm.memory_embedding import cosine_similarity
        exclude = set(policy.exclude_kinds or ())
        pool = [s for s in candidate_slices if s.kind not in exclude]
        if not pool:
            return []

        q_vec = self._compute_query_vector(request.query)
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        a_r = float(policy.alpha_recency)
        a_i = float(policy.alpha_importance)
        a_v = float(policy.alpha_relevance)
        require = set(policy.require_kinds or ())
        require_threshold = 0.3

        scored: list[ScoredSlice] = []
        decay = _stale_decay_factor()
        for sl in pool:
            try:
                s_vec = self.cache.get_or_compute(sl.id, sl.body, self.backend)
            except Exception as e:
                # Bubble up — caller distinguishes EmbeddingError from
                # other failures (HybridRetriever fails open, EmbeddingRetriever
                # propagates so MemoryContextBuilder can fail-open keyword).
                raise e
            sim = max(0.0, cosine_similarity(q_vec, s_vec))
            rec = _recency_score(sl, now)
            imp = max(0.0, min(1.0, float(sl.importance or 0.0)))
            score = a_r * rec + a_i * imp + a_v * sim
            if sl.kind in require:
                score = max(score, require_threshold)
            reason_bits = [f"cos={sim:.2f}", f"rec={rec:.2f}", f"imp={imp:.2f}"]
            if sl.kind in require:
                reason_bits.append("require")
            item = ScoredSlice(
                slice=sl,
                score=score,
                recency_score=rec,
                importance_score=imp,
                relevance_score=sim,
                reason=",".join(reason_bits),
            )
            if sl.stale and decay < 1.0:
                item = ScoredSlice(
                    slice=sl,
                    score=item.score * decay,
                    recency_score=rec,
                    importance_score=imp,
                    relevance_score=sim * decay,
                    reason=item.reason + ",stale",
                )
            scored.append(item)

        scored.sort(key=lambda x: x.score, reverse=True)
        top_k = int(policy.top_k or 6)
        return scored[:top_k]


class HybridRetriever:
    """Linear-fusion retriever (REQ-28): score = α·cos_sim + (1-α)·BM25_norm.

    Flow:

      1. Run :class:`KeywordRetriever` to get an *oversized* pool of size
         ``policy.top_k × policy.embedding_top_k_multiplier``.
      2. Embed the query + each slice in the pool, compute cosine.
      3. Fuse: ``hybrid_alpha * cos_sim + (1 - hybrid_alpha) * keyword_relevance``.
      4. Re-apply stale decay on the fused score.
      5. Return top_k.

    Fail-open contract: any :class:`EmbeddingError` mid-flow ⇒ return
    pure KeywordRetriever output (already computed) so the caller never
    sees an empty memory context just because the embedding service is
    down. The caller (:mod:`memory_context`) flips ``audit.fail_open=True``.
    """

    def __init__(self, keyword: "KeywordRetriever", embedding: "EmbeddingRetriever") -> None:
        self.keyword = keyword
        self.embedding = embedding

    def retrieve(
        self,
        request: RetrievalRequest,
        policy: InjectionPolicy,
        candidate_slices: list[MemorySlice],
    ) -> list[ScoredSlice]:
        if not candidate_slices:
            return []
        from larkhelm.memory_embedding import EmbeddingError, cosine_similarity

        # Stage 1 — oversized BM25 pool.
        multiplier = max(1, int(policy.embedding_top_k_multiplier or 3))
        pool_top_k = max(1, int(policy.top_k or 6) * multiplier)
        # Stash original top_k so we can shrink for the keyword pre-pass.
        pooled_policy = dataclasses_replace_top_k(policy, pool_top_k)
        keyword_scored = self.keyword.retrieve(request, pooled_policy, candidate_slices)
        if not keyword_scored:
            return []

        # Stage 2 — embed query + every slice in the pool. Any backend
        # failure ⇒ fall back to the keyword pool truncated to top_k.
        try:
            q_vec = self.embedding._compute_query_vector(request.query)
        except EmbeddingError as e:
            _debug_log(
                f"[MemoryRetriever] EmbeddingHybrid: query embed failed (fail-open): {e}"
            )
            top_k = int(policy.top_k or 6)
            return keyword_scored[:top_k]

        alpha = float(policy.hybrid_alpha)
        if alpha < 0.0:
            alpha = 0.0
        if alpha > 1.0:
            alpha = 1.0
        decay = _stale_decay_factor()

        fused: list[ScoredSlice] = []
        for item in keyword_scored:
            sl = item.slice
            try:
                s_vec = self.embedding.cache.get_or_compute(sl.id, sl.body, self.embedding.backend)
                sim = max(0.0, cosine_similarity(q_vec, s_vec))
            except EmbeddingError as e:
                _debug_log(
                    f"[MemoryRetriever] EmbeddingHybrid: slice embed failed (fail-open): {e}"
                )
                top_k = int(policy.top_k or 6)
                return keyword_scored[:top_k]
            # Note: keyword_scored may already carry a stale-decayed
            # ``relevance_score``. We undo the decay locally so the fusion
            # math operates on the raw BM25 norm, then re-apply once at the
            # end. Without this a stale slice would be discounted twice.
            raw_kw_relevance = item.relevance_score
            if sl.stale and decay > 0:
                raw_kw_relevance = item.relevance_score / decay if decay != 0 else 0.0
            fused_relevance = alpha * sim + (1.0 - alpha) * raw_kw_relevance
            # Recompute total score with policy alphas on the fused relevance,
            # using the same recency/importance components keyword path emitted.
            score = (
                float(policy.alpha_recency) * item.recency_score
                + float(policy.alpha_importance) * item.importance_score
                + float(policy.alpha_relevance) * fused_relevance
            )
            reason = f"hybrid α={alpha:.2f},cos={sim:.2f},bm25={raw_kw_relevance:.2f}"
            new_item = ScoredSlice(
                slice=sl,
                score=score,
                recency_score=item.recency_score,
                importance_score=item.importance_score,
                relevance_score=fused_relevance,
                reason=reason,
            )
            if sl.stale and decay < 1.0:
                new_item = ScoredSlice(
                    slice=sl,
                    score=new_item.score * decay,
                    recency_score=item.recency_score,
                    importance_score=item.importance_score,
                    relevance_score=fused_relevance * decay,
                    reason=reason + ",stale",
                )
            fused.append(new_item)

        fused.sort(key=lambda x: x.score, reverse=True)
        top_k = int(policy.top_k or 6)
        return fused[:top_k]


def dataclasses_replace_top_k(policy: InjectionPolicy, top_k: int) -> InjectionPolicy:
    """Helper: clone ``policy`` with a different ``top_k`` (frozen dataclass).

    Kept at module scope rather than inline so tests can monkeypatch it if
    they want to assert pool sizing.
    """
    import dataclasses as _dc
    return _dc.replace(policy, top_k=int(top_k))


# ── Mode resolution + retriever factory ────────────────────────────────────


def resolve_actual_mode(
    policy: InjectionPolicy,
    chat_id: str,
    config: dict[str, Any] | None = None,
) -> ActualRetrievalMode:
    """Decide which physical retriever to dispatch for this (policy, chat).

    Algorithm (design.md §1.3):

      1. start with policy.retrieval_mode (or config override if not "auto")
      2. embedding_traffic gate ON ⇒ force "hybrid"
      3. embedding_backend == "none" ⇒ collapse to "keyword"
      4. anything other than the 3 actual modes ⇒ "keyword"
    """
    cfg = config if config is not None else (getattr(_cfg, "config", {}) or {})
    override = str(cfg.get("memory_retriever_mode", "auto") or "auto").lower()
    if override == "auto":
        mode = str(policy.retrieval_mode or "keyword").lower()
    else:
        mode = override

    # Stage B grading: embedding_traffic active for this chat ⇒ force hybrid.
    # We re-implement the bucket math locally (rather than calling
    # ``hash_traffic_active``) so the config dict passed in here is the source
    # of truth — important for tests that drive resolve_actual_mode directly
    # without mutating ``larkhelm.config.config``.
    try:
        if bool(cfg.get("embedding_enabled", False)):
            traffic = float(cfg.get("embedding_traffic", 0.0) or 0.0)
            active = False
            if traffic >= 1.0:
                active = True
            elif traffic > 0.0:
                import hashlib as _hl
                digest = _hl.md5(str(chat_id).encode("utf-8")).hexdigest()
                bucket = int(digest[:8], 16) % 10000
                active = bucket < int(traffic * 10000)
            if active:
                mode = "hybrid"
    except Exception as e:
        _debug_log(f"[MemoryRetriever] embedding traffic gate failed: {e}")

    backend_name = str(cfg.get("embedding_backend", "none") or "none").lower()
    if backend_name == "none":
        # Embedding off — collapse to keyword regardless.
        mode = "keyword"

    if mode not in ("keyword", "embedding", "hybrid"):
        mode = "keyword"
    return mode  # type: ignore[return-value]


def get_retriever(
    mode: ActualRetrievalMode,
    *,
    backend: EmbeddingBackend | None = None,
) -> MemoryRetriever:
    """Construct a retriever instance for the given mode.

    When ``mode`` is ``embedding``/``hybrid`` but ``backend`` is ``None``,
    silently downgrades to ``KeywordRetriever`` — easier to reason about
    than raising at the dispatch layer (caller has fail-open semantics).
    """
    if mode == "embedding":
        if backend is None:
            return KeywordRetriever()
        return EmbeddingRetriever(backend)
    if mode == "hybrid":
        if backend is None:
            return KeywordRetriever()
        emb = EmbeddingRetriever(backend)
        return HybridRetriever(KeywordRetriever(), emb)
    return KeywordRetriever()


# ── Composition ────────────────────────────────────────────────────────────

def _layer_meter_line_local(chars: int, max_chars: int) -> str:
    """Local copy of memory._layer_meter_line so we don't trigger circular
    imports during memory_context bootstrap."""
    if max_chars <= 0:
        pct = 0
    else:
        pct = chars * 100 // max_chars
    base = f"[{chars}/{max_chars} chars, {pct}%]"
    if pct >= 90:
        base += " ⚠️ near limit"
    return base


def compose_slices_to_context(
    scored: list[ScoredSlice],
    policy: InjectionPolicy,
    *,
    cwd: str | None = None,
) -> str:
    """Group slices by layer → smart_truncate per layer → wrap with tags.

    Output shape is byte-compatible with v2's ``MemoryContextBuilder.build()``:
    each present layer becomes
    ``[<LAYER> MEMORY[ — cwd]]\\n<meter>\\n<body>\\n[/<LAYER> MEMORY]``
    joined by ``\\n\\n``."""
    if not scored:
        return ""
    from larkhelm.memory_context import smart_truncate

    layer_order: list[SliceLayer] = ["global", "project", "session"]
    by_layer: dict[str, list[ScoredSlice]] = {l: [] for l in layer_order}
    for s in scored:
        by_layer.setdefault(s.slice.layer, []).append(s)

    total_budget = max(1, int(policy.token_budget or 1))
    weights = dict(policy.layer_weights or {})

    # Per-layer caps (fall back to module-level v2 caps when memory available).
    try:
        from larkhelm.memory import (
            GLOBAL_MAX_CHARS, PROJECT_MAX_CHARS, SESSION_MAX_CHARS,
        )
    except Exception:
        GLOBAL_MAX_CHARS, PROJECT_MAX_CHARS, SESSION_MAX_CHARS = 800, 1500, 2000

    caps = {
        "global": GLOBAL_MAX_CHARS,
        "project": PROJECT_MAX_CHARS,
        "session": SESSION_MAX_CHARS,
    }

    parts: list[tuple[str, str, str, int]] = []
    for layer in layer_order:
        layer_slices = by_layer.get(layer, [])
        if not layer_slices:
            continue
        w = float(weights.get(layer, 0.0))
        if w <= 0:
            continue
        budget = max(1, int(total_budget * w))
        sections: list[str] = []
        for s in layer_slices:
            head = f"## {s.slice.title}\n" if s.slice.title else ""
            sections.append(f"{head}{s.slice.body}".strip())
        merged = "\n\n".join(x for x in sections if x).strip()
        if not merged:
            continue
        trimmed = smart_truncate(merged, budget)
        if layer == "global":
            open_tag, close_tag = "[GLOBAL MEMORY]", "[/GLOBAL MEMORY]"
        elif layer == "project":
            scope = cwd or ""
            open_tag = f"[PROJECT MEMORY — {scope}]" if scope else "[PROJECT MEMORY]"
            close_tag = "[/PROJECT MEMORY]"
        else:
            open_tag, close_tag = "[SESSION MEMORY]", "[/SESSION MEMORY]"
        parts.append((open_tag, trimmed, close_tag, caps[layer]))

    if not parts:
        return ""

    return "\n\n".join(
        f"{o}\n{_layer_meter_line_local(len(c), cap)}\n{c}\n{cl}"
        for o, c, cl, cap in parts
    )


# ── Gating ─────────────────────────────────────────────────────────────────

def _retriever_active(chat_id: str) -> bool:
    """True iff Phase D retriever should run for this chat (flag + traffic)."""
    return hash_traffic_active(
        chat_id, "memory_retriever_enabled", "memory_retriever_traffic",
    )


# ── Audit (async JSONL) ────────────────────────────────────────────────────

_AUDIT_QUEUE: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=100)
_AUDIT_THREAD: threading.Thread | None = None
_AUDIT_THREAD_LOCK = threading.Lock()
_AUDIT_DROP_LOGGED = False


def _resolve_audit_path() -> Path:
    cfg = getattr(_cfg, "config", {}) or {}
    custom = cfg.get("memory_retriever_audit_path") or ""
    if custom:
        return Path(custom)
    data_dir = getattr(_cfg, "DATA_DIR", None)
    if data_dir is None:
        return Path(tempfile.gettempdir()) / "memory_retriever_audit.jsonl"
    return Path(data_dir) / "memory_retriever_audit.jsonl"


def _audit_writer_loop() -> None:
    while True:
        try:
            record = _AUDIT_QUEUE.get()
        except Exception:
            return
        if record is None:
            return
        try:
            path = _resolve_audit_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(
                str(path), os.O_APPEND | os.O_WRONLY | os.O_CREAT, 0o600,
            )
            try:
                try:
                    os.fchmod(fd, 0o600)
                except OSError:
                    pass
                os.write(
                    fd,
                    (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8"),
                )
            finally:
                os.close(fd)
        except Exception as e:
            _debug_log(f"[MemoryRetriever] audit write failed: {e}")


def _ensure_audit_thread() -> None:
    global _AUDIT_THREAD
    with _AUDIT_THREAD_LOCK:
        if _AUDIT_THREAD is not None and _AUDIT_THREAD.is_alive():
            return
        t = threading.Thread(
            target=_audit_writer_loop,
            name="memory-retriever-audit",
            daemon=True,
        )
        t.start()
        _AUDIT_THREAD = t


def _audit_decision(record: dict[str, Any]) -> None:
    """Enqueue one audit record; drops on overflow with one-shot debug log."""
    global _AUDIT_DROP_LOGGED
    _ensure_audit_thread()
    try:
        _AUDIT_QUEUE.put_nowait(record)
    except queue.Full:
        if not _AUDIT_DROP_LOGGED:
            _AUDIT_DROP_LOGGED = True
            _debug_log(
                "[MemoryRetriever] audit queue full, dropped 1 record"
            )


def _build_audit_record(
    request: RetrievalRequest,
    policy: InjectionPolicy,
    scored: list[ScoredSlice],
    candidate_count: int,
    elapsed_ms: int,
    selected_chars: int,
    fail_open: bool,
) -> dict[str, Any]:
    """Phase 1 audit record (preserved for legacy callers / tests)."""
    return {
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chat_id": request.chat_id,
        "agent_type": request.agent_type,
        "mode": policy.retrieval_mode,
        "query_head": (request.query or "")[:80],
        "candidate_slice_count": candidate_count,
        "selected_slice_ids": [s.slice.id for s in scored],
        "selected_token_chars": selected_chars,
        "elapsed_ms": elapsed_ms,
        "fail_open": fail_open,
    }


def build_audit_record_v2(
    request: RetrievalRequest,
    policy: InjectionPolicy,
    scored: list[ScoredSlice],
    candidate_count: int,
    elapsed_ms: int,
    selected_chars: int,
    fail_open: bool,
    actual_mode: ActualRetrievalMode,
    declared_mode: RetrievalMode | None = None,
) -> dict[str, Any]:
    """Phase 2 audit record (schema_version="2").

    All Phase 1 fields are preserved so the legacy reader can still parse
    these records (extra fields are ignored). Phase 2 additions follow
    design.md §3.4.
    """
    if declared_mode is None:
        declared_mode = policy.retrieval_mode
    selected_ids = [s.slice.id for s in scored]
    stale_hits = sum(1 for s in scored if getattr(s.slice, "stale", False))
    return {
        "schema_version": "2",
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chat_id": request.chat_id,
        "agent_type": request.agent_type,
        "mode": actual_mode,
        "declared_mode": declared_mode,
        "hybrid_alpha": float(policy.hybrid_alpha),
        "query_head": (request.query or "")[:80],
        "query_token_count": len(_tokenise(request.query or "")),
        "candidate_slice_count": int(candidate_count),
        "selected_slice_ids": selected_ids,
        "selected_token_chars": int(selected_chars),
        "top_k_returned": len(selected_ids),
        "elapsed_ms": int(elapsed_ms),
        "fail_open": bool(fail_open),
        "stale_hit_count": stale_hits,
    }


# ── Audit rotation (REQ-39) ───────────────────────────────────────────────


def _audit_dir_files() -> list[Path]:
    """Return all files in the audit directory matching the audit name.

    The "primary" file (``memory_retriever_audit.jsonl``) plus every rotated
    archive (``memory_retriever_audit-YYYY-MM-DD.jsonl`` and
    ``-YYYY-MM-DD.NN.jsonl`` for 32MiB rollover suffixes).
    """
    primary = _resolve_audit_path()
    parent = primary.parent
    base = primary.stem  # "memory_retriever_audit"
    suffix = primary.suffix  # ".jsonl"
    out: list[Path] = []
    try:
        for p in parent.glob(f"{base}*{suffix}"):
            out.append(p)
    except Exception:
        pass
    return out


def rotate_audit_files() -> None:
    """Daily + 32 MiB rotation + retention sweep.

    Idempotent; safe to call concurrently because we use ``os.rename`` for
    the rollover and ``unlink(missing_ok=True)`` for retention.

    1. Daily roll: if today's primary file's mtime is on a previous day,
       rename it to ``memory_retriever_audit-<that-day>.jsonl``.
    2. Size roll: if today's primary file is over the configured MB limit,
       rename to ``-<today>.<N>.jsonl`` (N = next free integer).
    3. Retention: unlink rotated files older than
       ``memory_audit_retain_days`` (default 30).
    """
    cfg = getattr(_cfg, "config", {}) or {}
    rotate_max_mb = int(cfg.get("memory_audit_rotate_max_mb", 32) or 32)
    retain_days = int(cfg.get("memory_audit_retain_days", 30) or 30)
    primary = _resolve_audit_path()

    # 1) Daily roll
    try:
        if primary.exists():
            st = primary.stat()
            mtime_dt = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).astimezone()
            today = datetime.now().astimezone().date()
            if mtime_dt.date() < today:
                day_str = mtime_dt.date().isoformat()
                rotated = primary.with_name(f"{primary.stem}-{day_str}{primary.suffix}")
                # If yesterday's already exists (e.g. crash recovery), pick a suffix.
                if rotated.exists():
                    idx = 1
                    while True:
                        candidate = primary.with_name(
                            f"{primary.stem}-{day_str}.{idx}{primary.suffix}"
                        )
                        if not candidate.exists():
                            rotated = candidate
                            break
                        idx += 1
                os.rename(str(primary), str(rotated))
    except Exception as e:
        _debug_log(f"[MemoryAudit] daily rotate failed: {e}")

    # 2) Size roll (after the daily roll so we're operating on today's file).
    try:
        if primary.exists():
            st = primary.stat()
            if st.st_size >= rotate_max_mb * 1024 * 1024:
                today_str = datetime.now().astimezone().date().isoformat()
                idx = 1
                while True:
                    candidate = primary.with_name(
                        f"{primary.stem}-{today_str}.{idx}{primary.suffix}"
                    )
                    if not candidate.exists():
                        break
                    idx += 1
                os.rename(str(primary), str(candidate))
    except Exception as e:
        _debug_log(f"[MemoryAudit] size rotate failed: {e}")

    # 3) Retention sweep — unlink rotated archives older than retain_days.
    try:
        cutoff = time.time() - retain_days * 86400
        for p in _audit_dir_files():
            try:
                if p == primary:
                    continue  # never unlink the live file
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
            except Exception as inner:
                _debug_log(f"[MemoryAudit] retention unlink failed for {p.name}: {inner}")
    except Exception as e:
        _debug_log(f"[MemoryAudit] retention sweep failed: {e}")


def iter_audit_records(
    window: Any,
    chat_id: str | None = None,
) -> "Iterator[dict[str, Any]]":
    """Yield audit records (dict) emitted within the given ``window``.

    ``window`` may be a :class:`datetime.timedelta` (preferred) or an int
    (interpreted as seconds for back-compat). Both rotated archives and
    the live file are scanned, oldest → newest. Each line is parsed with
    :func:`json.loads`; unparseable lines are silently skipped. Records
    without a ``ts`` field are skipped.
    """
    from datetime import timedelta as _td
    if isinstance(window, _td):
        delta = window
    else:
        try:
            delta = _td(seconds=float(window))
        except Exception:
            delta = _td(days=1)
    cutoff = datetime.now(timezone.utc) - delta

    paths = sorted(_audit_dir_files(), key=lambda p: p.stat().st_mtime if p.exists() else 0)
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    ts = record.get("ts", "")
                    if not ts:
                        continue
                    try:
                        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    except Exception:
                        continue
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if dt < cutoff:
                        continue
                    if chat_id is not None and record.get("chat_id") != chat_id:
                        continue
                    yield record
        except Exception as e:
            _debug_log(f"[MemoryAudit] iter_audit_records read failed for {path.name}: {e}")


__all__ = [
    "POLICY_TABLE",
    "get_policy",
    "load_slices",
    "KeywordRetriever",
    "EmbeddingRetriever",
    "HybridRetriever",
    "compose_slices_to_context",
    "_retriever_active",
    "_audit_decision",
    "_build_audit_record",
    "build_audit_record_v2",
    "resolve_actual_mode",
    "get_retriever",
    "rotate_audit_files",
    "iter_audit_records",
]
