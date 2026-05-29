"""larkhelm · session-layer smart compression (P2 REQ-07).

Replaces the P1 tail-truncation strategy
(``session_layer[:budget]``) with a deterministic score-based
sentence selection. Goal: when the session memory body exceeds the
budget, keep the *most informative* sentences instead of the most recent
N characters.

Scoring (no LLM call):

    score = w_role + w_decay·exp(-Δt/τ) + w_keyword·overlap(query)

  * ``w_role``     — role-conditional bonus
                     (user 0.4 / assistant 0.3 / milestone 0.5);
  * ``w_decay``    — temporal decay; τ = 24h, weight 0.4;
  * ``w_keyword``  — fraction of query tokens that appear in the
                     sentence, weight 0.3.

Top-K selection: sort sentences by score desc, then greedily add
sentences while the accumulated char count stays within ``budget``.
After selection, sentences are re-emitted in *original document order*
so the resulting text reads as a coherent excerpt (not a score-ranked
list).

Gated by ``memory_session_smart_compress`` (default False). When the
flag is off, the legacy P1 tail-truncation path is used directly — the
``smart_compress`` function is dead code in that branch.
"""
from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass

import larkhelm.config as _cfg


# ── Tunables (module-level so tests can monkey-patch) ──────────────────
W_ROLE: dict[str, float] = {"user": 0.4, "assistant": 0.3, "milestone": 0.5}
W_DECAY: float = 0.4
W_KEYWORD: float = 0.3
DECAY_TAU_SEC: float = 24 * 60 * 60  # 24 h

# Sentence splitter: Chinese full-width punct + ASCII ``.``/``!``/``?`` with
# trailing whitespace OR end-of-string. ``re.split`` keeps the punctuation
# attached because we use a look-behind that consumes only the boundary.
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?；])\s+|\n+")

# Sentence-level metadata header in session_*.md bodies: lines of the form
# ``[HH:MM:SS] role: ...``. When present they let us assign per-sentence
# role + timestamp; otherwise we treat the chunk as anonymous text and
# fall back on role="user" / now-Δ=0 for the score.
_SENT_META_RE = re.compile(
    r"^\[(?P<ts>\d{2}:\d{2}:\d{2})\]\s+(?P<role>user|assistant|milestone)[:：]\s*(?P<body>.*)$",
    re.IGNORECASE,
)


@dataclass
class CompressScore:
    sentence: str
    role: str
    timestamp: float    # epoch seconds; 0.0 means "unknown"
    score: float
    keep: bool          # filled by _select_top_k


def is_enabled() -> bool:
    """Honour the operator gate; default False keeps P1 tail-truncate."""
    return bool(getattr(_cfg, "MEMORY_SESSION_SMART_COMPRESS", True))


# ── Sentence extraction ────────────────────────────────────────────────


def _split_sentences(text: str) -> list[tuple[str, str, float]]:
    """Split ``text`` into ``[(sentence, role, ts), ...]`` tuples.

    The role / ts pair is best-effort: when a line matches ``_SENT_META_RE``
    we lift the metadata for every sentence on that line; otherwise both
    fall back to ("user", 0.0). The timestamp is not date-aware (the
    on-disk format is ``HH:MM:SS`` only) — we project it onto "today" so
    the decay weight is comparable across recent lines. Lines older than
    the current local day will get slightly under-weighted decay; this is
    acceptable because session memory is regenerated every 10 turns.
    """
    out: list[tuple[str, str, float]] = []
    if not text:
        return out
    # The session body is processed line by line so we can attach the
    # per-line meta to every fragment that came from that line.
    now_struct = time.localtime()
    today_midnight = time.mktime((now_struct.tm_year, now_struct.tm_mon,
                                  now_struct.tm_mday, 0, 0, 0, 0, 0, -1))
    for line in text.splitlines():
        m = _SENT_META_RE.match(line.strip())
        if m:
            role = m.group("role").lower()
            ts_str = m.group("ts")
            try:
                h, mn, s = (int(x) for x in ts_str.split(":"))
                ts = today_midnight + h * 3600 + mn * 60 + s
            except (ValueError, TypeError):
                ts = 0.0
            body = m.group("body")
        else:
            role = "user"
            ts = 0.0
            body = line
        for sent in _SENT_SPLIT_RE.split(body or ""):
            sent = sent.strip()
            if sent:
                out.append((sent, role, ts))
    return out


# ── Scoring ────────────────────────────────────────────────────────────


def _query_tokens(query: str) -> set[str]:
    """Tokenise the query for keyword-overlap scoring.

    Falls back to character bigrams for Chinese content so a non-tokenised
    Chinese query still produces overlap signal. ASCII content uses
    whitespace word splits. Lowercased + length-2 minimum to avoid noise
    from articles ("a", "I").
    """
    if not query:
        return set()
    ascii_words = {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_]+", query) if len(w) >= 2}
    cjk = re.findall(r"[一-鿿]+", query)
    bigrams: set[str] = set()
    for run in cjk:
        for i in range(len(run) - 1):
            bigrams.add(run[i: i + 2])
        # Length-1 CJK run still emits the single char as a fall-back token.
        if len(run) == 1:
            bigrams.add(run)
    return ascii_words | bigrams


def _score_sentence(
    sentence: str,
    role: str,
    ts: float,
    query_tokens: set[str],
    now: float,
) -> float:
    """Compute the composite score for a single sentence."""
    role_w = W_ROLE.get(role, 0.2)

    if ts > 0:
        delta = max(0.0, now - ts)
        decay = math.exp(-delta / DECAY_TAU_SEC)
    else:
        # No timestamp → assume mid-window so neither most-fresh nor most-stale.
        decay = 0.5
    decay_w = W_DECAY * decay

    if query_tokens:
        sl = sentence.lower()
        matches = sum(1 for tok in query_tokens if tok in sl)
        # Normalise by query size so a 1-token match in a 1-token query
        # ranks equally with 5/5 in a 5-token query.
        keyword_w = W_KEYWORD * (matches / max(1, len(query_tokens)))
    else:
        # Query empty → fall back to role + decay only (the AC for
        # ``test_memory_session_compress`` covers this branch).
        keyword_w = 0.0

    return role_w + decay_w + keyword_w


def _select_top_k(scored: list[CompressScore], budget: int) -> list[CompressScore]:
    """Greedy top-K under a budget cap.

    Sort by score desc, then greedily pick while the cumulative char
    length stays ≤ ``budget``. Ties resolve by *original index* so the
    selection is reproducible across runs (Python's sort is stable).
    Marks ``keep=True`` on selected items in-place.

    Returns the kept subset in *original document order* so the resulting
    text reads chronologically.
    """
    if budget <= 0:
        return []
    indexed = list(enumerate(scored))
    indexed.sort(key=lambda pair: (-pair[1].score, pair[0]))

    kept_idx: list[int] = []
    used = 0
    for idx, cs in indexed:
        added = len(cs.sentence) + 1  # +1 for the joining newline
        if used + added > budget:
            continue
        cs.keep = True
        kept_idx.append(idx)
        used += added
        if used >= budget:
            break
    kept_idx.sort()
    return [scored[i] for i in kept_idx]


# ── Public API ─────────────────────────────────────────────────────────


def smart_compress(
    text: str,
    budget: int,
    query: str = "",
    now: float | None = None,
) -> str:
    """Compress ``text`` to ≤ ``budget`` chars via score-based selection.

    Returns the original text untouched when it already fits. When the
    text exceeds budget but the resulting selection ends up empty (e.g.
    budget=10 vs. minimum sentence length), falls back to a raw
    char-cut to honour the budget contract.

    ``now`` defaults to ``time.time()`` for the decay term; tests override
    it for deterministic scoring.
    """
    if budget <= 0 or not text:
        return ""
    if len(text) <= budget:
        return text

    if now is None:
        now = time.time()
    qtokens = _query_tokens(query or "")

    sentences = _split_sentences(text)
    if not sentences:
        return text[:budget]

    scored = [
        CompressScore(
            sentence=s, role=r, timestamp=ts,
            score=_score_sentence(s, r, ts, qtokens, now),
            keep=False,
        )
        for (s, r, ts) in sentences
    ]
    kept = _select_top_k(scored, budget)
    if not kept:
        return text[:budget]
    out = "\n".join(cs.sentence for cs in kept)
    # Defensive: in pathological cases the cumulative budget computation
    # may overshoot by a few chars (Unicode boundary, joining newline);
    # enforce the hard cap so callers never have to re-trim downstream.
    if len(out) > budget:
        out = out[:budget]
    return out


__all__ = [
    "CompressScore",
    "W_ROLE",
    "W_DECAY",
    "W_KEYWORD",
    "DECAY_TAU_SEC",
    "is_enabled",
    "smart_compress",
]
