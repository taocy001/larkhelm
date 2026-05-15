"""larkhelm · memory recall evaluation — Phase D / Phase 2.

Offline metric helpers for recall@k / MRR over a JSONL golden set.

The runtime memory retriever is non-deterministic only in that the audit
queue is async; the retrieval algorithm itself is deterministic given
the same slice set and config. This module loads slices via
:func:`load_slices`, runs the configured retriever (hybrid > keyword
fallback), and produces a compact metrics dict suitable for CI gating.

CI does not force this module to run — there is no automatic ``score
against golden`` pytest. Operators run it manually:

    python -m larkhelm.memory_eval tests/fixtures/golden_recall.jsonl

(Implementation note: the CLI is intentionally trivial — wider operator
surface lives in :mod:`larkhelm.__main__`. We keep this module
side-effect-free so it imports cleanly during pytest collection.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from larkhelm.memory_retriever import (
    KeywordRetriever,
    get_policy,
    get_retriever,
    load_slices,
    resolve_actual_mode,
)
from larkhelm.memory_slice import RetrievalRequest


def _recall_at_k(retrieved_titles: list[str], expected_titles: list[str], k: int) -> float:
    """Return |retrieved[:k] ∩ expected| / |expected|, or 0 if expected is empty."""
    if not expected_titles:
        return 0.0
    expected = {str(t).strip() for t in expected_titles if t}
    top = {str(t).strip() for t in retrieved_titles[:k] if t}
    if not expected:
        return 0.0
    return len(expected & top) / len(expected)


def _mrr(retrieved_titles: list[str], expected_titles: list[str]) -> float:
    """Mean Reciprocal Rank — uses the position of the first hit (1-indexed)."""
    if not expected_titles or not retrieved_titles:
        return 0.0
    expected = {str(t).strip() for t in expected_titles if t}
    for rank, t in enumerate(retrieved_titles, start=1):
        if str(t).strip() in expected:
            return 1.0 / rank
    return 0.0


def score_against_golden(golden_path: Path) -> dict:
    """Read ``golden_path`` (JSONL) and compute aggregate metrics.

    Each line schema::

        {"query": "...",
         "expected_slice_titles": ["title1", "title2"],
         "chat_id": "...",       # optional — defaults to "eval"
         "cwd": "...",           # optional
         "agent_type": "dev"}   # optional — defaults to "chat"
    """
    path = Path(golden_path)
    if not path.exists():
        return {"recall_at_k": 0.0, "mrr": 0.0, "n": 0, "parse_errors": 0}

    try:
        from larkhelm.memory_embedding import get_embedding_backend
    except Exception:
        # NIT-03: avoid the ``_cfg`` parameter name — project-wide convention
        # is ``import larkhelm.config as _cfg``, so this shadowing was a
        # readability trap. ``_config`` is unambiguous.
        get_embedding_backend = lambda _config=None: None  # type: ignore[misc]

    n = 0
    parse_errors = 0
    recall_total = 0.0
    mrr_total = 0.0

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except Exception:
            parse_errors += 1
            continue
        query = str(row.get("query", "") or "").strip()
        expected = list(row.get("expected_slice_titles", []) or [])
        if not query or not expected:
            parse_errors += 1
            continue
        chat_id = str(row.get("chat_id", "eval") or "eval")
        cwd = row.get("cwd") or None
        agent_type = str(row.get("agent_type", "chat") or "chat")

        policy = get_policy(agent_type)
        try:
            slices = load_slices(chat_id, cwd)
        except Exception:
            slices = []
        request = RetrievalRequest(chat_id=chat_id, cwd=cwd, query=query, agent_type=agent_type)
        backend = None
        try:
            backend = get_embedding_backend()
        except Exception:
            backend = None
        actual_mode = resolve_actual_mode(policy, chat_id)
        if actual_mode in ("embedding", "hybrid") and backend is None:
            actual_mode = "keyword"
        try:
            retriever = get_retriever(actual_mode, backend=backend)
            scored = retriever.retrieve(request, policy, slices)
        except Exception:
            scored = KeywordRetriever().retrieve(request, policy, slices)

        titles = [s.slice.title for s in scored]
        recall_total += _recall_at_k(titles, expected, k=int(policy.top_k or 6))
        mrr_total += _mrr(titles, expected)
        n += 1

    if n == 0:
        return {"recall_at_k": 0.0, "mrr": 0.0, "n": 0, "parse_errors": parse_errors}
    return {
        "recall_at_k": recall_total / n,
        "mrr": mrr_total / n,
        "n": n,
        "parse_errors": parse_errors,
    }


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print("usage: python -m larkhelm.memory_eval <golden.jsonl>", file=sys.stderr)
        return 2
    result = score_against_golden(Path(args[0]))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = ["score_against_golden", "_recall_at_k", "_mrr", "main"]
