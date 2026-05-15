"""larkhelm · on-demand memory injection — pure data types.

Phase D / Phase 1 (keyword retriever) + Phase 2 (embedding / hybrid).

This module defines the data classes consumed by ``memory_retriever``,
``memory_context`` and ``memory_embedding``. Everything here is frozen +
stdlib-only so that the slice layer remains side-effect free and
trivially testable.

No I/O, no logging, no config access. Importing this module is safe in
any boot order. See ``.crew_workspace/design.md`` §3 and §4 for the full
specification of fields, default values, and POLICY_TABLE shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    # Forward-reference only — kept out of runtime so importing
    # ``larkhelm.memory_slice`` never drags numpy in.
    import numpy as np  # noqa: F401


SliceLayer = Literal["global", "project", "session"]

SliceKind = Literal[
    "fact",
    "preference",
    "decision",
    "task_history",
    "incident",
    "convention",
    "context_summary",
]

# Declared retrieval mode (super-set of what can be physically dispatched).
RetrievalMode = Literal["keyword", "embedding", "llm_router", "force", "hybrid"]

# Phase 2: narrowed to the 3 modes that can actually run end-to-end.
# Audit `mode` field is typed as this — `force`/`llm_router` are policy-level
# declarations and never appear in the executed retrieval path.
ActualRetrievalMode = Literal["keyword", "embedding", "hybrid"]


@dataclass(frozen=True)
class MemorySlice:
    """A single addressable memory unit, materialised from an H2 section of a
    layer file (or the whole file when no H2 is present)."""

    id: str
    layer: SliceLayer
    kind: SliceKind = "fact"
    title: str = ""
    body: str = ""
    importance: float = 0.5
    created_at: str = ""
    updated_at: str = ""
    keywords: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    char_len: int = 0
    # Phase 2: flagged True when ``memory_lifecycle`` finds this slice in
    # the layer's ``.meta.json`` ``stale_slice_ids`` list. Soft-delete:
    # retrievers multiply ``relevance`` by ``memory_stale_decay`` (default 0.5)
    # but never drop the slice outright — single-call unstale CLI can revive.
    stale: bool = False

    def with_score(self, score: float) -> "ScoredSlice":
        """Build a ScoredSlice wrapping this slice with the given total score.

        The factor components default to 0 — call sites that compute them
        should construct ScoredSlice directly with the full breakdown."""
        return ScoredSlice(slice=self, score=score)


@dataclass(frozen=True)
class ScoredSlice:
    """A slice plus its retrieval-time scoring breakdown."""

    slice: MemorySlice
    score: float
    recency_score: float = 0.0
    importance_score: float = 0.0
    relevance_score: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class RetrievalRequest:
    """Per-query input to ``MemoryRetriever.retrieve``.

    All fields default-valued so call sites can construct partial requests
    (eg. tests) without enumerating every flag."""

    chat_id: str
    cwd: str | None = None
    query: str = ""
    recent_turns: tuple[str, ...] = ()
    agent_type: str = "chat"
    sub_intent: str = ""
    complexity: str = "medium"
    confidence: float = 0.0
    is_explicit_cmd: bool = False
    has_doc_urls: bool = False
    has_images: bool = False
    has_code_fence: bool = False
    force_layers: tuple[SliceLayer, ...] = ()
    force_files: tuple[str, ...] = ()


@dataclass(frozen=True)
class InjectionPolicy:
    """How a given agent_type wants to assemble memory: budgets, weights, scoring."""

    agent_type: str
    token_budget: int
    layer_weights: dict[str, float]
    kind_priority: tuple[str, ...]
    require_kinds: tuple[str, ...] = ()
    exclude_kinds: tuple[str, ...] = ()
    retrieval_mode: RetrievalMode = "keyword"
    top_k: int = 6
    alpha_recency: float = 0.3
    alpha_importance: float = 0.3
    alpha_relevance: float = 0.4
    # Phase 2 — Hybrid linear-fusion weight: score = α·cos_sim + (1-α)·BM25_norm.
    hybrid_alpha: float = 0.6
    # Phase 2 — Multiplier applied to ``top_k`` when seeding the BM25 pool
    # for the embedding rerank stage. Pool size = ``top_k × multiplier``
    # (clipped at all candidates).
    embedding_top_k_multiplier: int = 3


class MemoryRetriever(Protocol):
    """Structural type for any retriever (keyword / embedding / hybrid)."""

    def retrieve(
        self,
        request: RetrievalRequest,
        policy: InjectionPolicy,
        candidate_slices: list[MemorySlice],
    ) -> list[ScoredSlice]:
        ...


class EmbeddingBackend(Protocol):
    """Structural type for any vector backend.

    Phase 2 contract: ``embed`` returns a ``(N, dim)`` float32 ndarray (string
    forward-ref so importing this module never drags numpy in). Failure modes
    are signalled by raising :class:`larkhelm.memory_embedding.EmbeddingError`.

    ``warm`` is a once-on-boot hook (lazy load model files, ping endpoint).
    Implementations MUST NOT raise inside ``warm`` — silent failure is the
    contract; the caller logs and continues.
    """

    name: str
    dim: int

    def embed(self, texts: list[str]) -> Any:
        """Return a ``(len(texts), dim)`` float32 ``np.ndarray``. Typed as
        ``Any`` so the protocol module never imports numpy at runtime; the
        concrete callers narrow on use."""
        ...

    def warm(self) -> None:
        ...


__all__ = [
    "SliceLayer",
    "SliceKind",
    "RetrievalMode",
    "ActualRetrievalMode",
    "MemorySlice",
    "ScoredSlice",
    "RetrievalRequest",
    "InjectionPolicy",
    "MemoryRetriever",
    "EmbeddingBackend",
]
