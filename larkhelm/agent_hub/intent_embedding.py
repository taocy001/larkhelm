"""larkhelm · agent_hub.intent_embedding — REQ-03 cosine-based L2 classifier.

When ``intent_layer2_strategy="embedding"`` is set, the intent router
delegates the L2 step to :class:`EmbeddingIntentClassifier` instead of
the cheap-LLM JSON path:

1. Precompute a dense vector for each registered agent's
   ``description`` once.
2. On each incoming query, embed the query text via the same
   :class:`EmbeddingBackend` and pick the agent with the highest
   cosine similarity. If the top score is below ``threshold`` we
   return ``None`` so the caller falls back to the LLM path.

The classifier owns no I/O — it sees only an :class:`EmbeddingBackend`
reference and a precomputed table. The router decides when (and
whether) to invoke it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from larkhelm.agent_hub.intent_types import IntentResult
from larkhelm.log import safe_log as _safe_log

if TYPE_CHECKING:
    from larkhelm.memory_slice import EmbeddingBackend


class EmbeddingIntentClassifier:
    """Cosine top-1 agent_type picker over precomputed description vectors."""

    def __init__(
        self,
        backend: "EmbeddingBackend",
        threshold: float = 0.30,
    ) -> None:
        self._backend = backend
        self._threshold = float(threshold)
        # Stored as plain dict for predictable iteration order;
        # vectors are ``np.ndarray`` but typed as ``object`` here so
        # numpy isn't pulled in at import time.
        self._agent_vectors: dict[str, object] = {}

    def precompute(self, agents: list[tuple[str, str]]) -> None:
        """Embed each ``(agent_type, description)`` pair and cache.

        Idempotent. Calling again replaces the cache so a re-registered
        agent picks up its new description on the next classify().
        """
        if not agents:
            self._agent_vectors = {}
            return
        types_list: list[str] = []
        texts: list[str] = []
        for agent_type, desc in agents:
            atype = str(agent_type or "").strip()
            text = str(desc or "").strip()
            if not atype or not text:
                continue
            types_list.append(atype)
            texts.append(text)
        if not texts:
            self._agent_vectors = {}
            return
        try:
            vectors = self._backend.embed(texts)
        except Exception as e:
            _safe_log(f"[IntentEmbedding] precompute failed: {e}")
            self._agent_vectors = {}
            return
        # vectors is an (N, dim) ndarray; index row-wise.
        self._agent_vectors = {
            types_list[i]: vectors[i] for i in range(len(types_list))
        }

    def classify(self, text: str) -> Optional[IntentResult]:
        """Return the best-match :class:`IntentResult` or ``None``.

        ``None`` is returned when the cache is empty, when ``backend.embed``
        raises, or when the top cosine score is below ``threshold`` —
        callers fall back to the LLM path in all three cases.
        """
        text = (text or "").strip()
        if not text or not self._agent_vectors:
            return None
        try:
            qvec_batch = self._backend.embed([text])
        except Exception as e:
            _safe_log(f"[IntentEmbedding] classify embed failed: {e}")
            return None
        try:
            qvec = qvec_batch[0]
        except Exception:
            return None

        best_type = ""
        best_score = -1.0
        for agent_type, avec in self._agent_vectors.items():
            try:
                score = self._cosine(qvec, avec)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_type = agent_type
        if not best_type or best_score < self._threshold:
            return None
        return IntentResult(
            agent_type=best_type,
            layer="L2-embed",
            confidence=float(best_score),
            raw_text=text,
            reasoning=f"cosine={best_score:.3f}",
        )

    @staticmethod
    def _cosine(a, b) -> float:
        """Numpy-free cosine: works against any sequence-like vector.

        Accepts ndarray, list, or tuple of floats. Falls back to a Python
        loop because the test stub may not actually use numpy.
        """
        try:
            # If they're numpy arrays this path is fast.
            import numpy as np  # type: ignore
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na == 0.0 or nb == 0.0:
                    return 0.0
                return float(np.dot(a, b) / (na * nb))
        except Exception:
            pass
        # Generic path.
        seq_a = list(a)
        seq_b = list(b)
        if len(seq_a) != len(seq_b) or not seq_a:
            return 0.0
        dot = sum(float(x) * float(y) for x, y in zip(seq_a, seq_b))
        norm_a = sum(float(x) * float(x) for x in seq_a) ** 0.5
        norm_b = sum(float(x) * float(x) for x in seq_b) ** 0.5
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)


__all__ = ["EmbeddingIntentClassifier"]
