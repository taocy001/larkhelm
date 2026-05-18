"""AC-03 — P3 REQ-03 intent embedding L2 classifier.

The classifier should:
1. Return ``IntentResult(layer='L2-embed', agent_type='dev')`` when the
   query embedding is closest to the dev description vector.
2. Return ``None`` (caller falls back to LLM) when no agent's cosine
   meets the threshold.
3. Be missing entirely when the embedding backend is unavailable — the
   router falls back to the LLM JSON path silently.
"""
from __future__ import annotations

import unittest

from larkhelm.agent_hub.intent_embedding import EmbeddingIntentClassifier


class _FakeEmbeddingBackend:
    """Maps known strings to handcrafted dim=4 vectors so cosine is predictable."""

    def __init__(self, table: dict[str, list[float]]):
        self.name = "fake"
        self.dim = 4
        self._table = table

    def embed(self, texts):  # noqa: D401 - protocol shim
        out = []
        for t in texts:
            vec = self._table.get(t, [0.0] * self.dim)
            out.append(vec)
        return out


class TestEmbeddingIntentClassifier(unittest.TestCase):

    def test_classify_picks_top_cosine(self) -> None:
        backend = _FakeEmbeddingBackend({
            "dev description": [1.0, 0.0, 0.0, 0.0],
            "chat description": [0.0, 1.0, 0.0, 0.0],
            "doc description": [0.0, 0.0, 1.0, 0.0],
            # Query lines up almost perfectly with 'dev description'.
            "实现一个 OAuth 模块": [0.9, 0.05, 0.05, 0.0],
        })
        classifier = EmbeddingIntentClassifier(backend, threshold=0.30)
        classifier.precompute([
            ("dev", "dev description"),
            ("chat", "chat description"),
            ("doc", "doc description"),
        ])
        result = classifier.classify("实现一个 OAuth 模块")
        self.assertIsNotNone(result)
        self.assertEqual(result.agent_type, "dev")
        self.assertEqual(result.layer, "L2-embed")
        self.assertGreater(result.confidence, 0.9)

    def test_below_threshold_returns_none(self) -> None:
        backend = _FakeEmbeddingBackend({
            "dev description": [1.0, 0.0, 0.0, 0.0],
            "chat description": [0.0, 1.0, 0.0, 0.0],
            # Query is uniformly small — every cosine well below 0.5.
            "天气如何": [0.1, 0.1, 0.1, 0.1],
        })
        classifier = EmbeddingIntentClassifier(backend, threshold=0.95)
        classifier.precompute([
            ("dev", "dev description"),
            ("chat", "chat description"),
        ])
        self.assertIsNone(classifier.classify("天气如何"))

    def test_empty_table_returns_none(self) -> None:
        backend = _FakeEmbeddingBackend({"anything": [0.1, 0.1, 0.1, 0.1]})
        classifier = EmbeddingIntentClassifier(backend, threshold=0.30)
        classifier.precompute([])  # no agents precomputed
        self.assertIsNone(classifier.classify("anything"))

    def test_backend_failure_returns_none(self) -> None:
        class _BrokenBackend:
            name = "broken"
            dim = 4

            def embed(self, texts):
                raise RuntimeError("embedding down")

        classifier = EmbeddingIntentClassifier(_BrokenBackend(), threshold=0.30)
        classifier.precompute([("dev", "dev description")])
        self.assertEqual(classifier._agent_vectors, {})
        # Even if precompute somehow populated, classify must handle backend errors.
        self.assertIsNone(classifier.classify("anything"))


class TestIntentRouterFallback(unittest.TestCase):

    def test_try_embedding_l2_returns_none_when_backend_unavailable(self) -> None:
        """When :func:`get_embedding_backend` returns ``None``, the helper
        must return ``None`` so the caller falls back to the LLM path."""
        from larkhelm.agent_hub import intent_router as ir
        from unittest.mock import patch

        with patch("larkhelm.memory_embedding.get_embedding_backend", return_value=None):
            result = ir._try_embedding_l2("讲个笑话", [("dev", "dev description")])
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
