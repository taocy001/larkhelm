"""Test fixture: deterministic stub EmbeddingBackend.

Used by ``test_memory_*`` so each test can run hybrid retrieval without
pulling onnxruntime/numpy. Vectors are stable across runs (deterministic
hash → float32 buckets) so tests can assert ordering.

Usage::

    from tests.fixtures.onnx_stub_backend import make_stub
    backend = make_stub(dim=8)
"""
from __future__ import annotations

import hashlib


def make_stub(dim: int = 8):
    """Build a :class:`larkhelm.memory_embedding.StubEmbedding` (which is
    itself deterministic) — wrapper exists so future evolution of the
    factory signature doesn't need search-replace across every test."""
    from larkhelm.memory_embedding import StubEmbedding
    return StubEmbedding(dim=dim)


def deterministic_vector(text: str, dim: int = 8):
    """Return the same vector StubEmbedding would produce for one text.

    Helpful for tests asserting cache reuse without going through
    StubEmbedding.embed (e.g. when we want to compare bytes).
    """
    try:
        import numpy as np
    except Exception as e:  # pragma: no cover — numpy is a dev dep here
        raise RuntimeError("tests/fixtures/onnx_stub_backend needs numpy") from e
    digest = hashlib.sha1((text or "").encode("utf-8")).digest()
    out = np.zeros((dim,), dtype=np.float32)
    for j in range(dim):
        out[j] = float(digest[j % len(digest)]) / 255.0 - 0.5
    n = float(np.linalg.norm(out))
    if n > 0:
        out /= n
    return out
