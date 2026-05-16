"""Unit tests for ``larkhelm.memory_embedding`` (Phase D / Phase 2).

Covers: stub determinism, lazy init, missing-dep fallback, HTTP circuit
breaker, timeout, cache hit/miss, dim invalidation, batch shape,
factory ``None`` return when backend == "none", and local fallback when
deps are missing.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest import mock

import pytest

# Skip everything if numpy is unavailable in this env.
np = pytest.importorskip("numpy")


from larkhelm.memory_embedding import (  # noqa: E402
    EmbeddingCache,
    EmbeddingError,
    HTTPEmbedding,
    LocalONNXEmbedding,
    StubEmbedding,
    get_embedding_backend,
)


def test_stub_dim_correct():
    b = StubEmbedding(dim=8)
    out = b.embed(["hello", "你好", ""])
    assert out.shape == (3, 8)
    assert out.dtype == np.float32


def test_stub_deterministic():
    a = StubEmbedding(dim=16).embed(["the same text"])
    b = StubEmbedding(dim=16).embed(["the same text"])
    assert np.allclose(a, b)


def test_local_missing_onnx_raises(tmp_path, unload_module):
    """When onnxruntime is missing, _lazy_init raises EmbeddingError."""
    p = tmp_path / "fake.onnx"
    p.write_bytes(b"")  # exists but won't load
    backend = LocalONNXEmbedding(model_path=str(p), dim=8)
    # Make ``import onnxruntime`` fail (ModuleNotFoundError) for this test only.
    # Migrated off the legacy sys.modules-mocking idiom — see REQ-09 of the
    # P0 PRD.
    unload_module("onnxruntime")
    with pytest.raises(EmbeddingError):
        backend.embed(["hi"])


def test_local_lazy_warm_does_not_raise(tmp_path, unload_module):
    """warm() must never raise — silent failure contract."""
    p = tmp_path / "nope.onnx"
    backend = LocalONNXEmbedding(model_path=str(p), dim=8)
    # Should NOT raise even with no file and no onnxruntime.
    unload_module("onnxruntime")
    backend.warm()


def test_http_circuit_breaker_opens_after_5_failures(monkeypatch):
    b = HTTPEmbedding(endpoint="http://invalid.example", dim=8, timeout=0.1)

    def _always_fail(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", _always_fail)
    for _ in range(5):
        with pytest.raises(EmbeddingError):
            b.embed(["x"])
    # 6th call: circuit OPEN — raises BEFORE attempting network.
    with pytest.raises(EmbeddingError) as ei:
        b.embed(["x"])
    assert "circuit open" in str(ei.value).lower()


def test_http_timeout_raises_embedding_error(monkeypatch):
    b = HTTPEmbedding(endpoint="http://example.invalid", dim=8, timeout=0.05)

    def _timeout(*_a, **_k):
        import urllib.error
        raise urllib.error.URLError("timeout")

    monkeypatch.setattr("urllib.request.urlopen", _timeout)
    with pytest.raises(EmbeddingError):
        b.embed(["x"])


def test_cache_hit_miss():
    cache = EmbeddingCache(maxsize=64)
    backend = StubEmbedding(dim=8)
    v1 = cache.get_or_compute("slice-A", "body text", backend)
    v2 = cache.get_or_compute("slice-A", "body text", backend)
    # Same object identity: second call comes from store (no recompute).
    assert v1 is v2
    assert len(cache) == 1


def test_cache_dim_change_invalidates():
    cache = EmbeddingCache(maxsize=64)
    b8 = StubEmbedding(dim=8)
    b16 = StubEmbedding(dim=16)
    v8 = cache.get_or_compute("slice-A", "body", b8)
    v16 = cache.get_or_compute("slice-A", "body", b16)
    assert v8.shape == (8,)
    assert v16.shape == (16,)
    assert len(cache) == 2


def test_embed_batch_shape():
    out = StubEmbedding(dim=4).embed(["a", "b", "c"])
    assert out.shape == (3, 4)
    out_empty = StubEmbedding(dim=4).embed([])
    assert out_empty.shape == (0, 4)


def test_get_embedding_backend_none_returns_none(monkeypatch):
    monkeypatch.setattr("larkhelm.memory_embedding._BACKEND_SINGLETON", None, raising=False)
    monkeypatch.setattr("larkhelm.memory_embedding._BACKEND_KEY", (), raising=False)
    assert get_embedding_backend({"embedding_backend": "none"}) is None


def test_get_embedding_backend_local_fallback_when_missing_deps(monkeypatch, tmp_path):
    monkeypatch.setattr("larkhelm.memory_embedding._BACKEND_SINGLETON", None, raising=False)
    monkeypatch.setattr("larkhelm.memory_embedding._BACKEND_KEY", (), raising=False)
    # Empty model path → LocalONNXEmbedding init raises EmbeddingError → factory degrades to None.
    out = get_embedding_backend({"embedding_backend": "local", "embedding_model_path": "", "embedding_dim": 8})
    assert out is None


def test_get_embedding_backend_stub_via_factory(monkeypatch):
    monkeypatch.setattr("larkhelm.memory_embedding._BACKEND_SINGLETON", None, raising=False)
    monkeypatch.setattr("larkhelm.memory_embedding._BACKEND_KEY", (), raising=False)
    b = get_embedding_backend({"embedding_backend": "stub", "embedding_dim": 8})
    assert b is not None
    assert b.name == "stub"
    assert b.dim == 8


def test_http_payload_too_large_records_failure(monkeypatch):
    b = HTTPEmbedding(endpoint="http://nope.example", dim=8, timeout=0.1)
    huge = "x" * (2 * 1024 * 1024)  # > 1MiB payload cap
    with pytest.raises(EmbeddingError):
        b.embed([huge])
    assert b._fail_count >= 1
