"""AC-08: per-backend doc inject TTL override tests."""
import time
import pytest

from larkhelm._context_cache import TTLCache, _doc_ttl_sec_for_backend


def test_ac08_ttl_override_shorter(monkeypatch):
    """AC-08a: ttl_override shorter than stored age → cache miss."""
    import larkhelm._context_cache as _cc
    cache: TTLCache[str, str] = TTLCache("test", ttl_sec=600)

    stored_at = time.monotonic()
    cache.put("key", "value")

    # Advance "now" by 90s relative to stored_at
    monkeypatch.setattr(_cc, "time", _make_time_mock(stored_at + 90))

    # ttl_override=60: 90s > 60s → expired → miss
    result = cache.get_with_age("key", ttl_override=60)
    assert result is None, f"Expected None (cache miss with ttl_override=60 at 90s), got {result!r}"


def test_ac08_ttl_override_longer(monkeypatch):
    """AC-08b: ttl_override longer than stored age → cache hit."""
    import larkhelm._context_cache as _cc
    cache: TTLCache[str, str] = TTLCache("test", ttl_sec=600)

    stored_at = time.monotonic()
    cache.put("key", "value")

    # Advance "now" by 90s relative to stored_at
    monkeypatch.setattr(_cc, "time", _make_time_mock(stored_at + 90))

    # ttl_override=300: 90s < 300s → still valid → hit
    result = cache.get_with_age("key", ttl_override=300)
    assert result is not None, "Expected cache hit with ttl_override=300 at 90s"
    payload, age = result
    assert payload == "value"
    assert age >= 0


def test_doc_ttl_sec_for_backend_kimi(monkeypatch):
    """_doc_ttl_sec_for_backend('kimi') reads DOC_INJECT_CACHE_TTL_SEC_KIMI from config."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "DOC_INJECT_CACHE_TTL_SEC_KIMI", 60, raising=False)

    result = _doc_ttl_sec_for_backend("kimi")
    assert result == 60.0, f"Expected 60.0, got {result}"


def test_doc_ttl_sec_for_backend_unknown_fallback():
    """_doc_ttl_sec_for_backend with unknown backend falls back to _doc_ttl_sec()."""
    from larkhelm._context_cache import _doc_ttl_sec
    result = _doc_ttl_sec_for_backend("unknown_backend_xyz")
    assert result == _doc_ttl_sec(), f"Expected fallback to _doc_ttl_sec(), got {result}"


def test_doc_ttl_sec_for_backend_case_insensitive(monkeypatch):
    """Backend name is uppercased before attribute lookup."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "DOC_INJECT_CACHE_TTL_SEC_GEMINI", 300, raising=False)

    assert _doc_ttl_sec_for_backend("gemini") == 300.0
    assert _doc_ttl_sec_for_backend("Gemini") == 300.0
    assert _doc_ttl_sec_for_backend("GEMINI") == 300.0


# ── helpers ────────────────────────────────────────────────────────────────


def _make_time_mock(now_value: float):
    """Return a mock time module where monotonic() returns now_value."""
    import types
    import time as _real_time

    mock = types.SimpleNamespace()
    mock.monotonic = lambda: now_value
    mock.time = _real_time.time
    return mock
