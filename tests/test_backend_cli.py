"""P1-7: tests for backend_cli runners (run_claude / run_gemini / run_kimi / run_deepseek)."""
from __future__ import annotations

import os
import threading
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import backend_cli as _bcli  # noqa: E402
from larkhelm.ai_runner import QueryCancelledError  # noqa: E402
from larkhelm.backend_registry import BackendSpec  # noqa: E402


def _spec(sid: str = "test-cli", provider: str = "claude_cli") -> BackendSpec:
    return BackendSpec(
        id=sid, provider=provider, display_name="Test",
        role="orchestrator", tags=[], command="claude",
        healthy=True, enabled=True,
    )


@pytest.fixture(autouse=True)
def _silence_recorder(monkeypatch):
    """Stub _record_outcome so tests don't hit BACKEND_REGISTRY."""
    monkeypatch.setattr(_bcli, "_record_outcome", lambda *a, **kw: None)


# ── run_claude ────────────────────────────────────────────────────────


def test_run_claude_normal(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_claude_proc",
                        lambda **kw: "hello world")
    out = _bcli.run_claude(_spec(), "c1", "hi", None, "/tmp")
    assert out == "hello world"


def test_run_claude_records_failure(monkeypatch):
    calls = []
    monkeypatch.setattr(_bcli, "_spawn_claude_proc",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(_bcli, "_record_outcome",
                        lambda sid, exc: calls.append((sid, type(exc).__name__)))
    with pytest.raises(RuntimeError):
        _bcli.run_claude(_spec(), "c1", "hi", None, "/tmp")
    assert calls and calls[0][1] == "RuntimeError"


def test_run_claude_cancelled(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_claude_proc",
                        lambda **kw: (_ for _ in ()).throw(
                            QueryCancelledError("cancelled")
                        ))
    with pytest.raises(QueryCancelledError):
        _bcli.run_claude(_spec(), "c1", "hi", None, "/tmp")


def test_run_claude_timeout(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_claude_proc",
                        lambda **kw: (_ for _ in ()).throw(TimeoutError("idle")))
    with pytest.raises(TimeoutError):
        _bcli.run_claude(_spec(), "c1", "hi", None, "/tmp")


# ── run_gemini ────────────────────────────────────────────────────────


def test_run_gemini_normal(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_gemini_proc", lambda **kw: "g out")
    out = _bcli.run_gemini(_spec(provider="gemini_cli"), "c1", "hi", None, "/tmp")
    assert out == "g out"


def test_run_gemini_failure(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_gemini_proc",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("g fail")))
    with pytest.raises(RuntimeError):
        _bcli.run_gemini(_spec(provider="gemini_cli"), "c1", "hi", None, "/tmp")


def test_run_gemini_cancelled(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_gemini_proc",
                        lambda **kw: (_ for _ in ()).throw(
                            QueryCancelledError("cancelled")
                        ))
    with pytest.raises(QueryCancelledError):
        _bcli.run_gemini(_spec(provider="gemini_cli"), "c1", "hi", None, "/tmp")


def test_run_gemini_timeout(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_gemini_proc",
                        lambda **kw: (_ for _ in ()).throw(TimeoutError("t")))
    with pytest.raises(TimeoutError):
        _bcli.run_gemini(_spec(provider="gemini_cli"), "c1", "hi", None, "/tmp")


# ── run_kimi ──────────────────────────────────────────────────────────


def test_run_kimi_normal(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_kimi_proc", lambda **kw: "k out")
    out = _bcli.run_kimi(_spec(provider="kimi_cli"), "c1", "hi", None, "/tmp")
    assert out == "k out"


def test_run_kimi_failure(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_kimi_proc",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("k fail")))
    with pytest.raises(RuntimeError):
        _bcli.run_kimi(_spec(provider="kimi_cli"), "c1", "hi", None, "/tmp")


def test_run_kimi_cancelled(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_kimi_proc",
                        lambda **kw: (_ for _ in ()).throw(QueryCancelledError("c")))
    with pytest.raises(QueryCancelledError):
        _bcli.run_kimi(_spec(provider="kimi_cli"), "c1", "hi", None, "/tmp")


def test_run_kimi_timeout(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_kimi_proc",
                        lambda **kw: (_ for _ in ()).throw(TimeoutError("t")))
    with pytest.raises(TimeoutError):
        _bcli.run_kimi(_spec(provider="kimi_cli"), "c1", "hi", None, "/tmp")


# ── run_deepseek ──────────────────────────────────────────────────────


def test_run_deepseek_normal(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_deepseek_proc", lambda **kw: "d out")
    out = _bcli.run_deepseek(_spec(provider="deepseek_api"), "c1", "hi", None, "/tmp")
    assert out == "d out"


def test_run_deepseek_failure(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_deepseek_proc",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("d fail")))
    with pytest.raises(RuntimeError):
        _bcli.run_deepseek(_spec(provider="deepseek_api"), "c1", "hi", None, "/tmp")


def test_run_deepseek_cancelled(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_deepseek_proc",
                        lambda **kw: (_ for _ in ()).throw(QueryCancelledError("c")))
    with pytest.raises(QueryCancelledError):
        _bcli.run_deepseek(_spec(provider="deepseek_api"), "c1", "hi", None, "/tmp")


def test_run_deepseek_timeout(monkeypatch):
    monkeypatch.setattr(_bcli, "_spawn_deepseek_proc",
                        lambda **kw: (_ for _ in ()).throw(TimeoutError("t")))
    with pytest.raises(TimeoutError):
        _bcli.run_deepseek(_spec(provider="deepseek_api"), "c1", "hi", None, "/tmp")
