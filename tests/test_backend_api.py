"""P1-7: tests for backend_api runners (run_anthropic / run_google / run_openai_compat)."""
from __future__ import annotations

import os
import threading
import types
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import backend_api as _bapi  # noqa: E402
from larkhelm.ai_runner import QueryCancelledError  # noqa: E402
from larkhelm.backend_registry import BackendSpec  # noqa: E402


def _spec(sid: str = "test-api", provider: str = "anthropic_api") -> BackendSpec:
    return BackendSpec(
        id=sid, provider=provider, display_name="Test",
        role="orchestrator", tags=[], api_key="sk-test",
        healthy=True, enabled=True,
    )


@pytest.fixture(autouse=True)
def _silence_recorder(monkeypatch):
    monkeypatch.setattr(_bapi, "_record_outcome", lambda *a, **kw: None)


# ── run_anthropic ─────────────────────────────────────────────────────


def _make_anthropic_stub(chunks=None, raises=None):
    """Build a tiny anthropic.Anthropic stub for streaming tests."""
    if chunks is None:
        chunks = ["hello", " world"]

    class _Stream:
        def __init__(self, _chunks):
            self._chunks = _chunks

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        @property
        def text_stream(self):
            if raises is not None:
                raise raises
            return iter(self._chunks)

    class _Messages:
        def stream(self, **kw):
            return _Stream(chunks)

    class _Anthropic:
        def __init__(self, **kw):
            self.messages = _Messages()

    mod = types.SimpleNamespace(Anthropic=_Anthropic)
    return mod


def test_run_anthropic_normal_stream():
    mod = _make_anthropic_stub(["foo", " bar"])
    out, hist = _bapi.run_anthropic(
        _spec(), "c1", "hi", history=[],
        _anthropic_module=mod,
    )
    assert out == "foo bar"
    assert hist[-2]["role"] == "user"
    assert hist[-1]["role"] == "assistant"


def test_run_anthropic_auth_error_raises():
    mod = _make_anthropic_stub(raises=RuntimeError("401 unauthorized"))
    with pytest.raises(RuntimeError):
        _bapi.run_anthropic(
            _spec(), "c1", "hi", history=[],
            _anthropic_module=mod,
        )


def test_run_anthropic_5xx_raises():
    mod = _make_anthropic_stub(raises=RuntimeError("503 server overloaded"))
    with pytest.raises(RuntimeError):
        _bapi.run_anthropic(_spec(), "c1", "hi", history=[],
                            _anthropic_module=mod)


def test_run_anthropic_cancel_midstream():
    ev = threading.Event()
    ev.set()
    mod = _make_anthropic_stub(["a", "b", "c"])
    with pytest.raises(QueryCancelledError):
        _bapi.run_anthropic(
            _spec(), "c1", "hi", history=[],
            cancel_ev=ev, _anthropic_module=mod,
        )


# ── run_google ────────────────────────────────────────────────────────


def _make_google_stub(chunks=None, raises=None):
    chunks = chunks or ["g", "o"]

    class _Chunk:
        def __init__(self, t): self.text = t

    class _Models:
        def generate_content_stream(self, model, contents, config=None):
            if raises is not None:
                raise raises
            return (_Chunk(c) for c in chunks)

    class _Client:
        def __init__(self, api_key=None): self.models = _Models()

    class _Part:
        def __init__(self, text=""): self.text = text

    class _Content:
        def __init__(self, role="user", parts=None): self.role = role; self.parts = parts or []

    class _Config:
        def __init__(self, system_instruction=""): self.system_instruction = system_instruction

    genai = types.SimpleNamespace(Client=_Client)
    genai_types = types.SimpleNamespace(
        Content=_Content, Part=_Part,
        GenerateContentConfig=_Config,
    )
    return types.SimpleNamespace(genai=genai, genai_types=genai_types)


def test_run_google_normal_stream():
    mod = _make_google_stub(["alpha ", "beta"])
    out, hist = _bapi.run_google(
        _spec(provider="google_api"), "c1", "hi", history=[],
        _google_module=mod,
    )
    assert out == "alpha beta"
    assert hist[-1]["role"] == "assistant"


def test_run_google_auth_error():
    mod = _make_google_stub(raises=RuntimeError("403 forbidden"))
    with pytest.raises(RuntimeError):
        _bapi.run_google(_spec(provider="google_api"), "c1", "hi",
                         history=[], _google_module=mod)


def test_run_google_5xx_error():
    mod = _make_google_stub(raises=RuntimeError("500"))
    with pytest.raises(RuntimeError):
        _bapi.run_google(_spec(provider="google_api"), "c1", "hi",
                         history=[], _google_module=mod)


def test_run_google_cancel_midstream():
    ev = threading.Event()
    ev.set()
    mod = _make_google_stub(["a", "b"])
    with pytest.raises(QueryCancelledError):
        _bapi.run_google(_spec(provider="google_api"), "c1", "hi",
                         history=[], cancel_ev=ev, _google_module=mod)


# ── run_openai_compat ─────────────────────────────────────────────────


def _install_fake_openai(monkeypatch, raises=None, chunks=None):
    chunks = chunks or [{"choices": [{"delta": {"content": "x"}}]},
                        {"choices": [{"delta": {"content": "y"}}]}]

    class _Delta:
        def __init__(self, content=""):
            self.content = content

    class _Choice:
        def __init__(self, content):
            self.delta = _Delta(content)

    class _Chunk:
        def __init__(self, content):
            self.choices = [_Choice(content)]

    class _Stream:
        def __init__(self, items):
            self._items = items

        def __enter__(self):
            return iter(self._items)

        def __exit__(self, *args):
            return False

        def __iter__(self):
            return iter(self._items)

    class _Completions:
        def create(self, model, messages, stream, **kw):
            if raises is not None:
                raise raises
            items = [
                _Chunk(c["choices"][0]["delta"].get("content", ""))
                for c in chunks
            ]
            return _Stream(items)

    class _Chat:
        def __init__(self): self.completions = _Completions()

    class _Client:
        def __init__(self, **kw):
            self.chat = _Chat()

    fake_openai = types.SimpleNamespace(OpenAI=_Client)
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_openai)


def test_run_openai_compat_normal(monkeypatch):
    _install_fake_openai(monkeypatch,
                         chunks=[
                             {"choices": [{"delta": {"content": "hi "}}]},
                             {"choices": [{"delta": {"content": "there"}}]},
                         ])
    out, hist = _bapi.run_openai_compat(
        _spec(provider="openai_compat_api"), "c1", "yo", history=[],
    )
    assert "hi" in out and "there" in out
    assert hist[-1]["role"] == "assistant"


def test_run_openai_compat_auth_error(monkeypatch):
    _install_fake_openai(monkeypatch, raises=RuntimeError("401"))
    with pytest.raises(RuntimeError):
        _bapi.run_openai_compat(
            _spec(provider="openai_compat_api"), "c1", "yo", history=[],
        )


def test_run_openai_compat_5xx_error(monkeypatch):
    _install_fake_openai(monkeypatch, raises=RuntimeError("502"))
    with pytest.raises(RuntimeError):
        _bapi.run_openai_compat(
            _spec(provider="openai_compat_api"), "c1", "yo", history=[],
        )


def test_run_openai_compat_cancel_midstream(monkeypatch):
    _install_fake_openai(monkeypatch,
                         chunks=[
                             {"choices": [{"delta": {"content": "x"}}]},
                             {"choices": [{"delta": {"content": "y"}}]},
                         ])
    ev = threading.Event()
    ev.set()
    with pytest.raises(QueryCancelledError):
        _bapi.run_openai_compat(
            _spec(provider="openai_compat_api"), "c1", "yo", history=[],
            cancel_ev=ev,
        )
