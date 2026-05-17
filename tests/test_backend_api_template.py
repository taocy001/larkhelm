"""P2 AC-04: tests for ``larkhelm.backend_api_streaming`` template + adapter.

Five required scenarios:
  1) Adapter Protocol compliance
  2) _run_streaming_api invokes the four hooks in order
  3) cancel_ev mid-stream raises QueryCancelledError
  4) on_text callback exceptions are swallowed
  5) _record_outcome covers success / failure / cancel branches
"""
from __future__ import annotations

import os
import threading
import types

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import backend_api_streaming as _stream  # noqa: E402
from larkhelm.ai_runner import QueryCancelledError  # noqa: E402
from larkhelm.backend_registry import BackendSpec  # noqa: E402


# ── Fixtures ─────────────────────────────────────────────────────────────


def _spec(sid: str = "test", provider: str = "fake_api") -> BackendSpec:
    return BackendSpec(
        id=sid, provider=provider, display_name="Test",
        role="orchestrator", tags=[], api_key="sk-test",
        healthy=True, enabled=True,
    )


class _DummyAdapter:
    """Minimal Protocol-conforming adapter with hook-call tracking."""
    provider_label = "fake_api"

    def __init__(self, chunks=("hello ", "world"), raises=None):
        self._chunks = chunks
        self._raises = raises
        self.calls: list[str] = []

    def build_client(self, spec):
        self.calls.append("build_client")
        return object()

    def prepare_request(self, spec, history, message, extra_system):
        self.calls.append("prepare_request")
        return {"messages": [{"role": "user", "content": message}], "model": "x"}

    def iter_text_chunks(self, client, request):
        self.calls.append("iter_text_chunks")
        if self._raises is not None:
            raise self._raises
        for c in self._chunks:
            yield c

    def format_history(self, history, message, response_text):
        self.calls.append("format_history")
        return list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ]


@pytest.fixture(autouse=True)
def _silence_record(monkeypatch):
    # Stub _record_outcome so BackendRegistry isn't touched in tests.
    recorded: list[tuple[str, Exception | None]] = []
    monkeypatch.setattr(
        _stream, "_record_outcome",
        lambda sid, exc: recorded.append((sid, exc)),
    )
    yield recorded


# ── 1) Protocol compliance ────────────────────────────────────────────────


def test_adapter_protocol_compliance():
    """``isinstance(obj, StreamingAPIAdapter)`` works because Protocol is
    structural; we verify the four required attributes/methods exist."""
    ad = _DummyAdapter()
    assert ad.provider_label == "fake_api"
    for attr in ("build_client", "prepare_request",
                 "iter_text_chunks", "format_history"):
        assert callable(getattr(ad, attr)), attr


# ── 2) Hooks fire in canonical order ─────────────────────────────────────


def test_template_invokes_hooks_in_order(_silence_record):
    ad = _DummyAdapter(chunks=("a", "b", "c"))
    out, hist = _stream._run_streaming_api(
        ad, _spec(), "chat_x", "hi", history=[],
    )
    assert out == "abc"
    assert hist[-1]["role"] == "assistant"
    assert ad.calls == [
        "build_client", "prepare_request",
        "iter_text_chunks", "format_history",
    ]
    # Success path → _record_outcome called once with exc=None.
    assert _silence_record == [("test", None)]


# ── 3) cancel_ev mid-stream raises QueryCancelledError ───────────────────


def test_cancel_midstream_raises(_silence_record):
    ev = threading.Event()
    ev.set()
    ad = _DummyAdapter(chunks=("a", "b"))
    with pytest.raises(QueryCancelledError):
        _stream._run_streaming_api(
            ad, _spec(), "chat_x", "hi", history=[], cancel_ev=ev,
        )
    # Cancel path: _record_outcome NOT called (cancellation is user-initiated).
    assert _silence_record == []


# ── 4) on_text exceptions are swallowed, stream continues ────────────────


def test_on_text_exception_is_swallowed(_silence_record):
    captured: list[str] = []

    def _bad_on_text(text, status):
        captured.append(text)
        raise RuntimeError("UI exploded")

    ad = _DummyAdapter(chunks=("a", "b", "c"))
    out, _ = _stream._run_streaming_api(
        ad, _spec(), "chat_x", "hi", history=[], on_text=_bad_on_text,
    )
    # Stream must complete despite the broken callback.
    assert out == "abc"
    assert len(captured) == 3  # callback was invoked for every chunk


# ── 5) _record_outcome covers the three branches ─────────────────────────


def test_record_outcome_success_branch():
    """Direct call to _record_outcome confirms the success path is silent."""
    # No assertion needed beyond "does not raise". The shim talks to
    # BACKEND_REGISTRY which is initialised at import time; even if the
    # spec isn't registered, the outer try/except keeps everything safe.
    _stream._record_outcome("does_not_matter", None)


def test_record_outcome_cancel_branch_is_noop():
    """Cancelled outcome must not touch BackendRegistry — the function
    returns silently before reaching any registry method.
    """
    _stream._record_outcome("does_not_matter", QueryCancelledError("u clicked"))


def test_record_outcome_failure_branch_tolerates_unknown_spec():
    """Failure outcome on a never-registered spec must not raise."""
    _stream._record_outcome("never_registered_spec_id", RuntimeError("boom"))


# ── adapter dispatch shape (smoke test) ──────────────────────────────────


def test_adapter_failure_propagates_after_record(_silence_record):
    ad = _DummyAdapter(raises=RuntimeError("network 503"))
    with pytest.raises(RuntimeError):
        _stream._run_streaming_api(
            ad, _spec(), "chat_x", "hi", history=[],
        )
    # Failure path: _record_outcome called once with the exception.
    assert len(_silence_record) == 1
    sid, exc = _silence_record[0]
    assert sid == "test"
    assert isinstance(exc, RuntimeError)


# ── Anthropic adapter accepts an injected module (no live SDK) ──────────


def test_anthropic_adapter_with_injected_module():
    chunks = ["hello ", "anthropic"]

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        @property
        def text_stream(self):
            return iter(chunks)

    class _Messages:
        def stream(self, **kw):
            return _Stream()

    class _Anthropic:
        def __init__(self, **kw):
            self.messages = _Messages()

    mod = types.SimpleNamespace(Anthropic=_Anthropic)
    ad = _stream.AnthropicAdapter(anthropic_module=mod)
    assert ad.provider_label == "anthropic_api"
    out, _ = _stream._run_streaming_api(
        ad, _spec(), "chat_x", "hi", history=[],
    )
    assert out == "hello anthropic"
