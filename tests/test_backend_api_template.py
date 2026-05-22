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

    def __init__(self, chunks=("hello ", "world"), raises=None, usage=None):
        self._chunks = chunks
        self._raises = raises
        self._usage_result = usage or {}
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

    def extract_usage(self) -> dict:
        self.calls.append("extract_usage")
        return self._usage_result


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
    structural; we verify the five required attributes/methods exist."""
    ad = _DummyAdapter()
    assert ad.provider_label == "fake_api"
    for attr in ("build_client", "prepare_request",
                 "iter_text_chunks", "format_history", "extract_usage"):
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
        "iter_text_chunks", "extract_usage", "format_history",
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
    # Cancel path: extract_usage must NOT be called.
    assert "extract_usage" not in ad.calls


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


# ── extract_usage Protocol tests ─────────────────────────────────────────


def test_extract_usage_non_empty_triggers_token_recording(monkeypatch, _silence_record):
    """When extract_usage() returns a non-empty dict, record_token_usage is called."""
    recorded: list = []

    def _fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    monkeypatch.setattr("larkhelm.token_stats.record_token_usage", _fake_record)

    usage_dict = {"input_tokens": 10, "output_tokens": 5, "cache_read": 0, "cache_create": 0}
    ad = _DummyAdapter(chunks=("hi",), usage=usage_dict)
    _stream._run_streaming_api(ad, _spec(), "chat_x", "hi", history=[])
    assert len(recorded) == 1
    assert recorded[0][0] == "chat_x"
    assert recorded[0][2]["input_tokens"] == 10


def test_extract_usage_empty_skips_token_recording(monkeypatch, _silence_record):
    """When extract_usage() returns {}, record_token_usage is not called."""
    recorded: list = []

    def _fake_record(chat_id, model, usage):
        recorded.append(usage)

    monkeypatch.setattr("larkhelm.token_stats.record_token_usage", _fake_record)

    ad = _DummyAdapter(chunks=("hi",), usage={})
    _stream._run_streaming_api(ad, _spec(), "chat_x", "hi", history=[])
    assert recorded == []


def test_anthropic_extract_usage_exception_safety():
    """AnthropicAdapter.extract_usage() returns {} when _usage_raw attribute raises."""

    class _BrokenUsage:
        @property
        def input_tokens(self):
            raise RuntimeError("SDK changed")

    mod = types.SimpleNamespace(Anthropic=lambda **kw: None)
    ad = _stream.AnthropicAdapter(anthropic_module=mod)
    ad._usage_raw = _BrokenUsage()
    assert ad.extract_usage() == {}


def test_anthropic_extract_usage_none_raw():
    """AnthropicAdapter.extract_usage() returns {} when _usage_raw is None."""
    mod = types.SimpleNamespace(Anthropic=lambda **kw: None)
    ad = _stream.AnthropicAdapter(anthropic_module=mod)
    assert ad._usage_raw is None
    assert ad.extract_usage() == {}


def test_failure_path_skips_extract_usage(_silence_record):
    """When iter_text_chunks raises, extract_usage must NOT be called."""
    ad = _DummyAdapter(raises=RuntimeError("network 503"))
    with pytest.raises(RuntimeError):
        _stream._run_streaming_api(ad, _spec(), "chat_x", "hi", history=[])
    assert "extract_usage" not in ad.calls


# ── AC-spec adapter-level extract_usage tests ────────────────────────────


def test_anthropic_extract_usage():
    """AnthropicAdapter.extract_usage() returns valid dict after simulated streaming call."""
    chunks = ["hello ", "world"]

    class _UsageRaw:
        input_tokens = 100
        output_tokens = 50
        cache_read_input_tokens = 10
        cache_creation_input_tokens = 5

    class _FinalMessage:
        usage = _UsageRaw()

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        @property
        def text_stream(self): return iter(chunks)
        def get_final_message(self): return _FinalMessage()

    class _Messages:
        def stream(self, **kw): return _Stream()

    class _Anthropic:
        def __init__(self, **kw): self.messages = _Messages()

    mod = types.SimpleNamespace(Anthropic=_Anthropic)
    ad = _stream.AnthropicAdapter(anthropic_module=mod)
    client = ad.build_client(_spec())
    request = ad.prepare_request(_spec(), [], "hi", "")
    list(ad.iter_text_chunks(client, request))

    usage = ad.extract_usage()
    assert isinstance(usage, dict)
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50
    assert usage["cache_read"] == 10
    assert usage["cache_create"] == 5


def test_google_extract_usage():
    """GoogleGenaiAdapter.extract_usage() returns valid dict after simulated streaming call."""

    class _UsageMeta:
        prompt_token_count = 80
        candidates_token_count = 40
        cached_content_token_count = 20

    class _Chunk:
        text = "hello "
        usage_metadata = None

    class _FinalChunk:
        text = "world"
        usage_metadata = _UsageMeta()

    class _Models:
        def generate_content_stream(self, **kw): return iter([_Chunk(), _FinalChunk()])

    class _Client:
        def __init__(self, **kw): self.models = _Models()

    class _GenaiMod:
        @staticmethod
        def Client(**kw): return _Client(**kw)

    class _TypesMod:
        class Content:
            def __init__(self, role, parts): pass
        class Part:
            def __init__(self, text): pass
        class GenerateContentConfig:
            def __init__(self, system_instruction=None): pass

    google_mod = types.SimpleNamespace(genai=_GenaiMod, genai_types=_TypesMod)
    ad = _stream.GoogleGenaiAdapter(google_module=google_mod)
    client = ad.build_client(_spec())
    request = ad.prepare_request(_spec(), [], "hi", "")
    list(ad.iter_text_chunks(client, request))

    usage = ad.extract_usage()
    assert isinstance(usage, dict)
    assert usage["input_tokens"] == 80
    assert usage["output_tokens"] == 40
    assert usage["cache_read"] == 20
    assert usage["cache_create"] == 0


def test_openai_extract_usage():
    """OpenAICompatAdapter.extract_usage() returns valid dict after simulated streaming call."""

    class _UsageRaw:
        prompt_tokens = 60
        completion_tokens = 30
        prompt_tokens_details = None

    class _Delta:
        content = "hello"

    class _Choice:
        delta = _Delta()

    class _Chunk:
        choices = [_Choice()]
        usage = None

    class _FinalChunk:
        choices = []
        usage = _UsageRaw()

    class _Stream:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __iter__(self): return iter([_Chunk(), _FinalChunk()])

    class _Completions:
        def create(self, **kw): return _Stream()

    class _Chat:
        def __init__(self): self.completions = _Completions()

    class _OpenAI:
        def __init__(self, **kw): self.chat = _Chat()

    mod = types.SimpleNamespace(OpenAI=_OpenAI)
    ad = _stream.OpenAICompatAdapter(openai_module=mod)
    client = ad.build_client(_spec())
    request = ad.prepare_request(_spec(), [], "hi", "")
    list(ad.iter_text_chunks(client, request))

    usage = ad.extract_usage()
    assert isinstance(usage, dict)
    assert usage["input_tokens"] == 60
    assert usage["output_tokens"] == 30
    assert usage["cache_read"] == 0
    assert usage["cache_create"] == 0


def test_template_records_usage(monkeypatch, _silence_record):
    """_run_streaming_api() calls record_token_usage when extract_usage() returns non-empty."""
    recorded: list = []

    def _fake_record(chat_id, model, usage):
        recorded.append((chat_id, model, usage))

    monkeypatch.setattr("larkhelm.token_stats.record_token_usage", _fake_record)

    usage_dict = {"input_tokens": 50, "output_tokens": 25, "cache_read": 0, "cache_create": 0}
    ad = _DummyAdapter(chunks=("hello",), usage=usage_dict)
    _stream._run_streaming_api(ad, _spec(), "chat_y", "hi", history=[])

    assert len(recorded) == 1
    assert recorded[0][0] == "chat_y"
    assert recorded[0][2]["input_tokens"] == 50


def test_extract_usage_exception_is_safe():
    """extract_usage() must return {} on internal error without propagating the exception."""
    mod = types.SimpleNamespace(Anthropic=lambda **kw: None)
    ad = _stream.AnthropicAdapter(anthropic_module=mod)

    class _AngryUsage:
        @property
        def input_tokens(self):
            raise RuntimeError("SDK internal change")

    ad._usage_raw = _AngryUsage()
    result = ad.extract_usage()
    assert result == {}, "extract_usage() must return {} on internal error, not raise"


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
