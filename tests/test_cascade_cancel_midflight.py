"""Tests for P1-5 midflight cancel of the memory cascade."""
from __future__ import annotations

import os
import threading
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import larkhelm.memory as _mem  # noqa: E402
from larkhelm.ai_runner import QueryCancelledError  # noqa: E402


def test_cascade_midflight_cancel_enabled_default():
    import larkhelm.config as _cfg
    _cfg.config = {}
    assert _mem._cascade_midflight_cancel_enabled() is True


def test_cascade_midflight_cancel_disabled():
    import larkhelm.config as _cfg
    _cfg.config = {"memory_cascade_midflight_cancel": False}
    assert _mem._cascade_midflight_cancel_enabled() is False


def test_midflight_on_text_raises_when_cancel_set():
    import larkhelm.config as _cfg
    _cfg.config = {"memory_cascade_midflight_cancel": True}
    ev = threading.Event()
    cb = _mem._cascade_midflight_on_text(ev)
    assert cb is not None
    # Not yet cancelled — should not raise
    cb("partial text")
    ev.set()
    with pytest.raises(QueryCancelledError):
        cb("more text")


def test_midflight_on_text_returns_none_when_disabled():
    import larkhelm.config as _cfg
    _cfg.config = {"memory_cascade_midflight_cancel": False}
    ev = threading.Event()
    cb = _mem._cascade_midflight_on_text(ev)
    assert cb is None


def test_get_cascade_stats_returns_copy():
    stats = _mem.get_cascade_stats()
    assert "active" in stats
    assert "dropped_total" in stats
    assert "midflight_cancelled_total" in stats
    # Mutating the returned dict should not affect internal state
    stats["active"] = 999
    again = _mem.get_cascade_stats()
    assert again["active"] != 999


def test_cascade_midflight_counter_increments_via_extract(monkeypatch):
    """End-to-end-ish: extract project triggers midflight cancel → counter++."""
    import larkhelm.config as _cfg
    _cfg.config = {"memory_cascade_midflight_cancel": True,
                   "memory_cascade_max_concurrent": 4}

    # Reset stats
    with _mem._cascade_stats_lock:
        _mem._cascade_stats.active = 0
        _mem._cascade_stats.dropped_total = 0
        _mem._cascade_stats.midflight_cancelled_total = 0

    fake_session = "## Work Context\nfoo\n## Key Decisions & Facts\nbar"

    # Build a cancel_ev that fires inside on_text (not before the function
    # is even called — the extract function's pre-LLM check would short-circuit).
    cancel_ev = threading.Event()

    def fake_dispatch_one_shot(spec, ns, prompt, on_text):
        # Simulate streaming: first chunk arrives, then cancel fires.
        cancel_ev.set()
        on_text("partial chunk", "typing")
        return "ignored"

    class FakeSpec:
        id = "fake-cheap"
        provider = "deepseek_api"

    class FakeRegistry:
        def get_by_tag(self, tags):
            return FakeSpec()
        def get_orchestrator(self):
            return FakeSpec()

    monkeypatch.setattr(_mem, "_dispatch_one_shot", fake_dispatch_one_shot)
    monkeypatch.setattr(
        "larkhelm.backend_registry.BACKEND_REGISTRY",
        FakeRegistry(),
    )

    monkeypatch.setattr(_mem, "_project_memory_file",
                        lambda cwd: __file__)  # any existing file path
    monkeypatch.setattr(_mem, "_load_md_frontmatter", lambda p: {})
    monkeypatch.setattr(_mem, "_should_skip_extract_by_hash", lambda fm, sc: False)
    monkeypatch.setattr(_mem, "load_project_memory", lambda cwd: "")

    with pytest.raises(QueryCancelledError):
        _mem._try_extract_project(fake_session, "/tmp/some/path", cancel_ev=cancel_ev)


def test_run_one_shot_does_not_fall_back_after_midflight_cancel(monkeypatch):
    """Regression for round-2 review MUST-FIX (memory.py:833).

    Bug: ``_run_one_shot``'s broad ``except Exception as cheap_err:``
    swallowed ``QueryCancelledError`` raised by the midflight check and
    routed the request to the orchestrator fallback path — defeating
    P1-5's whole purpose (stop burning tokens after cancel).

    Previous tests passed only because they returned the SAME spec for
    ``get_by_tag(["cheap"])`` and ``get_orchestrator``; the id-collision
    guard ``orch_spec.id == cheap_spec.id`` at line 854 re-raised. In
    production with DISTINCT cheap and orchestrator backends, cancel
    silently flipped to a fresh LLM call.

    This test uses distinct specs and asserts:
      • ``QueryCancelledError`` propagates from ``_run_one_shot``.
      • The orchestrator dispatch is NEVER invoked after cancel.
    """
    import larkhelm.config as _cfg
    _cfg.config = {"memory_cascade_midflight_cancel": True}

    cancel_ev = threading.Event()
    dispatch_calls: list[str] = []

    def fake_dispatch(spec, ns, prompt, on_text):
        dispatch_calls.append(spec.id)
        # Simulate the cheap LLM emitting one chunk, then user/system
        # cancel arriving before chunk 2. The midflight callback then
        # raises QueryCancelledError when on_text is called.
        cancel_ev.set()
        on_text("partial chunk", "typing")
        return "should-not-be-reached"

    class CheapSpec:
        id = "cheap-distinct"
        provider = "deepseek_api"

    class OrchSpec:
        id = "orch-distinct"          # ← distinct from CheapSpec
        provider = "anthropic_api"

    class FakeRegistry:
        def get_by_tag(self, tags):
            return CheapSpec()
        def get_orchestrator(self):
            return OrchSpec()

    monkeypatch.setattr(_mem, "_dispatch_one_shot", fake_dispatch)
    monkeypatch.setattr(
        "larkhelm.backend_registry.BACKEND_REGISTRY", FakeRegistry(),
    )

    with pytest.raises(QueryCancelledError):
        _mem._run_one_shot(
            prompt="anything", ns="t/ns",
            prefer_cheap=True, cancel_ev=cancel_ev,
        )

    # Only the cheap backend got called — orchestrator must NOT be invoked
    # after cancel. This is the bug round-2 review caught.
    assert dispatch_calls == ["cheap-distinct"], (
        f"orchestrator was invoked after cancel (calls={dispatch_calls}) — "
        f"P1-5 midflight cancel is being silently undone by the fallback path"
    )


def test_cascade_no_write_when_cancelled_midflight(monkeypatch):
    """When midflight cancel fires, save_project_memory must NOT be called."""
    import larkhelm.config as _cfg
    _cfg.config = {"memory_cascade_midflight_cancel": True}

    fake_session = "## Work Context\nfoo"
    cancel_ev = threading.Event()

    def fake_dispatch_one_shot(spec, ns, prompt, on_text):
        cancel_ev.set()
        on_text("any output", "typing")
        return "should-not-be-saved"

    class FakeSpec:
        id = "fake-cheap"; provider = "deepseek_api"

    class FakeRegistry:
        def get_by_tag(self, tags): return FakeSpec()
        def get_orchestrator(self): return FakeSpec()

    save_called = []
    monkeypatch.setattr(_mem, "_dispatch_one_shot", fake_dispatch_one_shot)
    monkeypatch.setattr("larkhelm.backend_registry.BACKEND_REGISTRY", FakeRegistry())
    monkeypatch.setattr(_mem, "_project_memory_file", lambda cwd: __file__)
    monkeypatch.setattr(_mem, "_load_md_frontmatter", lambda p: {})
    monkeypatch.setattr(_mem, "_should_skip_extract_by_hash", lambda fm, sc: False)
    monkeypatch.setattr(_mem, "load_project_memory", lambda cwd: "")
    monkeypatch.setattr(_mem, "save_project_memory",
                        lambda *a, **kw: save_called.append(True))

    with pytest.raises(QueryCancelledError):
        _mem._try_extract_project(fake_session, "/tmp/X", cancel_ev=cancel_ev)
    assert save_called == [], "save_project_memory was called despite midflight cancel"
