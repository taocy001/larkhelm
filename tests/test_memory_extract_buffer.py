"""P2 AC-06: tests for ``larkhelm.memory_extract_buffer``."""
from __future__ import annotations

import os
import threading
import time

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import memory_extract_buffer as _meb  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_buffer_state():
    _meb._reset_for_tests()
    yield
    _meb._reset_for_tests()


@pytest.fixture
def fake_cascade():
    """Replace the real ``_cascade_extract`` with a counter."""
    calls: list[tuple[str, str]] = []
    buf = _meb._get_buffer()
    buf.set_cascade_fn_for_tests(lambda content, chat_id: calls.append((chat_id, content)))
    yield calls
    buf.set_cascade_fn_for_tests(None)


@pytest.fixture
def force_window(monkeypatch):
    """Force a specific buffer window for the test."""
    def _set(sec: int):
        import larkhelm.config as _cfg
        monkeypatch.setattr(_cfg, "MEMORY_EXTRACT_BUFFER_WINDOW_SEC", sec, raising=False)
    return _set


# ── 1) window=0 → byte-compat with P1 ────────────────────────────────────


def test_buffer_disabled_byte_compatible(fake_cascade, force_window):
    """window=0 means cascade fires synchronously, once per update."""
    force_window(0)
    for i in range(5):
        _meb.record_session_update("oc_chat", f"summary {i}")
    # Every call → exactly one cascade. No buffering.
    assert len(fake_cascade) == 5
    assert all(cid == "oc_chat" for cid, _ in fake_cascade)
    # No state retained when buffering disabled.
    buf = _meb._get_buffer()
    with buf._lock:
        assert buf._states == {}


# ── 2) Five updates within the window → one flush ────────────────────────


def test_buffer_merges_within_window(fake_cascade, force_window):
    force_window(1)  # 1 second window
    for i in range(5):
        _meb.record_session_update("oc_chat", f"summary {i}")
    # Immediately after the burst: nothing has flushed yet, but the
    # latest content sits in the state.
    buf = _meb._get_buffer()
    with buf._lock:
        assert "oc_chat" in buf._states
        assert buf._states["oc_chat"].update_count == 5
    # Wait for the timer to fire (1s window + small overhead).
    time.sleep(1.4)
    # Exactly one cascade, containing only the LAST summary.
    assert len(fake_cascade) == 1
    assert fake_cascade[0] == ("oc_chat", "summary 4")


# ── 3) Manual flush triggers cascade ────────────────────────────────────


def test_timer_flush_via_manual_call(fake_cascade, force_window):
    force_window(60)  # long window so the manual flush is the trigger
    _meb.record_session_update("oc_chat", "summary X")
    _meb.flush("oc_chat", trigger="timer")
    assert len(fake_cascade) == 1
    assert fake_cascade[0] == ("oc_chat", "summary X")
    # State drained.
    buf = _meb._get_buffer()
    with buf._lock:
        assert "oc_chat" not in buf._states


# ── 4) Shutdown flush drains all pending slots ──────────────────────────


def test_flush_all_for_shutdown(fake_cascade, force_window):
    force_window(60)
    for cid in ("oc_a", "oc_b", "oc_c"):
        _meb.record_session_update(cid, f"summary for {cid}")
    _meb.flush_all_for_shutdown(timeout_sec=2.0)
    cascaded_ids = {cid for cid, _ in fake_cascade}
    assert cascaded_ids == {"oc_a", "oc_b", "oc_c"}


# ── 5) Buffer metric is incremented per flush ───────────────────────────


def test_extract_buffer_flush_counter_increments(fake_cascade, force_window, monkeypatch):
    pytest.importorskip("prometheus_client")
    force_window(60)
    from larkhelm import metrics as _met
    _met._reset_for_tests()
    _meb.record_session_update("oc_chat", "summary")
    _meb.flush("oc_chat", trigger="timer")
    body = _met.render_exposition()
    assert 'larkhelm_extract_buffer_flushes_total{trigger="timer"} 1' in body


# ── 6) Empty chat_id → no-op ────────────────────────────────────────────


def test_empty_chat_id_is_noop(fake_cascade, force_window):
    force_window(0)
    _meb.record_session_update("", "summary")
    assert fake_cascade == []


# ── 7) Concurrent updates from threads stay safe ────────────────────────


def test_concurrent_updates_thread_safe(fake_cascade, force_window):
    force_window(1)
    def _writer(idx: int):
        for i in range(20):
            _meb.record_session_update(f"oc_{idx}", f"chunk {i}")
    threads = [threading.Thread(target=_writer, args=(i,), daemon=True)
               for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    time.sleep(1.4)  # wait for timers
    # Each chat should have flushed exactly once (last value wins).
    cascaded_ids = {cid for cid, _ in fake_cascade}
    assert cascaded_ids == {f"oc_{i}" for i in range(5)}
