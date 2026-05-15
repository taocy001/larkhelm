"""Concurrency tests for ``crew/_state`` registries (active crew + breakpoint)."""
from __future__ import annotations

import threading
import time


def test_subscribe_crew_done_returns_set_when_no_active(init_test_config):
    """Race-safe: subscribe before any crew is registered → event is pre-set."""
    from larkhelm.crew._state import subscribe_crew_done
    ev = subscribe_crew_done("never_seen_chat")
    assert ev.is_set()


def test_register_and_signal_done(init_test_config):
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _signal_crew_done, subscribe_crew_done,
    )
    chat = "chat_xyz"
    with _active_crew_lock:
        _active_crew[chat] = "crewid_abc"
    try:
        ev = subscribe_crew_done(chat)
        assert not ev.is_set()
        with _active_crew_lock:
            _signal_crew_done(chat)
        # After signal_crew_done, the subscriber's event should fire
        assert ev.wait(timeout=0.5)
    finally:
        with _active_crew_lock:
            _active_crew.pop(chat, None)


def test_breakpoint_signal_resolves_event(init_test_config):
    from larkhelm.crew._state import (
        _breakpoint_events, _breakpoint_meta, _breakpoint_results,
        signal_breakpoint,
    )
    crew_id = "test_crew_001"
    ev = threading.Event()
    with _breakpoint_meta:
        _breakpoint_events[crew_id] = ev
        _breakpoint_results[crew_id] = False
    try:
        signal_breakpoint(crew_id, True)
        assert ev.is_set()
        with _breakpoint_meta:
            assert _breakpoint_results[crew_id] is True
    finally:
        with _breakpoint_meta:
            _breakpoint_events.pop(crew_id, None)
            _breakpoint_results.pop(crew_id, None)


def test_concurrent_subscribers_all_fire(init_test_config):
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _signal_crew_done, subscribe_crew_done,
    )
    chat = "concurrent_test"
    with _active_crew_lock:
        _active_crew[chat] = "crew_par_001"
    try:
        events: list[threading.Event] = []
        evlock = threading.Lock()

        def _sub():
            ev = subscribe_crew_done(chat)
            with evlock:
                events.append(ev)

        threads = [threading.Thread(target=_sub) for _ in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        with _active_crew_lock:
            _signal_crew_done(chat)
        # Every event must fire
        for ev in events:
            assert ev.wait(timeout=0.5)
    finally:
        with _active_crew_lock:
            _active_crew.pop(chat, None)


def test_register_crew_card_lru_eviction(init_test_config):
    """_register_crew_card caps at _CREW_CARD_INDEX_MAX entries."""
    from larkhelm.crew._state import (
        _crew_card_index, _crew_card_index_lock,
        _register_crew_card, _CREW_CARD_INDEX_MAX,
    )
    # Fill to cap+5
    for i in range(_CREW_CARD_INDEX_MAX + 5):
        _register_crew_card(f"mid_{i}", "test_chat", f"title_{i}", "summary")
    with _crew_card_index_lock:
        assert len(_crew_card_index) <= _CREW_CARD_INDEX_MAX
