"""Concurrency tests for ``crew/_scheduler`` topo helpers.

These tests reuse the existing scheduler API (``_topo_waves`` /
``_get_failed_dep``) and verify that calling the same function from many
threads with different inputs returns the right shape (no shared
mutable state). They are deliberately small and fast — they are meant
to catch regressions where someone introduces module-level caches
without proper synchronization.
"""
from __future__ import annotations

import threading


def test_topo_waves_thread_safe(fake_agent_spec):
    from larkhelm.crew._scheduler import _topo_waves
    specs = [
        fake_agent_spec(id="a", depends_on=[]),
        fake_agent_spec(id="b", depends_on=["a"]),
        fake_agent_spec(id="c", depends_on=["a"]),
        fake_agent_spec(id="d", depends_on=["b", "c"]),
    ]

    results: list[list[list[str]]] = []
    lock = threading.Lock()

    def _runner():
        out = _topo_waves(specs)
        with lock:
            results.append([[s.id for s in wave] for wave in out])

    threads = [threading.Thread(target=_runner) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 20
    # Every result should be the same shape: 3 waves of sizes 1,2,1
    for r in results:
        assert [len(w) for w in r] == [1, 2, 1]
        assert r[0] == ["a"]
        assert set(r[1]) == {"b", "c"}
        assert r[2] == ["d"]


def test_topo_waves_subset_thread_safe(fake_agent_spec):
    from larkhelm.crew._scheduler import _topo_waves_subset
    specs = [
        fake_agent_spec(id="a"), fake_agent_spec(id="b", depends_on=["a"]),
        fake_agent_spec(id="c", depends_on=["b"]),
    ]
    results: list[int] = []
    lock = threading.Lock()

    def _runner():
        waves = _topo_waves_subset(specs, {"b", "c"})
        with lock:
            results.append(sum(len(w) for w in waves))

    threads = [threading.Thread(target=_runner) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(n == 2 for n in results)


def test_detect_cycle_returns_none_for_dag():
    from larkhelm.crew._scheduler import _detect_cycle
    agents = [
        {"id": "a", "depends_on": []},
        {"id": "b", "depends_on": ["a"]},
        {"id": "c", "depends_on": ["b"]},
    ]
    assert _detect_cycle(agents) is None


def test_detect_cycle_finds_loop():
    from larkhelm.crew._scheduler import _detect_cycle
    agents = [
        {"id": "a", "depends_on": ["c"]},
        {"id": "b", "depends_on": ["a"]},
        {"id": "c", "depends_on": ["b"]},
    ]
    cycle = _detect_cycle(agents)
    assert cycle is not None
    assert len(cycle) >= 2
