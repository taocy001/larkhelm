"""Phase D — tests for ``larkhelm.memory_slice`` (data layer).

REQ-01 / REQ-04 / class diagram contract: frozen dataclasses, id hashing
stability, ``with_score`` factory, and the H2 split rules used downstream
by ``memory_retriever._slices_from_file``."""
from __future__ import annotations

import dataclasses

import pytest

from larkhelm.memory_slice import (
    InjectionPolicy,
    MemorySlice,
    RetrievalRequest,
    ScoredSlice,
)
from larkhelm.memory_retriever import (
    _slice_id,
    _slices_from_file,
)


def _make_slice(**overrides) -> MemorySlice:
    defaults = dict(
        id="abc123",
        layer="global",
        kind="fact",
        title="t",
        body="b",
    )
    defaults.update(overrides)
    return MemorySlice(**defaults)


def test_memoryslice_is_frozen():
    s = _make_slice()
    field_names = [f.name for f in dataclasses.fields(s)]
    assert "id" in field_names
    assert "layer" in field_names
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.title = "new"  # type: ignore[misc]


def test_scoredslice_is_frozen():
    s = _make_slice()
    ss = ScoredSlice(slice=s, score=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        ss.score = 0.6  # type: ignore[misc]


def test_retrievalrequest_default_values():
    r = RetrievalRequest(chat_id="c1")
    assert r.cwd is None
    assert r.recent_turns == ()
    assert r.agent_type == "chat"
    assert r.has_doc_urls is False


def test_injectionpolicy_required_minimal():
    p = InjectionPolicy(
        agent_type="x",
        token_budget=1000,
        layer_weights={"session": 1.0},
        kind_priority=("fact",),
    )
    assert p.alpha_recency == 0.3
    assert p.top_k == 6


def test_id_hash_stability():
    a = _slice_id("global", "user_a", "Preferences", 0)
    b = _slice_id("global", "user_a", "Preferences", 0)
    assert a == b
    assert len(a) == 12


def test_id_collision_avoidance_via_slice_idx():
    a = _slice_id("project", "cwd", "Same Title", 0)
    b = _slice_id("project", "cwd", "Same Title", 1)
    assert a != b


def test_h2_split_basic():
    raw = "## A\nbody A line\n## B\nbody B line\n"
    slices = _slices_from_file("project", "cwd_x", _FakePath("p.md"), raw)
    assert len(slices) == 2
    titles = [s.title for s in slices]
    assert titles == ["A", "B"]
    assert "body A line" in slices[0].body
    assert "body B line" in slices[1].body


def test_no_h2_monolith_single_slice():
    raw = "just a paragraph with no headings\nand a continuation"
    slices = _slices_from_file("global", "user", _FakePath("g.md"), raw)
    assert len(slices) == 1
    assert slices[0].title == ""
    # Global layer monolith → preference kind heuristic.
    assert slices[0].kind == "preference"


def test_h2_with_h3_nested():
    raw = "## A\nbody\n### A1\nx\n"
    slices = _slices_from_file("project", "cwd", _FakePath("p.md"), raw)
    assert len(slices) == 1
    assert slices[0].title == "A"
    assert "A1" in slices[0].body
    assert "### A1" in slices[0].body


def test_scored_slice_factory_with_score():
    s = _make_slice()
    ss = s.with_score(0.7)
    assert isinstance(ss, ScoredSlice)
    assert ss.score == 0.7
    assert ss.slice is s


def test_h2_split_skips_empty_sections():
    """A section with no body should be silently dropped."""
    raw = "## A\n\n## B\nbody B\n"
    slices = _slices_from_file("session", "chat_x", _FakePath("s.md"), raw)
    assert [s.title for s in slices] == ["B"]


class _FakePath:
    def __init__(self, name: str):
        self.name = name

    def stat(self):  # noqa: D401
        raise OSError("synthetic path, no stat")
