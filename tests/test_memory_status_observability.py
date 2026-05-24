"""Unit tests for ``/memory status`` observability triad (P1-1c).

Pins the three new lines inserted into ``_cmd_memory``'s
``if sub == "status":`` branch:

    1. **Cascade 断路器：** ...
    2. **L2 fallback：** ...
    3. **Cron 健康度：** ...

Each line must be partial-failure isolated (one row's exception cannot
drop the other two), and the section must fall back to ``n/a`` when
``metrics.is_prometheus_available()`` returns ``False``.
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest


class _FakeValue:
    """Stand-in for ``Counter._value`` (an unlabeled Counter's value cell)."""

    def __init__(self, v: float) -> None:
        self._v = v

    def get(self) -> float:
        return self._v


class _FakeSample:
    """Mimic ``prometheus_client.core.Sample`` (only the fields we read)."""

    def __init__(self, name: str, labels: dict, value: float) -> None:
        self.name = name
        self.labels = labels
        self.value = value


class _FakeMetric:
    def __init__(self, samples: list) -> None:
        self.samples = samples


class _FakeCollectingCounter:
    """Stand-in for a labeled Counter — exposes ``.collect()``."""

    def __init__(self, samples: list) -> None:
        self._samples = samples

    def collect(self) -> list:
        return [_FakeMetric(self._samples)]


def _layer_samples(*pairs):
    """Build ``intent_layer_total`` samples from ``(layer, outcome, value)`` triples.

    Each triple yields a ``_total`` sample (the value sample we want) plus the
    sibling ``_created`` sample that prometheus_client automatically emits.
    Production code must filter on ``name.endswith("_total")`` to ignore the
    latter — this fixture pins that behaviour.
    """
    out = []
    for layer, outcome, value in pairs:
        out.append(_FakeSample(
            "larkhelm_intent_layer_total",
            {"layer": layer, "outcome": outcome},
            value,
        ))
        out.append(_FakeSample(
            "larkhelm_intent_layer_created",
            {"layer": layer, "outcome": outcome},
            1_700_000_000.0,
        ))
    return out


def _build_registry(*, exhausted: int = 0, l2_fall: int = 0,
                    layer_samples: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        cascade_backoff_exhausted_total=SimpleNamespace(
            _value=_FakeValue(exhausted),
        ),
        intent_l2_fallback_total=SimpleNamespace(
            _value=_FakeValue(l2_fall),
        ),
        intent_layer_total=_FakeCollectingCounter(layer_samples or []),
    )


@pytest.fixture
def card_recorder(monkeypatch, fake_card_sender):
    """Bind ``fake_card_sender`` patches onto the module-level binding in
    ``larkhelm.commands`` — ``_cmd_memory`` calls the function via its
    top-level ``from larkhelm.lark_client import send_card_reply`` import.
    """
    import larkhelm.commands as _cmds
    import larkhelm.lark_client as _lc
    monkeypatch.setattr(_cmds, "send_card_reply", _lc.send_card_reply,
                        raising=False)
    return fake_card_sender


@pytest.fixture
def fake_status(monkeypatch):
    """Stub ``memory_io.get_memory_status`` so /memory status doesn't hit disk."""
    payload = {
        "n_chats": 0,
        "n_sessions": 0,
        "n_api_sessions": 0,
        "api_session_size": 0,
        "log_size": 0,
        "data_size": 0,
        "memory_files": 0,
        "memory_size": 0,
        "chats": [],
    }
    import larkhelm.memory_io as _mio
    monkeypatch.setattr(_mio, "get_memory_status",
                        lambda chat_id: payload, raising=False)
    return payload


@pytest.fixture
def fake_circuit(monkeypatch):
    """Stub ``memory_llm_router.circuit_state`` — flip via ``holder['state']``.

    Set ``holder['state']`` to a string ("closed"/"half_open"/"open") for the
    normal path, or to an Exception instance to simulate the breaker module
    blowing up (exercises the partial-failure path).
    """
    holder = {"state": "closed"}
    import larkhelm.memory_llm_router as _r

    def _stub():
        s = holder["state"]
        if isinstance(s, Exception):
            raise s
        return s

    monkeypatch.setattr(_r, "circuit_state", _stub, raising=False)
    return holder


@pytest.fixture
def fake_chat_store(monkeypatch):
    """Swap in a clean ``_chat_state_store`` + ``_state_lock`` to stage crons."""
    import larkhelm.chat_state as _cs
    new_store: dict = {}
    monkeypatch.setattr(_cs, "_chat_state_store", new_store, raising=False)
    monkeypatch.setattr(_cs, "_state_lock", threading.RLock(), raising=False)
    return new_store


@pytest.fixture
def stub_metrics(monkeypatch):
    """Patch ``metrics.get_registry`` + ``metrics.is_prometheus_available``.

    Returns a setter so each test stages exactly the registry it needs.
    """
    holder = {"available": True, "registry": _build_registry()}
    import larkhelm.metrics as _m
    monkeypatch.setattr(_m, "is_prometheus_available",
                        lambda: holder["available"], raising=False)
    monkeypatch.setattr(_m, "get_registry",
                        lambda: holder["registry"], raising=False)

    def _set(*, available: bool | None = None, **registry_kwargs):
        if available is not None:
            holder["available"] = available
        if registry_kwargs:
            holder["registry"] = _build_registry(**registry_kwargs)

    return _set


def _body_of(recorded: list) -> str:
    """Return the body of the first ``send_card_reply`` call captured."""
    return next(r for r in recorded if r.get("kind") == "send_card_reply")["body"]


pytestmark = pytest.mark.usefixtures("init_test_config")


# ── 1. ordering ──────────────────────────────────────────────────────────

def test_three_lines_present_in_order(card_recorder, fake_status, fake_circuit,
                                       fake_chat_store, stub_metrics):
    """All three lines render after 上次 GC and before 各 Chat 摘要."""
    fake_circuit["state"] = "closed"
    stub_metrics(
        available=True,
        exhausted=3,
        l2_fall=12,
        layer_samples=_layer_samples(
            ("l1", "hit", 50),
            ("l2", "hit", 37),
            ("l1", "abstain", 9),
        ),
    )
    fake_chat_store["chat-A"] = {"crons": [
        {"id": "1", "last_run_status": "ok"},
        {"id": "2", "last_run_status": "error"},
        {"id": "3"},  # never
    ]}

    from larkhelm.commands import _cmd_memory
    _cmd_memory("chat-A", "status", msg_id="m1")

    body = _body_of(card_recorder)
    i_stale = body.index("**Stale slice：**")
    i_cas = body.index("**Cascade 断路器：**")
    i_l2 = body.index("**L2 fallback：**")
    i_cron = body.index("**Cron 健康度：**")
    assert i_stale < i_cas < i_l2 < i_cron

    # And the values are formatted as the design pins them.
    assert "**Cascade 断路器：** ✅closed（backoff exhausted: 3 次）" in body
    assert "**L2 fallback：** 12 / 87 (14%)" in body
    assert "**Cron 健康度：** ✅1 · ❌1 · ❓1（共 3 条）" in body


# ── 2. breaker emoji map ────────────────────────────────────────────────

@pytest.mark.parametrize("state, emoji", [
    ("closed", "✅"),
    ("half_open", "⚠️ "),
    ("open", "🔥"),
])
def test_breaker_state_emoji(state, emoji, card_recorder, fake_status,
                              fake_circuit, fake_chat_store, stub_metrics):
    fake_circuit["state"] = state
    stub_metrics(available=True, exhausted=0)

    from larkhelm.commands import _cmd_memory
    _cmd_memory("chat-A", "status", msg_id="m1")

    body = _body_of(card_recorder)
    assert f"**Cascade 断路器：** {emoji}{state}" in body


# ── 3. L2 zero denominator ──────────────────────────────────────────────

def test_l2_zero_denominator(card_recorder, fake_status, fake_circuit,
                              fake_chat_store, stub_metrics):
    """When ``outcome="hit"`` total is 0 the line emits ``/ 0 (n/a)``."""
    stub_metrics(
        available=True,
        l2_fall=7,
        layer_samples=_layer_samples(
            ("l1", "abstain", 5),
            ("l2", "error", 3),
        ),
    )

    from larkhelm.commands import _cmd_memory
    _cmd_memory("chat-A", "status", msg_id="m1")

    body = _body_of(card_recorder)
    assert "**L2 fallback：** 7 / 0 (n/a)" in body


# ── 4. cron buckets ─────────────────────────────────────────────────────

def test_cron_buckets(card_recorder, fake_status, fake_circuit,
                       fake_chat_store, stub_metrics):
    """Mixed ok / error / missing buckets into ✅/❌/❓ across multiple chats."""
    fake_chat_store["chat-A"] = {"crons": [
        {"id": "1", "last_run_status": "ok"},
        {"id": "2", "last_run_status": "ok"},
        {"id": "3", "last_run_status": "error"},
        {"id": "4"},                          # missing field
        {"id": "5", "last_run_status": ""},   # empty string
    ]}
    fake_chat_store["chat-B"] = {"crons": [
        {"id": "10", "last_run_status": "ok"},
    ]}

    from larkhelm.commands import _cmd_memory
    _cmd_memory("chat-A", "status", msg_id="m1")
    body = _body_of(card_recorder)
    assert "**Cron 健康度：** ✅3 · ❌1 · ❓2（共 6 条）" in body

    # And — empty store renders the "无 cron 任务" branch.
    card_recorder.clear()
    fake_chat_store.clear()
    _cmd_memory("chat-A", "status", msg_id="m2")
    body = _body_of(card_recorder)
    assert "**Cron 健康度：** 无 cron 任务" in body


# ── 5. partial failure isolation ────────────────────────────────────────

def test_partial_failure_isolation(card_recorder, fake_status, fake_circuit,
                                    fake_chat_store, stub_metrics):
    """``circuit_state`` raises → only the Cascade line degrades to ``n/a``."""
    fake_circuit["state"] = RuntimeError("boom")
    stub_metrics(
        available=True,
        l2_fall=4,
        layer_samples=_layer_samples(("l1", "hit", 20)),
    )
    fake_chat_store["chat-A"] = {"crons": [
        {"id": "1", "last_run_status": "ok"},
    ]}

    from larkhelm.commands import _cmd_memory
    _cmd_memory("chat-A", "status", msg_id="m1")

    body = _body_of(card_recorder)
    assert "**Cascade 断路器：** n/a" in body
    # The two healthy lines must keep rendering.
    assert "**L2 fallback：** 4 / 20 (20%)" in body
    assert "**Cron 健康度：** ✅1 · ❌0 · ❓0（共 1 条）" in body
    # And the existing stale-slice line is untouched.
    assert "**Stale slice：**" in body


# ── 6. prometheus unavailable ───────────────────────────────────────────

def test_prometheus_unavailable_graceful(card_recorder, fake_status,
                                          fake_circuit, fake_chat_store,
                                          stub_metrics):
    """``is_prometheus_available()=False`` → lines 1+2 整体 n/a, cron unaffected."""
    stub_metrics(available=False)
    fake_circuit["state"] = "closed"  # would render ✅closed if reached
    fake_chat_store["chat-A"] = {"crons": [
        {"id": "1", "last_run_status": "ok"},
    ]}

    from larkhelm.commands import _cmd_memory
    _cmd_memory("chat-A", "status", msg_id="m1")

    body = _body_of(card_recorder)
    assert "**Cascade 断路器：** n/a" in body
    assert "**L2 fallback：** n/a" in body
    # Cron line does not depend on prometheus.
    assert "**Cron 健康度：** ✅1 · ❌0 · ❓0（共 1 条）" in body
