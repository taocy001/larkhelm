"""P1-5a / AC-02 + AC-03: pin every ``_bump_intent_layer`` call site inside
:func:`larkhelm.agent_hub.intent_router.resolve_intent`.

The router currently bumps from ten distinct branch points (explicit / L1
hit / L1 abstain / L1 error / microlearn hit / microlearn abstain /
microlearn error / L2 hit / L2 error / fallback hit) and additionally
calls :func:`larkhelm.metrics.inc_intent_l2_fallback` whenever the L2 path
collapses to ``layer=="fallback"``. Tests monkeypatch the
``_bump_intent_layer`` wrapper (a single symbol) and the L1 / microlearn /
L2 sub-resolvers to drive each branch deterministically.

Why monkeypatch the wrapper instead of the metric helper directly:
``_bump_intent_layer`` is the only spot the router touches metrics for the
layer counter — patching it lets the test stay neutral to whether
``prometheus-client`` is installed in the venv (no skips, no fixture).
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm.agent_hub import intent_router as _ir  # noqa: E402
from larkhelm.agent_hub.intent_types import IntentResult  # noqa: E402


@pytest.fixture
def bump_spy(monkeypatch):
    """Replace ``_bump_intent_layer`` with a recording spy. The returned
    list captures ``(layer, outcome)`` pairs in call order so tests can
    assert both count and sequence.
    """
    calls: list[tuple[str, str]] = []

    def _spy(layer: str, outcome: str) -> None:
        calls.append((layer, outcome))

    monkeypatch.setattr(_ir, "_bump_intent_layer", _spy)
    return calls


@pytest.fixture
def fallback_spy(monkeypatch):
    """Replace ``metrics.inc_intent_l2_fallback`` with a counter. The router
    imports the helper lazily inside the fallback branch
    (``from larkhelm.metrics import inc_intent_l2_fallback as _inc_fb``) so
    we patch the module-level symbol that the import resolves to.
    """
    from larkhelm import metrics as _met
    calls: list[None] = []

    def _spy() -> None:
        calls.append(None)

    monkeypatch.setattr(_met, "inc_intent_l2_fallback", _spy)
    return calls


# ── 1) Explicit slash prefix → ("explicit","hit") once ────────────────────


def test_explicit_command_bumps_explicit_hit(bump_spy):
    intent = _ir.resolve_intent("/dev 帮我实现 X")
    assert intent.is_explicit_command
    assert intent.agent_type == "dev"
    assert bump_spy == [("explicit", "hit")]


# ── 2) L1 hit → ("l1","hit") only ─────────────────────────────────────────


def test_l1_hit_bumps_l1_hit_only(bump_spy, monkeypatch):
    fake = IntentResult(agent_type="dev", layer="L1", confidence=0.8,
                        raw_text="任意文本")
    monkeypatch.setattr(_ir, "_resolve_l1", lambda *a, **kw: fake)
    intent = _ir.resolve_intent("不会被 explicit 命中的文本")
    assert intent is fake
    assert bump_spy == [("l1", "hit")]


# ── 3) L1 error → L2 hit: ("l1","error") + ("l2","hit") ───────────────────


def test_l1_error_then_l2_hit_bumps_two(bump_spy, monkeypatch):
    def _l1_boom(*_a, **_kw):
        raise RuntimeError("synthetic L1 failure")

    fake_l2 = IntentResult(agent_type="dev", layer="L2", confidence=0.7,
                           raw_text="任意文本")
    monkeypatch.setattr(_ir, "_resolve_l1", _l1_boom)
    monkeypatch.setattr(_ir, "_resolve_microlearn", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_l2", lambda *a, **kw: fake_l2)
    intent = _ir.resolve_intent("一段触发分类的文本")
    assert intent is fake_l2
    # ``("l1","abstain")`` must NOT appear — error and abstain are mutually
    # exclusive per call (the wrapper records exactly one of them).
    assert bump_spy == [("l1", "error"), ("microlearn", "abstain"), ("l2", "hit")]


# ── 4) L1 abstain → microlearn hit: ("l1","abstain") + ("microlearn","hit") ──


def test_l1_abstain_microlearn_hit_bumps_two(bump_spy, monkeypatch):
    fake_ml = IntentResult(agent_type="plan", layer="microlearn",
                           confidence=0.9, raw_text="任意文本")
    monkeypatch.setattr(_ir, "_resolve_l1", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_microlearn", lambda *a, **kw: fake_ml)
    intent = _ir.resolve_intent("一段触发分类的文本")
    assert intent is fake_ml
    assert bump_spy == [("l1", "abstain"), ("microlearn", "hit")]


# ── 5) full fallback chain → three abstains + fallback hit + fallback metric ──


def test_full_fallback_path_bumps_all_three_tiers(bump_spy, fallback_spy, monkeypatch):
    """AC-02 + AC-03: L1 abstain + microlearn abstain + L2 returns fallback
    must yield ``("l1","abstain")``, ``("microlearn","abstain")``,
    ``("fallback","hit")`` (in that order) AND a single
    ``inc_intent_l2_fallback()`` call.
    """
    fake_fb = _ir._fallback("一段无人认领的文本")  # layer="fallback"
    assert fake_fb.layer == "fallback"  # sanity for the helper's contract
    monkeypatch.setattr(_ir, "_resolve_l1", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_microlearn", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_l2", lambda *a, **kw: fake_fb)
    intent = _ir.resolve_intent("一段无人认领的文本")
    assert intent.layer == "fallback"
    assert intent.agent_type == "chat"
    assert bump_spy == [
        ("l1", "abstain"),
        ("microlearn", "abstain"),
        ("fallback", "hit"),
    ]
    assert len(fallback_spy) == 1


# ── 6) microlearn error → L2 hit: error outcome wins over abstain ─────────


def test_microlearn_error_branch_records_error_outcome(bump_spy, monkeypatch):
    def _ml_boom(*_a, **_kw):
        raise RuntimeError("synthetic microlearn failure")

    fake_l2 = IntentResult(agent_type="crew", layer="L2", confidence=0.7,
                           raw_text="任意文本")
    monkeypatch.setattr(_ir, "_resolve_l1", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_microlearn", _ml_boom)
    monkeypatch.setattr(_ir, "_resolve_l2", lambda *a, **kw: fake_l2)
    intent = _ir.resolve_intent("一段触发分类的文本")
    assert intent is fake_l2
    # ``("microlearn","abstain")`` must NOT appear — error path is exclusive.
    assert ("microlearn", "abstain") not in bump_spy
    assert bump_spy == [("l1", "abstain"), ("microlearn", "error"), ("l2", "hit")]


# ── 7) L2 raises → fallback path: ("l2","error") + ("fallback","hit") + fb metric ──


def test_l2_error_branch_falls_back_with_two_bumps(bump_spy, fallback_spy, monkeypatch):
    def _l2_boom(*_a, **_kw):
        raise RuntimeError("synthetic L2 failure")

    monkeypatch.setattr(_ir, "_resolve_l1", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_microlearn", lambda *a, **kw: None)
    monkeypatch.setattr(_ir, "_resolve_l2", _l2_boom)
    intent = _ir.resolve_intent("一段触发分类的文本")
    # The router converts the raised L2 to a fallback() before bumping the
    # ("fallback","hit") tier, so the outward-facing result is chat/fallback.
    assert intent.layer == "fallback"
    assert intent.agent_type == "chat"
    assert bump_spy == [
        ("l1", "abstain"),
        ("microlearn", "abstain"),
        ("l2", "error"),
        ("fallback", "hit"),
    ]
    assert len(fallback_spy) == 1


# ── 8) empty input → fallback short-circuit, no layer bumps ───────────────


def test_empty_input_short_circuits_without_layer_bump(bump_spy, fallback_spy):
    """``resolve_intent`` returns ``_fallback("")`` for empty / whitespace
    input BEFORE the layer pipeline runs. Confirms the bump wrapper isn't
    invoked on that short-circuit so dashboards don't inflate from upstream
    noise (empty cards, dedupe artefacts, …).
    """
    intent = _ir.resolve_intent("")
    assert intent.layer == "fallback"
    assert intent.agent_type == "chat"
    assert bump_spy == []
    assert fallback_spy == []
