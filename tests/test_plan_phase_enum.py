"""Regression tests for P3-7: ``PlanPhase`` enum migration.

The earlier ``phase: str`` field accepted any string at runtime; a typo
(``state.phase = "runing"``) silently flowed through the state machine
and the persistence layer. Migrating to an ``Enum`` makes typos a loud
``ValueError`` at the boundary and gives the IDE / mypy enough info to
catch bad comparisons.

These tests pin:
  • the 7 enum values match the pre-migration string set verbatim,
  • the dataclass coerces string ``phase=`` kwargs at construction time
    (so existing test fixtures and the persistence loader keep working),
  • unknown strings raise ``ValueError`` (not silently mismatched),
  • ``state.phase.value`` round-trips byte-identically through
    ``serialise_plan_record`` so existing on-disk plan_state.json files
    stay readable across the migration.
"""
from __future__ import annotations

from larkhelm.cmd_plan import MultiPlanState, PlanPhase, PlanStep, _coerce_phase


def test_planphase_values_match_pre_migration_strings():
    """The 7 values are the contract — changing one would orphan every
    on-disk plan_state.json. Pin them explicitly."""
    expected = {
        "PLANNING":   "planning",
        "CONFIRMING": "confirming",
        "RUNNING":    "running",
        "WAITING":    "waiting",
        "DONE":       "done",
        "CANCELLED":  "cancelled",
        "FAILED":     "failed",
    }
    for name, value in expected.items():
        assert PlanPhase[name].value == value
    assert {p.value for p in PlanPhase} == set(expected.values())


def test_coerce_phase_accepts_enum_passthrough():
    assert _coerce_phase(PlanPhase.RUNNING) is PlanPhase.RUNNING


def test_coerce_phase_accepts_string_value():
    assert _coerce_phase("running") is PlanPhase.RUNNING
    assert _coerce_phase("done") is PlanPhase.DONE


def test_coerce_phase_rejects_unknown_string():
    """The whole point of the migration: typos must be loud."""
    import pytest
    with pytest.raises(ValueError):
        _coerce_phase("runing")   # missing 'n'


def _make_state(**overrides) -> MultiPlanState:
    kwargs = dict(
        plan_id="t_phase", chat_id="oc_t", title="t",
        steps=[PlanStep(idx=0, type="dev", desc="x")],
    )
    kwargs.update(overrides)
    return MultiPlanState(**kwargs)


def test_state_default_phase_is_enum_running():
    state = _make_state()
    assert isinstance(state.phase, PlanPhase)
    assert state.phase is PlanPhase.RUNNING


def test_state_accepts_enum_kwarg():
    state = _make_state(phase=PlanPhase.PLANNING)
    assert state.phase is PlanPhase.PLANNING


def test_state_coerces_string_kwarg_for_back_compat():
    """Test fixtures and the persistence loader both pass raw strings;
    ``__post_init__`` must coerce them so the contract is enforced
    internally without breaking callers."""
    state = _make_state(phase="confirming")
    assert state.phase is PlanPhase.CONFIRMING


def test_state_post_init_raises_on_unknown_string():
    import pytest
    with pytest.raises(ValueError):
        _make_state(phase="bogus")


def test_serialise_plan_record_round_trips_phase_value(tmp_path, monkeypatch):
    """``save_plan_state`` must write ``phase`` as the raw string value
    so existing on-disk plan_state.json files stay readable. After load,
    ``MultiPlanState.__post_init__`` coerces back to the enum."""
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", str(tmp_path), raising=False)

    from larkhelm.plan_persistence import _serialise

    state = _make_state(phase=PlanPhase.WAITING, plan_id="rt_phase")
    rec = _serialise(state)
    assert rec["phase"] == "waiting", (
        "phase must be serialised as the raw string value, not the "
        "enum repr — otherwise existing JSON files break"
    )

    # And re-construct via the same coercion the loader uses:
    state2 = MultiPlanState(
        plan_id=rec["plan_id"], chat_id=state.chat_id,
        title=state.title, steps=state.steps,
        phase=rec["phase"],   # raw string from JSON
    )
    assert state2.phase is PlanPhase.WAITING
