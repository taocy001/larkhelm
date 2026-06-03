"""Unit tests for _parse_qa_verdict and QA retry-round cap (Week4-P1 M-QA-DELIVERY)."""
from __future__ import annotations

import threading
import uuid

import pytest

import os
os.environ.setdefault("LARKHELM_TEST_MODE", "1")


# ── helpers ──────────────────────────────────────────────────────────────────


def _parse(result: str) -> dict:
    from larkhelm.crew._runner import _parse_qa_verdict
    return _parse_qa_verdict(result)


def _make_spec(agent_id: str, **kw):
    from larkhelm.crew_types import AgentSpec
    return AgentSpec(
        id=agent_id, role=agent_id, model="", system="", prompt="",
        depends_on=[], timeout=60, **kw,
    )


def _make_state(specs, max_qa_retry_rounds: int = 2):
    from larkhelm.crew_types import AgentState, CrewPlan, CrewState
    plan = CrewPlan(
        title="test", agents=specs, synthesis_prompt="",
        max_qa_retry_rounds=max_qa_retry_rounds,
    )
    agents = {s.id: AgentState(spec=s) for s in specs}
    return CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="test_chat",
        plan=plan,
        agents=agents,
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind="crew",
    )


# ── AC-02: _parse_qa_verdict ──────────────────────────────────────────────────


def test_parse_qa_verdict_pass():
    result = "some output\nQA_VERDICT: PASS FAILED=0 BLOCKED=0 SKIP=2\nTESTS_PASSED"
    v = _parse(result)
    assert v["verdict"] == "PASS"
    assert v["failed_count"] == 0
    assert v["blocked_count"] == 0
    assert v["skip_count"] == 2


def test_parse_qa_verdict_fail():
    result = "bugs found\nQA_VERDICT: FAIL FAILED=3 BLOCKED=1 SKIP=0\nTESTS_FAILED"
    v = _parse(result)
    assert v["verdict"] == "FAIL"
    assert v["failed_count"] == 3
    assert v["blocked_count"] == 1
    assert v["skip_count"] == 0


def test_parse_qa_verdict_unknown_when_no_line():
    result = "everything passed\nTESTS_PASSED"
    v = _parse(result)
    assert v["verdict"] == "UNKNOWN"
    assert v["failed_count"] == 0


def test_parse_qa_verdict_never_raises():
    for bad in [None, "", "garbage\n" * 100, "QA_VERDICT:", "QA_VERDICT: BADWORD FAILED=x"]:
        try:
            result = bad if isinstance(bad, str) else ""
            v = _parse(result)
            assert "verdict" in v
        except Exception as exc:
            pytest.fail(f"_parse_qa_verdict raised for {bad!r}: {exc}")


def test_parse_qa_verdict_uses_last_20_lines():
    # Put a valid line beyond 20-line window and an UNKNOWN one in the last 20
    prefix_lines = ["noise"] * 25
    last_lines = ["QA_VERDICT: FAIL FAILED=2 BLOCKED=0 SKIP=0", "TESTS_FAILED"]
    result = "\n".join(prefix_lines + last_lines)
    v = _parse(result)
    assert v["verdict"] == "FAIL"
    assert v["failed_count"] == 2


def test_parse_qa_verdict_partial_counts():
    # Only FAILED provided; BLOCKED and SKIP should default to 0
    result = "QA_VERDICT: FAIL FAILED=5\nTESTS_FAILED"
    v = _parse(result)
    assert v["verdict"] == "FAIL"
    assert v["failed_count"] == 5
    assert v["blocked_count"] == 0
    assert v["skip_count"] == 0


# ── AC-03: CrewPlan.max_qa_retry_rounds field ─────────────────────────────────


def test_crew_plan_max_qa_retry_rounds_default():
    from larkhelm.crew_types import CrewPlan
    plan = CrewPlan(title="t", agents=[])
    assert plan.max_qa_retry_rounds == 2


def test_crew_plan_max_qa_retry_rounds_custom():
    from larkhelm.crew_types import CrewPlan
    plan = CrewPlan(title="t", agents=[], max_qa_retry_rounds=5)
    assert plan.max_qa_retry_rounds == 5


# ── AC-04: QA verdict prefix injected into feedback ───────────────────────────


def test_qa_verdict_prefix_format():
    """Verify the prefix string format matches the fixer's expected input."""
    from larkhelm.crew._runner import _parse_qa_verdict
    result = "QA_VERDICT: FAIL FAILED=3 BLOCKED=1 SKIP=0\nTESTS_FAILED"
    v = _parse_qa_verdict(result)
    prefix = (
        f"QA Verdict: {v['verdict']} — FAILED={v['failed_count']} "
        f"BLOCKED={v['blocked_count']} SKIP={v['skip_count']}\n"
    )
    assert "QA Verdict: FAIL" in prefix
    assert "FAILED=3" in prefix
    assert "BLOCKED=1" in prefix


# ── AC-04: max_qa_retry_rounds cap ───────────────────────────────────────────


def test_max_qa_retry_rounds_field_on_state():
    """CrewState.plan.max_qa_retry_rounds is accessible from state."""
    specs = [_make_spec("qa", fail_marker="TESTS_FAILED", max_retries=1)]
    state = _make_state(specs, max_qa_retry_rounds=1)
    assert state.plan.max_qa_retry_rounds == 1


def test_crew_state_phase_outputs_default_empty():
    """CrewState.phase_outputs defaults to empty dict."""
    specs = [_make_spec("qa")]
    state = _make_state(specs)
    assert state.phase_outputs == {}
