"""Round-trip producer↔consumer test for Phase C failure-card emission.

Same anti-pattern guard as ``test_phase_b_cascade_roundtrip.py``: synthetic
tests in ``test_crew_failure_card.py`` verify the PRODUCER half (state
mutation, redaction, OOM prefix) but never run the actual CONSUMER —
``crew_card._build_card(state)`` — to confirm the failure surfaces in
the user-visible card body.

A regression that, say, renamed ``AgentState.error`` to ``AgentState.err``
or moved the `f"❌ 失败：{a.error[:200]}"` rendering line in
``crew_card.py:165-166`` would pass all producer-side tests but make
failures invisible to the user. This file pins the chain end-to-end.

We exercise:
  emit_agent_failure(state, agent_id, stage, exc)
      ↓ (mutates state.agents[id].status / .error / .end_time)
      ↓ (also calls _crew_update_card, which calls _build_card)
  crew_card._build_card(state)
      ↓ (returns JSON card with the failure visible in element text)
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

# Bootstrap config so _cfg.config is initialised (required by some helpers
# called inside _build_card's lazy imports).
_TMP = tempfile.mkdtemp(prefix="larkhelm_failure_rt_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)


def _make_state(agent_ids: list[str], *, kind: str = "dev"):
    """Build a real CrewState (not a fake/mock) so the consumer side
    exercises production behaviour."""
    from larkhelm.crew_types import (
        AgentSpec, AgentState, AgentStatus, CrewPlan, CrewState,
    )
    specs = [
        AgentSpec(
            id=aid, role=f"{aid} role", model="", task_profile="engineer",
            system="", prompt=f"prompt for {aid}",
            depends_on=[], timeout=300,
        )
        for aid in agent_ids
    ]
    plan = CrewPlan(title="round-trip test", agents=specs,
                    synthesis_prompt="")
    agents = {s.id: AgentState(spec=s, status=AgentStatus.PENDING)
              for s in specs}
    return CrewState(
        crew_id=uuid.uuid4().hex[:8],
        chat_id="oc_rt_test",
        plan=plan,
        agents=agents,
        card_mid="fake_mid",
        cancel_ev=threading.Event(),
        phase="running",
        kind=kind,
        start_time=time.time() - 30,
    )


class FailureCardConsumerRoundTripTests(unittest.TestCase):
    """Run emit_*_failure → _build_card consumer end-to-end."""

    def setUp(self):
        # ``_build_card`` reads ``Path(_get_cwd(state.chat_id))`` looking
        # for a ``.crew_workspace`` dir. Stub _get_cwd to a tmpdir so the
        # test isn't affected by the operator's real cwd state.
        self._cwd_tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._cwd_tmp.cleanup)
        import larkhelm.crew_card as _cc
        self._patcher = patch.object(_cc, "_get_cwd",
                                     return_value=self._cwd_tmp.name)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    # ── emit_agent_failure → _build_card ───────────────────────────────

    def test_agent_failure_surfaces_in_built_card(self):
        """The user-facing card produced by ``_build_card`` must contain:
          1. The failed agent's role name (so the user knows WHICH agent)
          2. The failure error text (so the user knows WHY)
          3. The "失败" Chinese label that signals red-card status
        """
        from larkhelm.crew._failure_card import emit_agent_failure
        from larkhelm.crew_types import AgentStatus
        from larkhelm.crew_card import _build_card

        state = _make_state(["pm", "implementer"])
        # Avoid triggering the heartbeat network push (we don't have a
        # bridge running here). Patch the heartbeat hook used by emit_*.
        with patch("larkhelm.crew._failure_card._crew_update_card",
                   lambda s: None):
            emit_agent_failure(
                state, "implementer", "run",
                RuntimeError("subprocess hit unexpected EOF"),
            )

        # Producer-side invariants (pre-existing tests cover these).
        self.assertEqual(state.agents["implementer"].status, AgentStatus.FAILED)
        self.assertIn("subprocess hit unexpected EOF",
                      state.agents["implementer"].error)

        # CONSUMER-side invariants: the real _build_card output renders
        # the failure visibly.
        card_json = _build_card(state)
        text = json.dumps(card_json, ensure_ascii=False)
        # 1. Agent role mentioned.
        self.assertIn("implementer role", text,
                      "card omits the failed agent's role — user can't tell which agent failed")
        # 2. Failure error string surfaces.
        self.assertIn("subprocess hit unexpected EOF", text,
                      "card omits the actual error message — user has no idea what broke")
        # 3. Failure status label.
        self.assertIn("失败", text,
                      "card omits the 失败 label — Phase C UX guarantee broken")

    def test_agent_failure_oom_prefix_surfaces_in_card(self):
        """OOM-class errors get a friendlier '内存超限' prefix because the
        underlying ``rc=-9`` / ``killed by OS`` text isn't actionable.
        Pin that the rendered card shows the friendly prefix, NOT the
        raw OS text — otherwise the OOM-classification work in
        ``_failure_card._classify_oom`` is invisible to users."""
        from larkhelm.crew._failure_card import emit_agent_failure
        from larkhelm.crew_card import _build_card

        state = _make_state(["qa"])
        with patch("larkhelm.crew._failure_card._crew_update_card",
                   lambda s: None):
            emit_agent_failure(
                state, "qa", "run",
                RuntimeError("claude killed by OS (rc=-9, likely cgroup OOM)"),
            )

        card_json = _build_card(state)
        text = json.dumps(card_json, ensure_ascii=False)
        self.assertIn("内存超限", text,
                      "OOM-classified error must render the friendly Chinese prefix")

    def test_redacted_secret_not_in_card(self):
        """Redaction is producer-side, but pin it on the consumer side
        too: the user-visible CARD must not contain the original secret
        string. Catches a future refactor that bypasses ``redact_error``."""
        from larkhelm.crew._failure_card import emit_agent_failure
        from larkhelm.crew_card import _build_card

        state = _make_state(["pm"])
        with patch("larkhelm.crew._failure_card._crew_update_card",
                   lambda s: None):
            emit_agent_failure(
                state, "pm", "run",
                RuntimeError("auth failed: api_key=sk-secretvalue1234567890 expired"),
            )

        card_json = _build_card(state)
        text = json.dumps(card_json, ensure_ascii=False)
        self.assertNotIn("sk-secretvalue1234567890", text,
                         "raw secret leaked into the user-facing card — "
                         "redaction broken at producer↔consumer boundary")

    # ── breakpoint timeout → _build_card ───────────────────────────────

    def test_breakpoint_timeout_renders_cancelled_phase(self):
        """``emit_breakpoint_timeout`` flips ``state.phase`` to
        "cancelled". _build_card must then render the orange-cancelled
        title — proving the state-mutation→card-render pipeline closes."""
        from larkhelm.crew._failure_card import emit_breakpoint_timeout
        from larkhelm.crew_card import _build_card

        state = _make_state(["pm"])
        state.breakpoint_agent_id = "pm"
        with patch("larkhelm.crew._failure_card._crew_update_card",
                   lambda s: None), \
             patch("larkhelm.crew._failure_card.send_card",
                   lambda *a, **k: None):
            emit_breakpoint_timeout(state)

        self.assertEqual(state.phase, "cancelled")
        card_json = _build_card(state)
        text = json.dumps(card_json, ensure_ascii=False)
        self.assertIn("已取消", text,
                      "cancelled phase doesn't render the 已取消 title — "
                      "phase→card-title mapping broken")


if __name__ == "__main__":
    unittest.main()
