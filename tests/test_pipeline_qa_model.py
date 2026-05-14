"""Regression guard for the qa-step model in ``crew/_pipeline.py``.

History
-------
Before this commit, ``_make_dev_pipeline``'s qa step had ``model="gemini"``
hard-coded. With Gemini disabled in production (the standard 5-backend
config has all ``gemini*`` entries ``enabled: false``), every /dev run
that reached the qa stage failed permanently with:

    backend 'gemini' is disabled in config (set enabled=true in
    probe_models or use a different model)

The fix unified all six agents on ``claude``. This module pins that
invariant so a future "let's diversify the model per role" experiment
can't silently regress this — the qa step was a latent footgun for
months because there were no tests asserting which backend it ran on.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path

# Config bootstrap (mirrors tests/test_workspace_finalize.py): the
# pipeline factory reads ``_cfg.RESPONSE_TIMEOUT`` for the qa step's
# timeout, so config must be init'd before importing the factory.
_TMP = tempfile.mkdtemp(prefix="larkhelm_qa_model_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.crew._pipeline import _make_dev_pipeline


class DevPipelineModelTests(unittest.TestCase):
    """All ``/dev`` agents must run on a backend that's healthy in the
    standard production config — no hard-coded references to backends
    that ship disabled."""

    def setUp(self):
        # _make_dev_pipeline takes a requirement string + cwd; values don't
        # affect AgentSpec.model so any non-empty strings are fine.
        self.plan = _make_dev_pipeline(requirement="(test)", cwd="/tmp")

    def test_no_agent_uses_disabled_gemini_backend(self):
        """The exact bug we shipped a fix for: qa.model = 'gemini'
        while ``gemini*`` entries are ``enabled: false`` in
        ``larkhelm_config.example.json`` and the standard production
        config. Asserting on *no* agent (not just qa) so the rule
        scales to any future agent additions."""
        offenders = [(a.id, a.model) for a in self.plan.agents
                     if a.model == "gemini"]
        self.assertEqual(offenders, [],
            "no /dev agent may use model='gemini' — that backend is "
            "disabled in the standard config and the dispatch will "
            f"permanently fail. Offenders: {offenders!r}. If the user "
            "needs gemini, they must explicitly enable it in config + "
            "we should fall back to claude when it's disabled.")

    def test_qa_agent_uses_claude(self):
        """Direct pin on the formerly-broken agent — make a future
        regression noisy at the test level instead of in production."""
        qa = next((a for a in self.plan.agents if a.id == "qa"), None)
        self.assertIsNotNone(qa, "qa agent must exist in /dev pipeline")
        self.assertEqual(qa.model, "claude",
            "qa.model was 'gemini' (broken) — fix unified everyone on "
            "'claude'; do not revert without enabling gemini in config "
            "AND updating BackendRegistry.get_orchestrator fallback")

    def test_all_six_agents_present_and_run_claude(self):
        """End-to-end check on the canonical /dev agent list. If the
        pipeline gains a new agent (test step, security audit, etc),
        this test forces the author to decide its model explicitly
        rather than copy-pasting one and forgetting."""
        expected = {"pm", "architect", "implementer", "fixer", "qa", "reviewer"}
        actual = {a.id for a in self.plan.agents}
        self.assertEqual(actual, expected,
            f"/dev pipeline shape changed: {actual ^ expected}. Update "
            "this test if intentional.")
        for a in self.plan.agents:
            self.assertEqual(a.model, "claude",
                f"agent {a.id!r} uses model={a.model!r}; the standard "
                "config only ships claude / kimi / deepseek enabled. "
                "If diversifying, update this test + verify the chosen "
                "backend is healthy in production.")


if __name__ == "__main__":
    unittest.main()
