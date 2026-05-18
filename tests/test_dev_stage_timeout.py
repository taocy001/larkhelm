"""AC-10 — P3 REQ-10 dev_stage_timeouts override."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import larkhelm.config as _cfg
from larkhelm.crew._pipeline import _make_dev_pipeline


class TestDevStageTimeouts(unittest.TestCase):

    def setUp(self) -> None:
        # _make_dev_pipeline relies on _cfg.RESPONSE_TIMEOUT (set by
        # _init_runtime in production). Tests don't always run _init_runtime
        # so we set a baseline.
        self._prev_response_timeout = getattr(_cfg, "RESPONSE_TIMEOUT", None)
        _cfg.RESPONSE_TIMEOUT = 300

    def tearDown(self) -> None:
        if self._prev_response_timeout is None:
            try:
                delattr(_cfg, "RESPONSE_TIMEOUT")
            except AttributeError:
                pass
        else:
            _cfg.RESPONSE_TIMEOUT = self._prev_response_timeout

    def test_override_replaces_implementer_timeout(self) -> None:
        with patch.object(_cfg, "DEV_STAGE_TIMEOUTS", {"implementer": 7200}):
            plan = _make_dev_pipeline("写一个登录页", cwd="/tmp")
        timeouts = {spec.id: spec.timeout for spec in plan.agents}
        self.assertEqual(timeouts["implementer"], 7200)

    def test_unlisted_stages_keep_formula(self) -> None:
        with patch.object(_cfg, "DEV_STAGE_TIMEOUTS", {"implementer": 7200}):
            plan = _make_dev_pipeline("写一个登录页", cwd="/tmp")
        timeouts = {spec.id: spec.timeout for spec in plan.agents}
        self.assertNotEqual(timeouts["pm"], 7200)

    def test_empty_override_keeps_all_defaults(self) -> None:
        with patch.object(_cfg, "DEV_STAGE_TIMEOUTS", {}):
            plan = _make_dev_pipeline("hello", cwd="/tmp")
        for spec in plan.agents:
            self.assertGreater(spec.timeout, 0)


if __name__ == "__main__":
    unittest.main()
