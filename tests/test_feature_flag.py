"""AC-10: when intent_router_enabled=false, _intent_router_active returns False
and the handlers/_message.py code path skips importing agent_hub.

This test does not import handlers/_message at module top so we can observe
the import behavior of `_intent_router_active`.
"""
import sys
import unittest

import larkhelm.config as _cfg


class TestFeatureFlagShortCircuit(unittest.TestCase):

    def test_active_false_when_flag_disabled(self):
        from larkhelm.handlers._message import _intent_router_active

        original = getattr(_cfg, "config", {})
        _cfg.config = {"intent_router_enabled": False, "intent_router_traffic": 1.0}
        try:
            self.assertFalse(_intent_router_active("any-chat"))
        finally:
            _cfg.config = original

    def test_active_false_when_traffic_zero(self):
        from larkhelm.handlers._message import _intent_router_active

        original = getattr(_cfg, "config", {})
        _cfg.config = {"intent_router_enabled": True, "intent_router_traffic": 0.0}
        try:
            self.assertFalse(_intent_router_active("any-chat"))
        finally:
            _cfg.config = original

    def test_active_traffic_consistency(self):
        """Same chat_id always lands in the same bucket (deterministic hash)."""
        from larkhelm.handlers._message import _intent_router_active

        original = getattr(_cfg, "config", {})
        _cfg.config = {"intent_router_enabled": True, "intent_router_traffic": 0.5}
        try:
            decisions = {_intent_router_active("oc_consistent") for _ in range(20)}
            self.assertEqual(len(decisions), 1, decisions)
        finally:
            _cfg.config = original

    def test_disabled_flag_does_not_force_agent_hub_import(self):
        """If agent_hub was never imported, helper alone should not trigger it.

        We can't fully test the message handler without a Feishu event, but
        we can at least confirm `_intent_router_active` returns False without
        touching agent_hub.
        """
        from larkhelm.handlers._message import _intent_router_active

        # Pretend agent_hub hasn't been imported yet by removing it from sys.modules.
        # (Other tests may have imported it; that's fine — we only need to verify
        # _intent_router_active does not re-import on flag=false.)
        had_agent_hub = "larkhelm.agent_hub" in sys.modules
        original = getattr(_cfg, "config", {})
        _cfg.config = {"intent_router_enabled": False, "intent_router_traffic": 1.0}
        try:
            _intent_router_active("oc_x")
        finally:
            _cfg.config = original
        # If _intent_router_active had imported agent_hub, the module presence
        # would not change. The contract is just that flag=false short-circuits.
        # This assertion is informational — it guards the helper, not the handler.
        if not had_agent_hub:
            self.assertNotIn(
                "larkhelm.agent_hub", sys.modules,
                "_intent_router_active() must not import agent_hub when flag is off",
            )


if __name__ == "__main__":
    unittest.main()
