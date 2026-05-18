"""AC-07 — P3 REQ-07 plugin loader structured report + admin card."""
from __future__ import annotations

import types
import unittest
from unittest.mock import patch

from larkhelm.agent_hub import plugin_loader
from larkhelm.agent_hub.plugin_report import (
    PluginFailure,
    PluginLoadReport,
    emit_admin_card,
)


class _IsolatedRegistry:
    """Stand-in registry the loader writes into during tests."""

    def __init__(self) -> None:
        self._agents: dict[str, object] = {}

    def register(self, agent) -> None:
        self._agents[getattr(agent, "agent_type", repr(agent))] = agent

    def list_types(self):
        return sorted(self._agents)


class TestLoadPluginsReport(unittest.TestCase):

    def test_failed_import_recorded_in_report(self) -> None:
        def _boom(_name: str):
            raise ModuleNotFoundError("mymod_xyz_123")

        report = PluginLoadReport()
        plugin_loader._load_from_config(
            {"agent_plugins": ["mymod_xyz_123:Foo"]},
            _import_module_fn=_boom,
            report=report,
        )
        self.assertEqual(len(report.failed), 1)
        self.assertEqual(report.failed[0].spec, "mymod_xyz_123:Foo")
        self.assertEqual(report.failed[0].stage, "import")
        self.assertIn("mymod_xyz_123", report.failed[0].reason)

    def test_invalid_spec_recorded(self) -> None:
        report = PluginLoadReport()
        plugin_loader._load_from_config(
            {"agent_plugins": ["no_separator"]},
            report=report,
        )
        self.assertEqual(len(report.failed), 1)
        self.assertEqual(report.failed[0].stage, "import")

    def test_load_plugins_returns_report(self) -> None:
        report = plugin_loader.load_plugins({"agent_plugins": []})
        self.assertIsInstance(report, PluginLoadReport)
        self.assertEqual(report.failed, [])


class TestEmitAdminCard(unittest.TestCase):

    def test_emit_card_skipped_when_no_failures(self) -> None:
        report = PluginLoadReport(loaded=["ok"], failed=[])
        with patch("larkhelm.lark_client.send_card") as fake:
            emit_admin_card(report, "chat_admin")
        fake.assert_not_called()

    def test_emit_card_skipped_when_admin_chat_id_empty(self) -> None:
        report = PluginLoadReport(
            failed=[PluginFailure("mypkg:X", "import", "boom")],
        )
        with patch("larkhelm.lark_client.send_card") as fake:
            emit_admin_card(report, "")
        fake.assert_not_called()

    def test_emit_card_sends_orange_card_with_reason(self) -> None:
        report = PluginLoadReport(
            failed=[PluginFailure("mypkg:X", "import", "boom!")],
            duration_sec=0.05,
        )
        with patch("larkhelm.lark_client.send_card") as fake:
            emit_admin_card(report, "chat_admin")
        fake.assert_called_once()
        args, kwargs = fake.call_args
        # Inspect either positional or keyword form to be defensive.
        chat_id = args[0] if args else kwargs.get("chat_id")
        title = args[1] if len(args) > 1 else kwargs.get("title", "")
        body = args[2] if len(args) > 2 else kwargs.get("body", "")
        color = args[3] if len(args) > 3 else kwargs.get("color", "")
        self.assertEqual(chat_id, "chat_admin")
        self.assertIn("Plugin", title)
        self.assertIn("boom!", body)
        self.assertEqual(color, "orange")

    def test_emit_card_swallows_send_failures(self) -> None:
        report = PluginLoadReport(
            failed=[PluginFailure("mypkg:X", "import", "boom!")],
        )
        with patch("larkhelm.lark_client.send_card", side_effect=RuntimeError("net down")):
            # Must not raise.
            emit_admin_card(report, "chat_admin")


if __name__ == "__main__":
    unittest.main()
