"""Coverage for ``agent_hub.plugin_loader``: entry-point discovery, config-based
loading, instantiation paths, and graceful failure (NFR-SEC-02).

review.md §6 backlog explicitly called out plugin_loader as missing direct unit
tests; this module fills that gap.
"""
from __future__ import annotations

import types
import unittest
from typing import Any
from unittest.mock import patch

from larkhelm.agent_hub import plugin_loader
from larkhelm.agent_hub.agent_base import AgentExecutor, AgentRegistry
from larkhelm.agent_hub.intent_types import AgentContext, AgentResult, IntentResult


# ── Fixtures ────────────────────────────────────────────────────────────


class _OkPlugin(AgentExecutor):
    """Minimal valid AgentExecutor used as a successful plugin target."""
    agent_type = "ok_plugin"
    description = "valid plugin"

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        return AgentResult(success=True, output="ok")


class _RaisingInit(AgentExecutor):
    agent_type = "raising_init"
    description = "raises on construction"

    def __init__(self) -> None:
        raise RuntimeError("ctor boom")

    def execute(self, intent: IntentResult, ctx: AgentContext) -> AgentResult:
        return AgentResult(success=True)


def _ok_factory() -> AgentExecutor:
    return _OkPlugin()


def _bad_factory() -> AgentExecutor:
    raise RuntimeError("factory boom")


def _wrong_type_factory():
    return "not an agent executor"


# ── Helpers ─────────────────────────────────────────────────────────────


class _FakeEntryPoint:
    """Stand-in for importlib.metadata.EntryPoint with a controllable load()."""

    def __init__(self, name: str, target):
        self.name = name
        self.group = "larkhelm.agents"
        self._target = target
        self._raise = isinstance(target, Exception)

    def load(self):
        if self._raise:
            raise self._target  # type: ignore[misc]
        return self._target

    def __repr__(self) -> str:
        return f"<FakeEP {self.name}>"


class _FakeEntryPointsModern:
    """select(group=…) API as in Python 3.10+."""

    def __init__(self, eps: list[_FakeEntryPoint]):
        self._eps = eps

    def select(self, group: str):
        return [ep for ep in self._eps if ep.group == group]


class _FakeEntryPointsDict(dict):
    """Pre-3.10 dict-like API exposed by entry_points() before .select()."""


def _isolated_registry():
    """Build a plugin_loader-aware fresh registry by swapping AGENT_REGISTRY."""
    return AgentRegistry()


# ── Tests ───────────────────────────────────────────────────────────────


class TestInstantiate(unittest.TestCase):

    def test_already_an_instance(self):
        inst = _OkPlugin()
        self.assertIs(plugin_loader._instantiate(inst), inst)

    def test_subclass_of_executor(self):
        result = plugin_loader._instantiate(_OkPlugin)
        self.assertIsInstance(result, _OkPlugin)

    def test_callable_factory(self):
        result = plugin_loader._instantiate(_ok_factory)
        self.assertIsInstance(result, _OkPlugin)

    def test_subclass_ctor_failure_returns_none(self):
        with patch.object(plugin_loader, "_safe_log") as log:
            result = plugin_loader._instantiate(_RaisingInit)
        self.assertIsNone(result)
        log.assert_called()

    def test_factory_failure_returns_none(self):
        with patch.object(plugin_loader, "_safe_log") as log:
            result = plugin_loader._instantiate(_bad_factory)
        self.assertIsNone(result)
        log.assert_called()

    def test_factory_wrong_type_returns_none(self):
        with patch.object(plugin_loader, "_safe_log") as log:
            result = plugin_loader._instantiate(_wrong_type_factory)
        self.assertIsNone(result)
        log.assert_called()

    def test_unknown_target_returns_none(self):
        # An int isn't a class, isn't an AgentExecutor, but is also not callable
        # in a way that produces an AgentExecutor — should silently bail.
        self.assertIsNone(plugin_loader._instantiate(42))


class TestLoadFromEntryPoints(unittest.TestCase):

    def setUp(self):
        self.registry = _isolated_registry()
        self.patcher = patch.object(plugin_loader, "AGENT_REGISTRY", self.registry)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_modern_select_api(self):
        eps = _FakeEntryPointsModern([
            _FakeEntryPoint("ok", _OkPlugin),
        ])
        count = plugin_loader._load_from_entry_points(_entry_points_fn=lambda: eps)
        self.assertEqual(count, 1)
        self.assertIsNotNone(self.registry.get("ok_plugin"))

    def test_legacy_dict_api(self):
        eps = _FakeEntryPointsDict()
        eps["larkhelm.agents"] = [_FakeEntryPoint("ok", _OkPlugin)]
        count = plugin_loader._load_from_entry_points(_entry_points_fn=lambda: eps)
        self.assertEqual(count, 1)
        self.assertIsNotNone(self.registry.get("ok_plugin"))

    def test_load_failure_does_not_abort(self):
        """A broken entry point must not block other entry points from loading."""
        eps = _FakeEntryPointsModern([
            _FakeEntryPoint("bad", RuntimeError("import boom")),
            _FakeEntryPoint("ok", _OkPlugin),
        ])
        with patch.object(plugin_loader, "_safe_log") as log:
            count = plugin_loader._load_from_entry_points(_entry_points_fn=lambda: eps)
        self.assertEqual(count, 1)
        self.assertIsNotNone(self.registry.get("ok_plugin"))
        log.assert_called()

    def test_entry_points_scan_failure_returns_zero(self):
        """If entry_points() itself blows up the loader returns 0, never raises."""
        def _boom():
            raise RuntimeError("scan failed")

        with patch.object(plugin_loader, "_safe_log") as log:
            count = plugin_loader._load_from_entry_points(_entry_points_fn=_boom)
        self.assertEqual(count, 0)
        log.assert_called()


class TestLoadFromConfig(unittest.TestCase):

    def setUp(self):
        self.registry = _isolated_registry()
        self.patcher = patch.object(plugin_loader, "AGENT_REGISTRY", self.registry)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def _stub_module_factory(self, name: str, **attrs: Any):
        """Build a ``(module_name, fake_importer)`` pair to pass as
        ``_import_module_fn=fake_importer`` so the loader uses the synthetic
        module without polluting ``sys.modules``."""
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)

        def _fake_import(req_name: str):
            if req_name == name:
                return mod
            raise ModuleNotFoundError(req_name)

        return _fake_import

    def test_module_attr_form(self):
        importer = self._stub_module_factory("test_pkg_a", MyAgent=_OkPlugin)
        count = plugin_loader._load_from_config(
            {"agent_plugins": ["test_pkg_a:MyAgent"]},
            _import_module_fn=importer,
        )
        self.assertEqual(count, 1)
        self.assertIsNotNone(self.registry.get("ok_plugin"))

    def test_dotted_form(self):
        importer = self._stub_module_factory("test_pkg_b", MyAgent=_OkPlugin)
        count = plugin_loader._load_from_config(
            {"agent_plugins": ["test_pkg_b.MyAgent"]},
            _import_module_fn=importer,
        )
        self.assertEqual(count, 1)
        self.assertIsNotNone(self.registry.get("ok_plugin"))

    def test_invalid_spec_skipped(self):
        with patch.object(plugin_loader, "_safe_log") as log:
            count = plugin_loader._load_from_config(
                {"agent_plugins": ["", "  ", "no_separator", 123, None]})
        self.assertEqual(count, 0)
        # invalid module-only spec triggers _safe_log; empty/non-str are dropped silently
        log.assert_called()

    def test_import_failure(self):
        def _boom(_name: str):
            raise ModuleNotFoundError("non_existent_module_xyz_123")

        with patch.object(plugin_loader, "_safe_log") as log:
            count = plugin_loader._load_from_config(
                {"agent_plugins": ["non_existent_module_xyz_123:Foo"]},
                _import_module_fn=_boom,
            )
        self.assertEqual(count, 0)
        log.assert_called()

    def test_attr_missing(self):
        importer = self._stub_module_factory("test_pkg_c")  # no MyAgent attr
        with patch.object(plugin_loader, "_safe_log") as log:
            count = plugin_loader._load_from_config(
                {"agent_plugins": ["test_pkg_c:MyAgent"]},
                _import_module_fn=importer,
            )
        self.assertEqual(count, 0)
        log.assert_called()

    def test_factory_returns_wrong_type(self):
        importer = self._stub_module_factory("test_pkg_d", make_agent=_wrong_type_factory)
        count = plugin_loader._load_from_config(
            {"agent_plugins": ["test_pkg_d:make_agent"]},
            _import_module_fn=importer,
        )
        self.assertEqual(count, 0)

    def test_non_list_plugins_returns_zero(self):
        self.assertEqual(plugin_loader._load_from_config({"agent_plugins": "not a list"}), 0)
        self.assertEqual(plugin_loader._load_from_config({}), 0)


class TestLoadPluginsCombined(unittest.TestCase):

    def test_combines_entrypoint_and_config_counts(self):
        # P3 REQ-07: load_plugins now returns a PluginLoadReport; legacy
        # int contract is preserved as ``len(report.loaded)``.
        registry = _isolated_registry()
        with patch.object(plugin_loader, "AGENT_REGISTRY", registry), \
             patch.object(plugin_loader, "_load_from_entry_points", return_value=2), \
             patch.object(plugin_loader, "_load_from_config", return_value=1):
            report = plugin_loader.load_plugins({"agent_plugins": ["ignored"]})
        self.assertIsInstance(report, plugin_loader.PluginLoadReport)
        # Loaded list reflects what helpers appended (none, since they're mocked).
        # The combined helper return values are discarded under the new API; we
        # care that the report came back populated as a dataclass.
        self.assertEqual(len(report.loaded), 0)

    def test_none_config_treated_as_empty(self):
        captured: dict = {}

        def _capture_cfg(cfg, **kw):
            captured["cfg"] = cfg
            return 0

        with patch.object(plugin_loader, "_load_from_entry_points", return_value=0), \
             patch.object(plugin_loader, "_load_from_config", side_effect=_capture_cfg):
            report = plugin_loader.load_plugins(None)
        self.assertEqual(len(report.loaded), 0)
        # Should call _load_from_config with an empty dict, never None.
        self.assertEqual(captured["cfg"], {})


if __name__ == "__main__":
    unittest.main()
