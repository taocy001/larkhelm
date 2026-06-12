"""Tests for the P1-2 ``_init_runtime`` split (4 sub-functions)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")


@pytest.fixture
def fresh_cfg(tmp_path: Path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID": "TEST_APP", "APP_SECRET": "TEST_SECRET",
        "response_timeout": 30, "hard_timeout": 120,
    }))
    return cfg_path, tmp_path


def test_init_paths_sets_globals(fresh_cfg):
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_paths(str(cfg_path), str(tmp_path))
    assert _cfg.CONFIG_PATH == Path(cfg_path)
    assert _cfg.DATA_DIR == Path(tmp_path)
    assert _cfg.SESSION_DIR.exists()
    assert _cfg.LOG_DIR.exists()


def test_init_app_config_loads_keys(fresh_cfg):
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_paths(str(cfg_path), str(tmp_path))
    _cfg._init_app_config()
    assert _cfg.APP_ID == "TEST_APP"
    assert _cfg.APP_SECRET == "TEST_SECRET"
    # P1-3 / P1-5 / P1-6 / P1-8 new globals exist with defaults
    assert _cfg.HEALTH_ENDPOINT_PORT == 0
    assert _cfg.HEALTH_BIND_ADDR == "127.0.0.1"
    assert _cfg.MEMORY_CASCADE_MIDFLIGHT_CANCEL in (True, False)


def test_init_backends_sets_registry(fresh_cfg):
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_paths(str(cfg_path), str(tmp_path))
    _cfg._init_app_config()
    _cfg._init_backends()
    assert hasattr(_cfg, "BACKEND_REGISTRY")
    assert _cfg.BACKEND_REGISTRY is not None


def test_init_plugins_is_noop_in_test_mode(fresh_cfg):
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_paths(str(cfg_path), str(tmp_path))
    _cfg._init_app_config()
    # Must not raise in test mode (skips plugin loader + memory_gc)
    _cfg._init_plugins()


def test_init_runtime_facade_produces_runtime_snapshot(fresh_cfg):
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_runtime(str(cfg_path), str(tmp_path))
    assert _cfg._runtime is not None
    assert _cfg._runtime.APP_ID == "TEST_APP"
    assert _cfg._runtime.CONFIG_PATH == Path(cfg_path)


def test_init_runtime_idempotent_equivalent_snapshot(fresh_cfg):
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_runtime(str(cfg_path), str(tmp_path))
    snap_a = _cfg._runtime
    _cfg._init_runtime(str(cfg_path), str(tmp_path))
    snap_b = _cfg._runtime
    # Both runs hit setdefault on the in-memory dict, so basic scalar
    # equivalence holds across the boundary fields most consumers read.
    assert snap_a.APP_ID == snap_b.APP_ID
    assert snap_a.RESPONSE_TIMEOUT == snap_b.RESPONSE_TIMEOUT
    assert snap_a.HARD_TIMEOUT == snap_b.HARD_TIMEOUT
    assert snap_a.DEFAULT_MODEL == snap_b.DEFAULT_MODEL


def test_init_app_config_picks_up_new_p1_keys(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID": "X", "APP_SECRET": "Y",
        "health_endpoint_port": 9300,
        "memory_cascade_midflight_cancel": False,
    }))
    import larkhelm.config as _cfg
    _cfg._init_paths(str(cfg_path), str(tmp_path))
    _cfg._init_app_config()
    assert _cfg.HEALTH_ENDPOINT_PORT == 9300
    assert _cfg.MEMORY_CASCADE_MIDFLIGHT_CANCEL is False


def test_layered_cache_config_defaults_graduated(fresh_cfg):
    """2026-06-12 graduation: layered Anthropic cache_control is default-on.

    Pins setdefault + module globals so a silent rollback of the defaults
    (anthropic_layered_cache_control_enabled=true /
    anthropic_layered_cache_traffic=1.0) fails loudly. Opt-out path
    (false / 0.0) is covered by tests/test_backend_api_template.py AC-05.
    """
    cfg_path, tmp_path = fresh_cfg
    import larkhelm.config as _cfg
    _cfg._init_runtime(str(cfg_path), str(tmp_path))
    assert _cfg.config.get("anthropic_layered_cache_control_enabled") is True
    assert _cfg.config.get("anthropic_layered_cache_traffic") == 1.0
    assert _cfg.ANTHROPIC_LAYERED_CACHE_CONTROL is True
    assert _cfg.ANTHROPIC_LAYERED_CACHE_TRAFFIC == 1.0


def test_layered_cache_config_opt_out_respected(tmp_path):
    """Operator-set false / 0.0 must survive the new defaults (rollback knob)."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID": "X", "APP_SECRET": "Y",
        "anthropic_layered_cache_control_enabled": False,
        "anthropic_layered_cache_traffic": 0.0,
    }))
    import larkhelm.config as _cfg
    _cfg._init_runtime(str(cfg_path), str(tmp_path))
    assert _cfg.ANTHROPIC_LAYERED_CACHE_CONTROL is False
    assert _cfg.ANTHROPIC_LAYERED_CACHE_TRAFFIC == 0.0
