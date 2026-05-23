"""P1-8: validate larkhelm_config.example.json against config_schema.json."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

jsonschema = pytest.importorskip("jsonschema")

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "larkhelm" / "config_schema.json"
EXAMPLE_PATH = REPO_ROOT / "larkhelm_config.example.json"


def _load_json_with_comments(p: Path) -> dict:
    """Strip ``_comment_*`` keys (used as inline docs) before validation."""
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_comment")}


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_schema_loads_as_valid_jsonschema():
    schema = _load_schema()
    # Validator construction must not raise.
    jsonschema.Draft7Validator.check_schema(schema)


def test_example_config_passes_schema():
    schema = _load_schema()
    example = _load_json_with_comments(EXAMPLE_PATH)
    # validate raises on failure; absence of an exception = pass.
    jsonschema.validate(example, schema)


def test_invalid_default_model_rejected():
    schema = _load_schema()
    bad = {"default_model": "xxx"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_invalid_response_timeout_rejected():
    schema = _load_schema()
    bad = {"response_timeout": -1}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_invalid_health_endpoint_port_rejected():
    schema = _load_schema()
    bad = {"health_endpoint_port": 99999}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_invalid_voice_engine_rejected():
    schema = _load_schema()
    bad = {"voice_engine": "whisper-cpp"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_invalid_doc_write_backend_rejected():
    schema = _load_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"doc_write_backend": "s3"}, schema)


def test_session_layer_budgets_min_floor_enforced():
    schema = _load_schema()
    bad = {"memory_session_layer_budgets": {"work_context": 10}}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_traffic_above_one_rejected():
    schema = _load_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({"memory_retriever_traffic": 1.5}, schema)


def test_unknown_keys_allowed():
    schema = _load_schema()
    # additionalProperties: true → unknown keys validate.
    jsonschema.validate({"my_custom_key": "anything"}, schema)


# ── Default-value regression: P0+P1 caching audit (2026-05-22) ───────────


def test_doc_inject_cache_ttl_default_300(tmp_path):
    """An empty operator config must resolve to TTL=300s — aligned with
    Anthropic 5min ephemeral cache. Previously was 60s.
    """
    import larkhelm.config as _cfg

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))
    _cfg._init_runtime(config_path=str(cfg_file), data_dir=str(tmp_path))
    assert _cfg.DOC_INJECT_CACHE_TTL_SEC == 300


def test_anthropic_extended_cache_default_true(tmp_path):
    """The 1h Anthropic extended-cache opt-in defaults to ON for any
    operator who hasn't pinned it. A first-call rejection auto-disables
    process-wide, so leaving the toggle on is safe."""
    import larkhelm.config as _cfg

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))
    _cfg._init_runtime(config_path=str(cfg_file), data_dir=str(tmp_path))
    assert _cfg.ANTHROPIC_EXTENDED_CACHE_ENABLED is True


def test_anthropic_extended_cache_opt_out_honoured(tmp_path):
    """A pinned ``false`` survives _init_app_config."""
    import larkhelm.config as _cfg

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({
        "APP_ID": "x", "APP_SECRET": "x",
        "anthropic_extended_cache_enabled": False,
    }))
    _cfg._init_runtime(config_path=str(cfg_file), data_dir=str(tmp_path))
    assert _cfg.ANTHROPIC_EXTENDED_CACHE_ENABLED is False
