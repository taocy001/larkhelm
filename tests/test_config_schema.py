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
