"""Tests for the voice engine dispatcher (M3.2 commit β).

Coverage:
* `transcribe.transcribe()` honors `_cfg.VOICE_ENGINE` to route between
  faster_whisper and dashscope.
* `is_ready()` semantics differ per engine:
  - faster_whisper: VOICE_ENABLED + not _LOAD_FAILED
  - dashscope:      VOICE_ENABLED + VOICE_API_KEY non-empty
* DashScope adapter handles the "happy path" (mocked SDK), missing API
  key, missing SDK (ImportError), non-200 status, and empty result.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import larkhelm.config as _cfg

# `from larkhelm.voice.transcribe import …` in voice/__init__.py shadows the
# submodule with the function of the same name. Reach the submodule via
# sys.modules so we can poke its module-level state from tests.
import larkhelm.voice.transcribe  # noqa: F401 — registers in sys.modules
import larkhelm.voice._engine_dashscope as ds_mod
t_mod = sys.modules["larkhelm.voice.transcribe"]


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_voice_state():
    """Reset module state before each test."""
    t_mod._MODEL = None
    t_mod._LOAD_FAILED = False
    t_mod._LOAD_LOGGED_SUCCESS = False
    # snapshot + restore config attrs
    snap = {
        k: getattr(_cfg, k, None)
        for k in ("VOICE_ENABLED", "VOICE_ENGINE", "VOICE_API_KEY",
                  "VOICE_MODEL_SIZE", "VOICE_COMPUTE_TYPE")
    }
    yield
    for k, v in snap.items():
        if v is not None:
            setattr(_cfg, k, v)


# ── is_ready() per engine ─────────────────────────────────────────────────


def test_is_ready_faster_whisper_default():
    _cfg.VOICE_ENABLED = True
    _cfg.VOICE_ENGINE = "faster_whisper"
    assert t_mod.is_ready() is True


def test_is_ready_faster_whisper_after_load_failed():
    _cfg.VOICE_ENABLED = True
    _cfg.VOICE_ENGINE = "faster_whisper"
    t_mod._LOAD_FAILED = True
    assert t_mod.is_ready() is False


def test_is_ready_dashscope_with_key():
    _cfg.VOICE_ENABLED = True
    _cfg.VOICE_ENGINE = "dashscope"
    _cfg.VOICE_API_KEY = "sk-fake"
    # _LOAD_FAILED on faster_whisper path doesn't matter for dashscope
    t_mod._LOAD_FAILED = True
    assert t_mod.is_ready() is True


def test_is_ready_dashscope_without_key():
    _cfg.VOICE_ENABLED = True
    _cfg.VOICE_ENGINE = "dashscope"
    _cfg.VOICE_API_KEY = ""
    assert t_mod.is_ready() is False


def test_is_ready_disabled_globally():
    _cfg.VOICE_ENABLED = False
    _cfg.VOICE_ENGINE = "dashscope"
    _cfg.VOICE_API_KEY = "sk-fake"
    assert t_mod.is_ready() is False


# ── transcribe() dispatch ─────────────────────────────────────────────────


def test_transcribe_disabled_returns_disabled_tag():
    _cfg.VOICE_ENABLED = False
    r = t_mod.transcribe("/tmp/whatever.opus", lang="zh")
    assert r["ok"] is False
    assert r["error"] == "disabled"


def test_transcribe_dispatches_to_dashscope_when_engine_set():
    _cfg.VOICE_ENABLED = True
    _cfg.VOICE_ENGINE = "dashscope"
    _cfg.VOICE_API_KEY = "sk-fake"

    captured = {}

    def fake_dashscope(audio_path, *, lang):
        captured["audio_path"] = audio_path
        captured["lang"] = lang
        return {
            "ok": True, "text": "你好世界", "duration": 0.5,
            "lang": lang, "error": None,
        }

    with patch.object(ds_mod, "transcribe_via_dashscope", side_effect=fake_dashscope):
        # Patch the lookup site too — transcribe.py imports lazily inside the function
        with patch("larkhelm.voice._engine_dashscope.transcribe_via_dashscope",
                   side_effect=fake_dashscope):
            r = t_mod.transcribe("/tmp/test.opus", lang="zh")
    assert r["ok"] is True
    assert r["text"] == "你好世界"
    assert captured["audio_path"] == "/tmp/test.opus"


def test_transcribe_falls_back_to_faster_whisper_for_unknown_engine():
    """An invalid VOICE_ENGINE shouldn't crash the dispatcher; default path."""
    _cfg.VOICE_ENABLED = True
    _cfg.VOICE_ENGINE = "totally_made_up"  # not whitelisted, but normalized only at config load
    # _run_inference path: stub _load_model to return None → load_failed
    with patch.object(t_mod, "_load_model", return_value=None):
        r = t_mod.transcribe("/tmp/x.opus", lang="zh")
    # Falls into faster_whisper branch → _run_inference → load_failed
    assert r["ok"] is False
    assert r["error"] == "load_failed"


# ── DashScope adapter behavior ────────────────────────────────────────────


def test_dashscope_no_key_returns_clear_error():
    _cfg.VOICE_API_KEY = ""
    r = ds_mod.transcribe_via_dashscope("/tmp/x.opus", lang="zh")
    assert r["ok"] is False
    assert r["error"] == "dashscope_no_api_key"


def test_dashscope_sdk_missing_returns_clear_error():
    _cfg.VOICE_API_KEY = "sk-fake"
    # Make `import dashscope` raise ImportError
    with patch.dict(sys.modules, {"dashscope": None}):
        r = ds_mod.transcribe_via_dashscope("/tmp/x.opus", lang="zh")
    assert r["ok"] is False
    assert r["error"].startswith("dashscope_sdk_missing")


def test_dashscope_happy_path_with_mocked_sdk():
    """Inject a fake `dashscope` module and verify result parsing."""
    _cfg.VOICE_API_KEY = "sk-fake"

    fake_sentence = [{"text": "你好"}, {"text": "世界"}]
    fake_output = SimpleNamespace(sentence=fake_sentence)
    fake_result = SimpleNamespace(
        status_code=200, output=fake_output, message=""
    )

    # Real DashScope SDK: Recognition is a class. Constructor takes streaming
    # params; .call(file=...) is the instance method that runs ASR. The mock
    # mirrors that shape so the test catches Recognition.call() vs
    # recognition.call() classmethod-vs-instance bugs.
    class _FakeRecognition:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        def call(self, *, file):
            return fake_result
    fake_recognition = _FakeRecognition
    fake_asr = SimpleNamespace(Recognition=fake_recognition)
    fake_audio = SimpleNamespace(asr=fake_asr)
    fake_dashscope = SimpleNamespace(
        api_key="", audio=fake_audio, __version__="1.20.0",
    )

    with patch.dict(sys.modules, {
        "dashscope": fake_dashscope,
        "dashscope.audio": fake_audio,
        "dashscope.audio.asr": fake_asr,
    }):
        r = ds_mod.transcribe_via_dashscope("/tmp/test.opus", lang="zh")
    assert r["ok"] is True, f"unexpected: {r}"
    assert r["text"] == "你好世界"
    assert r["lang"] == "zh"


def test_dashscope_non_200_status():
    _cfg.VOICE_API_KEY = "sk-bad"

    fake_result = SimpleNamespace(
        status_code=401, message="Invalid Authentication", output=None,
    )
    class _FakeRecognition:
        def __init__(self, **kw): pass
        def call(self, *, file): return fake_result
    fake_asr = SimpleNamespace(Recognition=_FakeRecognition)
    fake_audio = SimpleNamespace(asr=fake_asr)
    fake_dashscope = SimpleNamespace(api_key="", audio=fake_audio, __version__="1.20.0")

    with patch.dict(sys.modules, {
        "dashscope": fake_dashscope,
        "dashscope.audio": fake_audio,
        "dashscope.audio.asr": fake_asr,
    }):
        r = ds_mod.transcribe_via_dashscope("/tmp/x.opus", lang="zh")
    assert r["ok"] is False
    assert r["error"].startswith("dashscope_status_401")
    assert "Invalid Authentication" in r["error"]


def test_dashscope_empty_result_text():
    _cfg.VOICE_API_KEY = "sk-fake"
    fake_output = SimpleNamespace(sentence=[])
    fake_result = SimpleNamespace(status_code=200, output=fake_output)
    class _FakeRecognition:
        def __init__(self, **kw): pass
        def call(self, *, file): return fake_result
    fake_asr = SimpleNamespace(Recognition=_FakeRecognition)
    fake_audio = SimpleNamespace(asr=fake_asr)
    fake_dashscope = SimpleNamespace(api_key="", audio=fake_audio, __version__="1.20.0")

    with patch.dict(sys.modules, {
        "dashscope": fake_dashscope,
        "dashscope.audio": fake_audio,
        "dashscope.audio.asr": fake_asr,
    }):
        r = ds_mod.transcribe_via_dashscope("/tmp/x.opus", lang="zh")
    assert r["ok"] is False
    assert r["error"] == "dashscope_empty_result"


def test_dashscope_call_raises():
    _cfg.VOICE_API_KEY = "sk-fake"

    class _FakeRecognition:
        def __init__(self, **kw): pass
        def call(self, *, file):
            raise ConnectionError("network down")
    fake_asr = SimpleNamespace(Recognition=_FakeRecognition)
    fake_audio = SimpleNamespace(asr=fake_asr)
    fake_dashscope = SimpleNamespace(api_key="", audio=fake_audio, __version__="1.20.0")

    with patch.dict(sys.modules, {
        "dashscope": fake_dashscope,
        "dashscope.audio": fake_audio,
        "dashscope.audio.asr": fake_asr,
    }):
        r = ds_mod.transcribe_via_dashscope("/tmp/x.opus", lang="zh")
    assert r["ok"] is False
    assert r["error"].startswith("dashscope_call_failed:ConnectionError")


# ── Config validation (already in config.py — smoke check) ───────────────


def test_config_dashscope_no_key_disables_voice(tmp_path, monkeypatch):
    """voice_engine='dashscope' + voice_api_key empty → VOICE_ENABLED forced False with warn."""
    import json
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID": "x", "APP_SECRET": "y",
        "voice_enabled": True,
        "voice_engine": "dashscope",
        # no voice_api_key, no DASHSCOPE_API_KEY env
    }))
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    _cfg._init_runtime(config_path=str(cfg_path), data_dir=str(tmp_path))
    assert _cfg.VOICE_ENGINE == "dashscope"
    assert _cfg.VOICE_API_KEY == ""
    assert _cfg.VOICE_ENABLED is False  # forced off due to missing key
