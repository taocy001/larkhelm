"""Tests for ``larkhelm.voice.transcribe`` (M3.2 commit 3).

Coverage map → PRD acceptance criteria:

| AC          | Test                                            |
| ----------- | ----------------------------------------------- |
| AC-02       | test_init_no_eager_load                         |
| AC-04       | test_lock_present                               |
| AC-05       | test_executor_serial                            |
| AC-06 (a/b) | test_load_failed_fallback_returns_error         |
| AC-06 (c)   | test_load_failed_warn_called                    |
| REQ-05      | test_disabled_state_skips_retry                 |
| API         | test_is_ready_when_disabled                     |

CI never sees a real ``faster_whisper`` install — every failure-path
test injects ``sys.modules["faster_whisper"] = None`` (or a custom
``__import__`` hook) so ``import faster_whisper`` raises ImportError
deterministically.
"""
from __future__ import annotations

import atexit
import builtins
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Config init (mirrors tests/test_phase1_modules.py bootstrap) ────────────
_TMP_DIR = tempfile.mkdtemp(prefix="larkhelm_voice_test_")
atexit.register(shutil.rmtree, _TMP_DIR, ignore_errors=True)

_DUMMY_CONFIG = {
    "APP_ID": "test_app",
    "APP_SECRET": "test_secret",
    "default_model": "claude",
    "default_cwd": _TMP_DIR,
}
_cfg_file = Path(_TMP_DIR) / "config.json"
_cfg_file.write_text(json.dumps(_DUMMY_CONFIG))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP_DIR)

# Import AFTER config init so module-level state binds against the test runtime.
import importlib

import larkhelm.voice  # noqa: F401  — exercises AC-02 (no eager faster-whisper import)
# ``larkhelm.voice.__init__`` re-exports the ``transcribe`` *function*, which
# shadows the ``larkhelm.voice.transcribe`` *module* attribute on the package.
# Reach the submodule via ``import_module`` (which goes to ``sys.modules``)
# rather than attribute access.
transcribe_mod = importlib.import_module("larkhelm.voice.transcribe")


def _reset_module_state() -> None:
    """Drop the singleton + DISABLED flag so each test starts fresh.

    Mirrors the risk-mitigation note in design.md §8 — without this
    a prior test's `_LOAD_FAILED=True` would short-circuit subsequent
    tests into the disabled branch.
    """
    transcribe_mod._MODEL = None
    transcribe_mod._LOAD_FAILED = False
    transcribe_mod._LOAD_LOGGED_SUCCESS = False


class _VoiceTestBase(unittest.TestCase):
    """Shared setUp/tearDown that protects ``_cfg.VOICE_ENABLED`` across cases."""

    def setUp(self) -> None:
        self._saved_enabled = _cfg.VOICE_ENABLED
        _cfg.VOICE_ENABLED = True
        _reset_module_state()
        # Ensure the failure-path tests don't inherit a real install.
        sys.modules.pop("faster_whisper", None)

    def tearDown(self) -> None:
        _cfg.VOICE_ENABLED = self._saved_enabled
        _reset_module_state()
        sys.modules.pop("faster_whisper", None)


class TestInitNoEagerLoad(_VoiceTestBase):
    """AC-02 — ``import larkhelm.voice`` must not load faster-whisper or build a model."""

    def test_init_no_eager_load(self) -> None:
        # The package was imported at module top; assert state is virgin.
        self.assertIsNone(transcribe_mod._MODEL)
        self.assertFalse(transcribe_mod._LOAD_FAILED)
        # No faster_whisper module should have been pulled in by ``import larkhelm.voice``.
        self.assertNotIn("faster_whisper", sys.modules)


class TestLoadFailedFallback(_VoiceTestBase):
    """AC-06 (a)(b) — first call after import-failure returns load_failed + flips VOICE_ENABLED."""

    def test_load_failed_fallback_returns_error(self) -> None:
        # ``sys.modules[name] = None`` makes ``import name`` raise ImportError.
        sys.modules["faster_whisper"] = None
        result = transcribe_mod.transcribe("/tmp/never_read.ogg", lang="zh")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "load_failed")
        self.assertEqual(result["text"], "")
        self.assertEqual(result["duration"], 0.0)
        self.assertFalse(_cfg.VOICE_ENABLED)
        self.assertTrue(transcribe_mod._LOAD_FAILED)


class TestLoadFailedWarnCalled(_VoiceTestBase):
    """AC-06 (c) — failure path emits a ``[Voice]``-prefixed warn line."""

    def test_load_failed_warn_called(self) -> None:
        sys.modules["faster_whisper"] = None
        with patch("larkhelm.log.warn") as mock_warn:
            transcribe_mod.transcribe("/tmp/never_read.ogg", lang="zh")
        self.assertGreaterEqual(mock_warn.call_count, 1)
        first_arg = mock_warn.call_args_list[0].args[0]
        self.assertTrue(
            first_arg.startswith("[Voice]"),
            f"warn first arg should start with '[Voice]', got: {first_arg!r}",
        )


class TestDisabledStateSkipsRetry(_VoiceTestBase):
    """REQ-05 — DISABLED is terminal; second call must not re-import faster_whisper."""

    def test_disabled_state_skips_retry(self) -> None:
        real_import = builtins.__import__
        counter = {"count": 0}

        def counting_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "faster_whisper":
                counter["count"] += 1
                raise ImportError("synthetic — voice tests")
            return real_import(name, globals, locals, fromlist, level)

        with patch.object(builtins, "__import__", counting_import):
            r1 = transcribe_mod.transcribe("/tmp/a.ogg", lang="zh")
            r2 = transcribe_mod.transcribe("/tmp/b.ogg", lang="zh")

        # First call surfaced the load failure; second call short-circuited.
        self.assertEqual(r1["error"], "load_failed")
        self.assertEqual(r2["error"], "disabled")
        # The actual contract: ``import faster_whisper`` was attempted exactly once.
        self.assertEqual(
            counter["count"], 1,
            f"expected exactly 1 faster_whisper import attempt, saw {counter['count']}",
        )


class TestStaticSourceGuards(unittest.TestCase):
    """AC-04 / AC-05 — static guards on the source so refactors can't drop the lock or widen the pool."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.source = Path(transcribe_mod.__file__).read_text(encoding="utf-8")

    def test_lock_present(self) -> None:
        self.assertGreaterEqual(
            self.source.count("threading.Lock"), 1,
            "transcribe.py must keep a threading.Lock for the load-model critical section (AC-04)",
        )

    def test_executor_serial(self) -> None:
        self.assertRegex(
            self.source,
            r"ThreadPoolExecutor\([^)]*max_workers\s*=\s*1",
            "transcribe.py must instantiate ThreadPoolExecutor with max_workers=1 (AC-05)",
        )


class TestIsReadyWhenDisabled(_VoiceTestBase):
    """API completeness — ``is_ready`` must reflect ``VOICE_ENABLED`` without triggering a load."""

    def test_is_ready_when_disabled(self) -> None:
        _cfg.VOICE_ENABLED = False
        self.assertFalse(transcribe_mod.is_ready())
        # Sanity: ``is_ready`` does not lazy-load.
        self.assertIsNone(transcribe_mod._MODEL)
        self.assertNotIn("faster_whisper", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
