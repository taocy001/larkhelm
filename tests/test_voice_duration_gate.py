"""AC-01 — P3 REQ-01 voice duration gate.

When the audio duration probe reports a value over
``VOICE_MAX_DURATION_MS``, :func:`transcribe` must short-circuit
without invoking ``_run_inference``.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import sys

import larkhelm.config as _cfg
import larkhelm.voice.transcribe  # noqa: F401 — registers the submodule in sys.modules

# ``larkhelm.voice.__init__`` re-exports the ``transcribe`` *function* and
# shadows the submodule on the package; grab the real module from sys.modules.
_t = sys.modules["larkhelm.voice.transcribe"]


class TestVoiceDurationGate(unittest.TestCase):

    def setUp(self) -> None:
        # The bridge usually populates VOICE_* attributes via _init_runtime.
        # Tests run unconditionally, so we assign defaults if missing then
        # restore on teardown.
        self._touched_attrs: dict[str, object] = {}
        for attr, value in (
            ("VOICE_ENABLED", True),
            ("VOICE_ENGINE", "faster_whisper"),
            ("VOICE_MAX_DURATION_MS", 60_000),
        ):
            self._touched_attrs[attr] = getattr(_cfg, attr, "_unset")
            setattr(_cfg, attr, value)

    def tearDown(self) -> None:
        for attr, prev in self._touched_attrs.items():
            if prev == "_unset":
                try:
                    delattr(_cfg, attr)
                except AttributeError:
                    pass
            else:
                setattr(_cfg, attr, prev)

    def test_duration_over_max_rejects_without_inference(self) -> None:
        with patch.object(_t, "_probe_duration_ms", return_value=180_000):
            with patch.object(_t, "_run_inference") as fake_infer:
                result = _t.transcribe("/tmp/fake_audio.m4a", lang="zh")
        self.assertFalse(result["ok"])
        self.assertEqual(result["error"], "duration_exceeded")
        fake_infer.assert_not_called()

    def test_duration_under_max_calls_inference(self) -> None:
        ok_result = _t.TranscribeResult(
            ok=True, text="hi", duration=2.0, lang="zh", error=None,
        )
        with patch.object(_t, "_probe_duration_ms", return_value=2000), \
             patch.object(_t, "_run_inference", return_value=ok_result) as fake_infer:
            result = _t.transcribe("/tmp/short.m4a", lang="zh")
        self.assertTrue(result["ok"])
        fake_infer.assert_called_once()

    def test_unknown_duration_falls_through_to_inference(self) -> None:
        """If the probe fails (returns None) we fail-open and let inference run."""
        ok_result = _t.TranscribeResult(
            ok=True, text="hi", duration=2.0, lang="zh", error=None,
        )
        with patch.object(_t, "_probe_duration_ms", return_value=None), \
             patch.object(_t, "_run_inference", return_value=ok_result) as fake_infer:
            result = _t.transcribe("/tmp/weird.m4a", lang="zh")
        self.assertTrue(result["ok"])
        fake_infer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
