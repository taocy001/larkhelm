"""Pin kimi-cli 1.43 stream-json schema (probe + runner).

Real-world breakdown
--------------------
Pre-fix audit on 2026-05-19 (host kimi-cli 1.43.x) showed:

* ``_probe_kimi`` sent ``{"type":"user","message":"."}`` as stdin and
  scanned stdout for ``ev.get("type") in ("system","assistant","result")``.
  kimi 1.43 expects ``{"role":"user","content":"."}`` (kosong Message
  format) and emits ``role`` at the TOP LEVEL of each event with no
  ``type`` field and no terminal ``result`` envelope. Net effect: every
  probe landed in Step-3 ``rc=0 but no stream-json event observed`` →
  INDETERMINATE forever.

* ``runner_kimi.parse_stdout_event`` was already mostly compatible
  with the 1.43 shape (``role == "assistant"`` + ``content`` list), but
  (a) ``role == "tool"`` did ``str(content)`` on a list and produced an
  ugly Python repr blob, and (b) session continuity broke because no
  ``session_id`` field appears anywhere in stdout — kimi 1.43 emits the
  sid **only** on stderr as ``To resume this session: kimi -r <uuid>``.

Each test below pins one specific real-1.43 envelope or stderr fragment.
The fixtures are verbatim captures from a live ``kimi --print
--output-format stream-json --input-format stream-json`` invocation.
"""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch, MagicMock

# Use the project conftest.py bootstrap (LARKHELM_TEST_MODE=1 already set).
import larkhelm.config as _cfg  # noqa: F401 — keep import for side-effect order
from larkhelm import model_probe
from larkhelm.backend_registry import BackendSpec


# ── Real kimi 1.43 stream-json captures ──────────────────────────────────────

# Plain-text reply (no tools): a "think" prefix then a final "text" answer.
ASSISTANT_REPLY = {
    "role": "assistant",
    "content": [
        {"type": "think", "think": "The user wants me to reply OK.", "encrypted": None},
        {"type": "text", "text": "OK"},
    ],
}

# Tool-call envelope: the model decides to invoke Shell. Note ``tool_calls``
# lives at the TOP LEVEL alongside ``content`` — kimi keeps the OpenAI-style
# function-calling shape even though everything else is kosong-Message.
ASSISTANT_TOOL_CALL = {
    "role": "assistant",
    "content": [
        {"type": "think", "think": "Need to run `ls /`.", "encrypted": None},
    ],
    "tool_calls": [
        {
            "type": "function",
            "id": "tool_fDfRhO8CUybWOQhepYPLDo6J",
            "function": {
                "name": "Shell",
                "arguments": '{"command": "ls /"}',
            },
        },
    ],
}

# Tool result: role=="tool", tool_call_id at top level, content is an array.
TOOL_RESULT = {
    "role": "tool",
    "content": [
        {"type": "text", "text": "<system>Command executed successfully.</system>"},
        {"type": "text", "text": "bin\nboot\netc\nhome\n"},
    ],
    "tool_call_id": "tool_fDfRhO8CUybWOQhepYPLDo6J",
}

# Stderr suffix kimi prints after every successful turn.
STDERR_RESUME_HINT = "\nTo resume this session: kimi -r f464a922-8680-4c23-a873-b8b61e0090e4\n"


# ─────────────────────────────────────────────────────────────────────────────
# Probe schema
# ─────────────────────────────────────────────────────────────────────────────


class ProbeStdinFormatTests(unittest.TestCase):
    """``_probe_kimi`` must hand kimi-cli the kosong-Message stdin shape."""

    def test_probe_stdin_is_role_content_format(self):
        captured: list[bytes | str] = []

        class _FakeResult:
            returncode = 0
            stdout = json.dumps(ASSISTANT_REPLY) + "\n"
            stderr = STDERR_RESUME_HINT

        def _fake_run(cmd, input=None, **kw):
            captured.append(input)
            return _FakeResult()

        spec = BackendSpec(
            id="kimi", provider="kimi_cli", display_name="Kimi",
            role="worker", tags=["tools"], command="kimi",
        )
        with patch.object(model_probe.subprocess, "run", side_effect=_fake_run):
            ok, err = model_probe._probe_kimi(spec)

        self.assertTrue(ok, f"valid 1.43 envelope must mark healthy. got {(ok, err)!r}")
        self.assertEqual(len(captured), 1)
        payload = json.loads(captured[0].splitlines()[0])
        # Pre-fix sent {"type":"user","message":"."} which kimi 1.43 silently
        # ignored. The kosong-Message shape is the contract; pin it here.
        self.assertEqual(payload.get("role"), "user",
                         f"probe stdin must be kosong Message; got {payload!r}")
        self.assertEqual(payload.get("content"), ".",
                         f"probe stdin must use 'content', not 'message'; got {payload!r}")
        self.assertNotIn("type", payload,
                         "probe stdin must NOT carry a top-level 'type' field for kimi 1.43")
        self.assertNotIn("message", payload,
                         "probe stdin must NOT carry the legacy 'message' field")


class ProbeStdoutDetectionTests(unittest.TestCase):
    """``_probe_kimi`` must detect ``role`` at the top level (1.43 schema)."""

    def _run_with_stdout(self, stdout: str) -> tuple[bool | None, str]:
        class _FakeResult:
            returncode = 0

        _FakeResult.stdout = stdout
        _FakeResult.stderr = ""
        spec = BackendSpec(
            id="kimi", provider="kimi_cli", display_name="Kimi",
            role="worker", tags=["tools"], command="kimi",
        )
        with patch.object(model_probe.subprocess, "run", return_value=_FakeResult()):
            return model_probe._probe_kimi(spec)

    def test_role_assistant_marks_healthy(self):
        """The real kimi 1.43 'assistant' envelope must flip healthy=True."""
        ok, err = self._run_with_stdout(json.dumps(ASSISTANT_REPLY) + "\n")
        self.assertTrue(ok, f"role=='assistant' must mark healthy. got {(ok, err)!r}")

    def test_role_tool_marks_healthy(self):
        """Tool-result envelopes are also a positive signal (the call
        reached the model AND a tool round-trip completed)."""
        ok, err = self._run_with_stdout(json.dumps(TOOL_RESULT) + "\n")
        self.assertTrue(ok, f"role=='tool' must mark healthy. got {(ok, err)!r}")

    def test_legacy_type_system_still_marks_healthy(self):
        """Backwards-compat: if a future kimi version reverts to
        Anthropic-style ``{type:"system",...}``, probe should still pass."""
        ok, err = self._run_with_stdout('{"type": "system", "msg": "hello"}\n')
        self.assertTrue(ok, f"legacy type=='system' must stay healthy. got {(ok, err)!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Runner parse_stdout_event
# ─────────────────────────────────────────────────────────────────────────────


def _make_runner():
    """Construct a KimiRunner without spawning a subprocess.

    All side-effecting helpers (_save_sid, _record_tokens, on_text, on_tool,
    on_tool_result) are replaced with MagicMocks so each test can assert
    against the actual call shape.
    """
    from larkhelm.runner_kimi import KimiRunner

    r = KimiRunner(
        chat_id="chat_test",
        message="hi",
        sid=None,
        cwd="/tmp",
        on_text=MagicMock(),
        on_tool=MagicMock(),
        on_tool_result=MagicMock(),
    )
    # Don't actually persist sids to disk during tests.
    return r


class ParseAssistantTextTests(unittest.TestCase):
    """``role == "assistant"`` with a kimi 1.43 content array."""

    def test_text_parts_streamed_to_on_text(self):
        r = _make_runner()
        ret = r.parse_stdout_event(ASSISTANT_REPLY)
        # No terminal result envelope ever in 1.43, so parse must NOT
        # signal done — let the outer loop drain stdout to EOF.
        self.assertFalse(ret, "kimi 1.43 'assistant' is non-terminal")
        # 'think' parts must be dropped — only 'text' is user-visible.
        self.assertEqual(r._result_text, "OK")
        r.on_text.assert_called_once_with("OK", status="typing")

    def test_think_only_envelope_emits_nothing(self):
        """An assistant envelope that contains *only* 'think' (e.g. mid-
        reasoning before the model emits its first text part) must not
        trigger on_text — would otherwise render empty cards mid-stream."""
        r = _make_runner()
        ret = r.parse_stdout_event({
            "role": "assistant",
            "content": [{"type": "think", "think": "...", "encrypted": None}],
        })
        self.assertFalse(ret)
        self.assertEqual(r._result_text, "")
        r.on_text.assert_not_called()


class ParseAssistantToolCallTests(unittest.TestCase):
    """``tool_calls`` at the top level: OpenAI-style function calling."""

    def test_tool_call_forwarded_to_on_tool(self):
        r = _make_runner()
        r.parse_stdout_event(ASSISTANT_TOOL_CALL)
        # 'Shell' maps to 'Bash' via _KIMI_TOOL_MAP — the user-facing card
        # uses the canonical claude/anthropic tool name.
        r.on_tool.assert_called_once()
        args, kwargs = r.on_tool.call_args
        self.assertEqual(args[0], "Bash",
                         "Shell must be remapped to Bash for card display")
        # Argument summary is the command preview, not the JSON repr.
        self.assertIn("ls /", args[1])
        self.assertEqual(kwargs.get("tool_id"), "tool_fDfRhO8CUybWOQhepYPLDo6J")


class ParseToolResultTests(unittest.TestCase):
    """``role == "tool"`` with content array of typed parts.

    Pre-fix this path did ``str(content)`` on the list which produced
    ``"[{'type': 'text', 'text': '...'}, ...]"`` — an ugly Python repr
    blob that leaked into the user-facing tool-result card.
    """

    def test_text_extracted_from_content_array(self):
        r = _make_runner()
        r.parse_stdout_event(TOOL_RESULT)
        r.on_tool_result.assert_called_once()
        args, _kwargs = r.on_tool_result.call_args
        # tc_id, pretty_text, is_error, elapsed
        tc_id, pretty, is_error, elapsed = args
        self.assertEqual(tc_id, "tool_fDfRhO8CUybWOQhepYPLDo6J")
        self.assertNotIn("[{",
                         f"pretty output must NOT be a list repr. got {pretty!r}")
        self.assertIn("bin", pretty)
        self.assertIn("boot", pretty)
        self.assertFalse(is_error)
        self.assertGreaterEqual(elapsed, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Session ID extraction (stderr)
# ─────────────────────────────────────────────────────────────────────────────


class StderrSessionExtractionTests(unittest.TestCase):
    """kimi 1.43 emits the session ID **only** on stderr."""

    def test_resume_hint_captures_sid(self):
        r = _make_runner()
        with patch("larkhelm.runner_kimi._save_sid") as save:
            # _drain_stderr in the base runner forwards each line through
            # _on_stderr_line (after rstrip). Simulate that contract.
            r._on_stderr_line(
                "To resume this session: kimi -r f464a922-8680-4c23-a873-b8b61e0090e4"
            )
        self.assertEqual(r._new_sid, "f464a922-8680-4c23-a873-b8b61e0090e4")
        save.assert_called_once()
        ns, sid, key = save.call_args.args[:3]
        self.assertEqual(sid, "f464a922-8680-4c23-a873-b8b61e0090e4")
        self.assertEqual(key, "kimi")

    def test_unrelated_stderr_does_not_set_sid(self):
        r = _make_runner()
        with patch("larkhelm.runner_kimi._save_sid") as save:
            r._on_stderr_line("npm warn: deprecated something")
        self.assertIsNone(r._new_sid)
        save.assert_not_called()

    def test_first_match_wins(self):
        """If the stderr stream happens to emit two resume hints (re-run
        after self-retry), only the first should be persisted — second
        would re-write _save_sid for a session we never actually used."""
        r = _make_runner()
        with patch("larkhelm.runner_kimi._save_sid") as save:
            r._on_stderr_line("To resume this session: kimi -r aaaaaaaa")
            r._on_stderr_line("To resume this session: kimi -r bbbbbbbb")
        self.assertEqual(r._new_sid, "aaaaaaaa")
        self.assertEqual(save.call_count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end: cleanup_extra still writes char-count tokens after parsing
# ─────────────────────────────────────────────────────────────────────────────


class CleanupExtraTokenFallbackTests(unittest.TestCase):
    """After parsing a real-1.43 turn (no usage envelope), cleanup_extra
    must still produce a non-zero estimate so /stats doesn't read 0."""

    def test_estimate_after_real_assistant_parse(self):
        r = _make_runner()
        # 16 ASCII chars + 4 CJK chars exercises both estimator branches.
        # The CJK ideographs count 1 token each; the ASCII falls back to
        # len // 4. "OK" alone (2 chars) would have rounded to 0 — too
        # short to exercise the non-zero path; pad with realistic text.
        longer = {
            "role": "assistant",
            "content": [
                {"type": "think", "think": "...", "encrypted": None},
                {"type": "text",
                 "text": "OK, here is the result: 你好世界世"},
            ],
        }
        r.parse_stdout_event(longer)
        self.assertIn("你好世界世", r._result_text)
        self.assertFalse(r._tokens_recorded)
        with patch.object(r, "_record_tokens") as rec:
            r.cleanup_extra()
        self.assertEqual(rec.call_count, 1)
        model_label, payload, cost = rec.call_args.args
        self.assertEqual(model_label, "kimi")
        self.assertTrue(payload.get("estimated"),
                        "estimated=True flag must propagate to stats record")
        # 4 CJK chars must contribute at least 4 tokens via the
        # CJK-aware path; pre-fix would have given (24 // 4) = 6 for the
        # whole string and under-counted CJK by ~4×.
        self.assertGreaterEqual(payload.get("output_tokens", 0), 4,
                                f"CJK-aware estimate too low: {payload!r}")
        self.assertGreaterEqual(payload.get("input_tokens", 0), 0)
        self.assertEqual(cost, 0.0)

    def test_estimate_zero_for_empty_reply(self):
        """A turn that emitted only 'think' parts (no user-visible text)
        and an empty prompt must NOT fabricate token counts — the
        ``not text and not prompt`` short-circuit guards this."""
        from larkhelm.runner_kimi import KimiRunner
        r = KimiRunner(
            chat_id="chat_test", message="", sid=None, cwd="/tmp",
            on_text=MagicMock(), on_tool=MagicMock(), on_tool_result=MagicMock(),
        )
        # 'think' is dropped, so _result_text stays "".
        r.parse_stdout_event({
            "role": "assistant",
            "content": [{"type": "think", "think": "silent reasoning"}],
        })
        with patch.object(r, "_record_tokens") as rec:
            r.cleanup_extra()
        rec.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
