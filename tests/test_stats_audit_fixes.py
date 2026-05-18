"""Regression tests for the /stats audit fixes (2026-05-19).

Four independent AI auditors (claude/kimi/gemini/deepseek) found the
same six P0 defects in the token-stats pipeline. Each fix in this round
gets its own narrowly-targeted test below so a future refactor that
re-introduces any of them trips loudly. Reports archived at
``.crew_workspace/stats_audit_{claude,kimi,gemini,deepseek}.md``.
"""
from __future__ import annotations

import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import larkhelm.config as _cfg
from larkhelm import token_stats


class TestPersistentReadIncludesRotationBackup(unittest.TestCase):
    """Fix #4: ``get_token_stats_persistent`` must read BOTH ``all.jsonl``
    AND ``all.jsonl.1`` (the single rotation backup ``log.py`` retains)
    or the "累计（全部）" window silently drops every record that was
    in the live file at rotation time."""

    def setUp(self):
        # Redirect LOG_DIR to a tmp dir so writes don't touch the real one.
        # ``_cfg.LOG_DIR`` may not be set yet under test (no ``_init_runtime``
        # call in this thin test module). Use ``setattr`` directly with
        # save/restore so we don't depend on the attribute pre-existing.
        self._tmp = Path(__file__).resolve().parent / "_stats_audit_tmp"
        self._tmp.mkdir(exist_ok=True)
        for f in (self._tmp / "all.jsonl", self._tmp / "all.jsonl.1"):
            if f.exists():
                f.unlink()
        self._orig_log_dir = getattr(_cfg, "LOG_DIR", None)
        _cfg.LOG_DIR = self._tmp

    def tearDown(self):
        if self._orig_log_dir is None:
            try:
                delattr(_cfg, "LOG_DIR")
            except AttributeError:
                pass
        else:
            _cfg.LOG_DIR = self._orig_log_dir
        for f in (self._tmp / "all.jsonl", self._tmp / "all.jsonl.1"):
            if f.exists():
                f.unlink()
        if self._tmp.exists():
            self._tmp.rmdir()

    def _write_token_record(self, path: Path, chat_id: str, model: str,
                            input_tokens: int, ts: str) -> None:
        record = {
            "ts": ts, "role": "token", "chat_id": chat_id, "model": model,
            "input_tokens": input_tokens, "output_tokens": 0,
            "cache_read": 0, "cache_create": 0, "cost_usd": 0.0,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_backup_records_included_in_aggregation(self):
        # Pre-rotation history lives in all.jsonl.1; live file has today's.
        self._write_token_record(
            self._tmp / "all.jsonl.1", "rot_chat", "claude", 100, "2026-04-15T10:00:00",
        )
        self._write_token_record(
            self._tmp / "all.jsonl",   "rot_chat", "claude",  50, "2026-05-19T10:00:00",
        )
        result = token_stats.get_token_stats_persistent("rot_chat")
        self.assertEqual(
            result["claude"]["input_tokens"], 150,
            "expected 100 (backup) + 50 (live) = 150; got %d — backup "
            "file isn't being read, every rotation permanently hides "
            "history from /stats" % result["claude"]["input_tokens"]
        )
        self.assertEqual(result["claude"]["calls"], 2)

    def test_missing_backup_does_not_break(self):
        """No rotation has happened yet → only live file exists."""
        self._write_token_record(
            self._tmp / "all.jsonl", "live_only", "claude", 33, "2026-05-19T10:00:00",
        )
        result = token_stats.get_token_stats_persistent("live_only")
        self.assertEqual(result["claude"]["input_tokens"], 33)

    def test_missing_live_with_backup_only(self):
        """Edge case: tooling could expose a state where the live file
        was unlinked but the backup persists. Don't crash."""
        self._write_token_record(
            self._tmp / "all.jsonl.1", "backup_only", "kimi", 7, "2026-04-01T10:00:00",
        )
        result = token_stats.get_token_stats_persistent("backup_only")
        self.assertEqual(result["kimi"]["input_tokens"], 7)

    def test_strict_chat_id_match_no_prefix_leak(self):
        """Cross-chat leak guard (audit P1): a record for ``chat__suffix``
        must NOT roll into chat ``chat``."""
        self._write_token_record(
            self._tmp / "all.jsonl", "chat__leak", "claude", 999, "2026-05-19T10:00:00",
        )
        result = token_stats.get_token_stats_persistent("chat")
        self.assertNotIn("claude", result)


class TestFmtTokenBlockCacheArithmetic(unittest.TestCase):
    """Fix #3: ``_fmt_token_block`` must include cache_read + cache_create
    in the displayed total, and the hit-pct denominator must be
    ``cache_read + input_tokens`` (the two are disjoint on Claude)."""

    def _format(self, **fields):
        from larkhelm.commands import _fmt_token_block
        data = {"claude": {
            "input_tokens":  fields.get("inp", 0),
            "output_tokens": fields.get("out", 0),
            "cache_read":    fields.get("cr", 0),
            "cache_create":  fields.get("cc", 0),
            "cost_usd":      fields.get("cost", 0.0),
            "calls":         fields.get("calls", 1),
        }}
        return _fmt_token_block("test", data)

    def test_total_includes_all_four_buckets(self):
        """The "合计 N tokens" line must add input + output + cache_read
        + cache_create. Previously it summed only input + output, which
        for a 50k-cache-hit query under-counted by ~34×.
        """
        out = self._format(inp=1000, out=500, cr=50000, cc=200)
        self.assertIn(f"{1000 + 500 + 50000 + 200:,}", out)

    def test_hit_pct_denominator_is_cr_plus_inp(self):
        """Pre-fix the denominator was ``inp`` alone, so cr=9000 / inp=1000
        rendered ``900%``. Correct denominator yields 9000/10000 = 90%."""
        out = self._format(inp=1000, cr=9000)
        # Must contain "（90%）" — not 900% or higher.
        self.assertIn("（90%）", out)
        self.assertNotIn("（900%）", out)

    def test_zero_cost_shown_as_dollars_not_dash(self):
        """``cost == 0.0`` is a real value (full-cache-hit / free tier),
        distinct from "we don't know". Previously rendered ``"—"``,
        making it look like data was missing.
        """
        out = self._format(inp=1, cost=0.0)
        self.assertIn("$0.0000", out)
        # And the dash must NOT appear in the cost column.
        # (Dashes can legitimately appear elsewhere, so we check via the
        # row prefix ``费用 **...``.)
        self.assertIn("费用 **$0.0000**", out)


class TestKimiCleanupExtraEstimate(unittest.TestCase):
    """Fix #6: Kimi CLI never emits a usage envelope on stdout, so the
    ``_record_tokens`` call inside ``parse_stdout_event`` was dead code.
    ``cleanup_extra`` now emits an estimated record from prompt /
    response char counts, marked ``estimated=True`` for downstream
    distinguishability."""

    def test_cleanup_extra_records_estimated_tokens(self):
        from larkhelm.runner_kimi import KimiRunner

        captured: list = []
        with patch.object(KimiRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(
                              (model, dict(usage), cost))):
            r = KimiRunner.__new__(KimiRunner)
            r.message = "Hello kimi" * 10   # 100 chars → ~25 input tokens
            r._result_text = "Reply text!" * 20   # 220 chars → ~55 output
            r.cleanup_extra()

        self.assertEqual(len(captured), 1, "cleanup_extra should record once")
        model, usage, cost = captured[0]
        self.assertEqual(model, "kimi")
        self.assertGreater(usage["input_tokens"], 0)
        self.assertGreater(usage["output_tokens"], 0)
        self.assertTrue(usage.get("estimated"),
            "kimi cleanup record must be flagged estimated=True so "
            "downstream tooling can tell it apart from precise SDK usage"
        )
        self.assertEqual(cost, 0.0)

    def test_cleanup_extra_noops_when_no_input_or_output(self):
        from larkhelm.runner_kimi import KimiRunner

        captured: list = []
        with patch.object(KimiRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(usage)):
            r = KimiRunner.__new__(KimiRunner)
            r.message = ""
            r._result_text = ""
            r.cleanup_extra()
        self.assertEqual(captured, [],
            "empty-prompt / empty-output query shouldn't manufacture a "
            "phantom record"
        )


class TestDeepSeekStreamOptions(unittest.TestCase):
    """Fix #2: DeepSeek's streaming body MUST set
    ``stream_options.include_usage=true``, otherwise the terminal SSE
    block omits ``usage`` and every query records zero tokens."""

    def test_streaming_body_requests_usage_chunk(self):
        # Read the source to confirm the body literal contains the
        # required key. (Functional integration is covered by the live
        # _record_tokens path elsewhere; here we're pinning the body
        # constructor against re-introducing the bug.)
        source = (Path(__file__).resolve().parents[1] /
                  "larkhelm" / "runner_deepseek.py").read_text(encoding="utf-8")
        self.assertIn(
            '"stream_options": {"include_usage": True}', source,
            "DeepSeek streaming body lost stream_options.include_usage — "
            "production calls will silently stop reporting tokens"
        )


class TestGeminiStatsSchema(unittest.TestCase):
    """Fix #1: Gemini CLI 0.41.x emits ``stats``, not ``usage``, on the
    terminal envelope. Pin both the schema reader and the cached/
    non-cached split."""

    def test_gemini_parse_stdout_event_reads_stats_field(self):
        from larkhelm.runner_gemini import GeminiRunner

        captured: list = []
        with patch.object(GeminiRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(
                              (model, dict(usage), cost))):
            r = GeminiRunner.__new__(GeminiRunner)
            # Stub minimal state _record_tokens path needs.
            r.use_session = False
            # ``_no_checkpointing`` is a @property reading ``_extra_args``;
            # set the backing field instead. With use_session=False the
            # _save_sid branch is short-circuited anyway, so the property
            # value doesn't matter — we just need _extra_args to exist.
            r._extra_args = []
            r._new_sid = None
            # The terminal envelope as observed in audit.
            ev = {
                "type": "result",
                "session_id": "gsid_test",
                "stats": {
                    "input_tokens":  12184,
                    "output_tokens": 19,
                    "cached":        11555,
                    "input":         629,    # 12184 - 11555
                    "total_tokens":  12203,
                },
            }
            r.parse_stdout_event(ev)

        self.assertEqual(len(captured), 1, "stats envelope should record once")
        model, usage, cost = captured[0]
        self.assertEqual(model, "gemini")
        self.assertEqual(usage["input_tokens"], 629,
            "non-cached input (stats.input) must populate input_tokens"
        )
        self.assertEqual(usage["cache_read"], 11555,
            "stats.cached must populate cache_read"
        )
        self.assertEqual(usage["output_tokens"], 19)
        self.assertEqual(usage["cache_create"], 0,
            "gemini doesn't distinguish cache creation; this stays 0"
        )

    def test_gemini_ignores_legacy_usage_field(self):
        """If a future tool / mock emits the OLD ``usage`` field, the
        new code must NOT pick it up (avoids double-counting if both
        somehow appear)."""
        from larkhelm.runner_gemini import GeminiRunner

        captured: list = []
        with patch.object(GeminiRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(usage)):
            r = GeminiRunner.__new__(GeminiRunner)
            r.use_session = False
            # ``_no_checkpointing`` is a @property reading ``_extra_args``;
            # set the backing field instead. With use_session=False the
            # _save_sid branch is short-circuited anyway, so the property
            # value doesn't matter — we just need _extra_args to exist.
            r._extra_args = []
            r._new_sid = None
            ev = {"type": "result", "session_id": "x",
                  "usage": {"input_tokens": 999, "output_tokens": 999}}
            r.parse_stdout_event(ev)

        self.assertEqual(captured, [],
            "legacy 'usage' envelope must not be re-parsed (would re-"
            "introduce the wrong-schema bug if test fixtures still use it)"
        )


class TestIntentCostLineSuppressedWhenZero(unittest.TestCase):
    """Fix #5: ``_cmd_stats_intent`` was always rendering "成本：$0.0000"
    because none of the 5 builtin agents populates ``cost_usd``. The
    fake number was indistinguishable from a real "free tier" total —
    suppress the line until at least one agent emits a non-zero value."""

    def test_zero_cost_suppresses_line(self):
        from larkhelm.commands import _cmd_stats_intent

        sent: list = []
        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            sent.append({"title": title, "body": body, "color": color})

        fake_agg = {
            "total": 5, "success_rate": 1.0, "avg_duration": 0.5,
            "total_cost": 0.0, "per_agent": {}, "date": "2026-05-19",
        }
        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.agent_hub.agent_audit.aggregate_daily",
                       return_value=fake_agg):
                _cmd_stats_intent("oc_test", "msg_1")

        self.assertEqual(len(sent), 1)
        self.assertNotIn("成本", sent[0]["body"],
            "成本 line must be omitted when total_cost is 0 — the "
            "previous '成本：$0.0000' was a UI lie (no agent fills "
            "cost_usd; the field is hardcoded to 0.0)"
        )

    def test_nonzero_cost_shown(self):
        from larkhelm.commands import _cmd_stats_intent

        sent: list = []
        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            sent.append({"body": body})

        fake_agg = {
            "total": 5, "success_rate": 0.9, "avg_duration": 1.2,
            "total_cost": 0.1234, "per_agent": {}, "date": "2026-05-19",
        }
        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.agent_hub.agent_audit.aggregate_daily",
                       return_value=fake_agg):
                _cmd_stats_intent("oc_test", "msg_1")

        self.assertEqual(len(sent), 1)
        self.assertIn("成本：$0.1234", sent[0]["body"])


if __name__ == "__main__":
    unittest.main()
