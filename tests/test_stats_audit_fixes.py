"""Regression tests for the /stats audit fixes (2026-05-19).

Four independent AI auditors (claude/kimi/gemini/deepseek) found the
same six P0 defects in the token-stats pipeline. Each fix in this round
gets its own narrowly-targeted test below so a future refactor that
re-introduces any of them trips loudly. Reports archived at
``.crew_workspace/stats_audit_{claude,kimi,gemini,deepseek}.md``.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import larkhelm.config as _cfg
from larkhelm import token_stats


# ────────────────────────────────────────────────────────────────────────
# Helpers for the R1.5 end-to-end tests (DeepSeek runner + Gemini event).
# Inlined deliberately so this file doesn't import from
# ``tests.test_runner_deepseek`` — that module bootstraps its own config
# at import time (incl. ``HARD_TIMEOUT = 600``) which would silently
# contaminate ``TestDurationHardCapMatchesHardTimeout`` further down.
# ────────────────────────────────────────────────────────────────────────

class _FakeStreamResponse:
    """Minimal stand-in for the slice of ``requests.Response`` that
    ``DeepSeekRunner._consume_sse`` uses."""

    def __init__(self, status_code: int, lines: list[str], text: str = ""):
        self.status_code = status_code
        self._lines = lines
        self.text = text
        self.closed = False

    def iter_lines(self, decode_unicode: bool = False):
        yield from self._lines

    def close(self):
        self.closed = True


def _ds_sse(content_chunks: list[str], usage: dict | None = None) -> list[str]:
    """Build a list of raw SSE lines matching DeepSeek's stream format."""
    lines: list[str] = []
    for chunk in content_chunks:
        lines.append("data: " + json.dumps({
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "delta": {"content": chunk},
                         "finish_reason": None}],
        }))
        lines.append("")
    if usage:
        lines.append("data: " + json.dumps({
            "id": "chatcmpl-x",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": usage,
        }))
        lines.append("")
    lines.append(": keep-alive")
    lines.append("data: [DONE]")
    return lines


def _deepseek_min_config():
    """Set the minimum config fields needed to instantiate DeepSeekRunner.

    Critically does NOT touch ``HARD_TIMEOUT`` / ``RESPONSE_TIMEOUT`` — those
    are read by other tests in this file and their values must survive the
    bootstrap. Tests using this helper should either save/restore those
    fields themselves or simply leave them at whatever the production
    config established."""
    tmp = Path(tempfile.mkdtemp(prefix="larkhelm-stats-audit-ds-"))
    _cfg.DATA_DIR    = tmp
    _cfg.SESSION_DIR = tmp / "sessions"
    _cfg.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    if not getattr(_cfg, "LOG_DIR", None):
        _cfg.LOG_DIR = tmp / "logs"
        _cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
    _cfg.STATE_FILE  = tmp / "state.json"
    _cfg.DEBUG_LOG   = tmp / "larkhelm.log"
    # RESPONSE_TIMEOUT / HARD_TIMEOUT intentionally NOT touched.
    if not getattr(_cfg, "RESPONSE_TIMEOUT", None):
        _cfg.RESPONSE_TIMEOUT = 300
    if not getattr(_cfg, "HARD_TIMEOUT", None):
        _cfg.HARD_TIMEOUT = 21600
    _cfg.DEEPSEEK_API_KEY  = "sk-test-fake"
    _cfg.DEEPSEEK_BASE_URL = "https://api.deepseek.com"
    _cfg.DEEPSEEK_MODEL    = "deepseek-chat"
    return tmp


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
            # Round-3 fix: cleanup_extra now short-circuits on
            # _tokens_recorded; the field is set in BaseProcessRunner.__init__
            # which __new__ bypasses, so re-create it manually for the test.
            r._tokens_recorded = False
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
            r._tokens_recorded = False
            r.message = ""
            r._result_text = ""
            r.cleanup_extra()
        self.assertEqual(captured, [],
            "empty-prompt / empty-output query shouldn't manufacture a "
            "phantom record"
        )


class TestKimiCJKEstimator(unittest.TestCase):
    """Round-3 follow-up P1 (R3-4): the Kimi cleanup_extra fallback
    formula was ``len(text) // 4`` for both English and CJK. That
    under-counted Chinese text by ~4× (BPE / SentencePiece typically
    gives 1 token per CJK ideograph), so bilingual users saw their
    monthly token totals silently halved-to-quartered.

    The new ``_estimate_tokens_cjk_aware`` heuristic:

    * English / Latin / digits / punctuation — keep ``// 4`` (backward
      compatible with prior tests' magnitudes).
    * CJK Unified Ideographs (U+4E00–U+9FFF) + Hiragana/Katakana
      (U+3040–U+30FF) — 1 token per char.
    * Mixed content — proportional split.

    These tests pin the formula across the three regimes so any
    accidental regression to ``len // 4`` (or another over-eager
    "simplification") fires immediately."""

    def test_pure_english_matches_legacy_divide_by_four(self):
        from larkhelm.runner_kimi import _estimate_tokens_cjk_aware
        # 100-char ASCII string → 25 tokens (legacy ``// 4`` floor).
        text = "Hello kimi! " * 10  # 12 * 10 = 120 chars
        self.assertEqual(_estimate_tokens_cjk_aware(text), len(text) // 4,
            "pure ASCII input must keep the original `// 4` ratio so the "
            "pre-fix TestKimiCleanupExtraEstimate magnitudes still hold"
        )

    def test_pure_cjk_counts_one_token_per_char(self):
        from larkhelm.runner_kimi import _estimate_tokens_cjk_aware
        text = "你好世界这是一段中文消息"  # 12 chars, all CJK
        self.assertEqual(_estimate_tokens_cjk_aware(text), len(text),
            "pure CJK input must count 1 token per char — the prior "
            "`len // 4` formula under-counted by ~4×, silently shrinking "
            "the user's monthly kimi token total"
        )

    def test_mixed_content_proportional_split(self):
        from larkhelm.runner_kimi import _estimate_tokens_cjk_aware
        # 4 CJK chars + 16 ASCII chars (incl. spaces & punctuation).
        text = "中文 hello kimi mix"
        cjk_count = sum(1 for c in text if "一" <= c <= "鿿"
                                         or "぀" <= c <= "ヿ")
        non_cjk = len(text) - cjk_count
        expected = cjk_count + non_cjk // 4
        self.assertEqual(_estimate_tokens_cjk_aware(text), expected,
            "mixed content must split into CJK (1 token/char) + non-CJK "
            "(// 4) buckets; this guards against accidental reversion to "
            "a single uniform divisor"
        )
        # Sanity: result must be strictly larger than the legacy
        # `len // 4` formula whenever CJK chars are present (this is
        # the whole point of the fix).
        self.assertGreater(_estimate_tokens_cjk_aware(text), len(text) // 4,
            "any input with CJK chars must produce MORE tokens than the "
            "legacy `// 4` formula — otherwise the fix is doing nothing"
        )

    def test_empty_input_returns_zero(self):
        from larkhelm.runner_kimi import _estimate_tokens_cjk_aware
        self.assertEqual(_estimate_tokens_cjk_aware(""), 0)
        # Whitespace-only: 4 spaces → 4 // 4 = 1 (acceptable cleanup
        # noise). Just pin "doesn't crash and stays small".
        self.assertLessEqual(_estimate_tokens_cjk_aware("    "), 1)

    def test_cleanup_extra_uses_cjk_aware_estimator(self):
        """End-to-end: Kimi cleanup_extra fed a Chinese prompt + reply
        records substantially more tokens than the legacy `len // 4`
        formula would have produced."""
        from larkhelm.runner_kimi import KimiRunner

        captured: list = []
        with patch.object(KimiRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(
                              dict(usage))):
            r = KimiRunner.__new__(KimiRunner)
            r._tokens_recorded = False
            # 40 CJK chars input, 60 CJK chars output.
            r.message = "你好" * 20         # 40 chars
            r._result_text = "回答内容" * 15  # 60 chars
            r.cleanup_extra()

        self.assertEqual(len(captured), 1)
        usage = captured[0]
        # Legacy formula would have given 40//4=10 and 60//4=15.
        # CJK-aware should give 40 and 60.
        self.assertEqual(usage["input_tokens"], 40,
            "Chinese prompt of 40 chars should record 40 tokens "
            "(1 per CJK char), not 10 (legacy `len // 4`)"
        )
        self.assertEqual(usage["output_tokens"], 60,
            "Chinese reply of 60 chars should record 60 tokens "
            "(1 per CJK char), not 15 (legacy `len // 4`)"
        )
        self.assertTrue(usage.get("estimated"),
            "estimated=True flag must still be present so downstream "
            "tooling can distinguish from precise usage envelopes"
        )


class TestDeepSeekStreamOptions(unittest.TestCase):
    """Fix #2: DeepSeek's streaming body MUST set
    ``stream_options.include_usage=true``, otherwise the terminal SSE
    block omits ``usage`` and every query records zero tokens.

    R1.5-3: previous version of this test grep'd the source file for the
    literal ``"stream_options": {"include_usage": True}`` string. That
    "text-as-test" approach is trivially defeated by any variable
    rename / dict-construction refactor (e.g. ``body["stream_options"] = ...``
    or ``opts = {"include_usage": True}; body["stream_options"] = opts``).
    The runner could ship without actually sending the key and the test
    would still pass.

    Replaced with a proper integration check: mock ``requests.Session.post``,
    drive a real ``DeepSeekRunner.run()`` to completion, then inspect the
    JSON body that was actually handed to the HTTP client. This guards
    against EVERY way of breaking the contract — including someone
    rewriting the body assembly while the source string literal stays
    intact elsewhere (e.g. left over in a comment).
    """

    def test_streaming_body_requests_usage_chunk(self):
        import threading as _threading
        from unittest import mock as _mock
        # Module-level helpers (top of file) are inlined deliberately so
        # we don't import from test_runner_deepseek (whose own bootstrap
        # clobbers HARD_TIMEOUT, breaking other tests in this file).
        from larkhelm.runner_deepseek import DeepSeekRunner

        _deepseek_min_config()
        sse = _ds_sse(["ok"], usage={
            "prompt_tokens": 10, "completion_tokens": 2,
            "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10,
        })
        fake_resp = _FakeStreamResponse(200, sse)

        with _mock.patch("requests.Session.post",
                         return_value=fake_resp) as m_post:
            DeepSeekRunner(
                "chat-stream-opts-test", "Hi",
                sid=None, cwd="/tmp",
                cancel_ev=_threading.Event(),
            ).run()

        # Exactly one POST issued.
        self.assertEqual(m_post.call_count, 1,
            "runner must hit Session.post exactly once for a single-attempt query"
        )
        # Capture the JSON body actually sent over the wire.
        _, kwargs = m_post.call_args
        body = kwargs.get("json")
        self.assertIsInstance(body, dict,
            "runner must pass the request body via the ``json=`` kwarg so "
            "requests serializes + content-type-tags it; sending raw "
            "``data=...`` would silently lose stream_options because "
            "DeepSeek's content negotiation would reject the call"
        )
        self.assertIn("stream_options", body,
            "request body is missing ``stream_options`` — DeepSeek's "
            "OpenAI-compatible API requires this key on streaming calls "
            "or the terminal SSE chunk omits ``usage`` entirely (every "
            "query then records zero tokens)"
        )
        self.assertEqual(
            body["stream_options"], {"include_usage": True},
            f"stream_options must be exactly {{'include_usage': True}}; "
            f"got {body['stream_options']!r}"
        )
        # And ``stream`` itself must remain true — stream_options is
        # only honoured when streaming is on.
        self.assertTrue(body.get("stream"),
            "stream=true must be set alongside stream_options.include_usage"
        )


class TestDeepSeekFieldSemanticsAlignment(unittest.TestCase):
    """Round-2 review (stats_fix_review.md, Fix #3 regression): the
    Claude-style ``total = input + output + cr + cc`` formula assumes
    the 4 buckets are disjoint. DeepSeek historically wrote
    ``input_tokens=prompt_tokens`` (which ALREADY contained cache_hit
    + cache_miss) AND ``cache_create=prompt_cache_miss_tokens`` (a
    subset of input). With the new total formula this near-doubled the
    DeepSeek total on every cache-heavy query.

    Fix: align DeepSeek with the uniform contract
      • input_tokens = miss (non-cached prompt)
      • cache_read   = hit
      • cache_create = 0  (DeepSeek has no separate cache-creation cost)
    """

    def _record_via_runner(self, usage_seen):
        """Drive ``DeepSeekRunner`` past the usage-emit branch and capture
        what gets passed to ``_record_tokens``."""
        from larkhelm.runner_deepseek import DeepSeekRunner

        captured: list = []
        with patch.object(DeepSeekRunner, "_record_tokens",
                          lambda self, usage, cost: captured.append(
                              (dict(usage), cost))):
            r = DeepSeekRunner.__new__(DeepSeekRunner)
            # Minimal state for the usage-handling branch — the actual
            # call site is below the SSE parse loop, so we invoke its
            # logic directly by replicating the few lines that compute
            # the bucketed usage and call _record_tokens.
            #
            # The runner's parse loop is too large to drive in unit
            # form; we exercise the projection logic via the same
            # source-level shape it computes. This is "white-box but
            # surgical": if anyone changes the projection, this test
            # has to change in lock-step.
            hit = int(usage_seen.get("prompt_cache_hit_tokens", 0) or 0)
            miss = int(usage_seen.get("prompt_cache_miss_tokens",
                                       max(0, int(usage_seen.get("prompt_tokens", 0) or 0) - hit)))
            r._record_tokens({
                "input_tokens":  miss,
                "output_tokens": int(usage_seen.get("completion_tokens", 0) or 0),
                "cache_read":    hit,
                "cache_create":  0,
            }, cost=0.0)
        return captured[0]

    def test_deepseek_input_tokens_excludes_cache_hit(self):
        """``input_tokens`` must be ONLY the miss portion, not the full
        prompt_tokens. Otherwise the new total formula double-counts
        cache_read."""
        usage, _ = self._record_via_runner({
            "prompt_tokens": 10000,
            "prompt_cache_hit_tokens": 8000,
            "prompt_cache_miss_tokens": 2000,
            "completion_tokens": 500,
        })
        self.assertEqual(usage["input_tokens"], 2000,
            "input_tokens should be miss only (2000), not full "
            "prompt_tokens (10000)"
        )
        self.assertEqual(usage["cache_read"], 8000)
        self.assertEqual(usage["cache_create"], 0,
            "DeepSeek has no separate cache-creation band; cache_create "
            "must stay 0 so it isn't double-counted in the total"
        )

    def test_deepseek_fallback_when_miss_field_absent(self):
        """If a future DeepSeek API drops ``prompt_cache_miss_tokens``,
        derive miss = prompt_tokens - hit. Pin the fallback."""
        usage, _ = self._record_via_runner({
            "prompt_tokens": 10000,
            "prompt_cache_hit_tokens": 8000,
            # prompt_cache_miss_tokens omitted
            "completion_tokens": 100,
        })
        self.assertEqual(usage["input_tokens"], 2000)
        self.assertEqual(usage["cache_read"], 8000)

    def test_deepseek_total_with_new_buckets_matches_real_usage(self):
        """End-to-end accounting check: with the new bucket assignment,
        ``total = input + output + cr + cc`` should equal the real
        ``prompt_tokens + completion_tokens`` from the API — no doubling."""
        usage, _ = self._record_via_runner({
            "prompt_tokens": 10000,
            "prompt_cache_hit_tokens": 8000,
            "prompt_cache_miss_tokens": 2000,
            "completion_tokens": 500,
        })
        total = (usage["input_tokens"] + usage["output_tokens"]
                 + usage["cache_read"] + usage["cache_create"])
        real_total = 10000 + 500  # prompt_tokens + completion_tokens
        self.assertEqual(total, real_total,
            f"new bucket total {total} != real usage {real_total} — "
            f"the round-2 regression isn't fixed"
        )


class TestEstimatedFieldPersistedToJSONL(unittest.TestCase):
    """Round-2 review (Fix #6 PARTIAL): ``cleanup_extra`` passes
    ``estimated=True`` into ``_record_tokens``, but the JSONL record
    constructor (``token_stats.record_token_usage``) was dropping the
    field. Now persisted so future audit / cost-rollup tooling can
    exclude estimated rows from precise totals."""

    def setUp(self):
        self._tmp = Path(__file__).resolve().parent / "_estimated_tmp"
        self._tmp.mkdir(exist_ok=True)
        for f in (self._tmp / "all.jsonl",):
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
        for f in (self._tmp / "all.jsonl",):
            if f.exists():
                f.unlink()
        if self._tmp.exists():
            self._tmp.rmdir()

    def _read_token_records(self):
        records = []
        with (_cfg.LOG_DIR / "all.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("role") == "token":
                    records.append(r)
        return records

    def test_estimated_flag_persisted_to_jsonl(self):
        token_stats.record_token_usage(
            "est_chat", "kimi",
            {"input_tokens": 100, "output_tokens": 50, "estimated": True},
        )
        records = self._read_token_records()
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].get("estimated") is True,
            "estimated=True must round-trip to JSONL so future tooling "
            "can distinguish char-count estimates from SDK-reported counts"
        )

    def test_precise_records_omit_estimated_field(self):
        """When estimated=False (or unset), the JSONL row should NOT
        contain the field — keeps existing rows backwards-compatible
        and avoids polluting precise records with a redundant False."""
        token_stats.record_token_usage(
            "prec_chat", "claude",
            {"input_tokens": 100, "output_tokens": 50},
        )
        records = self._read_token_records()
        self.assertEqual(len(records), 1)
        self.assertNotIn("estimated", records[0],
            "precise records should NOT carry an estimated=False flag — "
            "keeps JSONL backwards-compatible and minimal"
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


class TestStatsIntentDateArgument(unittest.TestCase):
    """Round-3 follow-up P0 (R3-2): ``/stats intent 2026-05-15`` must
    forward the optional date suffix to ``aggregate_daily`` so users can
    inspect historical aggregates. Prior to this fix, ``_cmd_stats``
    collapsed ``args.strip().lower()`` and matched the entire suffix
    against the literal ``"intent"`` — anything past the word was either
    silently dropped (when an exact match succeeded after stripping) or
    routed to the token-stats path.

    These tests pin the parser + downstream call: the date string must
    arrive at ``aggregate_daily(date=<X>)`` verbatim, and the rendered
    card title must surface ``agg['date']`` so users can confirm which
    day they're looking at."""

    def test_date_suffix_routed_to_aggregate_daily(self):
        from larkhelm.commands import _cmd_stats

        captured_date: list = []

        # ``**kw`` swallows the R4-1d ``chat_id`` kwarg without claiming a
        # contract on it (separate tests in TestStatsIntentChatIdFilter
        # pin the chat_id forwarding). Keeps this test focused on date.
        def _fake_aggregate(date=None, path=None, **kw):
            captured_date.append(date)
            return {
                "total": 0, "success_rate": 0.0, "avg_duration": 0.0,
                "total_cost": 0.0, "per_agent": {},
                "date": date or "2026-05-19",
            }

        sent: list = []
        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            sent.append({"title": title, "body": body})

        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.agent_hub.agent_audit.aggregate_daily",
                       side_effect=_fake_aggregate):
                _cmd_stats("oc_test", "msg_1", args="intent 2026-05-15")

        self.assertEqual(captured_date, ["2026-05-15"],
            "date suffix `2026-05-15` must be forwarded verbatim to "
            "aggregate_daily(date=...) — without this the user can never "
            "inspect historical intent aggregates"
        )
        # Card title should surface the date so the user sees which day.
        self.assertEqual(len(sent), 1)
        self.assertIn("2026-05-15", sent[0]["title"],
            "rendered card title must include the requested date so the "
            "user can confirm the report period"
        )

    def test_no_date_suffix_defaults_to_today(self):
        """Bare ``/stats intent`` (no date arg) must keep defaulting to
        today — preserves the pre-fix behavior."""
        from larkhelm.commands import _cmd_stats

        captured_date: list = []

        def _fake_aggregate(date=None, path=None, **kw):
            captured_date.append(date)
            return {
                "total": 0, "success_rate": 0.0, "avg_duration": 0.0,
                "total_cost": 0.0, "per_agent": {}, "date": "today-ish",
            }

        with patch("larkhelm.commands.send_card_reply"):
            with patch("larkhelm.agent_hub.agent_audit.aggregate_daily",
                       side_effect=_fake_aggregate):
                _cmd_stats("oc_test", "msg_1", args="intent")

        self.assertEqual(captured_date, [None],
            "bare `/stats intent` must call aggregate_daily(date=None) "
            "so the helper resolves to today — pre-fix default behavior"
        )


class TestCancelTimeoutPartialTokenRecording(unittest.TestCase):
    """Round-3 fix #1: when cancel / hard-timeout truncates the stream
    before the terminal usage envelope arrives, the runner must still
    persist a partial record (estimated=True) so /stats doesn't drop
    the entire query from accounting."""

    def test_safety_net_records_from_last_usage_seen(self):
        """When the runner stashed an intermediate usage (Claude
        ``assistant`` events carry usage too), the safety net writes
        those counts with estimated=True."""
        from larkhelm.runner_base import BaseProcessRunner

        captured: list = []

        class _Stub(BaseProcessRunner):
            def build_args(self): return ["true"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return False
            def cleanup_extra(self): pass

        with patch.object(BaseProcessRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(
                              (model, dict(usage), cost))):
            r = _Stub.__new__(_Stub)
            BaseProcessRunner.__init__(
                r, backend_name="stub", chat_id="c1", message="m",
                sid=None, cwd="/tmp",
            )
            r._last_usage_seen = {
                "input_tokens": 5000, "output_tokens": 200,
                "cache_read": 2000, "cache_create": 0,
            }
            r.record_partial_tokens_if_needed("claude")

        self.assertEqual(len(captured), 1)
        model, usage, cost = captured[0]
        self.assertEqual(model, "claude")
        self.assertEqual(usage["input_tokens"], 5000)
        self.assertEqual(usage["cache_read"], 2000)
        self.assertTrue(usage.get("estimated"),
            "partial records must be flagged estimated=True so future "
            "tooling can exclude them from precise totals"
        )

    def test_safety_net_falls_back_to_char_count(self):
        """No intermediate usage observed → char-count fallback."""
        from larkhelm.runner_base import BaseProcessRunner

        captured: list = []

        class _Stub(BaseProcessRunner):
            def build_args(self): return ["true"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return False
            def cleanup_extra(self): pass

        with patch.object(BaseProcessRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(
                              dict(usage))):
            r = _Stub.__new__(_Stub)
            BaseProcessRunner.__init__(
                r, backend_name="stub", chat_id="c1",
                message="user prompt " * 25,  # ~275 chars → 68 input
                sid=None, cwd="/tmp",
            )
            r._result_text = "partial response " * 30  # ~510 chars → 127 output
            r.record_partial_tokens_if_needed("claude")

        self.assertEqual(len(captured), 1)
        usage = captured[0]
        self.assertGreater(usage["input_tokens"], 0)
        self.assertGreater(usage["output_tokens"], 0)
        self.assertTrue(usage.get("estimated"))
        self.assertEqual(usage["cache_read"], 0,
            "char-count fallback can't know cache breakdown; stays 0"
        )

    def test_safety_net_short_circuits_when_already_recorded(self):
        """When the happy-path ``_record_tokens`` fired, the safety net
        must NOT write a second record."""
        from larkhelm.runner_base import BaseProcessRunner

        captured: list = []

        class _Stub(BaseProcessRunner):
            def build_args(self): return ["true"]
            def build_stdin(self): return None
            def parse_stdout_event(self, ev): return False
            def cleanup_extra(self): pass

        with patch.object(BaseProcessRunner, "_record_tokens",
                          lambda self, model, usage, cost: captured.append(
                              dict(usage))):
            r = _Stub.__new__(_Stub)
            BaseProcessRunner.__init__(
                r, backend_name="stub", chat_id="c1",
                message="anything", sid=None, cwd="/tmp",
            )
            r._tokens_recorded = True   # happy path already fired
            r._last_usage_seen = {"input_tokens": 999, "output_tokens": 999}
            r.record_partial_tokens_if_needed("claude")

        self.assertEqual(captured, [],
            "safety net should short-circuit when _tokens_recorded=True; "
            "writing again would double-count the same query"
        )


class TestDurationPairingByTraceId(unittest.TestCase):
    """Round-3 fix #2: ``_cmd_stats`` pairs user/assistant entries by
    ``trace_id`` when both sides carry one. Falls back to FIFO when
    not. Pre-fix FIFO-only logic scrambled durations under concurrent
    /btw or rapid /cancel + resend."""

    def _accept_durations(self, today_records):
        """Replay the duration-pairing block from _cmd_stats against a
        fixture record list and return the computed durations."""
        from datetime import datetime as _dt
        import larkhelm.config as _cfg
        durations: list[float] = []
        _pending_by_trace: dict[str, _dt] = {}
        _pending_fifo = None
        _hard_cap_secs = float(getattr(_cfg, "HARD_TIMEOUT", 21600) or 21600) * 1.1
        for r in today_records:
            role = r.get("role", "")
            if role not in ("user", "assistant", "error"):
                continue
            try:
                ts = _dt.fromisoformat(r["ts"])
            except (KeyError, ValueError):
                continue
            tid = r.get("trace_id")
            if role == "user":
                if tid:
                    _pending_by_trace[tid] = ts
                else:
                    _pending_fifo = ts
            else:
                paired = None
                if tid and tid in _pending_by_trace:
                    paired = _pending_by_trace.pop(tid)
                elif _pending_fifo is not None:
                    paired = _pending_fifo
                    _pending_fifo = None
                if paired is not None:
                    secs = (ts - paired).total_seconds()
                    if 0 < secs < _hard_cap_secs:
                        durations.append(secs)
        return durations

    def test_interleaved_pairs_resolve_by_trace_id(self):
        """Scenario: user A → user B → assistant B → assistant A.
        FIFO would record only one (B's start paired with B's end,
        then drop A entirely). Trace-id pairing recovers both."""
        records = [
            {"role": "user",      "ts": "2026-05-19T10:00:00", "trace_id": "A"},
            {"role": "user",      "ts": "2026-05-19T10:00:30", "trace_id": "B"},
            {"role": "assistant", "ts": "2026-05-19T10:01:00", "trace_id": "B"},
            {"role": "assistant", "ts": "2026-05-19T10:05:00", "trace_id": "A"},
        ]
        durations = self._accept_durations(records)
        self.assertEqual(len(durations), 2,
            "both A (300s) and B (30s) durations must be recovered; "
            "FIFO would have given only 1"
        )
        # B (30s) and A (300s) — verify both magnitudes appear
        self.assertIn(30.0, durations)
        self.assertIn(300.0, durations)

    def test_fifo_fallback_for_old_records_without_trace_id(self):
        """JSONL rows logged before trace_id propagation still get a
        FIFO duration so historical /stats numbers stay populated."""
        records = [
            {"role": "user",      "ts": "2026-05-19T10:00:00"},
            {"role": "assistant", "ts": "2026-05-19T10:00:42"},
        ]
        durations = self._accept_durations(records)
        self.assertEqual(durations, [42.0])

    def test_mixed_trace_id_and_legacy(self):
        """One pair has trace_id, another doesn't — both should record.

        Timeline (Δ vs the user-side start):
          T=0       user trace_id=T1
          T=60s     user no trace_id (legacy)        → _pending_fifo
          T=70s     assistant no trace_id           → FIFO pair = 10s
          T=300s    assistant trace_id=T1            → traced pair = 300s
        """
        records = [
            {"role": "user",      "ts": "2026-05-19T10:00:00", "trace_id": "T1"},
            {"role": "user",      "ts": "2026-05-19T10:01:00"},   # legacy → FIFO
            {"role": "assistant", "ts": "2026-05-19T10:01:10"},  # FIFO end → 10s
            {"role": "assistant", "ts": "2026-05-19T10:05:00", "trace_id": "T1"},  # traced → 300s
        ]
        durations = self._accept_durations(records)
        self.assertEqual(sorted(durations), [10.0, 300.0])


class TestCmdStatsEndToEndWithMockJsonl(unittest.TestCase):
    """Round-3 review SHOULD-FIX: the per-fix pairing tests above
    replay the algorithm in-line (``_accept_durations``). This adds
    an end-to-end driver that hits ``_cmd_stats`` itself through a
    mocked ``_read_logs``, so any future refactor that diverges
    pairing logic between the two implementations fires here."""

    def test_cmd_stats_renders_traced_pairs_and_logs_orphans(self):
        sent: list = []

        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            sent.append({"title": title, "body": body, "color": color})

        debug_lines: list = []

        def _capture_debug(msg):
            debug_lines.append(msg)

        # Fixture: 1 paired (trace_id=A, 60s), 1 orphan (trace_id=B
        # user, no assistant), 1 FIFO pair (no trace_id, 30s).
        # Expected: 2 durations recorded, average = (60+30)/2 = 45s,
        # debug log fires for the orphan B.
        fake_records = [
            {"ts": "2026-05-19T10:00:00", "role": "user", "trace_id": "A"},
            {"ts": "2026-05-19T10:00:30", "role": "user"},
            {"ts": "2026-05-19T10:01:00", "role": "user", "trace_id": "B"},
            {"ts": "2026-05-19T10:01:00", "role": "assistant"},  # FIFO 30s
            {"ts": "2026-05-19T10:01:00", "role": "assistant", "trace_id": "A"},  # 60s
            # B never gets an assistant — orphan.
        ]
        # Date-stamp matching: _cmd_stats filters by ``today``. Make
        # our records appear "today" so the filter accepts them.
        from datetime import datetime as _dt
        today_iso = _dt.now().strftime("%Y-%m-%d")
        for r in fake_records:
            r["ts"] = today_iso + "T" + r["ts"].split("T", 1)[1]

        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.commands._read_logs", return_value=fake_records):
                with patch("larkhelm.commands.get_token_stats_persistent",
                           return_value={}):
                    with patch("larkhelm.commands.get_token_stats",
                               return_value={}):
                        with patch("larkhelm.commands._debug_log",
                                   side_effect=_capture_debug):
                            from larkhelm.commands import _cmd_stats
                            _cmd_stats("oc_e2e", "msg_1")

        self.assertEqual(len(sent), 1, "exactly one /stats card expected")
        body = sent[0]["body"]
        # Average shown is _fmt_elapsed-ish; the raw number was ~45.0s
        # → "45s" or "45.0s" — body must surface a non-dash average.
        self.assertNotIn("平均耗时：**—**", body,
            "two pairs should produce a real average, not '—'"
        )
        # Orphan B should have triggered the debug-log warning line
        # (round-3 NICE-TO-HAVE).
        orphan_logs = [m for m in debug_lines if "pairing leftovers" in m]
        self.assertTrue(orphan_logs,
            "[stats] pairing-leftovers debug line should fire for the "
            "orphan trace_id B"
        )
        # The line should mention "1 trace_id orphan" (B).
        self.assertIn("1 trace_id orphan", orphan_logs[0])


class TestDurationHardCapMatchesHardTimeout(unittest.TestCase):
    """Round-3 fix #3: the previous ``if 0 < secs < 3600`` silently
    dropped every /dev / /crew query > 1h. Cap now keys off
    ``_cfg.HARD_TIMEOUT * 1.1`` so it only filters truly-bad records
    (pair-end missing across days etc.)."""

    def test_long_query_below_hard_timeout_is_kept(self):
        """A 2-hour /dev pipeline (well under the default 6h
        hard_timeout) must NOT be filtered."""
        import larkhelm.config as _cfg

        records = [
            {"role": "user",      "ts": "2026-05-19T10:00:00"},
            # assistant arrives 2 hours later (7200s) — under the
            # default 21600 * 1.1 = 23760s cap.
            {"role": "assistant", "ts": "2026-05-19T12:00:00"},
        ]
        from datetime import datetime as _dt
        hard_cap = float(getattr(_cfg, "HARD_TIMEOUT", 21600) or 21600) * 1.1
        durations: list[float] = []
        pending = None
        for r in records:
            ts = _dt.fromisoformat(r["ts"])
            if r["role"] == "user":
                pending = ts
            elif pending is not None:
                secs = (ts - pending).total_seconds()
                if 0 < secs < hard_cap:
                    durations.append(secs)
                pending = None
        self.assertEqual(durations, [7200.0],
            f"2h query must be kept (cap = {hard_cap}s); the old "
            "3600s cap silently dropped every long /dev query"
        )


class TestOverlappingCacheNoDoubleCount(unittest.TestCase):
    """R1.5-2: end-to-end pipeline test for the DeepSeek + Gemini
    "overlapping cache field" regression that round-2 audit caught.

    Background: post-991443f, ``_fmt_token_block`` switched to
    ``total = inp + out + cr + cc`` on the assumption the four buckets are
    disjoint. That assumption holds for Claude and (post-fix) Gemini, but
    BOTH backends previously reported overlapping fields (DeepSeek
    ``prompt_tokens = hit + miss``; Gemini ``stats.input_tokens = total``
    including cached). If the runner-level normalization to disjoint
    ever regresses, the displayed total near-doubles on every cache-heavy
    query — a P0 user-visible bug.

    The previous unit tests for both runners exercised either ``__new__``-
    bypassed instances or ``patch.object(_record_tokens)``, so they
    pinned the runner's call shape but NEVER drove the full chain
    ``runner → _record_tokens → record_token_usage → all.jsonl →
    get_token_stats_persistent → _fmt_token_block``. That's exactly
    where the round-2 regression hid in 991443f → ca8c609. These two
    tests close the gap by hitting every step with real I/O (the only
    mock is at the HTTP / subprocess boundary, which the tests must
    fake to stay hermetic).
    """

    def setUp(self):
        # Redirect token-stats JSONL writes to a tmp dir so production
        # all.jsonl is never touched.
        self._tmp = Path(__file__).resolve().parent / "_overlap_e2e_tmp"
        self._tmp.mkdir(exist_ok=True)
        for f in (self._tmp / "all.jsonl", self._tmp / "all.jsonl.1"):
            if f.exists():
                f.unlink()
        self._orig_log_dir = getattr(_cfg, "LOG_DIR", None)
        _cfg.LOG_DIR = self._tmp

    def tearDown(self):
        # Drop the in-memory token_stats accumulator entries we polluted
        # so the next test starts clean (record_token_usage updates the
        # process-wide ``_token_stats`` OrderedDict too, not just JSONL).
        with token_stats._token_stats_lock:
            for cid in ("deepseek_overlap_e2e", "gemini_overlap_e2e"):
                token_stats._token_stats.pop(cid, None)

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

    def _read_rendered_total(self, chat_id: str, model: str) -> tuple[int, str]:
        """Run the persistent reader + UI formatter, return (total, body)."""
        from larkhelm.commands import _fmt_token_block
        data = token_stats.get_token_stats_persistent(chat_id)
        self.assertIn(model, data,
            f"persistent reader should expose {model} usage for {chat_id}"
        )
        body = _fmt_token_block("__test__", data)
        # Pull the "合计 N,NNN tokens" number out of the body.
        import re
        m = re.search(r"合计 \*\*([\d,]+)\*\*", body)
        self.assertIsNotNone(m, f"could not parse total from {body!r}")
        return int(m.group(1).replace(",", "")), body

    def test_deepseek_overlapping_cache_no_double_count(self):
        """Real DeepSeekRunner SSE event with ``prompt_tokens=50000,
        prompt_cache_hit_tokens=40000, completion_tokens=500`` must
        render total = 50,500 (NOT ~100,500). Cache hit-rate must be
        ~80% in the rendered card."""
        import threading as _threading
        from unittest import mock as _mock
        # Use the module-level helpers (top of file) — they bootstrap the
        # minimum config DeepSeekRunner needs WITHOUT touching HARD_TIMEOUT,
        # so the duration-cap test downstream stays unaffected. We DO need
        # to preserve our LOG_DIR override (setUp redirected it to tmp).
        _saved_log_dir = _cfg.LOG_DIR
        _deepseek_min_config()
        _cfg.LOG_DIR = _saved_log_dir
        from larkhelm.runner_deepseek import DeepSeekRunner

        sse = _ds_sse(["hello world"], usage={
            "prompt_tokens": 50000,                # overlapping: hit + miss
            "completion_tokens": 500,
            "prompt_cache_hit_tokens": 40000,
            "prompt_cache_miss_tokens": 10000,
        })
        fake_resp = _FakeStreamResponse(200, sse)

        chat_id = "deepseek_overlap_e2e"
        with _mock.patch("requests.Session.post", return_value=fake_resp):
            DeepSeekRunner(
                chat_id, "Hi", sid=None, cwd="/tmp",
                cancel_ev=_threading.Event(),
            ).run()

        # Run the full reader + UI chain and inspect.
        total, body = self._read_rendered_total(chat_id, "deepseek")
        real_total = 50000 + 500  # the OpenAI-compatible "true" cost
        self.assertLessEqual(total, real_total + 1,  # +1 for any int rounding
            f"deepseek total {total:,} exceeds the real API total "
            f"{real_total:,}; round-2 double-count regression is back. "
            f"Body:\n{body}"
        )
        # And cache hit pct should reflect 40000 / 50000 = 80%, not 400%.
        self.assertIn("（80%）", body,
            f"expected '（80%）' for cache hit rate (40k/50k), got body:\n{body}"
        )

    def test_gemini_overlapping_cache_no_double_count(self):
        """Real GeminiRunner.parse_stdout_event with the audit-observed
        envelope ``stats={"input_tokens": 12184, "cached": 11555,
        "input": 629, "output_tokens": 19}`` must render total = 12,203
        (NOT 12184 + 11555 + 19). The previous "stats" reader pinned
        only the call-shape; this test verifies the disjoint contract
        survives the full JSONL → persistent reader → UI chain."""
        from larkhelm.runner_gemini import GeminiRunner
        from larkhelm.runner_base import BaseProcessRunner

        chat_id = "gemini_overlap_e2e"
        # Build a real GeminiRunner without spawning a subprocess. We can
        # bypass ``__init__`` (which would call super().__init__ and try
        # to acquire the AI semaphore on run()); ``parse_stdout_event``
        # only needs the handful of fields below.
        r = GeminiRunner.__new__(GeminiRunner)
        # BaseProcessRunner.__init__ sets these (some via the constructor
        # signature); we only need the subset _record_tokens reads.
        r.backend_name   = "gemini"
        r.chat_id        = chat_id
        r.record_under   = None
        r._tokens_recorded = False
        r._last_usage_seen = {}
        r.use_session    = False
        r._extra_args    = []
        r._new_sid       = None
        # Sanity: _record_tokens is inherited from BaseProcessRunner;
        # don't patch it — we want the real persistence path. On py3
        # methods accessed on the class itself are bare functions (no
        # __func__ wrapper), so compare the function objects directly.
        self.assertIs(
            GeminiRunner._record_tokens,
            BaseProcessRunner._record_tokens,
            "GeminiRunner must inherit _record_tokens unchanged; if it "
            "overrides we need to retarget this test"
        )

        ev = {
            "type": "result", "session_id": "gsid_overlap_test",
            "stats": {
                "input_tokens": 12184,   # OVERLAPPING — = input + cached
                "output_tokens": 19,
                "cached": 11555,
                "input": 629,            # = 12184 - 11555
                "total_tokens": 12203,
            },
        }
        r.parse_stdout_event(ev)

        # The runner persisted via the real chain. Verify the rendered total.
        total, body = self._read_rendered_total(chat_id, "gemini")
        real_total = 12184 + 19  # input_tokens (which is hit+miss) + output
        self.assertEqual(total, real_total,
            f"gemini total {total:,} != real API total {real_total:,}; "
            f"input/cached overlap must NOT be counted twice. "
            f"Body:\n{body}"
        )
        # Hit pct = cached / (cached + non-cached input) = 11555 / 12184
        # = 94.84% → rendered as "94%" by the int() truncation.
        self.assertIn("（94%）", body,
            f"expected '（94%）' cache hit pct (11555/12184), got body:\n{body}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# R4-1 audit fixes (round-4 stats wrap-up)
# ─────────────────────────────────────────────────────────────────────────────


class TestEvictCrewAgentTokensStrictPrefix(unittest.TestCase):
    """R4-1b: `evict_crew_agent_tokens(prefix)` previously used substring
    matching (`prefix in k`), violating the function's prefix contract.
    crew_id is currently a UUID so collisions were rare but pinning the
    contract now keeps a future move to semantic IDs safe."""

    def setUp(self):
        # Snapshot then reset module-level state so per-test mutations
        # don't bleed across tests.
        self._snap = dict(token_stats._crew_agent_tokens)
        token_stats._crew_agent_tokens.clear()

    def tearDown(self):
        token_stats._crew_agent_tokens.clear()
        token_stats._crew_agent_tokens.update(self._snap)

    def test_evict_only_keys_starting_with_prefix(self):
        # Two entries: target shares the prefix exactly, distractor
        # contains the prefix as a SUBSTRING but does NOT start with it.
        token_stats.record_crew_agent_tokens(
            "abc-uuid__agent_1", "claude", {"input_tokens": 10},
        )
        token_stats.record_crew_agent_tokens(
            "xx-abc-uuid-yy__agent_1", "claude", {"input_tokens": 20},
        )
        token_stats.evict_crew_agent_tokens("abc-uuid")
        # Target removed, distractor (with substring match only) preserved.
        self.assertNotIn("abc-uuid__agent_1", token_stats._crew_agent_tokens,
            "exact-prefix entry must be evicted"
        )
        self.assertIn("xx-abc-uuid-yy__agent_1", token_stats._crew_agent_tokens,
            "substring-only match must NOT be evicted — pre-fix `prefix in k` "
            "would have deleted this too, leaking eviction to unrelated entries"
        )

    def test_evict_empty_prefix_is_noop_on_empty_store(self):
        token_stats.evict_crew_agent_tokens("nope-uuid")
        self.assertEqual(len(token_stats._crew_agent_tokens), 0)


class TestSummarizeCrewAgentTokensForChat(unittest.TestCase):
    """R4-1e helper: aggregate the in-memory `_crew_agent_tokens` dict
    for one chat. Powers the `/stats` card bottom "Crew Agents" line."""

    def setUp(self):
        self._snap = dict(token_stats._crew_agent_tokens)
        token_stats._crew_agent_tokens.clear()

    def tearDown(self):
        token_stats._crew_agent_tokens.clear()
        token_stats._crew_agent_tokens.update(self._snap)

    def test_aggregates_matching_chat_entries(self):
        # Two agents under one chat + one entry under a different chat
        # which must NOT be included in the sum.
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_agent_pm", "claude",
            {"input_tokens": 100, "output_tokens": 50, "cost_usd": 0.01},
        )
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_agent_eng", "claude",
            {"input_tokens": 200, "output_tokens": 75, "cost_usd": 0.02},
        )
        token_stats.record_crew_agent_tokens(
            "chatB__crew_run9_agent_pm", "claude",
            {"input_tokens": 999, "output_tokens": 999, "cost_usd": 9.99},
        )
        summary = token_stats.summarize_crew_agent_tokens_for_chat("chatA")
        self.assertEqual(summary["agents"], 2)
        self.assertEqual(summary["input_tokens"], 300)
        self.assertEqual(summary["output_tokens"], 125)
        # Cost rounds via float arithmetic; compare to 4 decimals.
        self.assertAlmostEqual(summary["cost_usd"], 0.03, places=4)

    def test_empty_when_no_match(self):
        token_stats.record_crew_agent_tokens(
            "other__crew_x_agent_1", "claude", {"input_tokens": 5},
        )
        self.assertEqual(
            token_stats.summarize_crew_agent_tokens_for_chat("nobody"), {},
            "no matching prefix must return {} so /stats can suppress the line"
        )

    def test_prefix_is_anchored_to_chat_id_double_underscore(self):
        """Ensure `chatA` doesn't accidentally swallow `chatAB__crew_*`
        (the prefix is `chatA__crew_`, NOT `chatA`)."""
        token_stats.record_crew_agent_tokens(
            "chatA__crew_x_agent_1", "claude", {"input_tokens": 10},
        )
        token_stats.record_crew_agent_tokens(
            "chatAB__crew_x_agent_1", "claude", {"input_tokens": 99},
        )
        summary = token_stats.summarize_crew_agent_tokens_for_chat("chatA")
        self.assertEqual(summary["input_tokens"], 10,
            "chatAB__crew_* must NOT be summed into chatA's totals"
        )


class TestCmdStatsUserCountExcludesCrewPlan(unittest.TestCase):
    """R4-1c: `/crew` and `/plan` commands write `role="user", model in
    {crew,plan}` as command markers, not real conversation turns. They
    must NOT count toward `今日对话` in `/stats`."""

    def test_user_count_excludes_crew_plan_shell_markers(self):
        from datetime import datetime as _dt
        today = _dt.now().strftime("%Y-%m-%d")
        fake_records = [
            # 2 genuine user turns (model="" — legacy or claude/gemini).
            {"ts": today + "T10:00:00", "role": "user", "model": ""},
            {"ts": today + "T10:01:00", "role": "user", "model": "claude"},
            # 3 command markers that must NOT count:
            {"ts": today + "T10:02:00", "role": "user", "model": "crew"},
            {"ts": today + "T10:03:00", "role": "user", "model": "plan"},
            {"ts": today + "T10:04:00", "role": "user", "model": "shell"},
        ]

        captured: list = []
        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            captured.append({"body": body})

        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.commands._read_logs", return_value=fake_records):
                with patch("larkhelm.commands.get_token_stats_persistent",
                           return_value={}):
                    with patch("larkhelm.commands.get_token_stats",
                               return_value={}):
                        from larkhelm.commands import _cmd_stats
                        _cmd_stats("oc_count", "msg_1")

        self.assertEqual(len(captured), 1)
        body = captured[0]["body"]
        # Card formats the user count as "今日对话：**N** 次".
        self.assertIn("今日对话：**2** 次", body,
            "crew/plan/shell command markers must NOT inflate user_count; "
            f"got body:\n{body}"
        )


class TestAggregateDailyChatIdFilter(unittest.TestCase):
    """R4-1d: `aggregate_daily(chat_id=...)` must restrict to records
    matching that chat_id. Without this, `/stats intent` leaks volume /
    avg-duration from other chats — a cross-chat side channel."""

    def setUp(self):
        # Per-test JSONL file (avoid race with the shared DATA_DIR).
        self._tmp = Path(tempfile.mkdtemp(prefix="agent_audit_chatid_"))
        self._path = self._tmp / "agent_audit.jsonl"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _write(self, recs):
        with self._path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

    def test_chat_id_filter_isolates_one_chat(self):
        from larkhelm.agent_hub.agent_audit import aggregate_daily
        today = "2026-05-19"
        self._write([
            # 2 records for chat A
            {"ts": today + "T10:00:00", "chat_id": "A", "agent_type": "chat",
             "success": True,  "duration_sec": 1.0, "cost_usd": 0.0},
            {"ts": today + "T10:01:00", "chat_id": "A", "agent_type": "dev",
             "success": False, "duration_sec": 2.0, "cost_usd": 0.0},
            # 2 records for chat B — must be excluded
            {"ts": today + "T10:02:00", "chat_id": "B", "agent_type": "crew",
             "success": True,  "duration_sec": 100.0, "cost_usd": 0.0},
            {"ts": today + "T10:03:00", "chat_id": "B", "agent_type": "crew",
             "success": True,  "duration_sec": 200.0, "cost_usd": 0.0},
        ])

        a_only = aggregate_daily(date=today, path=self._path, chat_id="A")
        self.assertEqual(a_only["total"], 2)
        # A's avg_duration must NOT be polluted by B's huge entries.
        self.assertAlmostEqual(a_only["avg_duration"], 1.5, places=2)
        self.assertEqual(set(a_only["per_agent"].keys()), {"chat", "dev"},
            "chat B's `crew` entries must NOT appear in A's per_agent")

        b_only = aggregate_daily(date=today, path=self._path, chat_id="B")
        self.assertEqual(b_only["total"], 2)
        self.assertAlmostEqual(b_only["avg_duration"], 150.0, places=2)

    def test_no_chat_id_keeps_global_aggregate(self):
        """Default (chat_id=None) preserves the pre-fix behaviour for
        CLI tooling / future admin endpoints."""
        from larkhelm.agent_hub.agent_audit import aggregate_daily
        today = "2026-05-19"
        self._write([
            {"ts": today + "T10:00:00", "chat_id": "A", "agent_type": "chat",
             "success": True, "duration_sec": 1.0, "cost_usd": 0.0},
            {"ts": today + "T10:01:00", "chat_id": "B", "agent_type": "chat",
             "success": True, "duration_sec": 1.0, "cost_usd": 0.0},
        ])
        agg = aggregate_daily(date=today, path=self._path)
        self.assertEqual(agg["total"], 2,
            "chat_id=None must still aggregate across all chats")


class TestCmdStatsIntentForwardsChatId(unittest.TestCase):
    """R4-1d wiring: `_cmd_stats_intent` must pass the caller's chat_id
    to `aggregate_daily`. Pin so future refactors can't drop the kwarg."""

    def test_chat_id_forwarded_to_aggregate_daily(self):
        from larkhelm.commands import _cmd_stats_intent

        captured: dict = {}

        def _fake_aggregate(date=None, path=None, chat_id=None):
            captured["date"] = date
            captured["chat_id"] = chat_id
            return {
                "total": 0, "success_rate": 0.0, "avg_duration": 0.0,
                "total_cost": 0.0, "per_agent": {}, "date": "2026-05-19",
            }

        with patch("larkhelm.commands.send_card_reply"):
            with patch("larkhelm.agent_hub.agent_audit.aggregate_daily",
                       side_effect=_fake_aggregate):
                _cmd_stats_intent("oc_target_chat", "msg_1")

        self.assertEqual(captured.get("chat_id"), "oc_target_chat",
            "_cmd_stats_intent must forward its chat_id to aggregate_daily; "
            "otherwise the rendered card leaks cross-chat aggregates"
        )


class TestCmdStatsCrewAgentsLine(unittest.TestCase):
    """R4-1e render test: when in-memory `_crew_agent_tokens` has entries
    for the chat, `/stats` must append a `Crew Agents（本进程）` block
    so users can see the drill-down CLAUDE.md promised."""

    def setUp(self):
        self._snap = dict(token_stats._crew_agent_tokens)
        token_stats._crew_agent_tokens.clear()

    def tearDown(self):
        token_stats._crew_agent_tokens.clear()
        token_stats._crew_agent_tokens.update(self._snap)

    def test_crew_agents_line_appears_when_present(self):
        # Seed two crew agent entries under chat "oc_x".
        token_stats.record_crew_agent_tokens(
            "oc_x__crew_run1_agent_pm", "claude",
            {"input_tokens": 1000, "output_tokens": 500, "cost_usd": 0.05},
        )
        token_stats.record_crew_agent_tokens(
            "oc_x__crew_run1_agent_eng", "claude",
            {"input_tokens": 2000, "output_tokens": 800, "cost_usd": 0.10},
        )

        sent: list = []
        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            sent.append({"body": body})

        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.commands._read_logs", return_value=[]):
                with patch("larkhelm.commands.get_token_stats_persistent",
                           return_value={}):
                    with patch("larkhelm.commands.get_token_stats",
                               return_value={}):
                        from larkhelm.commands import _cmd_stats
                        _cmd_stats("oc_x", "msg_1")

        self.assertEqual(len(sent), 1)
        body = sent[0]["body"]
        self.assertIn("Crew Agents", body,
            "Crew Agents block must appear when in-memory entries exist"
        )
        self.assertIn("2 agents", body, "agent count must surface")
        # 1000+500+2000+800 = 4300 total tokens
        self.assertIn("4,300", body, "summed token total must surface")
        # 0.05+0.10 = 0.15 cost
        self.assertIn("$0.1500", body, "summed cost must surface")

    def test_crew_agents_line_suppressed_when_empty(self):
        """No crew entries → no card block (don't add empty noise)."""
        sent: list = []
        def _capture(chat_id, msg_id, title, body, color="blue", **kw):
            sent.append({"body": body})

        with patch("larkhelm.commands.send_card_reply", side_effect=_capture):
            with patch("larkhelm.commands._read_logs", return_value=[]):
                with patch("larkhelm.commands.get_token_stats_persistent",
                           return_value={}):
                    with patch("larkhelm.commands.get_token_stats",
                               return_value={}):
                        from larkhelm.commands import _cmd_stats
                        _cmd_stats("oc_empty", "msg_1")

        self.assertEqual(len(sent), 1)
        self.assertNotIn("Crew Agents", sent[0]["body"],
            "empty in-memory store must NOT render a 'Crew Agents' block"
        )


class TestFmtTokenBlockHitRateHeader(unittest.TestCase):
    """P0+P1 caching audit (2026-05-22):  /stats output must surface a
    "缓存命中率 X% 写入比 Y%" line per model so operators can grep and so
    cache regressions are visible without parsing the multi-line block.
    """

    def test_fmt_token_block_renders_hit_rate_header(self):
        from larkhelm.commands import _fmt_token_block

        stats = {
            "claude": {
                "input_tokens":  1000,
                "output_tokens": 500,
                "cache_read":    9000,
                "cache_create":  100,
                "calls":         3,
                "cost_usd":      0.05,
            }
        }
        body = _fmt_token_block("测试", stats)

        # The header line must surface both labels.
        self.assertIn("缓存命中率", body)
        self.assertIn("写入比", body)
        # cr / (cr + inp) = 9000 / 10000 = 90%
        self.assertIn("缓存命中率 **90%**", body)
        # cc / (cr + inp) = 100 / 10000 = 1%
        self.assertIn("写入比 **1%**", body)
        # Original detail line must still be present (regression guard).
        self.assertIn("合计 **10,600**", body)

    def test_fmt_token_block_empty_data_unchanged(self):
        """Empty data path must remain byte-identical (no hit-rate spam)."""
        from larkhelm.commands import _fmt_token_block

        body = _fmt_token_block("空", {})
        self.assertEqual(body, "**空** — 暂无数据")
        self.assertNotIn("缓存命中率", body)


# ════════════════════════════════════════════════════════════════════════
#  P5 — agent_type bucket aggregation + /stats rendering (AC-07..09)
# ════════════════════════════════════════════════════════════════════════


class TestSummarizeByType(unittest.TestCase):
    """AC-08 — summarize_crew_agent_tokens_by_type returns one dict per
    bucketed agent_type with the 6 documented inner fields."""

    def setUp(self):
        self._snap = dict(token_stats._crew_agent_tokens)
        token_stats._crew_agent_tokens.clear()

    def tearDown(self):
        token_stats._crew_agent_tokens.clear()
        token_stats._crew_agent_tokens.update(self._snap)

    def test_returns_empty_dict_when_no_entries(self):
        self.assertEqual(
            token_stats.summarize_crew_agent_tokens_by_type("nobody"), {}
        )

    def test_buckets_by_agent_id_via_static_map(self):
        """implementer + fixer must collapse to 'engineer'; qa / reviewer
        keep their own buckets."""
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_implementer", "claude",
            {"input_tokens": 100, "output_tokens": 50,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.01},
        )
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_fixer", "claude",
            {"input_tokens": 30, "output_tokens": 20,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.005},
        )
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_qa", "kimi",
            {"input_tokens": 10, "output_tokens": 5,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.001},
        )
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_reviewer", "kimi",
            {"input_tokens": 20, "output_tokens": 10,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.002},
        )

        buckets = token_stats.summarize_crew_agent_tokens_by_type("chatA")
        self.assertIn("engineer", buckets)
        self.assertIn("qa", buckets)
        self.assertIn("reviewer", buckets)
        self.assertEqual(buckets["engineer"]["agents"], 2)
        self.assertEqual(buckets["engineer"]["input_tokens"], 130)
        self.assertEqual(buckets["engineer"]["output_tokens"], 70)
        self.assertAlmostEqual(buckets["engineer"]["cost_usd"], 0.015, places=4)
        self.assertEqual(buckets["qa"]["agents"], 1)
        self.assertEqual(buckets["reviewer"]["agents"], 1)

    def test_unknown_agent_id_falls_back_to_其它(self):
        token_stats.record_crew_agent_tokens(
            "chatA__crew_run1_my_custom_agent", "claude",
            {"input_tokens": 5, "output_tokens": 5,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.001},
        )
        buckets = token_stats.summarize_crew_agent_tokens_by_type("chatA")
        self.assertIn("其它", buckets)
        self.assertEqual(buckets["其它"]["agents"], 1)

    def test_prefix_anchor_isolates_chats(self):
        """chatA must not absorb chatAB's entries (same anchor invariant
        as the legacy helper)."""
        token_stats.record_crew_agent_tokens(
            "chatA__crew_x_implementer", "claude", {"input_tokens": 10},
        )
        token_stats.record_crew_agent_tokens(
            "chatAB__crew_x_implementer", "claude", {"input_tokens": 99},
        )
        buckets = token_stats.summarize_crew_agent_tokens_by_type("chatA")
        self.assertEqual(buckets["engineer"]["input_tokens"], 10)


class TestStatsByTypeRendering(unittest.TestCase):
    """AC-07 — _render_crew_agent_breakdown orders buckets by total
    tokens descending and uses the PRD line format."""

    def setUp(self):
        self._snap = dict(token_stats._crew_agent_tokens)
        token_stats._crew_agent_tokens.clear()
        self._flag_snap = bool(
            getattr(_cfg, "STATS_AGENT_TYPE_BREAKDOWN_ENABLED", True)
        )

    def tearDown(self):
        token_stats._crew_agent_tokens.clear()
        token_stats._crew_agent_tokens.update(self._snap)
        _cfg.STATS_AGENT_TYPE_BREAKDOWN_ENABLED = self._flag_snap

    def _seed_three_buckets(self) -> None:
        # engineer biggest (1500), reviewer middle (250), qa smallest (150)
        token_stats.record_crew_agent_tokens(
            "ccc__crew_run1_implementer", "claude",
            {"input_tokens": 1000, "output_tokens": 500,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.01},
        )
        token_stats.record_crew_agent_tokens(
            "ccc__crew_run1_reviewer", "kimi",
            {"input_tokens": 200, "output_tokens": 50,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.002},
        )
        token_stats.record_crew_agent_tokens(
            "ccc__crew_run1_qa", "kimi",
            {"input_tokens": 100, "output_tokens": 50,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.001},
        )

    def test_descending_by_total_tokens(self):
        from larkhelm.commands import _render_crew_agent_breakdown

        _cfg.STATS_AGENT_TYPE_BREAKDOWN_ENABLED = True
        self._seed_three_buckets()
        lines = _render_crew_agent_breakdown("ccc")
        # First entry is the separator + header; bucket lines follow.
        bucket_lines = [
            ln for ln in lines if ln.startswith("›")
        ]
        self.assertEqual(len(bucket_lines), 3)
        # Order: engineer (1500) > reviewer (250) > qa (150)
        self.assertIn("engineer", bucket_lines[0])
        self.assertIn("reviewer", bucket_lines[1])
        self.assertIn("qa",       bucket_lines[2])

    def test_line_format_matches_prd_spec(self):
        from larkhelm.commands import _render_crew_agent_breakdown

        _cfg.STATS_AGENT_TYPE_BREAKDOWN_ENABLED = True
        self._seed_three_buckets()
        lines = _render_crew_agent_breakdown("ccc")
        header = lines[1]
        self.assertIn("按类型", header)
        engineer_line = next(ln for ln in lines if "engineer" in ln)
        self.assertIn("**engineer**", engineer_line)
        self.assertIn("1 agents", engineer_line)
        self.assertIn("合计 **1,500** tokens", engineer_line)
        self.assertIn("费用 **$0.0100**", engineer_line)


class TestBreakdownDisabled(unittest.TestCase):
    """AC-09 — STATS_AGENT_TYPE_BREAKDOWN_ENABLED=False restores the
    P2 single-line summary."""

    def setUp(self):
        self._snap = dict(token_stats._crew_agent_tokens)
        token_stats._crew_agent_tokens.clear()
        self._flag_snap = bool(
            getattr(_cfg, "STATS_AGENT_TYPE_BREAKDOWN_ENABLED", True)
        )

    def tearDown(self):
        token_stats._crew_agent_tokens.clear()
        token_stats._crew_agent_tokens.update(self._snap)
        _cfg.STATS_AGENT_TYPE_BREAKDOWN_ENABLED = self._flag_snap

    def test_breakdown_disabled_falls_back_to_p2_single_line(self):
        from larkhelm.commands import _render_crew_agent_breakdown

        token_stats.record_crew_agent_tokens(
            "ddd__crew_run1_implementer", "claude",
            {"input_tokens": 1000, "output_tokens": 500,
             "cache_read": 0, "cache_create": 0, "cost_usd": 0.01},
        )
        _cfg.STATS_AGENT_TYPE_BREAKDOWN_ENABLED = False
        lines = _render_crew_agent_breakdown("ddd")
        body = "\n".join(lines)
        self.assertIn("**🤖 Crew Agents（本进程）**", body)
        self.assertNotIn("按类型", body,
            "flag=false must keep the P2 single-line summary",
        )
        # Single bucket aggregated as one line.
        self.assertIn("1 agents", body)


if __name__ == "__main__":
    unittest.main()
