"""Coverage for ``log._prune_content`` + ``_get_recent_turns`` pruning.

Acceptance criteria mapping (PRD §AC-01..AC-10):

    AC-01  list_tool_result_truncated        test_list_tool_result_truncated
    AC-02  tool_use_untouched                test_tool_use_untouched
    AC-03  plain_text_untouched              test_plain_text_untouched
    AC-04  mixed_blocks                      test_mixed_blocks
    AC-05  serialized_json_content           test_serialized_json_content
    AC-06  token_savings_ratio 30..50%       test_token_savings_ratio
    AC-07  zero-regression                   test_reference_preserved
    AC-08  robustness_garbage_input          test_robustness_garbage_input
    AC-09  debug_log_emitted                 test_debug_log_emitted
    AC-10  placeholder_format                test_placeholder_format

Additional coverage:

    test_stats_singleton                     module-import-path determinism
    test_short_tool_result_not_truncated     ≤ 500 bytes left untouched
    test_json_load_failure_fallback          malformed JSON-like str → str
"""
from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import log as larkhelm_log
from larkhelm.log import (
    _TOOL_RESULT_PLACEHOLDER_FMT,
    _get_recent_turns,
    _maybe_rehydrate_json,
    _prune_content,
    _pruning_stats,
)


_PLACEHOLDER_RE = re.compile(r"^\[tool_result truncated — \d+ bytes\]$")


def _big_text(n_bytes: int) -> str:
    """ASCII filler of exactly n_bytes bytes (UTF-8)."""
    return "A" * n_bytes


# ── 1. Pure-function unit tests (no I/O) ─────────────────────────────────


class TestPrunePure(unittest.TestCase):
    """Direct ``_prune_content`` semantics — no log writes, no fixtures."""

    # AC-01
    def test_list_tool_result_truncated(self):
        big = _big_text(2000)
        content = [
            {"type": "tool_result", "tool_use_id": "u1", "content": big},
        ]
        pruned = _prune_content(content)
        self.assertIsNot(pruned, content)
        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0]["type"], "tool_result")
        self.assertEqual(pruned[0]["tool_use_id"], "u1")
        self.assertRegex(pruned[0]["content"], _PLACEHOLDER_RE)

    # AC-02
    def test_tool_use_untouched(self):
        content = [
            {
                "type": "tool_use",
                "id": "u1",
                "name": "Read",
                "input": {"path": "/x", "blob": _big_text(2000)},
            }
        ]
        pruned = _prune_content(content)
        # tool_use must not be replaced — identity preserved
        self.assertIs(pruned, content)
        self.assertEqual(pruned[0]["input"]["blob"], content[0]["input"]["blob"])

    # AC-03
    def test_plain_text_untouched(self):
        content = "hello world — this is plain dialog"
        self.assertIs(_prune_content(content), content)

    # AC-04
    def test_mixed_blocks(self):
        big = _big_text(1500)
        content = [
            {"type": "text", "text": "preface line"},
            {"type": "tool_use", "id": "u1", "name": "Bash", "input": {"cmd": "ls"}},
            {"type": "tool_result", "tool_use_id": "u1", "content": big},
            {"type": "text", "text": "trailing note"},
        ]
        pruned = _prune_content(content)
        self.assertIsNot(pruned, content)  # at least one replacement
        self.assertEqual(pruned[0], content[0])  # text untouched (same dict ref)
        self.assertIs(pruned[1], content[1])     # tool_use identity preserved
        self.assertRegex(pruned[2]["content"], _PLACEHOLDER_RE)
        self.assertEqual(pruned[3], content[3])

    # AC-05
    def test_serialized_json_content(self):
        big = _big_text(2000)
        inner = [{"type": "tool_result", "tool_use_id": "u1", "content": big}]
        as_str = json.dumps(inner, ensure_ascii=False)
        # serialized form is what JSONL ``content`` field actually carries
        pruned_str = _prune_content(as_str)
        self.assertIsInstance(pruned_str, str)
        # The pruned string is shorter than the original
        self.assertLess(len(pruned_str), len(as_str))
        decoded = json.loads(pruned_str)
        self.assertRegex(decoded[0]["content"], _PLACEHOLDER_RE)

    # AC-07
    def test_reference_preserved(self):
        # Plain string
        s = "ok"
        self.assertIs(_prune_content(s), s)
        # List with only short tool_results — must keep identity
        small = [{"type": "tool_result", "tool_use_id": "u", "content": "ok"}]
        self.assertIs(_prune_content(small), small)
        # Nested dict with no tool_result at all
        d = {"role": "user", "content": "hi"}
        self.assertIs(_prune_content(d), d)

    # AC-08
    def test_robustness_garbage_input(self):
        # All of these must return *something* (the input itself) without
        # raising — even though they aren't sensible content shapes.
        for garbage in (None, 123, 1.5, True, False, b"\x00\x01", [1, 2, 3], (1, 2)):
            try:
                _prune_content(garbage)
            except Exception as e:  # pragma: no cover
                self.fail(f"_prune_content raised on {garbage!r}: {e}")
        # dict missing 'content' field
        d = {"type": "tool_result", "tool_use_id": "u"}
        self.assertIs(_prune_content(d), d)
        # 10-level deep nesting must not raise nor recurse forever
        nested = {"l": 0}
        cur = nested
        for i in range(1, 12):
            cur["next"] = {"l": i}
            cur = cur["next"]
        # Deeply nested with a big tool_result at the bottom — exceeds
        # _PRUNE_MAX_DEPTH but must NOT raise; deep node is left as-is.
        cur["type"] = "tool_result"
        cur["content"] = _big_text(2000)
        try:
            result = _prune_content(nested)
        except Exception as e:  # pragma: no cover
            self.fail(f"deep nesting raised: {e}")
        self.assertIsNotNone(result)

    # AC-10
    def test_placeholder_format(self):
        # Format string carries an em-dash, not a plain hyphen.
        self.assertIn("—", _TOOL_RESULT_PLACEHOLDER_FMT)
        # Formatting in plausible call shape produces the expected output.
        s = _TOOL_RESULT_PLACEHOLDER_FMT.format(n=1234)
        self.assertRegex(s, _PLACEHOLDER_RE)
        # The placeholder must NOT have thousands separators.
        self.assertEqual(s, "[tool_result truncated — 1234 bytes]")

    def test_short_tool_result_not_truncated(self):
        small = "x" * 100  # ≤ 500 bytes
        content = [{"type": "tool_result", "tool_use_id": "u", "content": small}]
        pruned = _prune_content(content)
        self.assertIs(pruned, content)
        self.assertEqual(pruned[0]["content"], small)

    def test_json_load_failure_fallback(self):
        # Long string, looks JSON-ish but is malformed → return original.
        s = "[not valid json " + "x" * 80
        self.assertIs(_prune_content(s), s)
        # And the rehydration helper itself returns None.
        self.assertIsNone(_maybe_rehydrate_json(s))


class TestRehydrationHeuristic(unittest.TestCase):
    """``_maybe_rehydrate_json`` gates JSON parsing on a cheap heuristic."""

    def test_short_string_skipped(self):
        # Below 50 chars → never attempts parse, returns None.
        self.assertIsNone(_maybe_rehydrate_json('[1,2,3]'))

    def test_non_json_prefix_skipped(self):
        s = "hello world " + "x" * 100
        self.assertIsNone(_maybe_rehydrate_json(s))

    def test_valid_json_parsed(self):
        s = json.dumps([{"k": "v"}] * 6)
        self.assertEqual(_maybe_rehydrate_json(s), [{"k": "v"}] * 6)

    def test_non_str_returns_none(self):
        self.assertIsNone(_maybe_rehydrate_json(123))
        self.assertIsNone(_maybe_rehydrate_json(None))


# ── 2. Stats singleton determinism ────────────────────────────────────────


class TestStatsSingleton(unittest.TestCase):
    """The module-level ``_pruning_stats`` must be the same object across
    every import path (design §8 risk #5)."""

    def test_stats_singleton(self):
        from larkhelm import log as via_pkg
        from larkhelm.log import _pruning_stats as via_attr
        self.assertIs(via_pkg._pruning_stats, _pruning_stats)
        self.assertIs(via_attr, _pruning_stats)

    def test_zero_byte_record_ignored(self):
        # The 0-byte guard in PruningStats.record() must drop the sample.
        snapshot_window = _pruning_stats.summary()["window"]
        _pruning_stats.record(0, 0)
        self.assertEqual(_pruning_stats.summary()["window"], snapshot_window)


# ── 3. Fixture-driven _get_recent_turns tests ─────────────────────────────


class _PruningIOBase(unittest.TestCase):
    """Shared fixture: tmp LOG_DIR + DEBUG_LOG; patches _cfg on log module."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="larkhelm_test_prune_"))
        self.log_dir = self.tmp / "logs"
        self.debug_log = self.tmp / "larkhelm.log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log.write_text("", encoding="utf-8")
        self.cfg_patches = [
            patch.object(larkhelm_log._cfg, "LOG_DIR", self.log_dir, create=True),
            patch.object(larkhelm_log._cfg, "DEBUG_LOG", self.debug_log, create=True),
        ]
        for p in self.cfg_patches:
            p.start()
        # Reset the singleton's deque between tests so each test sees a
        # clean ring buffer for its assertions.
        with _pruning_stats.lock:
            _pruning_stats.calls.clear()

    def tearDown(self):
        for p in self.cfg_patches:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_jsonl(self, records):
        with (self.log_dir / "all.jsonl").open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


class TestGetRecentTurnsPruning(_PruningIOBase):

    # AC-06
    def test_token_savings_ratio(self):
        """Construct a fixture where pruning saves between 30% and 50%
        of the recent-turns byte budget (PRD G2)."""
        chat = "chat_AC06"
        big = _big_text(1200)  # > 500 bytes → triggers placeholder
        records = []
        # 8 plain-dialog turns: stays untouched (anchors the denominator
        # so the ratio lands in 30-50% rather than the saturation regime).
        for i in range(8):
            records.append({
                "ts": f"2026-01-01T00:00:0{i}",
                "chat_id": chat, "role": "user" if i % 2 == 0 else "assistant",
                "content": "plain dialog " * 25,  # ~325 bytes each
                "model": "claude",
            })
        # 2 turns whose content is a serialized list with a big tool_result.
        for i in range(2):
            inner = [{
                "type": "tool_result", "tool_use_id": f"u{i}", "content": big,
            }]
            records.append({
                "ts": f"2026-01-01T00:00:1{i}",
                "chat_id": chat, "role": "assistant",
                "content": json.dumps(inner, ensure_ascii=False),
                "model": "claude",
            })
        self._write_jsonl(records)

        result = _get_recent_turns(chat, max_turns=10, max_chars=100_000)
        self.assertIn("[Recent conversation]", result)

        summary = _pruning_stats.summary()
        self.assertGreater(summary["before_sum"], 0)
        saved = summary["before_sum"] - summary["after_sum"]
        ratio = saved / summary["before_sum"]
        self.assertGreaterEqual(ratio, 0.30,
                                f"savings ratio {ratio:.2f} < 0.30")
        self.assertLessEqual(ratio, 0.50,
                             f"savings ratio {ratio:.2f} > 0.50")

    # AC-09
    def test_debug_log_emitted(self):
        chat = "chat_AC09"
        big = _big_text(2000)
        records = [
            {
                "ts": "2026-01-01T00:00:00", "chat_id": chat, "role": "user",
                "content": "tell me", "model": "claude",
            },
            {
                "ts": "2026-01-01T00:00:01", "chat_id": chat, "role": "assistant",
                "content": json.dumps([{
                    "type": "tool_result", "tool_use_id": "u1", "content": big,
                }], ensure_ascii=False),
                "model": "claude",
            },
        ]
        self._write_jsonl(records)
        _ = _get_recent_turns(chat, max_turns=5, max_chars=100_000)
        debug_contents = self.debug_log.read_text(encoding="utf-8")
        self.assertRegex(
            debug_contents,
            r"\[Log\] _get_recent_turns pruned chat=" + re.escape(chat[:8])
            + r" blocks=\d+ saved=\d+",
        )

    def test_zero_regression_on_plain_dialog(self):
        """Plain user/assistant text gets no pruning, no debug log, and the
        dialog substring is preserved verbatim (subject to 400-char cap)."""
        chat = "chat_plain"
        records = [
            {
                "ts": "2026-01-01T00:00:00", "chat_id": chat, "role": "user",
                "content": "what is 2 + 2?", "model": "claude",
            },
            {
                "ts": "2026-01-01T00:00:01", "chat_id": chat, "role": "assistant",
                "content": "It is 4.", "model": "claude",
            },
        ]
        self._write_jsonl(records)
        before_log = self.debug_log.read_text(encoding="utf-8")
        result = _get_recent_turns(chat, max_turns=5, max_chars=100_000)
        after_log = self.debug_log.read_text(encoding="utf-8")
        self.assertIn("User: what is 2 + 2?", result)
        self.assertIn("Assistant: It is 4.", result)
        # No pruning → no debug log
        self.assertEqual(before_log, after_log)


# ── 4. Concurrency smoke test on PruningStats ─────────────────────────────


class TestPruningStatsThreadSafety(unittest.TestCase):
    """The single threading.Lock must protect the deque from races."""

    def test_concurrent_record_no_crash(self):
        stats = larkhelm_log.PruningStats(capacity=200)
        # Override deque maxlen to match capacity for this isolated instance.
        from collections import deque as _deque
        stats.calls = _deque(maxlen=200)

        def writer():
            for _ in range(500):
                stats.record(10, 5)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        s = stats.summary()
        self.assertGreater(s["window"], 0)
        self.assertLessEqual(s["window"], 200)


if __name__ == "__main__":
    unittest.main()
