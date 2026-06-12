"""Coverage for ``memory._aggregate_memory_observation`` + capacity meter.

Tests ride the PRD AC-01..AC-11 acceptance criteria for the ``/memory observe``
feature. Uses fixture files (not LLM mocks) so the observe path is exercised
purely as I/O + string parsing, matching the design's "pure function" goal.
"""
from __future__ import annotations

import shutil
import tempfile
import timeit
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm import memory as larkhelm_memory
from larkhelm import log as larkhelm_log


def _write_md(path: Path, body: str, *, updated_at: str | None = None,
              extra_fm: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fm_lines = ["---"]
    if updated_at:
        fm_lines.append(f'updated_at: "{updated_at}"')
    if extra_fm:
        fm_lines.append(extra_fm.rstrip("\n"))
    fm_lines.append("---")
    path.write_text("\n".join(fm_lines) + "\n\n" + body, encoding="utf-8")


class _ObserveTestBase(unittest.TestCase):
    """Shared scaffolding: tmp DEBUG_LOG, MEMORY_HOME_DIR, LOG_DIR."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="larkhelm_test_observe_"))
        self.memory_dir = self.tmp / "memory"
        self.log_dir = self.tmp / "logs"
        self.debug_log = self.tmp / "larkhelm.log"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log.write_text("", encoding="utf-8")
        # _cfg is the same module object in memory.py and log.py — patch once
        # at the module level so per-test inner patches stack predictably.
        self.cfg_patches = [
            patch.object(larkhelm_memory, "MEMORY_HOME_DIR", self.memory_dir),
            patch.object(larkhelm_memory._cfg, "DEBUG_LOG", self.debug_log, create=True),
            patch.object(larkhelm_memory._cfg, "LOG_DIR", self.log_dir, create=True),
            patch.object(larkhelm_memory._cfg, "DATA_DIR", self.tmp, create=True),
            # _get_cwd is consulted for the project layer; isolate it so tests
            # don't pull in real per-chat state.
            patch("larkhelm.chat_state._get_cwd", return_value=str(self.tmp / "proj")),
        ]
        for p in self.cfg_patches:
            p.start()
        # _load_md_body uses an mtime-keyed in-process cache; clear it so the
        # fixture rewrites between tests are observed.
        larkhelm_memory._mem_body_cache.clear()
        # _query_sender_open_id is a process-wide ContextVar that
        # ``handlers/_message.handle_message`` ``.set()``s without saving a
        # reset Token. In production each asyncio task gets its own context
        # copy so this is harmless; in pytest every test shares the main
        # context and an earlier ``test_file_message`` run leaks
        # ``"user_open_id"`` here, making ``_global_memory_file`` resolve
        # ``chat_M``'s global memory to ``global_user_open_id.md`` (which
        # this fixture doesn't write) and silently dropping the global
        # layer. Reset it on setUp + restore on tearDown so the slot/legacy
        # path (P5-OPT6 flipped slot ON by default) sees the expected file.
        from larkhelm.memory import _query_sender_open_id
        self._sender_open_id_var = _query_sender_open_id
        self._sender_open_id_token = _query_sender_open_id.set("")

    def tearDown(self):
        for p in self.cfg_patches:
            p.stop()
        larkhelm_memory._mem_body_cache.clear()
        try:
            self._sender_open_id_var.reset(self._sender_open_id_token)
        except Exception:
            # ContextVar reset can fail if the Token belongs to a different
            # context (rare; happens when a test mutates the contextvars
            # state mid-run). Swallow — clearing the leak partially is still
            # better than leaving it untouched.
            pass
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestAggregateObservation(_ObserveTestBase):
    """AC-02 + AC-05: structure + counter correctness across fixtures."""

    def test_empty_logs_returns_zero_counters(self):
        """No DEBUG_LOG matches → all counts == 0, ratios == 0.0."""
        r = larkhelm_memory._aggregate_memory_observation("chat_X")
        self.assertEqual(
            {"layers", "recent_window", "fallback", "last_successful_update",
             "chat_id", "cwd", "trends", "pruning"},
            set(r.keys()),
        )
        self.assertEqual(r["recent_window"]["unchanged_count"], 0)
        self.assertEqual(r["fallback"]["count"], 0)
        self.assertEqual(r["recent_window"]["unchanged_ratio"], 0.0)
        self.assertFalse(r["recent_window"]["unavailable"])
        self.assertFalse(r["fallback"]["unavailable"])

    def test_layers_dimensions(self):
        """Each layer must report chars/max_chars/pct/near_limit/updated_at."""
        r = larkhelm_memory._aggregate_memory_observation("chat_X")
        for name in ("global", "project", "session"):
            with self.subTest(layer=name):
                layer = r["layers"][name]
                self.assertEqual(
                    {"chars", "max_chars", "pct", "near_limit", "updated_at"},
                    set(layer.keys()),
                )

    def test_counter_ratios_match_log(self):
        """AC-05: 5 cheap-fail + 20 unchanged + 75 saves → ratios match."""
        lines = []
        for i in range(5):
            lines.append(f"[12:00:{i:02d}] [Memory] cheap backend deepseek failed (RuntimeError: x); retrying")
        for i in range(20):
            lines.append(f"[12:01:{i:02d}] [Memory] project extract rejected non-useful output for /a")
        for i in range(75):
            lines.append(f"[12:02:{i:02d}] [Memory] saved session_chat_X.md ({100 + i} chars)")
        self.debug_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

        r = larkhelm_memory._aggregate_memory_observation("chat_X")
        self.assertEqual(r["fallback"]["count"], 5)
        self.assertEqual(r["recent_window"]["unchanged_count"], 20)
        # unchanged_ratio = 20 / (20 + 75) = 0.2105…
        self.assertAlmostEqual(r["recent_window"]["unchanged_ratio"], 0.2105, places=2)
        # fallback ratio = 5 / (5 + 75) = 0.0625
        self.assertAlmostEqual(r["fallback"]["ratio"], 0.0625, places=3)
        self.assertEqual(r["fallback"]["last_ts"], "12:00:04")

    def test_layer_chars_and_updated_at_read_from_file(self):
        """Layer chars + updated_at must reflect the on-disk fixture."""
        _write_md(
            self.memory_dir / "session_chat_Y.md",
            "x" * 1850,
            updated_at="2026-05-12T03:14:22",
            extra_fm="chat_id: chat_Y\nturns: 3\nversion: 1\n",
        )
        r = larkhelm_memory._aggregate_memory_observation("chat_Y")
        s = r["layers"]["session"]
        self.assertEqual(s["chars"], 1850)
        self.assertEqual(s["max_chars"], larkhelm_memory.SESSION_MAX_CHARS)
        self.assertGreaterEqual(s["pct"], 90)
        self.assertTrue(s["near_limit"])
        self.assertEqual(s["updated_at"], "2026-05-12T03:14:22")
        # last_successful_update falls back to session frontmatter
        self.assertEqual(r["last_successful_update"], "2026-05-12T03:14:22")


class TestMeterInjection(_ObserveTestBase):
    """P5-OPT1: meter line is **never** injected into ``get_memory_context``.

    The meter sat on the second line of every layer; when session_n grew by
    a single char the meter rotated and busted Anthropic prompt-cache prefix
    for the entire system prompt. The ``/memory observe`` card still calls
    ``_layer_meter_line`` directly via ``_aggregate_memory_observation``, so
    the per-layer capacity readout for humans is unaffected.

    """

    def _setup_layers(self, *, global_n: int, project_n: int, session_n: int):
        cwd = str(self.tmp / "proj")
        Path(cwd).mkdir(parents=True, exist_ok=True)

        # global keyed by open_id — register it in chat_state
        from larkhelm import chat_state
        chat_state._chat_state_store.setdefault("chat_M", {})["sender_open_id"] = "ou_test"

        if global_n > 0:
            _write_md(self.memory_dir / "global_ou_test.md", "g" * global_n,
                      updated_at="2026-05-10T10:00:00")
        if project_n > 0:
            import hashlib
            h = hashlib.md5(str(Path(cwd).resolve()).encode()).hexdigest()[:16]
            _write_md(self.memory_dir / f"project_{h}.md", "p" * project_n,
                      updated_at="2026-05-10T10:00:00",
                      extra_fm=f'cwd: "{Path(cwd).resolve()}"\n')
        if session_n > 0:
            _write_md(self.memory_dir / "session_chat_M.md", "s" * session_n,
                      updated_at="2026-05-10T10:00:00",
                      extra_fm="chat_id: chat_M\nturns: 3\nversion: 1\n")
        return cwd

    def test_50pct_no_meter_line(self):
        cwd = self._setup_layers(global_n=400, project_n=750, session_n=1000)
        ctx = larkhelm_memory.get_memory_context("chat_M", cwd=cwd)
        # Layer envelopes remain — only the meter line was stripped.
        self.assertIn("[GLOBAL MEMORY]", ctx)
        self.assertIn("[PROJECT MEMORY", ctx)
        self.assertIn("[SESSION MEMORY]", ctx)
        # No meter, no warning — body content still rendered.
        self.assertNotIn("[400/800 chars", ctx)
        self.assertNotIn("[750/1500 chars", ctx)
        self.assertNotIn("[1000/2000 chars", ctx)
        self.assertNotIn("chars, 50%]", ctx)
        self.assertNotIn("⚠️", ctx)
        # Body characters made it through (gggg... ssss... etc.)
        self.assertIn("g" * 400, ctx)
        self.assertIn("s" * 1000, ctx)

    def test_92pct_no_warning_in_injected_ctx(self):
        cwd = self._setup_layers(global_n=0, project_n=0, session_n=1850)
        ctx = larkhelm_memory.get_memory_context("chat_M", cwd=cwd)
        self.assertNotIn("[1850/2000 chars", ctx)
        self.assertNotIn("⚠️ near limit", ctx)
        self.assertIn("s" * 1850, ctx)

    def test_100pct_no_warning_in_injected_ctx(self):
        cwd = self._setup_layers(global_n=800, project_n=0, session_n=0)
        ctx = larkhelm_memory.get_memory_context("chat_M", cwd=cwd)
        self.assertNotIn("[800/800 chars", ctx)
        self.assertNotIn("⚠️ near limit", ctx)
        self.assertIn("g" * 800, ctx)

    def test_layer_meter_line_function_still_works(self):
        """``_layer_meter_line`` is still consumed by the observe card path."""
        self.assertEqual(
            larkhelm_memory._layer_meter_line(400, 800),
            "[400/800 chars, 50%]",
        )
        # near-limit warning preserved for the observe card formatter
        self.assertIn(
            "⚠️ near limit",
            larkhelm_memory._layer_meter_line(1850, 2000),
        )


class TestBudgetUnchanged(_ObserveTestBase):
    """AC-07: full three-layer load + tag overhead → still ≤ layer caps + slack."""

    def test_total_within_budget_after_meter_injection(self):
        cwd = str(self.tmp / "proj")
        Path(cwd).mkdir(parents=True, exist_ok=True)

        from larkhelm import chat_state
        chat_state._chat_state_store.setdefault("chat_B", {})["sender_open_id"] = "ou_b"

        _write_md(self.memory_dir / "global_ou_b.md", "g" * larkhelm_memory.GLOBAL_MAX_CHARS)
        import hashlib
        h = hashlib.md5(str(Path(cwd).resolve()).encode()).hexdigest()[:16]
        _write_md(self.memory_dir / f"project_{h}.md", "p" * larkhelm_memory.PROJECT_MAX_CHARS,
                  extra_fm=f'cwd: "{Path(cwd).resolve()}"\n')
        _write_md(self.memory_dir / "session_chat_B.md", "s" * larkhelm_memory.SESSION_MAX_CHARS,
                  extra_fm="chat_id: chat_B\nturns: 1\nversion: 1\n")

        ctx = larkhelm_memory.get_memory_context("chat_B", cwd=cwd)
        # Combined raw content = 800 + 1500 + 2000 = 4300 chars; allow ~90
        # chars/layer of slack for the [LAYER]…[/LAYER] tags + separators.
        raw_total = (larkhelm_memory.GLOBAL_MAX_CHARS
                     + larkhelm_memory.PROJECT_MAX_CHARS
                     + larkhelm_memory.SESSION_MAX_CHARS)
        upper_bound = raw_total + 3 * 90
        self.assertLessEqual(len(ctx), upper_bound,
                             f"len(ctx)={len(ctx)} exceeds {upper_bound}")
        # The content must still be visible — trimming hasn't replaced the
        # whole body with the meter line.
        self.assertIn("[GLOBAL MEMORY]", ctx)
        self.assertIn("[PROJECT MEMORY", ctx)
        self.assertIn("[SESSION MEMORY]", ctx)


class TestDebugLogMissing(_ObserveTestBase):
    """AC-09: DEBUG_LOG missing → unavailable flags, no exception."""

    def test_missing_debug_log_unavailable_flags(self):
        # Point DEBUG_LOG at a nonexistent path
        bogus = self.tmp / "does_not_exist.log"
        with patch.object(larkhelm_memory._cfg, "DEBUG_LOG", bogus, create=True):
            r = larkhelm_memory._aggregate_memory_observation("nope")
        self.assertTrue(r["recent_window"]["unavailable"])
        self.assertTrue(r["fallback"]["unavailable"])
        self.assertEqual(r["recent_window"]["reason"], "debug_log_missing")
        self.assertEqual(r["fallback"]["reason"], "debug_log_missing")


class TestObservePerformance(_ObserveTestBase):
    """AC-11: 1 MB jsonl + 2 MB debug log → 5-run average < 200 ms."""

    def test_performance_under_threshold(self):
        # 2 MB debug log
        line = "[12:34:56] [Memory] saved session_x.md (200 chars)\n"
        with self.debug_log.open("w", encoding="utf-8") as f:
            count = (2 * 1024 * 1024) // len(line) + 1
            for _ in range(count):
                f.write(line)
        # 1 MB jsonl
        import json
        with (self.log_dir / "all.jsonl").open("w", encoding="utf-8") as f:
            payload = json.dumps({"ts": "T", "chat_id": "p_chat", "role": "user",
                                  "content": "x" * 200, "model": "claude"}) + "\n"
            count = (1 * 1024 * 1024) // len(payload) + 1
            for _ in range(count):
                f.write(payload)

        durations = timeit.repeat(
            lambda: larkhelm_memory._aggregate_memory_observation("p_chat"),
            number=1, repeat=5,
        )
        avg = sum(durations) / len(durations)
        self.assertLess(avg, 0.2, f"observe avg {avg:.3f}s ≥ 0.2s budget")


if __name__ == "__main__":
    unittest.main(verbosity=2)
