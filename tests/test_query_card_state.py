"""Unit tests for QueryCardState (S2 / P1 #8).

Extracting the streaming-card state machine from ``_do_query`` into a
self-contained class is meaningful only if the class is actually
unit-testable. These tests verify the core invariants without spinning
up the chat lock, backend resolution, or Feishu API stack — the very
testability that motivated the split.
"""
import threading
import time
import unittest

from larkhelm.handlers._query_card_state import (
    QueryCardState,
    ToolRecord,
    _fmt_desc,
)


def _new_state(model_name: str = "Claude") -> QueryCardState:
    return QueryCardState(chat_id="chat-x", model_name=model_name,
                          start_time=time.time())


class ScalarStateTests(unittest.TestCase):
    """Scalar mutator + reader contract."""

    def test_initial_state(self):
        s = _new_state()
        dirty, cursor, text, last_body, _, in_bg = s.get_state_snapshot()
        self.assertFalse(dirty)
        self.assertEqual(cursor, 0)
        self.assertEqual(text, "")
        self.assertEqual(last_body, "")
        self.assertFalse(in_bg)

    def test_set_current_text_marks_dirty(self):
        s = _new_state()
        s.set_current_text("hello")
        self.assertEqual(s.current_text, "hello")
        self.assertTrue(s.dirty)

    def test_set_in_background_marks_dirty(self):
        s = _new_state()
        self.assertFalse(s.in_background)
        s.set_in_background(True)
        self.assertTrue(s.in_background)
        self.assertTrue(s.dirty)

    def test_tick_cursor_wraps_around(self):
        s = _new_state()
        from larkhelm.handlers._query_card_state import CURSOR_FRAMES
        for i in range(len(CURSOR_FRAMES) * 2 + 1):
            s.tick_cursor()
        # After N*2+1 ticks we should be at (N*2+1) % N == 1
        _, cursor, *_ = s.get_state_snapshot()
        self.assertEqual(cursor, (len(CURSOR_FRAMES) * 2 + 1) % len(CURSOR_FRAMES))

    def test_set_dirty_clears_flag(self):
        s = _new_state()
        s.set_current_text("foo")
        self.assertTrue(s.dirty)
        s.set_dirty(False)
        self.assertFalse(s.dirty)

    def test_update_heartbeat_moves_timestamp_forward(self):
        s = _new_state()
        t0 = s.last_heartbeat
        time.sleep(0.01)
        s.update_heartbeat()
        self.assertGreater(s.last_heartbeat, t0)

    def test_update_model_name(self):
        s = _new_state(model_name="Claude")
        s.update_model_name("Kimi")
        self.assertEqual(s.model_name, "Kimi")


class ToolTrackingTests(unittest.TestCase):
    """The two callback paths (on_tool / on_tool_result) and snapshot helpers."""

    def test_on_tool_creates_active_record(self):
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        self.assertEqual(len(s.active_tools), 1)
        rec = s.active_tools["tool-1"]
        self.assertEqual(rec.name, "Read")
        self.assertEqual(rec.desc, "/tmp/x")
        self.assertTrue(s.dirty)

    def test_on_tool_flushes_previous_active(self):
        # If a second on_tool fires before on_tool_result, the previous
        # active tool should land in completed_tools (without a result).
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        s.on_tool("Write", "/tmp/y", "tool-2")
        self.assertEqual(len(s.active_tools), 1)
        self.assertEqual(list(s.active_tools.keys()), ["tool-2"])
        completed = s.snapshot_completed_tools()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].name, "Read")
        self.assertEqual(completed[0].result, "")

    def test_on_tool_result_moves_active_to_completed(self):
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        s.on_tool_result("tool-1", "file contents", False, 0.42)
        self.assertEqual(len(s.active_tools), 0)
        completed = s.snapshot_completed_tools()
        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].result, "file contents")
        self.assertEqual(completed[0].elapsed, 0.42)
        self.assertFalse(completed[0].is_error)

    def test_on_tool_result_for_unknown_id_is_noop(self):
        s = _new_state()
        s.on_tool_result("nonexistent", "result", False, 1.0)
        self.assertEqual(len(s.active_tools), 0)
        self.assertEqual(s.n_completed_tools(), 0)

    def test_long_result_stashes_full_result_snippet(self):
        s = _new_state()
        long_result = "x" * 1000
        s.on_tool("Bash", "ls", "tool-1")
        s.on_tool_result("tool-1", long_result, False, 0.1)
        completed = s.snapshot_completed_tools()
        # Above 200 chars triggers the larger full_result snapshot
        self.assertEqual(completed[0].full_result, long_result[:5000])

    def test_short_result_skips_full_result(self):
        s = _new_state()
        s.on_tool("Bash", "ls", "tool-1")
        s.on_tool_result("tool-1", "ok", False, 0.1)
        completed = s.snapshot_completed_tools()
        self.assertEqual(completed[0].full_result, "")

    def test_snapshot_active_tools_as_completed(self):
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        s.on_tool("Read", "/tmp/y", "tool-2")
        # Without flushing, completed has just one (from on_tool's auto-flush)
        self.assertEqual(s.n_completed_tools(), 1)
        s.snapshot_active_tools_as_completed()
        # Now both tools should be in completed
        self.assertEqual(s.n_completed_tools(), 2)
        self.assertEqual(len(s.active_tools), 0)

    def test_on_text_sets_current_text(self):
        # on_text is a callback shim that mirrors set_current_text.
        s = _new_state()
        s.on_text("streaming output")
        self.assertEqual(s.current_text, "streaming output")
        self.assertTrue(s.dirty)

    def test_on_text_accepts_status_kwarg(self):
        # Backend runners pass a `status` kwarg; must be tolerated.
        s = _new_state()
        s.on_text("hi", status="typing")  # must not raise


class RenderTests(unittest.TestCase):
    """``render_body`` purity and title selection."""

    def test_idle_state_renders_thinking(self):
        s = _new_state(model_name="Claude")
        r = s.render_body()
        self.assertIn("思考中", r.title)
        self.assertIn("Claude", r.title)
        self.assertEqual(r.response_md, "> 正在思考...")
        self.assertIsNone(r.tools_md)

    def test_streaming_text_renders_typing_title(self):
        s = _new_state()
        s.on_text("Hello")
        r = s.render_body()
        self.assertIn("回应中", r.title)
        # Cursor frame is appended; verify the prefix
        self.assertTrue(r.response_md.startswith("Hello"))

    def test_active_tool_renders_tool_title(self):
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        r = s.render_body()
        self.assertIn("工具调用中", r.title)
        self.assertIn("Read", r.tools_md)
        # Active tool icon
        self.assertIn("🔧", r.tools_md)

    def test_completed_tool_with_result(self):
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        s.on_tool_result("tool-1", "ok", False, 0.1)
        r = s.render_body()
        self.assertIn("✓", r.tools_md)
        self.assertIn("Read", r.tools_md)

    def test_completed_tool_with_error(self):
        s = _new_state()
        s.on_tool("Read", "/tmp/x", "tool-1")
        s.on_tool_result("tool-1", "ENOENT", True, 0.1)
        r = s.render_body()
        self.assertIn("✗", r.tools_md)

    def test_background_prefix_appears_when_in_bg(self):
        s = _new_state()
        s.on_text("output")
        s.set_in_background(True)
        r = s.render_body()
        self.assertIn("后台·", r.title)

    def test_tools_history_cap_truncates_with_marker(self):
        from larkhelm.handlers._query_card_state import TOOL_HISTORY_CAP
        s = _new_state()
        # Generate TOOL_HISTORY_CAP+3 completed tools so we get a hidden count
        for i in range(TOOL_HISTORY_CAP + 3):
            s.on_tool(f"T{i}", "", f"id-{i}")
            s.on_tool_result(f"id-{i}", "ok", False, 0.01)
        r = s.render_body()
        # Marker "+3 条更早记录已隐藏" appears
        self.assertIn("条更早记录已隐藏", r.tools_md)
        self.assertIn(f"+{3}", r.tools_md)

    def test_render_is_pure(self):
        # Calling render twice without state change must produce identical
        # output. Cursor index is NOT bumped by render (only by tick_cursor).
        s = _new_state()
        s.on_text("hello")
        r1 = s.render_body()
        r2 = s.render_body()
        # Title contains elapsed time; one render may produce different
        # title if the wall clock advanced — accept that for title but
        # check response_md is stable.
        self.assertEqual(r1.response_md, r2.response_md)
        self.assertEqual(r1.tools_md, r2.tools_md)


class PushDecisionTests(unittest.TestCase):
    """``should_push`` / ``mark_pushed`` delta detection."""

    def test_first_render_should_push_when_dirty(self):
        s = _new_state()
        s.on_text("hi")  # sets dirty
        r = s.render_body()
        need, combined = s.should_push(r)
        self.assertTrue(need)
        self.assertIn("hi", combined)

    def test_force_overrides_clean(self):
        s = _new_state()
        s.on_text("hi")
        r = s.render_body()
        # Mark as pushed so dirty=False
        _, combined = s.should_push(r)
        s.mark_pushed(combined)
        self.assertFalse(s.dirty)
        # force=True still triggers push
        need, _ = s.should_push(r, force=True)
        self.assertTrue(need)

    def test_unchanged_body_skips_push(self):
        s = _new_state()
        s.on_text("hi")
        r = s.render_body()
        _, combined = s.should_push(r)
        s.mark_pushed(combined)
        # Same render again, dirty is now False
        need, _ = s.should_push(r)
        self.assertFalse(need)

    def test_changed_body_triggers_push_even_after_mark(self):
        s = _new_state()
        s.on_text("hi")
        r1 = s.render_body()
        _, c1 = s.should_push(r1)
        s.mark_pushed(c1)
        # New text → new body
        s.on_text("hi there")
        r2 = s.render_body()
        need, _ = s.should_push(r2)
        self.assertTrue(need)


class ConcurrencyTests(unittest.TestCase):
    """Lock-acquiring methods must be safe to call from multiple threads."""

    def test_concurrent_on_text_no_lost_updates(self):
        # The serialisation guarantee: every set_current_text completes
        # atomically and the last writer wins; no torn reads of the
        # text field.
        s = _new_state()
        N = 200

        def writer(idx: int):
            for i in range(N):
                s.set_current_text(f"thread-{idx}-iter-{i}")

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        # Final text is some valid winner from one of the threads —
        # parse it back and verify format integrity
        final = s.current_text
        self.assertTrue(final.startswith("thread-"))
        self.assertIn("-iter-", final)

    def test_concurrent_tool_events_no_race(self):
        # Two threads firing on_tool / on_tool_result must produce a
        # consistent state (sum of active + completed == sum of starts).
        s = _new_state()
        N = 100

        def runner(prefix: str):
            for i in range(N):
                tid = f"{prefix}-{i}"
                s.on_tool(f"T{i}", "", tid)
                s.on_tool_result(tid, "ok", False, 0.001)

        threads = [threading.Thread(target=runner, args=(p,))
                   for p in ("a", "b", "c")]
        for t in threads: t.start()
        for t in threads: t.join()
        # After all threads finish: zero active (every tid resolved), 3*N completed
        self.assertEqual(len(s.active_tools), 0)
        self.assertEqual(s.n_completed_tools(), 3 * N)


class FmtDescTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(_fmt_desc(""), "")

    def test_single_line_inline(self):
        # Single-line description → inline code with leading "  \n"
        r = _fmt_desc("hello")
        self.assertIn("`hello`", r)
        self.assertTrue(r.startswith("  \n`"))

    def test_multiline_fenced(self):
        # Multi-line description → fenced code block
        r = _fmt_desc("line1\nline2")
        self.assertIn("```", r)
        self.assertIn("line1\nline2", r)


if __name__ == "__main__":
    unittest.main()
