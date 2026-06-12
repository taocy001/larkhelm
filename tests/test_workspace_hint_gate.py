"""
P3 — workspace-hint gate + passive phrasing unit tests

Covers AC-02 (keyword gate) and AC-03 (metric outcomes) from
`.crew_workspace/prd.md`. Exercises `_build_workspace_hint` directly so
no Feishu SDK / runtime bridge is needed.

MCR update: pins the tightened gate regex (English keywords with \\b word
boundaries; high-frequency code-task substrings removed) and the file
listing cap (max 8, newest-first, >30-day files dropped, "等 N 个文件"
overflow suffix).
"""
import atexit
import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# ── Initialize config so _cfg attribute lookups succeed ──────────────
_TMP = tempfile.mkdtemp(prefix="larkhelm_wshint_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)

_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

os.environ.setdefault("LARKHELM_TEST_MODE", "1")
import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.handlers import _message as _msg
from larkhelm import metrics as _metrics


def _make_workspace(parent: Path, files: list[str]) -> Path:
    ws = parent / ".crew_workspace"
    ws.mkdir()
    for name in files:
        (ws / name).write_text("payload")
    return ws


class TestBuildWorkspaceHint(unittest.TestCase):
    """AC-02 / AC-03 — gate + outcome metric label per branch."""

    def setUp(self) -> None:
        # Snapshot the gate flag so tests can't bleed into each other.
        self._gate_snap = bool(
            getattr(_cfg, "WORKSPACE_HINT_KEYWORD_GATE", False)
        )

    def tearDown(self) -> None:
        _cfg.WORKSPACE_HINT_KEYWORD_GATE = self._gate_snap

    def test_default_gate_off_injects_when_files_exist(self):
        """gate=False + prd.md present → passive injection."""
        with tempfile.TemporaryDirectory() as td:
            _make_workspace(Path(td), ["prd.md", "tasks.json"])
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = False
                prefix, outcome = _msg._build_workspace_hint("c1", "hello")
        self.assertTrue(prefix, "non-empty prefix on injection branch")
        self.assertIn("[工作区]", prefix)
        self.assertIn("prd.md", prefix)
        self.assertIn("如果与本次问题相关", prefix,
            "P3 REQ-01 passive phrasing must be present",
        )
        self.assertEqual(outcome, "injected_passive")

    def test_gate_on_keyword_missing_skips(self):
        """gate=True + non-matching message → empty prefix, skipped_by_gate."""
        with tempfile.TemporaryDirectory() as td:
            _make_workspace(Path(td), ["prd.md"])
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = True
                prefix, outcome = _msg._build_workspace_hint("c1", "现在几点")
        self.assertEqual(prefix, "")
        self.assertEqual(outcome, "skipped_by_gate")

    def test_gate_on_keyword_present_injects(self):
        """gate=True + matching message (prd/design/任务/...) → injected_passive."""
        with tempfile.TemporaryDirectory() as td:
            _make_workspace(Path(td), ["prd.md", "design.md"])
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = True
                # 'prd' must match case-insensitively (design.md §3.3)
                prefix, outcome = _msg._build_workspace_hint("c1", "帮我看 PRD")
        self.assertTrue(prefix)
        self.assertIn("design.md", prefix)
        self.assertEqual(outcome, "injected_passive")

    def test_gate_on_chinese_keyword(self):
        """gate=True + Chinese keyword '任务' → injected_passive."""
        with tempfile.TemporaryDirectory() as td:
            _make_workspace(Path(td), ["tasks.json"])
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = True
                prefix, outcome = _msg._build_workspace_hint("c1", "看一下任务列表")
        self.assertTrue(prefix)
        self.assertEqual(outcome, "injected_passive")

    def test_gate_word_boundary_blocks_substrings(self):
        """English keywords need \\b boundaries: 'fixture' / 'unicode' /
        'qatar' / 'predesign' must NOT open the gate (MCR fix)."""
        for text in (
            "the fixture is broken",
            "unicode normalization question",
            "flights to qatar",
            "predesigned template please",
            "barcode scanner",
        ):
            with tempfile.TemporaryDirectory() as td:
                _make_workspace(Path(td), ["prd.md"])
                with patch.object(_msg, "_get_cwd", return_value=td):
                    _cfg.WORKSPACE_HINT_KEYWORD_GATE = True
                    prefix, outcome = _msg._build_workspace_hint("c1", text)
            self.assertEqual(prefix, "", f"gate leaked for: {text!r}")
            self.assertEqual(outcome, "skipped_by_gate")

    def test_gate_dropped_code_task_keywords_no_longer_match(self):
        """P2a code-task substrings (code/edit/fix/implement/refactor/
        debug) were removed from the gate vocabulary (MCR fix)."""
        for text in (
            "please fix this",
            "edit the file",
            "implement a parser",
            "refactor everything",
            "debug session",
            "write some code",
        ):
            with tempfile.TemporaryDirectory() as td:
                _make_workspace(Path(td), ["prd.md"])
                with patch.object(_msg, "_get_cwd", return_value=td):
                    _cfg.WORKSPACE_HINT_KEYWORD_GATE = True
                    prefix, outcome = _msg._build_workspace_hint("c1", text)
            self.assertEqual(prefix, "", f"gate matched dropped kw: {text!r}")
            self.assertEqual(outcome, "skipped_by_gate")

    def test_gate_whole_word_english_keywords_match(self):
        """Standalone workspace vocabulary still opens the gate."""
        for text in (
            "show me the workspace",
            "design doc update",
            "QA pass please",
            "crew status",
            "tasks overview",
            "review the plan",
        ):
            with tempfile.TemporaryDirectory() as td:
                _make_workspace(Path(td), ["prd.md"])
                with patch.object(_msg, "_get_cwd", return_value=td):
                    _cfg.WORKSPACE_HINT_KEYWORD_GATE = True
                    prefix, outcome = _msg._build_workspace_hint("c1", text)
            self.assertTrue(prefix, f"gate missed keyword in: {text!r}")
            self.assertEqual(outcome, "injected_passive")

    def test_listing_capped_at_8_newest_with_overflow_suffix(self):
        """>8 eligible files → only the 8 newest are listed (mtime desc)
        plus a '等 N 个文件' suffix for the remainder."""
        with tempfile.TemporaryDirectory() as td:
            names = [f"f{i:02d}.md" for i in range(12)]
            ws = _make_workspace(Path(td), names)
            now = time.time()
            # f00 is the newest, f11 the oldest (all within 30 days).
            for i, name in enumerate(names):
                ts = now - i * 60
                os.utime(ws / name, (ts, ts))
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = False
                prefix, outcome = _msg._build_workspace_hint("c1", "hello")
        self.assertEqual(outcome, "injected_passive")
        for name in names[:8]:
            self.assertIn(name, prefix)
        for name in names[8:]:
            self.assertNotIn(name, prefix)
        self.assertIn("等 4 个文件", prefix)
        # Newest-first ordering inside the listing.
        self.assertLess(prefix.index("f00.md"), prefix.index("f07.md"))

    def test_listing_no_overflow_suffix_at_or_below_cap(self):
        """≤8 files → every name listed, no '等 N 个文件' suffix."""
        with tempfile.TemporaryDirectory() as td:
            _make_workspace(Path(td), ["a.md", "b.json", "c.md"])
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = False
                prefix, outcome = _msg._build_workspace_hint("c1", "hello")
        self.assertEqual(outcome, "injected_passive")
        for name in ("a.md", "b.json", "c.md"):
            self.assertIn(name, prefix)
        self.assertNotIn("等", prefix.split("。")[0].split("：")[1])

    def test_files_older_than_30_days_excluded(self):
        """Files with mtime >30 days ago are dropped from the listing;
        if ALL files are stale the hint degrades to skipped_empty."""
        with tempfile.TemporaryDirectory() as td:
            ws = _make_workspace(Path(td), ["fresh.md", "stale.md"])
            old = time.time() - 31 * 86400
            os.utime(ws / "stale.md", (old, old))
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = False
                prefix, outcome = _msg._build_workspace_hint("c1", "hello")
        self.assertEqual(outcome, "injected_passive")
        self.assertIn("fresh.md", prefix)
        self.assertNotIn("stale.md", prefix)

        with tempfile.TemporaryDirectory() as td:
            ws = _make_workspace(Path(td), ["stale.md"])
            old = time.time() - 31 * 86400
            os.utime(ws / "stale.md", (old, old))
            with patch.object(_msg, "_get_cwd", return_value=td):
                _cfg.WORKSPACE_HINT_KEYWORD_GATE = False
                prefix, outcome = _msg._build_workspace_hint("c1", "hello")
        self.assertEqual(prefix, "")
        self.assertEqual(outcome, "skipped_empty")

    def test_empty_workspace_returns_skipped_empty(self):
        """No .crew_workspace/ → skipped_empty."""
        with tempfile.TemporaryDirectory() as td:
            with patch.object(_msg, "_get_cwd", return_value=td):
                prefix, outcome = _msg._build_workspace_hint("c1", "anything")
        self.assertEqual(prefix, "")
        self.assertEqual(outcome, "skipped_empty")

    def test_workspace_with_only_checkpoint_returns_skipped_empty(self):
        """Only crew_checkpoint.json present → it's excluded → skipped_empty."""
        with tempfile.TemporaryDirectory() as td:
            _make_workspace(Path(td), ["crew_checkpoint.json"])
            with patch.object(_msg, "_get_cwd", return_value=td):
                prefix, outcome = _msg._build_workspace_hint("c1", "anything")
        self.assertEqual(prefix, "")
        self.assertEqual(outcome, "skipped_empty")

    def test_oserror_during_iterdir_degrades_to_skipped_empty(self):
        """A ``PermissionError`` (subclass of OSError) raised by
        ``Path.iterdir()`` must NOT propagate — the function degrades to
        ``("", "skipped_empty")`` so a flaky ``.crew_workspace/`` directory
        cannot break message handling.

        Closes the §6 review gap: the broad ``except OSError`` fallback
        had no regression test.
        """
        with tempfile.TemporaryDirectory() as td:
            # Workspace exists so the ``is_dir()`` short-circuit doesn't
            # fire — execution reaches ``iterdir()`` which we make raise.
            _make_workspace(Path(td), ["prd.md"])
            original_iterdir = Path.iterdir

            def _raising_iterdir(self):
                # Only the .crew_workspace directory should raise; let
                # tempfile / setup machinery use the real impl.
                if self.name == ".crew_workspace":
                    raise PermissionError("simulated EACCES")
                return original_iterdir(self)

            with patch.object(_msg, "_get_cwd", return_value=td), \
                    patch.object(Path, "iterdir", _raising_iterdir):
                prefix, outcome = _msg._build_workspace_hint("c1", "anything")
        self.assertEqual(prefix, "")
        self.assertEqual(outcome, "skipped_empty")


class TestMetricCounterRegistered(unittest.TestCase):
    """AC-03 — `inc_workspace_hint` accepts every documented outcome
    without raising, regardless of whether prometheus-client is loaded."""

    def test_all_four_outcomes_never_raise(self):
        for outcome in (
            "injected_passive",
            "injected_active_legacy",
            "skipped_by_gate",
            "skipped_empty",
        ):
            try:
                _metrics.inc_workspace_hint(outcome)
            except Exception as e:  # pragma: no cover — defensive
                self.fail(f"inc_workspace_hint({outcome!r}) raised: {e}")


if __name__ == "__main__":
    unittest.main()
