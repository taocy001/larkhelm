"""Regression guard for the partial-delivery escape in ``_get_failed_dep``.

History
-------
Pre-fix behaviour: when an upstream agent ended in ``AgentStatus.FAILED``,
all its downstream agents got marked FAILED with ``error="upstream X
failed"``. This was the right rule for crashes that prevented any output
— but it also fired in a common scenario where the upstream had
**already atomically written its declared ``output_file``** to
``.crew_workspace/`` and then died for an unrelated reason (claude CLI
OOM-killed mid-stream after the Write tool committed, Feishu API SSL
EOF during the post-write doc-sync step, etc).

Real example from commit b3a116f's morning /dev run:
  * implementer attempt 1: cgroup OOM (rc=-9) AFTER writing changes.md
  * implementer attempt 2: SSL EOF during Feishu doc sync
  * → implementer FAILED, fixer/qa/reviewer all skipped
  * → but ``changes.md`` was complete on disk and the code actually
     compiled + passed 61 tests

The partial-delivery rule: when ``workspace_path / dep.output_file``
exists and is non-empty, the failure is treated as "delivered, ignore"
for the purpose of blocking downstream. CANCELLED is unchanged —
user-cancel intent stays binding.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path

# Config bootstrap — _make_dev_pipeline reads RESPONSE_TIMEOUT
_TMP = tempfile.mkdtemp(prefix="larkhelm_pd_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.crew._scheduler import _get_failed_dep
from larkhelm.crew_types import (
    AgentSpec, AgentState, AgentStatus, CrewState, CrewPlan,
)


def _spec(id, depends_on=None, output_file="") -> AgentSpec:
    return AgentSpec(
        id=id, role="r", model="claude", system="", prompt="",
        depends_on=depends_on or [], timeout=60, output_file=output_file,
    )


def _state(specs: list[AgentSpec], statuses: dict[str, AgentStatus]) -> CrewState:
    return CrewState(
        crew_id="t", chat_id="oc_t",
        plan=CrewPlan(title="t", agents=specs),
        agents={s.id: AgentState(spec=s, status=statuses.get(s.id, AgentStatus.PENDING))
                for s in specs},
        cancel_ev=threading.Event(),
        lock=threading.Lock(),
    )


class PartialDeliveryTests(unittest.TestCase):

    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(prefix="larkhelm_ws_"))

    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)

    # ── Baseline (no workspace_path) — legacy behaviour preserved ──

    def test_no_workspace_path_keeps_legacy_strict_behaviour(self):
        """``workspace_path=None`` (the old call signature) must still
        return the FAILED upstream — callers that don't opt in to the
        new rule get the conservative behaviour."""
        impl  = _spec("impl",  output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        self.assertEqual(_get_failed_dep(state, fixer), "impl")
        # Even with a file present on disk, no ws_path → no rescue:
        (self.ws / "changes.md").write_text("real content")
        # (file exists but we don't pass workspace_path)
        self.assertEqual(_get_failed_dep(state, fixer), "impl")

    # ── The bug fix: FAILED + output_file present → not blocking ──

    def test_failed_upstream_with_delivered_output_does_not_block(self):
        """The b3a116f scenario: implementer FAILED but changes.md
        landed. fixer should be allowed to run."""
        impl  = _spec("impl",  output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        (self.ws / "changes.md").write_text("real implementer output")
        self.assertIsNone(_get_failed_dep(state, fixer, self.ws),
            "FAILED implementer with non-empty changes.md must NOT "
            "block fixer — fixer reads the file anyway")

    def test_failed_upstream_with_zero_byte_output_blocks(self):
        """Empty output_file = the Write tool started but didn't
        commit anything useful → still a real failure, must block."""
        impl  = _spec("impl",  output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        (self.ws / "changes.md").write_text("")  # 0 bytes
        self.assertEqual(_get_failed_dep(state, fixer, self.ws), "impl",
            "0-byte file is no delivery — must still block downstream")

    def test_failed_upstream_with_missing_output_blocks(self):
        """Output file declared but never written → real failure."""
        impl  = _spec("impl",  output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        # file deliberately not created
        self.assertEqual(_get_failed_dep(state, fixer, self.ws), "impl")

    def test_failed_upstream_without_output_file_declaration_blocks(self):
        """Some agents (e.g. ``trigger_only`` ones) don't declare an
        output_file. For those, fall back to strict legacy behaviour
        — no file to verify, no partial-delivery rescue."""
        impl  = _spec("impl",  output_file="")  # no output_file declared
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        # Even with files present in ws, no output_file declaration
        (self.ws / "anything.md").write_text("X")
        self.assertEqual(_get_failed_dep(state, fixer, self.ws), "impl")

    # ── Semantic: CANCELLED is NOT rescued ──

    def test_cancelled_upstream_still_blocks_even_with_output(self):
        """User pressed cancel — respect that intent regardless of
        whatever was on disk. The partial-delivery escape is FAILED-only."""
        impl  = _spec("impl",  output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.CANCELLED})
        (self.ws / "changes.md").write_text("real content")
        self.assertEqual(_get_failed_dep(state, fixer, self.ws), "impl",
            "CANCELLED is user-intent-binding; partial-delivery escape "
            "MUST NOT apply (would silently undo a cancel)")

    # ── Recursion: rescue at one level still checks upstream ──

    def test_partial_delivery_does_not_short_circuit_deeper_upstream_failure(self):
        """impl delivered changes.md (rescue applies), but pm above it
        FAILED without an output_file — the deeper failure should still
        propagate to reviewer (transitive dep)."""
        pm    = _spec("pm",         output_file="")  # no rescue available
        impl  = _spec("impl",       depends_on=["pm"], output_file="changes.md")
        rev   = _spec("reviewer",   depends_on=["impl"])
        state = _state([pm, impl, rev], {
            "pm":   AgentStatus.FAILED,
            "impl": AgentStatus.FAILED,
        })
        (self.ws / "changes.md").write_text("partial delivery")
        # reviewer checks impl → rescued; recurses to pm → pm has no
        # output_file → real block. Result: pm returned (or impl if
        # rescue logic chose to short-circuit).
        result = _get_failed_dep(state, rev, self.ws)
        self.assertEqual(result, "pm",
            "rescued FAILED must still recurse upstream; deeper "
            f"non-rescuable FAILED ('pm') must still block. got {result!r}")

    def test_partial_delivery_chained_when_both_upstream_delivered(self):
        """pm FAILED but design.md landed; impl FAILED but changes.md
        landed; reviewer should run with both partials."""
        pm    = _spec("pm",       output_file="design.md")
        impl  = _spec("impl",     depends_on=["pm"],   output_file="changes.md")
        rev   = _spec("reviewer", depends_on=["impl"])
        state = _state([pm, impl, rev], {
            "pm":   AgentStatus.FAILED,
            "impl": AgentStatus.FAILED,
        })
        (self.ws / "design.md").write_text("design content")
        (self.ws / "changes.md").write_text("changes content")
        self.assertIsNone(_get_failed_dep(state, rev, self.ws),
            "two partial-delivered FAILEDs in chain → reviewer can run")

    # ── Defensive ──

    def test_disk_io_error_falls_back_to_strict_blocking(self):
        """If we can't stat the file (perms, broken symlink, FS issue),
        be conservative — treat as not delivered. We can't risk false
        rescue when we don't know what's on disk."""
        impl  = _spec("impl", output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        # Pass a path that can't be a normal directory
        bad_ws = Path("/proc/self/this_does_not_exist/nope")
        self.assertEqual(_get_failed_dep(state, fixer, bad_ws), "impl")

    def test_needs_retry_upstream_does_not_block_regardless(self):
        """An upstream with ``needs_retry=True`` is the existing
        retry-feedback escape (orthogonal to this fix). It must still
        not block."""
        impl  = _spec("impl", output_file="changes.md")
        fixer = _spec("fixer", depends_on=["impl"])
        state = _state([impl, fixer], {"impl": AgentStatus.FAILED})
        state.agents["impl"].needs_retry = True
        # No output file written, but needs_retry overrides everything
        self.assertIsNone(_get_failed_dep(state, fixer, self.ws))


if __name__ == "__main__":
    unittest.main()
