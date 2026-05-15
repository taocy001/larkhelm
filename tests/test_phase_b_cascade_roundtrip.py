"""Round-trip producer↔consumer test for Phase B cascade short-circuit (S53).

Reproduces the recurring failure mode: synthetic tests use hand-rolled
dicts that match the IMPLEMENTOR's assumption rather than the actual
producer output. We've caught 6 instances of this pattern across
Phase B/C/D in independent expert reviews; the lesson is "tests should
exercise the REAL producer".

This file pins the Phase B cascade chain end-to-end:

  save_project_memory(cwd, body, extra_fm_pairs={hash, len})
      ↓ (real _save_md writes YAML frontmatter to disk)
      ↓ (real _load_md_frontmatter parses it back)
  _should_skip_extract_by_hash(fm, session_content)
      ↓ returns True iff frontmatter hash matches md5(session)[:16]
  _try_extract_project(session, cwd) — must short-circuit, NOT call LLM

Without this round-trip, a quoting / escaping / formatting drift between
``_save_md`` and ``_load_md_frontmatter`` would silently break cascade
short-circuit (causing every cascade to re-run the cheap LLM) — wasting
~80% of S53's token-savings claim. The bug would be invisible to the
existing tests in test_memory_optimization.py because those all patch
``_load_md_frontmatter`` to return a synthetic dict.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Bootstrap config (matches the pattern used by test_memory_optimization.py).
# Without this, ``_cfg.config`` is None and any ``patch.dict(_cfg.config, ...)``
# call raises ``AttributeError`` — caught the first time this test ran.
_TMP = tempfile.mkdtemp(prefix="larkhelm_cascade_rt_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm import memory as mem


class CascadeShortCircuitRoundTripTests(unittest.TestCase):
    """Exercise the cascade producer → consumer chain through REAL
    ``_save_md`` and REAL ``_load_md_frontmatter`` — no patching of
    the I/O boundary.
    """

    def setUp(self):
        # Real cwd path; the project memory file path is derived from cwd
        # via md5 hash, so we don't need a separate dir parameter.
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.cwd = self._tmpdir.name

        # Redirect MEMORY_HOME_DIR to an isolated tempdir so the test
        # never touches the operator's real ~/.larkhelm/memory/.
        self._mem_home = tempfile.TemporaryDirectory()
        self.addCleanup(self._mem_home.cleanup)
        self._orig_home = mem.MEMORY_HOME_DIR
        mem.MEMORY_HOME_DIR = Path(self._mem_home.name)
        self.addCleanup(lambda: setattr(mem, "MEMORY_HOME_DIR", self._orig_home))

    def _build_session(self, suffix: str = "") -> str:
        # Multi-line body so the H2 sections + Chinese characters round-
        # trip through YAML safely. Mirrors what generate_memory emits.
        return (
            "## Work Context\n"
            f"larkhelm Phase D - 测试 cascade short-circuit{suffix}\n\n"
            "## Key Decisions & Facts\n"
            "Use md5(session)[:16] as cascade idempotency key.\n\n"
            "## Next Steps\n"
            "继续 Phase D Stage C 灰度观察。\n"
        )

    # ── 1. Real round-trip: write then read produces the same hash key ──

    def test_save_md_then_load_md_preserves_session_hash(self):
        """The CORE invariant: writing ``extra_fm_pairs={hash, len}`` via
        the real ``save_project_memory`` (which uses ``_save_md``) and
        then reading it back via ``_load_md_frontmatter`` must yield the
        identical hash + len values. If quoting / escaping / line endings
        drift, this test fails BEFORE any short-circuit logic runs.
        """
        session = self._build_session()
        sess_hash = mem._session_hash(session)
        sess_len = str(len(session))

        # REAL save — no mock. Goes through _save_md → YAML frontmatter.
        mem.save_project_memory(
            self.cwd, "body content",
            extra_fm_pairs={
                "last_extracted_session_hash": sess_hash,
                "last_extracted_session_len":  sess_len,
            },
        )

        # REAL load — no mock. Parses the YAML frontmatter from disk.
        path = mem._project_memory_file(self.cwd)
        fm = mem._load_md_frontmatter(path)

        self.assertEqual(fm.get("last_extracted_session_hash"), sess_hash,
                         f"hash drift through save→load: {fm!r}")
        self.assertEqual(fm.get("last_extracted_session_len"), sess_len,
                         f"len drift through save→load: {fm!r}")

    # ── 2. End-to-end short-circuit: real frontmatter → real predicate ──

    def test_short_circuit_via_real_io_chain(self):
        """First cascade saves the hash; second cascade with the SAME
        session must short-circuit (no LLM call). All three components
        run for real:
          1. save_project_memory(extra_fm_pairs={hash, len})  ← producer
          2. _load_md_frontmatter(path)                       ← consumer #1
          3. _should_skip_extract_by_hash(fm, session)        ← consumer #2

        Patches only the cheap-LLM call (``_run_one_shot``) so we can
        assert it was/wasn't invoked, NOT the I/O.
        """
        session = self._build_session()
        sess_hash = mem._session_hash(session)
        sess_len = str(len(session))

        # Pre-populate the project memory with the hash from a previous
        # cascade — using REAL save (no mock).
        mem.save_project_memory(
            self.cwd, "prior cascade body",
            extra_fm_pairs={
                "last_extracted_session_hash": sess_hash,
                "last_extracted_session_len":  sess_len,
            },
        )

        # Now exercise the cascade path. _try_extract_project will
        # load the frontmatter, call _should_skip_extract_by_hash, and
        # return WITHOUT invoking _run_one_shot. The latter being not-
        # called is the proof of short-circuit.
        with patch.object(mem, "_run_one_shot") as run_one_shot, \
             patch.dict(_cfg.config, {"memory_cascade_shortcircuit": True}, clear=False):
            mem._try_extract_project(session, self.cwd)

        run_one_shot.assert_not_called()

    # ── 3. Negative case: different session → no short-circuit ──

    def test_no_short_circuit_when_session_changes(self):
        """Counterpart to the happy-path test: when the on-disk hash
        differs from md5(current_session), the cascade must FULLY run
        (i.e. invoke ``_run_one_shot``). Without this we could claim
        100% short-circuit-rate trivially by always returning True.
        """
        # Save with hash of session_v1.
        session_v1 = self._build_session(" v1")
        mem.save_project_memory(
            self.cwd, "stale body",
            extra_fm_pairs={
                "last_extracted_session_hash": mem._session_hash(session_v1),
                "last_extracted_session_len":  str(len(session_v1)),
            },
        )

        # Then trigger cascade with session_v2 — hash differs.
        session_v2 = self._build_session(" v2 — new content")
        with patch.object(mem, "_run_one_shot",
                          return_value="UNCHANGED") as run_one_shot:
            mem._try_extract_project(session_v2, self.cwd)

        # _run_one_shot WAS called this time (hash mismatch → no short-circuit).
        run_one_shot.assert_called_once()

    # ── 4. Document the (asymmetric) extra_fm_pairs encoding contract ──

    def test_extra_fm_pairs_safe_for_ascii_only_values(self):
        """Document what extra_fm_pairs **actually** guarantees.

        ``_save_md`` escapes only the ``"`` character (``safe_v.replace
        ('"', '\\"')`` in memory.py); ``_load_md_frontmatter`` parses
        with ``strip().strip('"')`` — it does NOT un-escape. So values
        containing ``"`` or ``\\`` do NOT round-trip; the reader sees
        the raw on-disk representation.

        Independent reviewer (commit fba975a round-1) verified this
        asymmetry is real. Production is safe because the only current
        callers pass md5 hex digests + integer-as-string-len; neither
        contains quotes or backslashes. This test pins the current
        contract literally: ASCII alphanumeric values DO round-trip;
        special-char values DO NOT. Anyone adding a new extra_fm_pairs
        value with quotes/backslashes must fix ``_load_md_frontmatter``
        first (add an unescape step).
        """
        # Case 1: ASCII alphanumeric — round-trips cleanly.
        mem.save_project_memory(
            self.cwd, "body-ascii",
            extra_fm_pairs={"ascii_field": "abc123def456"},
        )
        path = mem._project_memory_file(self.cwd)
        fm = mem._load_md_frontmatter(path)
        self.assertEqual(fm.get("ascii_field"), "abc123def456",
                         "ASCII alphanumeric must round-trip identically")

        # Case 2: value with backslash — survives but with the literal
        # backslashes from _save_md's escape step intact. This is the
        # asymmetry the reviewer caught; we pin it explicitly so any
        # future symmetric encoding change has to delete this assertion.
        weird = 'no-quote-but-back\\slash'
        mem.save_project_memory(
            self.cwd, "body-weird",
            extra_fm_pairs={"backslash_field": weird},
        )
        fm2 = mem._load_md_frontmatter(path)
        # Current implementation: backslash survives but only if writer
        # didn't insert any escapes. ``_save_md`` only escapes ``"``, so
        # a backslash-only value passes through unchanged.
        self.assertEqual(fm2.get("backslash_field"), weird,
                         "values without quotes round-trip even when they "
                         "contain backslashes — only `\"` triggers escape "
                         "drift in the current implementation")


if __name__ == "__main__":
    unittest.main()
