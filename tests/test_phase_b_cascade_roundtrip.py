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

    # ── 4. YAML edge case: hash with special chars round-trips intact ──

    def test_yaml_quoting_handles_special_chars_in_hash(self):
        """``_save_md`` writes frontmatter as ``key: "value"``. If the
        value contains a literal quote ``"``, the implementation must
        escape it (or use a different YAML form) — otherwise
        ``_load_md_frontmatter`` parses the field as the substring up to
        the first quote and the short-circuit silently breaks for any
        hash that happens to contain a quote character.

        md5 hex digests never contain quotes, BUT if the schema ever
        evolves to store a different field via extra_fm_pairs that does
        (e.g. a user-supplied label), this test pins the quoting
        contract."""
        # Manually injected special-char value through the same
        # extra_fm_pairs channel.
        weird = 'has"quote-and-\\\\backslash'
        mem.save_project_memory(
            self.cwd, "body",
            extra_fm_pairs={"odd_field": weird},
        )

        path = mem._project_memory_file(self.cwd)
        fm = mem._load_md_frontmatter(path)
        # The parsed value should match what we wrote (modulo any
        # documented escaping). If quoting drifts, the parsed value
        # will be a truncated prefix of `weird`.
        self.assertIn("quote", fm.get("odd_field", ""),
                      f"YAML quoting drift: stored={weird!r} loaded={fm.get('odd_field')!r}")


if __name__ == "__main__":
    unittest.main()
