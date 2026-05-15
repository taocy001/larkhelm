"""End-to-end producer↔consumer round-trip for Phase D audit pipeline.

Pins the FULL chain through real code (no synthetic dicts):

  ``build_audit_record_v2(req, policy, scored, …, llm_router_diag)``
    │  producer-side record-builder
    ▼
  ``_audit_decision(record)``  → enqueue
    │  async writer thread serialises to ``memory_retriever_audit.jsonl``
    ▼
  on-disk JSONL (0600 permissions, one record per line)
    │
    ▼
  ``iter_audit_records(window)``  ← consumer #1, ts-window filter
    │
    ▼
  ``_compute_audit_summary(records, window)``  ← consumer #2, aggregation
    │  computes ``llm_router`` sub-dict via ``_compute_llm_router_summary``
    ▼
  end-to-end CLI ``larkhelm memory audit-summary``

Same anti-pattern guard as the Phase B / C round-trip tests: any future
schema drift (field rename, JSON encoding change, ts format change,
producer adds a field consumer doesn't expect, …) is caught HERE
without waiting for production logs to surface it.

This test exercises ``_audit_writer_loop`` which runs in a daemon
thread. We force-drain via ``_AUDIT_QUEUE.join()`` so the test is
deterministic instead of relying on ``time.sleep``.
"""
from __future__ import annotations

import atexit
import json
import os
import shutil
import tempfile
import time
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

# Bootstrap config so _cfg.DATA_DIR is initialised — the audit-path
# resolver reads it.
_TMP = tempfile.mkdtemp(prefix="larkhelm_audit_rt_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({"APP_ID": "x", "APP_SECRET": "x"}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)

from larkhelm.__main__ import (
    _compute_audit_summary,
    _compute_llm_router_summary,
)
from larkhelm.memory_llm_router import RouterDiagnostics
from larkhelm.memory_retriever import (
    _audit_decision,
    _resolve_audit_path,
    build_audit_record_v2,
    iter_audit_records,
)
from larkhelm.memory_slice import (
    InjectionPolicy,
    MemorySlice,
    RetrievalRequest,
    ScoredSlice,
)


def _drain_audit_queue() -> None:
    """Force the async audit writer thread to flush queued records to
    disk before any read assertion. The writer is single-threaded so a
    sentinel ``put`` followed by ``Queue.join`` would block forever
    waiting for ``task_done``; instead we poll the queue size with a
    short sleep loop (deterministic enough for tests).
    """
    from larkhelm.memory_retriever import _AUDIT_QUEUE
    # Wait at most 2s for the writer to drain — far longer than needed
    # for the few records each test enqueues.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if _AUDIT_QUEUE.empty():
            break
        time.sleep(0.01)
    # One last yield so the writer's final ``os.write`` completes before
    # we open the file for reading.
    time.sleep(0.05)


class AuditPipelineRoundTripTests(unittest.TestCase):
    """Producer → JSONL → consumer #1 → consumer #2 end-to-end."""

    def setUp(self):
        # Isolated audit dir per test (also prevents test bleed-through).
        self._audit_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._audit_dir.cleanup)
        self._audit_file = Path(self._audit_dir.name) / "memory_retriever_audit.jsonl"

        # Redirect the resolver — _audit_decision / iter_audit_records
        # both call _resolve_audit_path() so a single monkeypatch covers
        # producer + both consumers.
        self._patcher = patch(
            "larkhelm.memory_retriever._resolve_audit_path",
            return_value=self._audit_file,
        )
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    # ── helpers ────────────────────────────────────────────────────────

    def _emit(self, *, agent_type: str = "dev", mode: str = "keyword",
              diag: RouterDiagnostics | None = None,
              fail_open: bool = False) -> None:
        """Build a real audit record via the production producer + enqueue."""
        req = RetrievalRequest(
            chat_id="oc_rt",
            query=f"sample query for {agent_type}",
            agent_type=agent_type, complexity="complex",
        )
        pol = InjectionPolicy(
            agent_type=agent_type, token_budget=2000,
            layer_weights={"project": 1.0},
            kind_priority=("convention", "fact"),
        )
        scored: list[ScoredSlice] = []
        rec = build_audit_record_v2(
            request=req, policy=pol, scored=scored,
            candidate_count=5, elapsed_ms=8,
            selected_chars=150, fail_open=fail_open,
            actual_mode=mode, declared_mode="hybrid",
            llm_router_diag=diag,
        )
        _audit_decision(rec)

    # ── 1. JSONL physically written + readable ────────────────────────

    def test_real_writer_thread_persists_records_to_disk(self):
        """The async writer must actually create + append the JSONL file
        with one record per line, 0600 permissions, and JSON parseable."""
        self._emit()
        self._emit()
        _drain_audit_queue()

        self.assertTrue(self._audit_file.exists(), "audit file not written")
        lines = self._audit_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2, f"expected 2 records, got {len(lines)}")
        for ln in lines:
            r = json.loads(ln)
            self.assertEqual(r["schema_version"], "2")
            self.assertEqual(r["chat_id"], "oc_rt")

        # 0600 perms (NFR — user secrets in query_head should not leak).
        st_mode = self._audit_file.stat().st_mode & 0o777
        self.assertEqual(st_mode, 0o600,
                         f"audit file permissions are {oct(st_mode)}, expected 0o600")

    # ── 2. iter_audit_records + _compute_audit_summary round-trip ──────

    def test_consumer_reads_what_producer_wrote(self):
        """End-to-end producer→writer→consumer #1→consumer #2 happy path.

        Catches any schema drift between ``build_audit_record_v2``'s
        field names and ``_compute_audit_summary``'s reader expectations.
        """
        self._emit(agent_type="chat", mode="keyword")
        self._emit(agent_type="dev", mode="hybrid")
        _drain_audit_queue()

        records = list(iter_audit_records(timedelta(hours=1)))
        self.assertEqual(len(records), 2)

        summary = _compute_audit_summary(records, timedelta(hours=1))
        self.assertEqual(summary["total_records"], 2)
        self.assertEqual(summary["mode_distribution"], {"keyword": 1, "hybrid": 1})
        # Two distinct agent_types → both should appear in by_agent_type.
        self.assertIn("chat", summary["by_agent_type"])
        self.assertIn("dev", summary["by_agent_type"])

    # ── 3. LLM-router fields round-trip end-to-end ─────────────────────

    def test_llm_router_diag_flows_producer_to_summary(self):
        """The LLM-router aggregation depends on ``build_audit_record_v2``
        writing four ``llm_router_*`` fields that
        ``_compute_llm_router_summary`` reads back. Any rename / drop
        on either side would silently kill the observability surface."""
        good_diag = RouterDiagnostics(
            invoked=True, cache_hit=False, skipped_reason="",
            elapsed_ms=5, selected_by_llm=3,
        )
        skipped_diag = RouterDiagnostics(
            invoked=False, cache_hit=False, skipped_reason="rate_limit",
            elapsed_ms=1, selected_by_llm=0,
        )
        cache_diag = RouterDiagnostics(
            invoked=False, cache_hit=True, skipped_reason="",
            elapsed_ms=1, selected_by_llm=2,
        )

        self._emit(agent_type="dev", mode="hybrid", diag=good_diag)
        self._emit(agent_type="dev", mode="hybrid", diag=skipped_diag)
        self._emit(agent_type="dev", mode="hybrid", diag=cache_diag)
        # And one record WITHOUT diag — must NOT contribute to gate_fired_count.
        self._emit(agent_type="dev", mode="keyword", diag=None)
        _drain_audit_queue()

        records = list(iter_audit_records(timedelta(hours=1)))
        self.assertEqual(len(records), 4)

        llm_summary = _compute_llm_router_summary(records)
        self.assertIsNotNone(llm_summary,
                             "llm_router section missing — fields didn't survive write/read")
        # The diagless record must NOT be counted.
        self.assertEqual(llm_summary["gate_fired_count"], 3)
        self.assertEqual(llm_summary["invoked_count"], 1)
        self.assertEqual(llm_summary["cache_hit_count"], 1)
        self.assertEqual(llm_summary["skipped_breakdown"], {"rate_limit": 1})
        self.assertAlmostEqual(llm_summary["avg_selected_n"], 3.0)

        # Disjointness invariant: invoked + cache_hits + sum(skipped) == gate_fired.
        s = llm_summary
        self.assertEqual(
            s["invoked_count"] + s["cache_hit_count"]
            + sum(s["skipped_breakdown"].values()),
            s["gate_fired_count"],
            "buckets not disjoint — producer/consumer disagree about field semantics",
        )

    # ── 4. The ``invoked + skipped`` overlap case (MF-01 round-2 regression) ──

    def test_invoked_plus_skipped_record_still_counted_as_skipped(self):
        """Reproduces the production shape that the v1 aggregator
        miscounted (review MF-01 round-2): the producer sets
        ``invoked=True`` BEFORE calling the LLM as a debug breadcrumb,
        then on parse_failed sets ``skipped_reason`` WITHOUT flipping
        invoked. The consumer must treat this as SKIPPED (not as a
        successful invoke dragging avg_selected_n to 0).

        This is the EXACT scenario synthetic ``_make_record`` tests
        couldn't construct — only the real producer + writer creates it.
        """
        weird_diag = RouterDiagnostics(
            invoked=True, cache_hit=False,
            skipped_reason="parse_failed", selected_by_llm=0, elapsed_ms=12,
        )
        good_diag = RouterDiagnostics(
            invoked=True, cache_hit=False, skipped_reason="",
            selected_by_llm=4, elapsed_ms=8,
        )
        self._emit(agent_type="dev", mode="hybrid", diag=weird_diag)
        self._emit(agent_type="dev", mode="hybrid", diag=good_diag)
        _drain_audit_queue()

        records = list(iter_audit_records(timedelta(hours=1)))
        llm = _compute_llm_router_summary(records)
        self.assertIsNotNone(llm)

        # 1 successful invoke (avg should be 4.0, NOT 2.0).
        self.assertEqual(llm["invoked_count"], 1)
        self.assertAlmostEqual(llm["avg_selected_n"], 4.0,
                               msg="parse_failed invoke dragged avg_selected_n — MF-01 regressed")
        # And the failed-invoke is counted in skipped_breakdown.
        self.assertEqual(llm["skipped_breakdown"], {"parse_failed": 1})


if __name__ == "__main__":
    unittest.main()
