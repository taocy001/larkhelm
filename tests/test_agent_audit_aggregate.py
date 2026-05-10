"""Coverage for ``agent_audit.aggregate_daily`` — multi-date filtering,
per-agent breakdown, and corrupted-line tolerance. These code paths back
``/stats intent`` and previously had no direct unit test (review.md §6 backlog).
"""
from __future__ import annotations

import datetime
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from larkhelm.agent_hub import agent_audit


def _record(
    *,
    ts: str,
    agent_type: str,
    success: bool = True,
    duration_sec: float = 0.0,
    cost_usd: float = 0.0,
) -> dict:
    return {
        "ts": ts,
        "chat_id": "oc_test",
        "agent_type": agent_type,
        "backend_id": "claude",
        "duration_sec": duration_sec,
        "cost_usd": cost_usd,
        "success": success,
        "layer": "L1",
        "confidence": 0.9,
        "trace_id": "abc12345",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


class TestAggregateDaily(unittest.TestCase):

    def test_missing_file_returns_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "missing.jsonl"
            out = agent_audit.aggregate_daily(date="2026-05-09", path=p)
        self.assertEqual(out["total"], 0)
        self.assertEqual(out["per_agent"], {})
        self.assertEqual(out["avg_duration"], 0.0)
        self.assertEqual(out["total_cost"], 0.0)
        self.assertEqual(out["success_rate"], 0.0)

    def test_filters_by_date(self):
        """Records on other dates must not leak into today's totals."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            _write_jsonl(p, [
                _record(ts="2026-05-08T10:00:00+08:00", agent_type="dev",
                        duration_sec=1.0, cost_usd=0.01),
                _record(ts="2026-05-09T08:00:00+08:00", agent_type="dev",
                        duration_sec=2.0, cost_usd=0.02),
                _record(ts="2026-05-09T09:00:00+08:00", agent_type="chat",
                        duration_sec=0.5, cost_usd=0.001),
                _record(ts="2026-05-10T08:00:00+08:00", agent_type="crew",
                        duration_sec=10.0, cost_usd=0.5),
            ])
            out = agent_audit.aggregate_daily(date="2026-05-09", path=p)
        self.assertEqual(out["date"], "2026-05-09")
        self.assertEqual(out["total"], 2)
        self.assertAlmostEqual(out["total_cost"], 0.021, places=6)
        self.assertAlmostEqual(out["avg_duration"], (2.0 + 0.5) / 2, places=6)
        self.assertEqual(set(out["per_agent"].keys()), {"dev", "chat"})

    def test_per_agent_breakdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            _write_jsonl(p, [
                _record(ts="2026-05-09T08:00:00+08:00", agent_type="dev",
                        success=True, duration_sec=1.0),
                _record(ts="2026-05-09T08:01:00+08:00", agent_type="dev",
                        success=False, duration_sec=3.0),
                _record(ts="2026-05-09T08:02:00+08:00", agent_type="dev",
                        success=True, duration_sec=2.0),
                _record(ts="2026-05-09T08:03:00+08:00", agent_type="chat",
                        success=True, duration_sec=0.5),
            ])
            out = agent_audit.aggregate_daily(date="2026-05-09", path=p)
        self.assertEqual(out["per_agent"]["dev"]["count"], 3)
        self.assertEqual(out["per_agent"]["dev"]["success"], 2)
        self.assertAlmostEqual(out["per_agent"]["dev"]["avg_duration"], 2.0, places=6)
        self.assertEqual(out["per_agent"]["chat"]["count"], 1)
        self.assertEqual(out["per_agent"]["chat"]["success"], 1)
        self.assertAlmostEqual(out["success_rate"], 3 / 4, places=6)

    def test_corrupted_lines_skipped(self):
        """A bad line must not break aggregation of the surrounding good lines."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            with p.open("w", encoding="utf-8") as f:
                f.write(json.dumps(_record(
                    ts="2026-05-09T08:00:00+08:00", agent_type="dev",
                    duration_sec=1.0)) + "\n")
                f.write("this is not json\n")
                f.write("{partial json\n")
                f.write("\n")  # blank line tolerated
                f.write(json.dumps(_record(
                    ts="2026-05-09T08:01:00+08:00", agent_type="dev",
                    duration_sec=2.0)) + "\n")
            out = agent_audit.aggregate_daily(date="2026-05-09", path=p)
        self.assertEqual(out["total"], 2)
        self.assertAlmostEqual(out["avg_duration"], 1.5, places=6)

    def test_unknown_agent_type_bucketed_as_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            with p.open("w", encoding="utf-8") as f:
                rec = _record(ts="2026-05-09T08:00:00+08:00", agent_type="dev")
                rec.pop("agent_type")
                f.write(json.dumps(rec) + "\n")
            out = agent_audit.aggregate_daily(date="2026-05-09", path=p)
        self.assertIn("unknown", out["per_agent"])

    def test_default_date_is_today(self):
        """When date=None we should use today's local date and ignore older rows."""
        today = datetime.datetime.now().astimezone().date().isoformat()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            _write_jsonl(p, [
                _record(ts="2000-01-01T00:00:00+08:00", agent_type="dev"),
                _record(ts=f"{today}T08:00:00+08:00", agent_type="dev"),
            ])
            out = agent_audit.aggregate_daily(path=p)
        self.assertEqual(out["date"], today)
        self.assertEqual(out["total"], 1)


class TestResolvePath(unittest.TestCase):

    def test_temp_fallback_when_data_dir_unset(self):
        """Without DATA_DIR or custom config we land in tempdir, not cwd."""
        with patch("larkhelm.config.config", {}, create=True), \
             patch("larkhelm.config.DATA_DIR", None, create=True):
            p = agent_audit._resolve_path()
        self.assertEqual(p.name, "agent_audit.jsonl")
        self.assertEqual(p.parent, Path(tempfile.gettempdir()))

    def test_custom_path_wins(self):
        with patch("larkhelm.config.config", {"intent_audit_path": "/var/log/x.jsonl"}, create=True):
            p = agent_audit._resolve_path()
        self.assertEqual(str(p), "/var/log/x.jsonl")


class TestAppendJsonlChmod(unittest.TestCase):

    def test_writes_with_0600(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "audit.jsonl"
            agent_audit._append_jsonl(p, _record(
                ts="2026-05-09T08:00:00+08:00", agent_type="dev"))
            mode = os.stat(p).st_mode & 0o777
        self.assertEqual(mode, 0o600)


if __name__ == "__main__":
    unittest.main()
