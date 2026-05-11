"""
Comprehensive tests for larkhelm.memory_io — export / import / helpers.

Coverage:
  - _redact_config: shallow keys, APP_ID, nested dicts, nested lists
  - export_memory: zip structure, manifest, config redaction, chat_ids filter
  - import_memory: merge mode, replace mode, dry_run, zip slip, format_version warning,
                   missing-entry graceful skip
  - _merge_state: new chat insertion, deep merge preserving live fields, cron union
  - _merge_jsonl: SHA-256 deduplication, new lines appended, no-new-lines path
  - _dest_for: valid data/ path, valid memory/ path, zip slip blocked, manifest skipped
  - pending_memory_import timestamp expiry logic (unit test of the check)
"""
import json
import os
import shutil
import tempfile
import time
import unittest
import zipfile
from pathlib import Path

# ── Minimal bootstrap so memory_io can resolve DATA_DIR without a real config ──
import larkhelm.memory_io as _mio

# Patch _MEMORY_HOME to a temp dir so tests never touch ~/.larkhelm/memory
_ORIG_MEMORY_HOME = _mio._MEMORY_HOME


def setUpModule():
    global _tmp_memory_home
    _tmp_memory_home = Path(tempfile.mkdtemp(prefix="larkhelm_test_mem_home_"))
    _mio._MEMORY_HOME = _tmp_memory_home


def tearDownModule():
    _mio._MEMORY_HOME = _ORIG_MEMORY_HOME
    shutil.rmtree(_tmp_memory_home, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_data_dir():
    """Return a fresh temp directory representing DATA_DIR."""
    d = Path(tempfile.mkdtemp(prefix="larkhelm_test_data_"))
    (d / "sessions").mkdir()
    (d / "logs").mkdir()
    return d


def _build_state(chats: dict) -> bytes:
    return json.dumps(chats, ensure_ascii=False, indent=2).encode()


def _build_jsonl(*records: dict) -> bytes:
    return b"\n".join(json.dumps(r).encode() for r in records) + b"\n"


def _export_and_open(data_dir: Path, **kwargs) -> zipfile.ZipFile:
    """Run export_memory into data_dir and return an open ZipFile for inspection."""
    out = data_dir / "out.zip"
    _mio.export_memory(out, data_dir=data_dir, **kwargs)
    return zipfile.ZipFile(out, "r")


# ─────────────────────────────────────────────────────────────────────────────
# _redact_config
# ─────────────────────────────────────────────────────────────────────────────

class TestRedactConfig(unittest.TestCase):

    def test_secret_key_redacted(self):
        result = _mio._redact_config({"APP_SECRET": "s3cr3t"})
        self.assertEqual(result["APP_SECRET"], "***")

    def test_api_key_redacted(self):
        result = _mio._redact_config({"deepseek_api_key": "sk-abc"})
        self.assertEqual(result["deepseek_api_key"], "***")

    def test_app_id_redacted(self):
        """APP_ID must be treated as sensitive (was missing before the fix)."""
        result = _mio._redact_config({"APP_ID": "cli_abc123"})
        self.assertEqual(result["APP_ID"], "***")

    def test_safe_key_preserved(self):
        result = _mio._redact_config({"voice_model_size": "medium"})
        self.assertEqual(result["voice_model_size"], "medium")

    def test_nested_dict_recursed(self):
        """api_key inside a nested dict must also be redacted."""
        raw = {"backends": {"deepseek": {"api_key": "sk-xxx", "model": "v3"}}}
        result = _mio._redact_config(raw)
        self.assertEqual(result["backends"]["deepseek"]["api_key"], "***")
        self.assertEqual(result["backends"]["deepseek"]["model"], "v3")

    def test_nested_list_of_dicts_recursed(self):
        """api_key inside a list of dicts must be redacted."""
        raw = {"probe_models": [{"id": "ds", "api_key": "sk-yyy"}, {"id": "kimi"}]}
        result = _mio._redact_config(raw)
        self.assertEqual(result["probe_models"][0]["api_key"], "***")
        self.assertEqual(result["probe_models"][1]["id"], "kimi")

    def test_non_string_list_items_preserved(self):
        raw = {"tags": ["fast", "cheap", 42]}
        result = _mio._redact_config(raw)
        self.assertEqual(result["tags"], ["fast", "cheap", 42])

    def test_token_redacted(self):
        result = _mio._redact_config({"access_token": "tok123"})
        self.assertEqual(result["access_token"], "***")


# ─────────────────────────────────────────────────────────────────────────────
# export_memory
# ─────────────────────────────────────────────────────────────────────────────

class TestExportMemory(unittest.TestCase):

    def setUp(self):
        self.data = _make_data_dir()

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    def test_creates_valid_zip(self):
        out = self.data / "test.zip"
        result = _mio.export_memory(out, data_dir=self.data)
        self.assertTrue(result.exists())
        self.assertTrue(zipfile.is_zipfile(result))

    def test_manifest_present_and_valid(self):
        out = self.data / "test.zip"
        _mio.export_memory(out, data_dir=self.data)
        with zipfile.ZipFile(out, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        self.assertEqual(manifest["format_version"], "1")
        self.assertIn("created_at", manifest)
        self.assertIn("files", manifest)

    def test_state_json_exported(self):
        state = {"chat1": {"model": "claude", "cwd": "/tmp"}}
        (self.data / "state.json").write_text(json.dumps(state))
        out = self.data / "test.zip"
        _mio.export_memory(out, data_dir=self.data)
        with zipfile.ZipFile(out, "r") as zf:
            self.assertIn("data/state.json", zf.namelist())
            exported = json.loads(zf.read("data/state.json"))
        self.assertEqual(exported, state)

    def test_config_redacted_in_zip(self):
        # Write a fake config alongside
        cfg_path = self.data / "config.json"
        cfg_path.write_text(json.dumps({"APP_ID": "cli_x", "APP_SECRET": "s3cr3t",
                                         "voice_model_size": "medium"}))
        # Patch config module to return this path
        import larkhelm.config as _cfg_mod
        orig = getattr(_cfg_mod, "CONFIG_PATH", None)
        try:
            _cfg_mod.CONFIG_PATH = cfg_path
            out = self.data / "test.zip"
            _mio.export_memory(out, data_dir=self.data)
        finally:
            if orig is None:
                del _cfg_mod.CONFIG_PATH
            else:
                _cfg_mod.CONFIG_PATH = orig

        with zipfile.ZipFile(out, "r") as zf:
            if "config.json" in zf.namelist():
                exported_cfg = json.loads(zf.read("config.json"))
                self.assertEqual(exported_cfg.get("APP_ID"), "***")
                self.assertEqual(exported_cfg.get("APP_SECRET"), "***")
                self.assertEqual(exported_cfg.get("voice_model_size"), "medium")

    def test_chat_ids_filter_state(self):
        state = {"chat_a": {"model": "claude"}, "chat_b": {"model": "gemini"}}
        (self.data / "state.json").write_text(json.dumps(state))
        out = self.data / "test.zip"
        _mio.export_memory(out, chat_ids=["chat_a"], data_dir=self.data)
        with zipfile.ZipFile(out, "r") as zf:
            exported = json.loads(zf.read("data/state.json"))
        self.assertIn("chat_a", exported)
        self.assertNotIn("chat_b", exported)

    def test_memory_md_included(self):
        (_tmp_memory_home / "global_u123.md").write_text("## global memory")
        out = self.data / "test.zip"
        _mio.export_memory(out, data_dir=self.data)
        with zipfile.ZipFile(out, "r") as zf:
            self.assertIn("memory/global_u123.md", zf.namelist())

    def test_output_suffix_added(self):
        out_no_suffix = self.data / "myexport"
        result = _mio.export_memory(out_no_suffix, data_dir=self.data)
        self.assertEqual(result.suffix, ".zip")

    def test_manifest_chat_ids_filter_recorded(self):
        out = self.data / "test.zip"
        _mio.export_memory(out, chat_ids=["chatX"], data_dir=self.data)
        with zipfile.ZipFile(out, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
        self.assertEqual(manifest["chat_ids_filter"], ["chatX"])

    def test_no_debug_log_by_default(self):
        out = self.data / "test.zip"
        _mio.export_memory(out, data_dir=self.data)
        with zipfile.ZipFile(out, "r") as zf:
            self.assertNotIn("data/larkhelm.log", zf.namelist())


# ─────────────────────────────────────────────────────────────────────────────
# import_memory
# ─────────────────────────────────────────────────────────────────────────────

def _make_zip(data_dir: Path, files: dict[str, bytes], fmt_version: str = "1") -> Path:
    """Build a minimal export zip with the given in-archive paths and contents."""
    out = data_dir / "_tmp_archive.zip"
    entries = [
        {"zip_path": k, "orig_path": str(data_dir / k), "role": _guess_role(k), "size": len(v)}
        for k, v in files.items()
    ]
    manifest = {
        "format_version": fmt_version,
        "created_at": "2026-01-01T00:00:00+00:00",
        "app_version": "test",
        "data_dir": str(data_dir),
        "memory_home_dir": str(_tmp_memory_home),
        "chat_ids_filter": None,
        "files": entries,
    }
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        for zpath, content in files.items():
            zf.writestr(zpath, content)
    return out


def _guess_role(zpath: str) -> str:
    if "state.json" in zpath:
        return "state"
    if ".jsonl" in zpath:
        return "jsonl"
    if ".sid" in zpath:
        return "session"
    if ".md" in zpath:
        return "log_md"
    return "unknown"


class TestImportMemory(unittest.TestCase):

    def setUp(self):
        self.data = _make_data_dir()

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    # ── replace mode ──────────────────────────────────────────────────────────

    def test_replace_writes_state(self):
        state = {"chat1": {"model": "claude", "cwd": "/tmp"}}
        archive = _make_zip(self.data, {"data/state.json": _build_state(state)})
        result = _mio.import_memory(archive, merge=False, data_dir=self.data)
        dest = self.data / "state.json"
        self.assertTrue(dest.exists())
        self.assertEqual(json.loads(dest.read_text()), state)
        # Use resolved paths to handle macOS /var → /private/var symlinks
        written_resolved = [str(Path(p).resolve()) for p in result["written"]]
        self.assertIn(str(dest.resolve()), written_resolved)

    def test_replace_overwrites_existing_state(self):
        (self.data / "state.json").write_bytes(_build_state({"old": {"model": "gemini"}}))
        archive = _make_zip(self.data, {"data/state.json": _build_state({"new": {"model": "claude"}})})
        _mio.import_memory(archive, merge=False, data_dir=self.data)
        loaded = json.loads((self.data / "state.json").read_text())
        self.assertNotIn("old", loaded)
        self.assertIn("new", loaded)

    # ── dry_run mode ──────────────────────────────────────────────────────────

    def test_dry_run_does_not_write(self):
        archive = _make_zip(self.data, {"data/state.json": _build_state({"c1": {}})})
        result = _mio.import_memory(archive, dry_run=True, data_dir=self.data)
        dest = self.data / "state.json"
        self.assertFalse(dest.exists(), "dry_run must not write files")
        self.assertGreater(len(result["written"]), 0, "dry_run should report what would be written")

    # ── config is always skipped ───────────────────────────────────────────────

    def test_config_redacted_skipped(self):
        content = json.dumps({"APP_ID": "***"}).encode()
        archive = _make_zip(self.data, {})
        # Manually inject a config_redacted entry
        with zipfile.ZipFile(archive, "a") as zf:
            cfg_entry = {"zip_path": "config.json", "orig_path": "", "role": "config_redacted", "size": len(content)}
            manifest = json.loads(zf.read("manifest.json"))
            manifest["files"].append(cfg_entry)
            zf.writestr("manifest.json", json.dumps(manifest))
            zf.writestr("config.json", content)
        result = _mio.import_memory(archive, data_dir=self.data)
        skipped_ids = [s[0] for s in result["skipped"]]
        self.assertIn("config.json", skipped_ids)

    # ── format_version mismatch → warning, not exception ──────────────────────

    def test_format_version_mismatch_warns_not_raises(self):
        """AC-05: mismatched format_version must warn and continue, not raise."""
        archive = _make_zip(self.data, {"data/state.json": _build_state({"c": {}})},
                            fmt_version="99")
        result = _mio.import_memory(archive, data_dir=self.data)
        self.assertTrue(any("format_version" in w for w in result["warnings"]))
        # Import still completed
        self.assertTrue((self.data / "state.json").exists())

    # ── missing entry in archive → per-entry skip, not abort ─────────────────

    def test_missing_zip_entry_skipped_gracefully(self):
        archive = _make_zip(self.data, {})
        # Inject a manifest entry pointing to a file not actually in the zip
        with zipfile.ZipFile(archive, "a") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            manifest["files"].append({
                "zip_path": "data/state.json",
                "orig_path": "",
                "role": "state",
                "size": 0,
            })
            zf.writestr("manifest.json", json.dumps(manifest))
        result = _mio.import_memory(archive, data_dir=self.data)
        skipped_ids = [s[0] for s in result["skipped"]]
        self.assertIn("data/state.json", skipped_ids)

    # ── zip slip protection ───────────────────────────────────────────────────

    def test_zip_slip_rejected(self):
        """data/ entries escaping DATA_DIR must be skipped (zip slip protection)."""
        dest = _mio._dest_for("data/../../../etc/passwd", self.data)
        self.assertIsNone(dest, "Zip slip path must be rejected by _dest_for")

    def test_valid_data_path_accepted(self):
        dest = _mio._dest_for("data/state.json", self.data)
        self.assertIsNotNone(dest)
        self.assertTrue(str(dest).startswith(str(self.data.resolve())))

    def test_manifest_json_returns_none(self):
        self.assertIsNone(_mio._dest_for("manifest.json", self.data))

    def test_unknown_prefix_returns_none(self):
        self.assertIsNone(_mio._dest_for("random/file.txt", self.data))


# ─────────────────────────────────────────────────────────────────────────────
# _merge_state
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeState(unittest.TestCase):

    def setUp(self):
        self.data = _make_data_dir()
        self.dest = self.data / "state.json"

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    def _merge(self, incoming: dict, existing: dict) -> dict:
        self.dest.write_bytes(_build_state(existing))
        result: dict = {"written": [], "skipped": [], "warnings": []}
        _mio._merge_state(_build_state(incoming), self.dest, result)
        return json.loads(self.dest.read_text())

    def test_new_chat_inserted(self):
        merged = self._merge({"new_chat": {"model": "claude"}}, {"old_chat": {"model": "gemini"}})
        self.assertIn("new_chat", merged)
        self.assertIn("old_chat", merged)

    def test_existing_chat_live_fields_preserved(self):
        """Live state fields not present in archive must be kept."""
        existing = {"c1": {"model": "claude", "turn_count": 42, "cwd": "/live"}}
        incoming = {"c1": {"model": "claude", "cwd": "/archive"}}
        merged = self._merge(incoming, existing)
        # turn_count was only in live → must be preserved
        self.assertEqual(merged["c1"]["turn_count"], 42)

    def test_crons_merged_by_id(self):
        cron_a = {"id": "aaa", "expr": "0 9 * * *", "query": "morning"}
        cron_b = {"id": "bbb", "expr": "0 18 * * *", "query": "evening"}
        existing = {"c1": {"crons": [cron_a]}}
        incoming = {"c1": {"crons": [cron_b]}}
        merged = self._merge(incoming, existing)
        cron_ids = {c["id"] for c in merged["c1"]["crons"]}
        self.assertIn("aaa", cron_ids)
        self.assertIn("bbb", cron_ids)

    def test_live_cron_wins_over_archived_cron(self):
        """If the same cron ID exists in both, the live version wins."""
        cron_live = {"id": "aaa", "expr": "0 10 * * *", "query": "live version"}
        cron_arch = {"id": "aaa", "expr": "0 9 * * *", "query": "archive version"}
        existing = {"c1": {"crons": [cron_live]}}
        incoming = {"c1": {"crons": [cron_arch]}}
        merged = self._merge(incoming, existing)
        crons = {c["id"]: c for c in merged["c1"]["crons"]}
        self.assertEqual(crons["aaa"]["query"], "live version")

    def test_invalid_json_in_archive_skips(self):
        self.dest.write_bytes(_build_state({"c1": {}}))
        result: dict = {"written": [], "skipped": [], "warnings": []}
        _mio._merge_state(b"not valid json!!!", self.dest, result)
        self.assertTrue(any("state.json" in s[0] for s in result["skipped"]))
        # Existing state must be untouched
        self.assertEqual(json.loads(self.dest.read_text()), {"c1": {}})

    def test_corrupted_existing_state_treated_as_empty(self):
        self.dest.write_bytes(b"broken{json")
        result: dict = {"written": [], "skipped": [], "warnings": []}
        _mio._merge_state(_build_state({"c1": {"model": "claude"}}), self.dest, result)
        self.assertIn("c1", json.loads(self.dest.read_text()))


# ─────────────────────────────────────────────────────────────────────────────
# _merge_jsonl
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeJsonl(unittest.TestCase):

    def setUp(self):
        self.data = _make_data_dir()
        self.dest = self.data / "logs" / "all.jsonl"

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    def _run(self, existing_records: list[dict], incoming_records: list[dict]) -> tuple[list[dict], dict]:
        self.dest.write_bytes(_build_jsonl(*existing_records))
        result: dict = {"written": [], "skipped": [], "warnings": []}
        _mio._merge_jsonl(_build_jsonl(*incoming_records), self.dest, result)
        lines = [json.loads(ln) for ln in self.dest.read_bytes().splitlines() if ln.strip()]
        return lines, result

    def test_duplicate_lines_not_appended(self):
        rec = {"chat_id": "c1", "role": "user", "content": "hello"}
        lines, result = self._run([rec], [rec])
        self.assertEqual(lines.count(rec), 1)

    def test_new_lines_appended(self):
        rec1 = {"chat_id": "c1", "role": "user", "content": "hello"}
        rec2 = {"chat_id": "c1", "role": "assistant", "content": "world"}
        lines, result = self._run([rec1], [rec1, rec2])
        self.assertIn(rec2, lines)
        self.assertEqual(len(lines), 2)

    def test_no_new_lines_no_warning(self):
        """When 0 new lines are appended, no warning should be emitted."""
        rec = {"chat_id": "c1", "role": "user"}
        _, result = self._run([rec], [rec])
        self.assertEqual(result["warnings"], [])

    def test_new_lines_emit_warning(self):
        rec1 = {"a": 1}
        rec2 = {"b": 2}
        _, result = self._run([rec1], [rec2])
        self.assertTrue(any("new line" in w.lower() for w in result["warnings"]))

    def test_idempotent_double_import(self):
        """Importing the same archive twice must not grow the file."""
        rec = {"x": 1}
        self.dest.write_bytes(_build_jsonl(rec))
        incoming = _build_jsonl(rec)
        r1: dict = {"written": [], "skipped": [], "warnings": []}
        r2: dict = {"written": [], "skipped": [], "warnings": []}
        _mio._merge_jsonl(incoming, self.dest, r1)
        _mio._merge_jsonl(incoming, self.dest, r2)
        lines = [ln for ln in self.dest.read_bytes().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 1)


# ─────────────────────────────────────────────────────────────────────────────
# _dest_for (zip slip)
# ─────────────────────────────────────────────────────────────────────────────

class TestDestFor(unittest.TestCase):

    def setUp(self):
        self.data = _make_data_dir()

    def tearDown(self):
        shutil.rmtree(self.data, ignore_errors=True)

    def test_data_prefix_valid(self):
        dest = _mio._dest_for("data/sessions/chat1.sid", self.data)
        self.assertIsNotNone(dest)
        self.assertIn("chat1.sid", dest.name)

    def test_memory_prefix_valid(self):
        dest = _mio._dest_for("memory/global_u1.md", self.data)
        self.assertIsNotNone(dest)
        self.assertTrue(str(dest).startswith(str(_tmp_memory_home.resolve())))

    def test_zip_slip_data(self):
        self.assertIsNone(_mio._dest_for("data/../../etc/passwd", self.data))

    def test_zip_slip_memory(self):
        self.assertIsNone(_mio._dest_for("memory/../../../etc/shadow", self.data))

    def test_manifest_none(self):
        self.assertIsNone(_mio._dest_for("manifest.json", self.data))

    def test_unrecognised_prefix_none(self):
        self.assertIsNone(_mio._dest_for("other/file.txt", self.data))


# ─────────────────────────────────────────────────────────────────────────────
# pending_memory_import timestamp expiry logic
# (unit-tests the check without needing the Feishu handler running)
# ─────────────────────────────────────────────────────────────────────────────

class TestPendingImportExpiry(unittest.TestCase):
    """Test the 10-minute TTL logic for pending_memory_import flag."""

    _TTL = 600  # seconds

    def _check_expired(self, pending_ts) -> bool:
        """Mirror of the logic in handlers/_message.py."""
        if not pending_ts:
            return True  # no flag at all → treat as expired/absent
        if isinstance(pending_ts, float) and time.time() - pending_ts > self._TTL:
            return True
        return False

    def test_fresh_timestamp_not_expired(self):
        ts = time.time()
        self.assertFalse(self._check_expired(ts))

    def test_old_timestamp_expired(self):
        ts = time.time() - (self._TTL + 1)
        self.assertTrue(self._check_expired(ts))

    def test_none_treated_as_absent(self):
        self.assertTrue(self._check_expired(None))

    def test_false_treated_as_absent(self):
        self.assertTrue(self._check_expired(False))

    def test_legacy_boolean_true_never_expires(self):
        """Boolean True (legacy) passes the isinstance(float) check → never expires."""
        ts = True
        # isinstance(True, float) is False, so the condition won't trigger
        self.assertFalse(self._check_expired(ts))

    def test_just_inside_boundary(self):
        # 1 second before expiry → NOT expired
        ts = time.time() - (self._TTL - 1)
        self.assertFalse(self._check_expired(ts))

    def test_just_outside_boundary(self):
        # 1 second past expiry → expired
        ts = time.time() - (self._TTL + 1)
        self.assertTrue(self._check_expired(ts))


# ─────────────────────────────────────────────────────────────────────────────
# Full round-trip: export → import
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundTrip(unittest.TestCase):

    def setUp(self):
        self.src = _make_data_dir()
        self.dst = _make_data_dir()

    def tearDown(self):
        shutil.rmtree(self.src, ignore_errors=True)
        shutil.rmtree(self.dst, ignore_errors=True)

    def test_state_survives_round_trip(self):
        state = {"chat_rt": {"model": "claude", "cwd": "/tmp", "turn_count": 5}}
        (self.src / "state.json").write_text(json.dumps(state))
        out = self.src / "rt.zip"
        _mio.export_memory(out, data_dir=self.src)
        _mio.import_memory(out, merge=False, data_dir=self.dst)
        loaded = json.loads((self.dst / "state.json").read_text())
        self.assertEqual(loaded["chat_rt"]["model"], "claude")
        self.assertEqual(loaded["chat_rt"]["turn_count"], 5)

    def test_jsonl_survives_round_trip(self):
        records = [{"chat_id": "c1", "role": "user", "content": f"msg{i}"} for i in range(10)]
        (self.src / "logs" / "all.jsonl").write_bytes(_build_jsonl(*records))
        out = self.src / "rt.zip"
        _mio.export_memory(out, data_dir=self.src)
        _mio.import_memory(out, merge=False, data_dir=self.dst)
        dst_lines = [(self.dst / "logs" / "all.jsonl").read_bytes().splitlines()]
        self.assertEqual(len(dst_lines[0]), 10)

    def test_idempotent_merge_import(self):
        """Importing the same archive twice in merge mode must not duplicate records."""
        records = [{"chat_id": "c1", "msg": f"m{i}"} for i in range(5)]
        (self.src / "logs" / "all.jsonl").write_bytes(_build_jsonl(*records))
        out = self.src / "rt.zip"
        _mio.export_memory(out, data_dir=self.src)

        _mio.import_memory(out, merge=True, data_dir=self.dst)
        _mio.import_memory(out, merge=True, data_dir=self.dst)
        lines = [ln for ln in (self.dst / "logs" / "all.jsonl").read_bytes().splitlines() if ln.strip()]
        self.assertEqual(len(lines), 5)

    def test_sid_file_survives_round_trip(self):
        sid = self.src / "sessions" / "chat1.sid"
        sid.write_text("ses_abc123")
        out = self.src / "rt.zip"
        _mio.export_memory(out, data_dir=self.src)
        _mio.import_memory(out, merge=False, data_dir=self.dst)
        dst_sid = self.dst / "sessions" / "chat1.sid"
        self.assertTrue(dst_sid.exists())
        self.assertEqual(dst_sid.read_text(), "ses_abc123")


if __name__ == "__main__":
    unittest.main(verbosity=2)
