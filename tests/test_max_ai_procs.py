"""Tests for round-3 OOM defense: dynamic ``MAX_AI_PROCS`` via cgroup probe.

Background
----------
Production was OOM-killed 3× on a 4-GB host with cgroup ``MemoryMax=2.8G``.
Round-3 of the OOM defense replaces the hard-coded ``MAX_AI_PROCS = 3`` with
a memory-budget-aware probe (``runner_base._compute_max_procs``) that derives
a safe cap from cgroup v2 ``memory.max`` (falling back to physical RAM).

Round-3 v1 (the now-stashed first attempt) had a P0 staleness bug: every
caller did ``from larkhelm.runner_base import _ai_proc_sem`` and held the
binding past ``_init_ai_sem`` rebuilding the sem. The v2 rewrite covered
here:

* exposes a ``get_ai_sem()`` getter that always returns the live sem
* exposes ``get_max_ai_procs()`` for read access
* keeps ``_ai_proc_sem`` / ``MAX_AI_PROCS`` as module-level mirrors for back
  compat — but new code must use the getters

Test design notes
-----------------
* No real ``/sys/fs/cgroup`` reads — we patch the low-level
  ``_detect_cgroup_memory_max`` / ``_detect_physical_ram_mb`` helpers so
  the suite is deterministic across hosts (cgroup v1 / WSL / macOS / Docker).
* P0 regression: the ``test_get_ai_sem_consistent_across_modules`` test
  asserts the cross-module invariant that's the whole reason for the
  getter refactor.
"""
from __future__ import annotations

import atexit
import json
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

_TMP = tempfile.mkdtemp(prefix="larkhelm_maxprocs_")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
_cfg_file = Path(_TMP) / "config.json"
_cfg_file.write_text(json.dumps({
    "APP_ID": "x", "APP_SECRET": "x",
}))

import larkhelm.config as _cfg
_cfg._init_runtime(config_path=str(_cfg_file), data_dir=_TMP)


class ComputeMaxProcsTests(unittest.TestCase):
    """``runner_base._compute_max_procs`` derives a memory-safe cap.

    Formula:
      budget = available - BRIDGE_BASELINE_MB - SAFETY_MARGIN_MB
      n = budget // worker_rss_mb
      clamped to [1, HARD_CEILING]
    """

    def test_compute_max_procs_uses_cgroup_when_available(self):
        """2.8 GB cgroup (real prod scenario) → 2 workers."""
        from larkhelm import runner_base as rb
        # 2.8 GB in bytes
        cgroup_bytes = int(2.8 * 1024 * 1024 * 1024)
        with patch.object(rb, "_detect_cgroup_memory_max", return_value=cgroup_bytes), \
             patch.object(rb, "_detect_physical_ram_mb", return_value=99999):  # should not be used
            n, reason = rb._compute_max_procs()
        # 2867 MB - 400 - 400 = 2067 MB ; 2067 // 800 = 2
        self.assertEqual(n, 2)
        self.assertIn("cgroup_max=", reason)

    def test_compute_max_procs_falls_back_to_physical_ram(self):
        """No cgroup limit → use /proc/meminfo MemTotal."""
        from larkhelm import runner_base as rb
        # 8 GB physical
        with patch.object(rb, "_detect_cgroup_memory_max", return_value=None), \
             patch.object(rb, "_detect_physical_ram_mb", return_value=8192):
            n, reason = rb._compute_max_procs()
        # 8192 - 400 - 400 = 7392 ; 7392 // 800 = 9 ; clamped to HARD_CEILING=8
        self.assertEqual(n, rb.HARD_CEILING)
        self.assertIn("physical_ram=", reason)

    def test_compute_max_procs_hits_hard_ceiling(self):
        """32 GB host → caps at HARD_CEILING (8) to keep scheduling sane."""
        from larkhelm import runner_base as rb
        big_bytes = 32 * 1024 * 1024 * 1024
        with patch.object(rb, "_detect_cgroup_memory_max", return_value=big_bytes):
            n, _ = rb._compute_max_procs()
        self.assertEqual(n, rb.HARD_CEILING)

    def test_compute_max_procs_floor_is_one(self):
        """Tiny host (256 MB) — budget goes negative, still get at least 1."""
        from larkhelm import runner_base as rb
        with patch.object(rb, "_detect_cgroup_memory_max", return_value=256 * 1024 * 1024):
            n, reason = rb._compute_max_procs()
        self.assertEqual(n, 1)
        # Negative-budget branch should annotate the reason
        self.assertIn("floored", reason)

    def test_compute_max_procs_with_custom_worker_rss(self):
        """Operator-tuned worker RSS estimate still flows through the formula."""
        from larkhelm import runner_base as rb
        with patch.object(rb, "_detect_cgroup_memory_max", return_value=4 * 1024 * 1024 * 1024):
            n_default, _ = rb._compute_max_procs()
            n_lean, _ = rb._compute_max_procs(worker_rss_mb=400)
        # 4096 - 400 - 400 = 3296 MB
        # default (800): 3296 // 800 = 4
        # lean (400):    3296 // 400 = 8
        self.assertEqual(n_default, 4)
        self.assertEqual(n_lean, rb.HARD_CEILING)  # 8, also == ceiling


class InitAiSemTests(unittest.TestCase):
    """``runner_base._init_ai_sem`` honours config override, else probes."""

    def setUp(self):
        # Snapshot pre-test state so we restore after each case — every test
        # in this class mutates the global cap and the live semaphore.
        from larkhelm import runner_base as rb
        self._rb = rb
        self._saved_cfg = getattr(_cfg, "MAX_AI_PROCS_CONFIG", None)
        self._saved_max = rb._MAX_AI_PROCS
        self._saved_sem = rb._ai_proc_sem
        self._saved_mirror = rb.MAX_AI_PROCS

    def tearDown(self):
        rb = self._rb
        rb._MAX_AI_PROCS = self._saved_max
        rb._ai_proc_sem = self._saved_sem
        rb.MAX_AI_PROCS = self._saved_mirror
        _cfg.MAX_AI_PROCS_CONFIG = self._saved_cfg

    def test_init_ai_sem_uses_config_override_when_set(self):
        """A positive int in config → use it verbatim, skip probing."""
        rb = self._rb
        _cfg.MAX_AI_PROCS_CONFIG = 5
        with patch.object(rb, "_compute_max_procs",
                          side_effect=AssertionError("probe should not be called")):
            rb._init_ai_sem()
        self.assertEqual(rb.get_max_ai_procs(), 5)
        self.assertEqual(rb.MAX_AI_PROCS, 5)
        # New sem should permit 5 immediate non-blocking acquires.
        acquired = []
        for _ in range(5):
            acquired.append(rb.get_ai_sem().acquire(blocking=False))
        try:
            self.assertEqual(acquired, [True] * 5)
            self.assertFalse(rb.get_ai_sem().acquire(blocking=False),
                             "6th acquire must block — capacity is 5")
        finally:
            for ok in acquired:
                if ok:
                    rb.get_ai_sem().release()

    def test_init_ai_sem_auto_detects_when_none(self):
        """Config=None → call _compute_max_procs."""
        rb = self._rb
        _cfg.MAX_AI_PROCS_CONFIG = None
        with patch.object(rb, "_compute_max_procs", return_value=(3, "test stub")) as probe:
            rb._init_ai_sem()
        probe.assert_called_once()
        self.assertEqual(rb.get_max_ai_procs(), 3)

    def test_init_ai_sem_invalid_config_falls_back_to_auto(self):
        """Non-int / non-positive values must trigger probe rather than crash."""
        rb = self._rb
        for bad in (0, -3, "auto", "twelve", [], {"n": 4}):
            _cfg.MAX_AI_PROCS_CONFIG = bad
            with patch.object(rb, "_compute_max_procs", return_value=(2, "probe")):
                rb._init_ai_sem()
            self.assertEqual(rb.get_max_ai_procs(), 2,
                             f"bad config value {bad!r} should fall back to probe")

    def test_init_ai_sem_rejects_bool_config_as_int(self):
        """``bool`` is a subclass of ``int`` in Python, so ``isinstance(True, int)``
        is True. Without an explicit ``not isinstance(raw, bool)`` guard,
        ``max_ai_procs: true`` in config.json silently becomes cap=1, which
        looks like correct behaviour but masks a user error. Regression guard
        for the round-3 review finding (#17)."""
        rb = self._rb
        for bad in (True, False):
            _cfg.MAX_AI_PROCS_CONFIG = bad
            with patch.object(rb, "_compute_max_procs", return_value=(5, "probe")):
                rb._init_ai_sem()
            self.assertEqual(rb.get_max_ai_procs(), 5,
                f"bool config value {bad!r} must NOT be coerced to int — "
                "must fall back to probe so the user sees the warning path "
                "in config._init_runtime")

    def test_init_ai_sem_idempotent_no_rebuild(self):
        """When the resolved value matches the current cap, the sem instance
        must NOT be rebuilt — that would orphan in-flight acquirers."""
        rb = self._rb
        _cfg.MAX_AI_PROCS_CONFIG = 4
        rb._init_ai_sem()
        sem_after_first = rb.get_ai_sem()
        self.assertEqual(rb.get_max_ai_procs(), 4)
        # Second call with same config — should be a no-op for the sem.
        rb._init_ai_sem()
        self.assertIs(rb.get_ai_sem(), sem_after_first,
                      "idempotent call must keep the same sem instance")


class CrossModuleStaleness_P0Tests(unittest.TestCase):
    """P0 regression: every module that needs the AI sem must see the LIVE one.

    The pre-fix bug was ``from larkhelm.runner_base import _ai_proc_sem`` —
    after ``_init_ai_sem()`` rebuilt the sem, importing modules still held
    the old binding. The fix routes all access through ``get_ai_sem()``
    (with ``__getattr__`` shim in ``ai_runner`` for back-compat).
    """

    def setUp(self):
        from larkhelm import runner_base as rb
        self._rb = rb
        self._saved_cfg = getattr(_cfg, "MAX_AI_PROCS_CONFIG", None)
        self._saved_max = rb._MAX_AI_PROCS
        self._saved_sem = rb._ai_proc_sem
        self._saved_mirror = rb.MAX_AI_PROCS

    def tearDown(self):
        rb = self._rb
        rb._MAX_AI_PROCS = self._saved_max
        rb._ai_proc_sem = self._saved_sem
        rb.MAX_AI_PROCS = self._saved_mirror
        _cfg.MAX_AI_PROCS_CONFIG = self._saved_cfg

    def test_get_ai_sem_consistent_across_runner_base_and_ai_runner(self):
        """ai_runner.get_ai_sem() and runner_base.get_ai_sem() must agree
        after a sem rebuild — this is the exact P0 bug."""
        from larkhelm import runner_base as rb
        from larkhelm import ai_runner as ar
        _cfg.MAX_AI_PROCS_CONFIG = 7
        rb._init_ai_sem()
        self.assertIs(rb.get_ai_sem(), ar.get_ai_sem(),
                      "ai_runner must see the rebuilt sem instance, not a stale "
                      "import-time copy. This was the round-3 v1 P0 bug.")

    def test_legacy_module_attr_shim_returns_live_sem(self):
        """``from ai_runner import _ai_proc_sem`` (legacy pattern, still in
        test_runner_refactoring / qa_verify) must resolve to the *current*
        sem via the module-level ``__getattr__`` shim."""
        from larkhelm import runner_base as rb
        from larkhelm import ai_runner as ar
        _cfg.MAX_AI_PROCS_CONFIG = 6
        rb._init_ai_sem()
        # Force the shim path
        legacy = ar._ai_proc_sem  # type: ignore[attr-defined]
        self.assertIs(legacy, rb.get_ai_sem(),
                      "ai_runner.__getattr__('_ai_proc_sem') must return live sem")

    def test_deepseek_runner_imports_getter_not_binding(self):
        """``runner_deepseek`` must NOT hold a frozen ``_ai_proc_sem`` binding
        (verifying the source change, since dynamic re-use under load is hard
        to assert at unit-test scope)."""
        import larkhelm.runner_deepseek as rd
        # The post-fix source imports get_ai_sem, not _ai_proc_sem.
        self.assertTrue(hasattr(rd, "get_ai_sem"),
                        "runner_deepseek must import get_ai_sem (post-fix)")
        # The stale binding should not be in the module's namespace.
        self.assertNotIn("_ai_proc_sem", vars(rd),
                         "runner_deepseek must not freeze the _ai_proc_sem binding — "
                         "use get_ai_sem() at acquire site instead")

def _make_fake_open(file_map: dict[str, str | Exception]):
    """Build a ``builtins.open`` replacement that serves a fixed virtual FS.

    ``file_map`` maps absolute paths to either a string (returned as a
    text-mode file's contents) or an Exception instance (raised when that
    path is opened). Any path not in the map raises ``FileNotFoundError`` —
    deliberately *not* falling back to the real filesystem so a test
    failure points cleanly at the missing mock rather than picking up a
    real ``/sys/fs/cgroup/...`` value from the host.
    """
    from io import StringIO

    def fake_open(path, *a, **kw):
        key = str(path)
        if key in file_map:
            val = file_map[key]
            if isinstance(val, Exception):
                raise val
            return StringIO(val)
        raise FileNotFoundError(key)

    return fake_open


class DetectorEdgeCasesTests(unittest.TestCase):
    """Low-level detectors — sanity over weird inputs."""

    def test_cgroup_detect_walks_self_cgroup_path(self):
        """Real prod scenario: process under /system.slice/larkhelm.service
        with a 2.8 GB memory.max at the leaf — function must read it directly,
        NOT fall back to the root /sys/fs/cgroup/memory.max."""
        from larkhelm import runner_base as rb
        fake = _make_fake_open({
            "/proc/self/cgroup": "0::/system.slice/larkhelm.service\n",
            "/sys/fs/cgroup/system.slice/larkhelm.service/memory.max": "2936012800\n",
            # Parent + root deliberately omitted from map; should not be reached.
        })
        with patch("builtins.open", side_effect=fake):
            self.assertEqual(rb._detect_cgroup_memory_max(), 2936012800)

    def test_cgroup_detect_walks_up_when_leaf_is_max(self):
        """Leaf says 'max' (no limit) → must walk up and find the parent's limit."""
        from larkhelm import runner_base as rb
        fake = _make_fake_open({
            "/proc/self/cgroup": "0::/system.slice/larkhelm.service\n",
            "/sys/fs/cgroup/system.slice/larkhelm.service/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/memory.max": "5368709120\n",
        })
        with patch("builtins.open", side_effect=fake):
            self.assertEqual(rb._detect_cgroup_memory_max(), 5368709120)

    def test_cgroup_detect_returns_none_when_all_max(self):
        """Every level in the hierarchy is 'max' (no limit) → return None,
        callers fall back to physical RAM."""
        from larkhelm import runner_base as rb
        fake = _make_fake_open({
            "/proc/self/cgroup": "0::/system.slice/larkhelm.service\n",
            "/sys/fs/cgroup/system.slice/larkhelm.service/memory.max": "max\n",
            "/sys/fs/cgroup/system.slice/memory.max": "max\n",
            "/sys/fs/cgroup/memory.max": "max\n",
        })
        with patch("builtins.open", side_effect=fake):
            self.assertIsNone(rb._detect_cgroup_memory_max())

    def test_cgroup_detect_returns_none_when_v1_format(self):
        """cgroup v1 host: /proc/self/cgroup is multi-line, none starting with
        '0::' — function must return None rather than misparse."""
        from larkhelm import runner_base as rb
        v1_contents = (
            "11:memory:/user.slice/user-1000.slice\n"
            "10:cpu,cpuacct:/user.slice\n"
            "9:devices:/user.slice\n"
        )
        fake = _make_fake_open({
            "/proc/self/cgroup": v1_contents,
        })
        with patch("builtins.open", side_effect=fake):
            self.assertIsNone(rb._detect_cgroup_memory_max())

    def test_cgroup_detect_returns_none_on_io_error(self):
        """/proc/self/cgroup unreadable (non-Linux, sandboxed env) → None,
        no exception propagates."""
        from larkhelm import runner_base as rb
        fake = _make_fake_open({
            "/proc/self/cgroup": FileNotFoundError("/proc/self/cgroup"),
        })
        with patch("builtins.open", side_effect=fake):
            self.assertIsNone(rb._detect_cgroup_memory_max())

    def test_detect_cgroup_memory_max_handles_max_sentinel(self):
        """cgroup v2 root 'max' sentinel = no limit → return None.

        Pre-bugfix this test used the root path directly; post-fix the
        function reads /proc/self/cgroup first, so we provide a v2 line
        pointing at root ('/') and put 'max' there.
        """
        from larkhelm import runner_base as rb
        fake = _make_fake_open({
            "/proc/self/cgroup": "0::/\n",
            "/sys/fs/cgroup/memory.max": "max\n",
        })
        with patch("builtins.open", side_effect=fake):
            self.assertIsNone(rb._detect_cgroup_memory_max())

    def test_detect_cgroup_memory_max_handles_missing_file(self):
        """FileNotFoundError at the leaf memory.max → keep walking; if every
        candidate is missing, return None."""
        from larkhelm import runner_base as rb
        fake = _make_fake_open({
            "/proc/self/cgroup": "0::/system.slice/larkhelm.service\n",
            # No memory.max anywhere in the chain → walk exhausts → None.
        })
        with patch("builtins.open", side_effect=fake):
            self.assertIsNone(rb._detect_cgroup_memory_max())

    def test_detect_physical_ram_mb_default_on_failure(self):
        """If both /proc/meminfo and `sysctl hw.memsize` are unavailable,
        return the conservative 4096 fallback."""
        from larkhelm import runner_base as rb
        import builtins
        real_open = builtins.open

        def fake_open(path, *a, **kw):
            if str(path) == "/proc/meminfo":
                raise OSError("simulated")
            return real_open(path, *a, **kw)

        # Also fail the macOS/BSD sysctl fallback so the final 4096 is reached.
        with patch("builtins.open", side_effect=fake_open), \
                patch("larkhelm.runner_base.subprocess.run",
                      side_effect=OSError("no sysctl")):
            self.assertEqual(rb._detect_physical_ram_mb(), 4096)

    def test_detect_physical_ram_mb_macos_sysctl_fallback(self):
        """On macOS (no /proc/meminfo) the probe reads `sysctl hw.memsize`."""
        from larkhelm import runner_base as rb
        import builtins
        from unittest.mock import MagicMock
        real_open = builtins.open

        def fake_open(path, *a, **kw):
            if str(path) == "/proc/meminfo":
                raise OSError("simulated")
            return real_open(path, *a, **kw)

        fake_proc = MagicMock(returncode=0, stdout=str(24 * 1024 * 1024 * 1024))
        with patch("builtins.open", side_effect=fake_open), \
                patch("larkhelm.runner_base.subprocess.run", return_value=fake_proc):
            self.assertEqual(rb._detect_physical_ram_mb(), 24 * 1024)


if __name__ == "__main__":
    unittest.main()
