"""Tests for ``larkhelm.voice.system_probe`` — the install-time capability probe.

Coverage:
* Static-only path (``--no-benchmark`` / ``with_benchmark=False``):
  - AVX2 + RAM ≥ 2.5GB → recommend medium, viable
  - AVX + RAM ≥ 1.5GB → recommend small, viable
  - SSE4 only + RAM ≥ 1.5GB → recommend small, viable, marginal note
  - No AVX/SSE4 with sufficient RAM → fall back to base, note about WER
  - No ffmpeg → not viable regardless of CPU/RAM
* Real-benchmark path:
  - RTF < threshold → viable
  - RTF ≥ threshold → not viable
* Config writeback: backup created + voice_enabled / voice_engine / voice_probe_done
  written; existing keys preserved.

Real model loading is **never** triggered in tests — ``run_benchmark`` is
patched. CPU-flag and meminfo probes are tested via monkey-patched ``open``.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from larkhelm.voice import system_probe as sp


# ── Stubbed /proc/* readers ──────────────────────────────────────────────


def _fake_cpuinfo(flags: str) -> str:
    """Build a minimal /proc/cpuinfo body with the supplied flag list."""
    return (
        "processor\t: 0\n"
        "vendor_id\t: GenuineIntel\n"
        f"flags\t\t: {flags}\n"
    )


def _fake_meminfo(total_kb: int, available_kb: int) -> str:
    return (
        f"MemTotal:       {total_kb} kB\n"
        f"MemFree:        100 kB\n"
        f"MemAvailable:   {available_kb} kB\n"
    )


# ── probe_cpu_flags / probe_memory ────────────────────────────────────────


def test_cpu_flags_avx2_present(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        lambda p, *a, **kw: io.StringIO(
            _fake_cpuinfo("fpu vme avx avx2 sse4_1 sse4_2 fma popcnt")
        ),
    )
    flags = sp.probe_cpu_flags()
    assert flags == {"avx2": True, "avx": True, "sse4": True, "fma": True}


def test_cpu_flags_no_avx_qemu_default(monkeypatch):
    """QEMU 默认 qemu64 模型只给 SSE4，没有 AVX/AVX2/FMA — 本机就是这种。"""
    # Prevent the Apple Silicon early-return path (arm64/Darwin → all-True).
    # platform is imported locally inside probe_cpu_flags; patch at stdlib level.
    with patch("platform.machine", return_value="x86_64"), \
         patch("platform.system", return_value="Linux"), \
         patch("builtins.open",
               lambda p, *a, **kw: io.StringIO(
                   _fake_cpuinfo("fpu sse sse2 sse4_1 sse4_2 popcnt aes")
               )):
        flags = sp.probe_cpu_flags()
    assert flags == {"avx2": False, "avx": False, "sse4": True, "fma": False}


def test_cpu_flags_no_proc(monkeypatch):
    """Non-Linux / unreadable proc → all False (conservative fallback)."""
    # Prevent the Apple Silicon early-return path (arm64/Darwin → all-True).
    def _raise(*a, **kw):
        raise FileNotFoundError("/proc/cpuinfo")
    with patch("platform.machine", return_value="x86_64"), \
         patch("platform.system", return_value="Linux"), \
         patch("builtins.open", _raise):
        flags = sp.probe_cpu_flags()
    assert flags == {"avx2": False, "avx": False, "sse4": False, "fma": False}


def test_memory_parse(monkeypatch):
    monkeypatch.setattr(
        "builtins.open",
        lambda p, *a, **kw: io.StringIO(_fake_meminfo(4_000_000, 2_500_000)),
    )
    total, available = sp.probe_memory()
    assert total == 4_000_000 // 1024  # ≈ 3906
    assert available == 2_500_000 // 1024  # ≈ 2441


# ── decide() — static-only branches (benchmark_ran=False) ─────────────────


def _result_with(*, ffmpeg=True, avx2=False, avx=False, sse4=True, fma=False,
                 mem_avail=2000) -> sp.ProbeResult:
    return sp.ProbeResult(
        ffmpeg_present=ffmpeg, ffmpeg_path="/usr/bin/ffmpeg" if ffmpeg else "",
        cpu_has_avx2=avx2, cpu_has_avx=avx, cpu_has_sse4=sse4, cpu_has_fma=fma,
        mem_total_mb=4096, mem_available_mb=mem_avail,
    )


def test_decide_no_ffmpeg_unviable():
    r = _result_with(ffmpeg=False)
    sp.decide(r)
    assert not r.viable
    assert "ffmpeg" in r.recommendation_reason.lower()


def test_decide_avx2_recommends_medium_and_viable_static():
    r = _result_with(avx2=True, avx=True, sse4=True, mem_avail=3000)
    sp.decide(r)
    assert r.viable
    assert r.recommended_size == "medium"
    assert "static probe" in r.recommendation_reason


def test_decide_avx_only_recommends_small():
    r = _result_with(avx2=False, avx=True, sse4=True, mem_avail=2000)
    sp.decide(r)
    assert r.viable
    assert r.recommended_size == "small"


def test_decide_sse4_only_marginal_note():
    """No AVX → still small viable but with a 'CPU lacks AVX' marginal note."""
    r = _result_with(avx2=False, avx=False, sse4=True, mem_avail=2000)
    sp.decide(r)
    assert r.viable  # static-only optimistic; user runs benchmark for real verdict
    assert r.recommended_size == "small"
    assert any("lacks AVX" in n for n in r.extra_notes)


def test_decide_low_ram_falls_to_base():
    """≥800 MB but <1500 MB → base, with WER warning note."""
    r = _result_with(avx=False, sse4=True, mem_avail=1000)
    sp.decide(r)
    assert r.recommended_size == "base"
    assert any("WER" in n for n in r.extra_notes)


def test_decide_too_low_ram_unviable():
    r = _result_with(avx=False, sse4=True, mem_avail=400)
    sp.decide(r)
    assert not r.viable
    assert "insufficient" in r.recommendation_reason


# ── decide() — real-benchmark branches (benchmark_ran=True) ───────────────


def test_decide_benchmark_below_threshold_viable():
    r = _result_with(avx2=True, avx=True, mem_avail=3000)
    r.benchmark_ran = True
    r.benchmark_audio_sec = 1.0
    r.benchmark_infer_sec = 0.4
    r.benchmark_rtf = 0.4
    sp.decide(r)
    assert r.viable
    assert "RTF=0.40" in r.recommendation_reason


def test_decide_benchmark_above_threshold_unviable():
    """RTF=1.5 → above 0.8 threshold → not viable even on AVX2 box."""
    r = _result_with(avx2=True, avx=True, mem_avail=3000)
    r.benchmark_ran = True
    r.benchmark_audio_sec = 1.0
    r.benchmark_infer_sec = 1.5
    r.benchmark_rtf = 1.5
    sp.decide(r)
    assert not r.viable
    assert "1.50" in r.recommendation_reason


def test_decide_benchmark_qemu_sse4_marginal_passes():
    """Benchmark §9.7: QEMU SSE4 small RTF=0.64 → passes 0.8 threshold."""
    r = _result_with(avx2=False, avx=False, sse4=True, mem_avail=2000)
    r.benchmark_ran = True
    r.benchmark_audio_sec = 1.0
    r.benchmark_infer_sec = 0.64
    r.benchmark_rtf = 0.64
    sp.decide(r)
    assert r.viable
    assert r.recommended_size == "small"


# ── apply_to_config — writeback + backup ──────────────────────────────────


def test_apply_to_config_writes_voice_fields(tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID": "x", "APP_SECRET": "y",
        "default_model": "claude",
    }, ensure_ascii=False, indent=2))

    r = _result_with(avx=True, sse4=True, mem_avail=2000)
    r.viable = True
    r.recommended_size = "small"
    written = sp.apply_to_config(r, cfg_path)

    assert written["voice_enabled"] is True
    assert written["voice_engine"] == "faster_whisper"
    assert written["voice_model_size"] == "small"
    assert written["voice_probe_done"] is True

    # Backup created
    backup = cfg_path.with_suffix(".json.bak-pre-voice-probe")
    assert backup.exists()

    # Existing keys preserved
    cfg = json.loads(cfg_path.read_text())
    assert cfg["APP_ID"] == "x"
    assert cfg["APP_SECRET"] == "y"
    assert cfg["default_model"] == "claude"
    assert cfg["voice_enabled"] is True
    assert cfg["voice_model_size"] == "small"


def test_apply_to_config_preserves_dashscope_engine_choice(tmp_path):
    """Re-running probe must NOT silently flip a user's voice_engine="dashscope"
    back to faster_whisper. Frugal-default invariant: probe is install-time
    only, the user's later config edits are the source of truth for engine."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "APP_ID": "x", "APP_SECRET": "y",
        "voice_engine": "dashscope",
        "voice_api_key": "${DASHSCOPE_API_KEY}",
    }, ensure_ascii=False, indent=2))

    r = _result_with(avx=True, sse4=True, mem_avail=2000)
    r.viable = True
    r.recommended_size = "small"
    sp.apply_to_config(r, cfg_path)

    cfg = json.loads(cfg_path.read_text())
    assert cfg["voice_engine"] == "dashscope"  # preserved
    assert cfg.get("voice_api_key") == "${DASHSCOPE_API_KEY}"  # preserved
    # voice_model_size NOT written when user is on dashscope (it'd be unused)
    assert "voice_model_size" not in cfg
    # voice_enabled and voice_probe_done still get written
    assert cfg["voice_enabled"] is True
    assert cfg["voice_probe_done"] is True


def test_apply_to_config_unviable_omits_size(tmp_path):
    """Not viable → don't pin a size (user will hand-edit if going dashscope)."""
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"APP_ID": "x"}, ensure_ascii=False, indent=2))

    r = _result_with(ffmpeg=False)
    r.viable = False
    written = sp.apply_to_config(r, cfg_path)

    assert written["voice_enabled"] is False
    assert "voice_model_size" not in written


def test_apply_to_config_missing_returns_empty(tmp_path):
    """Non-existent config.json → no-op, returns {} so caller can decide."""
    r = _result_with()
    r.viable = True
    written = sp.apply_to_config(r, tmp_path / "nonexistent.json")
    assert written == {}


# ── run_full_probe integration (with benchmark stubbed out) ───────────────


def test_run_full_probe_with_benchmark_stub(monkeypatch):
    """run_full_probe wires together flag + memory + benchmark + decide."""
    monkeypatch.setattr(sp, "probe_ffmpeg", lambda: (True, "/usr/bin/ffmpeg"))
    monkeypatch.setattr(sp, "probe_cpu_flags", lambda: {
        "avx2": False, "avx": False, "sse4": True, "fma": False,
    })
    monkeypatch.setattr(sp, "probe_memory", lambda: (3800, 2600))
    monkeypatch.setattr(sp, "run_benchmark", lambda size: (True, {
        "load_sec": 12.5, "infer_sec": 0.64, "audio_sec": 1.0,
        "rtf": 0.64, "error": "",
    }))

    r = sp.run_full_probe(with_benchmark=True)
    assert r.benchmark_ran
    assert r.benchmark_rtf == pytest.approx(0.64)
    assert r.viable  # 0.64 < 0.8 threshold
    assert r.recommended_size == "small"


def test_run_full_probe_no_benchmark_static_only(monkeypatch):
    monkeypatch.setattr(sp, "probe_ffmpeg", lambda: (True, "/usr/bin/ffmpeg"))
    monkeypatch.setattr(sp, "probe_cpu_flags", lambda: {
        "avx2": True, "avx": True, "sse4": True, "fma": True,
    })
    monkeypatch.setattr(sp, "probe_memory", lambda: (8000, 4000))

    r = sp.run_full_probe(with_benchmark=False)
    assert not r.benchmark_ran
    assert r.viable
    assert r.recommended_size == "medium"
