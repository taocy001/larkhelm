"""larkhelm voice · install-time system capability probe.

Run via ``larkhelm voice probe`` after install. Decides whether the local
faster-whisper STT path is viable on this machine and writes the verdict
into ``config.json`` so the bridge will honor it on next start.

Frugal-default principle (per user):
* DashScope (云 API) is **never** chosen automatically — it costs money.
* If local capability check passes → ``voice_enabled=true`` automatically.
* If local capability fails → ``voice_enabled`` stays ``false``; the
  printed report shows the user how to opt into DashScope manually.

Probe stages (skip with ``--no-benchmark`` to drop stage 4):
1. ffmpeg present (apt-installed binary) — required by faster-whisper
2. CPU flags from ``/proc/cpuinfo`` (AVX2 / AVX / SSE4 / FMA)
3. Available memory from ``/proc/meminfo``
4. Real benchmark: load small int8 + transcribe 1 second of synthesized
   audio, measure RTF (real-time factor). Threshold ``RTF < 0.8`` → viable.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# RTF threshold below which we consider local STT viable. 0.8 = transcribes
# faster than realtime with 20% headroom; small int8 on QEMU SSE4 measured
# 0.64 in benchmark §9.7 → marginal viable. Caller can override via
# ``RTF_THRESHOLD`` constant or by editing config.json post-probe.
RTF_THRESHOLD = 0.8

# Memory floors per faster-whisper model size (int8 quantization). Numbers
# include peak inference buffer; resting model RSS is roughly half. Source:
# CTranslate2 docs + benchmark §9.1 ("small 跑起来已只剩 956 MB free").
_MIN_RAM_MB = {
    "tiny":     500,
    "base":     800,
    "small":   1_500,
    "medium":  2_500,
    "large":   4_000,
}


@dataclass
class ProbeResult:
    """Aggregated results of a probe pass.

    Each ``ok`` bool is the per-stage verdict; ``viable`` is the overall
    AND of the relevant stages. ``decision`` carries the recommended
    config writeback.
    """
    ffmpeg_present: bool = False
    ffmpeg_path: str = ""
    cpu_has_avx2: bool = False
    cpu_has_avx: bool = False
    cpu_has_sse4: bool = False
    cpu_has_fma: bool = False
    mem_total_mb: int = 0
    mem_available_mb: int = 0
    benchmark_ran: bool = False
    benchmark_load_sec: float = 0.0
    benchmark_infer_sec: float = 0.0
    benchmark_audio_sec: float = 0.0
    benchmark_rtf: float = 0.0
    benchmark_error: str = ""
    # Decision
    viable: bool = False
    recommended_size: str = "small"
    recommendation_reason: str = ""
    extra_notes: list[str] = field(default_factory=list)


def probe_ffmpeg() -> tuple[bool, str]:
    """Locate ``ffmpeg`` in PATH. Required by faster-whisper for decoding."""
    p = shutil.which("ffmpeg")
    return (p is not None, p or "")


def probe_cpu_flags() -> dict[str, bool]:
    """Read CPU flags from ``/proc/cpuinfo`` (Linux only).

    Returns ``{"avx2": bool, "avx": bool, "sse4": bool, "fma": bool}``.
    On non-Linux or unreadable proc, returns all ``False`` (caller treats
    as no acceleration available — conservative).
    """
    out = {"avx2": False, "avx": False, "sse4": False, "fma": False}
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("flags") or line.startswith("Features"):
                    flags = set(line.split(":", 1)[1].split())
                    out["avx2"] = "avx2" in flags
                    out["avx"] = "avx" in flags
                    out["sse4"] = ("sse4_2" in flags) or ("sse4_1" in flags)
                    out["fma"] = "fma" in flags
                    break
    except (FileNotFoundError, PermissionError):
        pass
    return out


def probe_memory() -> tuple[int, int]:
    """Read total + available RAM from ``/proc/meminfo``, return (total_mb, available_mb)."""
    total_kb = available_kb = 0
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kb = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kb = int(line.split()[1])
    except (FileNotFoundError, PermissionError):
        pass
    return total_kb // 1024, available_kb // 1024


def synthesize_test_wav(out_path: Path, duration_sec: float = 1.0) -> bool:
    """Generate a 1-second sine wave WAV for benchmark input. Uses ffmpeg
    ``lavfi`` source (no extra dep). Returns True on success.
    """
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "quiet",
                "-f", "lavfi",
                "-i", f"sine=frequency=440:duration={duration_sec}",
                "-ar", "16000", "-ac", "1",
                str(out_path),
            ],
            check=True, timeout=10,
        )
        return out_path.exists() and out_path.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return False


def run_benchmark(model_size: str = "small") -> tuple[bool, dict[str, Any]]:
    """Load faster-whisper at ``model_size`` and transcribe a 1-sec audio.

    Returns ``(ok, info)`` where ``info`` carries timings + any error message.
    Designed to never raise — failures collapse into ``ok=False``.

    First-time call downloads ~244 MB (small) from HuggingFace, which can
    dominate ``load_sec`` on slow networks; subsequent calls hit local cache.
    """
    info: dict[str, Any] = {
        "load_sec": 0.0, "infer_sec": 0.0, "audio_sec": 1.0,
        "rtf": float("inf"), "error": "",
    }
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        info["error"] = f"faster-whisper not installed: {e}"
        return False, info

    tmpdir = Path("/tmp/larkhelm_voice_probe")
    tmpdir.mkdir(parents=True, exist_ok=True)
    wav_path = tmpdir / "probe_1s.wav"
    if not wav_path.exists() and not synthesize_test_wav(wav_path, 1.0):
        info["error"] = "ffmpeg sine synthesis failed; cannot benchmark"
        return False, info

    try:
        t0 = time.monotonic()
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        info["load_sec"] = time.monotonic() - t0

        t1 = time.monotonic()
        segments, _ = model.transcribe(str(wav_path), language="zh", beam_size=1, vad_filter=False)
        # Force generator drain to capture full inference time
        for _ in segments:
            pass
        info["infer_sec"] = time.monotonic() - t1
        info["rtf"] = info["infer_sec"] / info["audio_sec"] if info["audio_sec"] > 0 else float("inf")
        return True, info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        return False, info


def decide(result: ProbeResult) -> None:
    """Fill in ``result.viable`` / ``result.recommended_size`` / ``result.recommendation_reason``.

    Decision policy:
    * **No ffmpeg** → not viable (download/decode would fail at runtime).
    * **Benchmark ran**: trust RTF directly (``< RTF_THRESHOLD`` → viable).
    * **No benchmark + AVX2 + RAM ≥ 2.5GB** → recommend medium.
    * **No benchmark + AVX + RAM ≥ 1.5GB** → recommend small.
    * **No benchmark + SSE4 only** → recommend small but note "marginal".
    * Otherwise → not viable; suggest DashScope opt-in path.
    """
    notes = result.extra_notes
    if not result.ffmpeg_present:
        result.viable = False
        result.recommendation_reason = "ffmpeg not installed (apt install ffmpeg)"
        notes.append("DashScope path also needs no ffmpeg — see README.")
        return

    # Pick minimum size for which we have RAM
    sized: Optional[str] = None
    if result.cpu_has_avx2 and result.mem_available_mb >= _MIN_RAM_MB["medium"]:
        sized = "medium"
    elif result.cpu_has_avx and result.mem_available_mb >= _MIN_RAM_MB["small"]:
        sized = "small"
    elif result.cpu_has_sse4 and result.mem_available_mb >= _MIN_RAM_MB["small"]:
        sized = "small"  # marginal; note added below
        notes.append("CPU lacks AVX → CTranslate2 falls back to SSE4 path "
                     "(3-5× slower than AVX2 baseline). Real benchmark recommended.")
    elif result.mem_available_mb >= _MIN_RAM_MB["base"]:
        sized = "base"
        notes.append("base model accuracy is poor (~22% WER on Chinese); "
                     "consider DashScope for production.")

    if sized is None:
        result.viable = False
        result.recommendation_reason = (
            f"insufficient resources: avx={result.cpu_has_avx}, "
            f"sse4={result.cpu_has_sse4}, available_ram={result.mem_available_mb}MB"
        )
        return

    result.recommended_size = sized

    if result.benchmark_ran:
        if result.benchmark_rtf < RTF_THRESHOLD:
            result.viable = True
            result.recommendation_reason = (
                f"real benchmark RTF={result.benchmark_rtf:.2f} < {RTF_THRESHOLD} threshold"
            )
        else:
            result.viable = False
            result.recommendation_reason = (
                f"real benchmark RTF={result.benchmark_rtf:.2f} ≥ {RTF_THRESHOLD} threshold "
                f"(too slow for interactive use)"
            )
    else:
        # Static-only verdict — trust the flag/RAM heuristic; RTF unknown.
        result.viable = True
        result.recommendation_reason = (
            f"static probe: cpu={'avx2' if result.cpu_has_avx2 else 'avx' if result.cpu_has_avx else 'sse4'}, "
            f"available_ram={result.mem_available_mb}MB, recommended size={sized}; "
            f"run without --no-benchmark for RTF verification"
        )


def run_full_probe(*, with_benchmark: bool = True) -> ProbeResult:
    """Execute every probe stage and return a populated ``ProbeResult``."""
    result = ProbeResult()
    result.ffmpeg_present, result.ffmpeg_path = probe_ffmpeg()
    flags = probe_cpu_flags()
    result.cpu_has_avx2 = flags["avx2"]
    result.cpu_has_avx = flags["avx"]
    result.cpu_has_sse4 = flags["sse4"]
    result.cpu_has_fma = flags["fma"]
    result.mem_total_mb, result.mem_available_mb = probe_memory()

    if with_benchmark and result.ffmpeg_present:
        # Pre-decide size for benchmark — will refine after
        if result.cpu_has_avx2 and result.mem_available_mb >= _MIN_RAM_MB["medium"]:
            test_size = "small"  # always benchmark with small (faster, accurate signal)
        else:
            test_size = "small"
        ok, info = run_benchmark(test_size)
        result.benchmark_ran = ok
        result.benchmark_load_sec = info["load_sec"]
        result.benchmark_infer_sec = info["infer_sec"]
        result.benchmark_audio_sec = info["audio_sec"]
        result.benchmark_rtf = info["rtf"]
        result.benchmark_error = info["error"]

    decide(result)
    return result


def apply_to_config(result: ProbeResult, config_path: Path) -> dict:
    """Write probe verdict into config.json (with ``.bak`` backup).

    Returns the dict of fields written for the report.

    Fields written:
    * ``voice_enabled`` ← ``result.viable``
    * ``voice_engine`` ← ``"faster_whisper"`` (always; user changes manually for dashscope)
    * ``voice_model_size`` ← ``result.recommended_size`` (only if viable)
    * ``voice_probe_done`` ← ``True``

    Existing fields unrelated to voice are preserved verbatim.
    """
    if not config_path.exists():
        return {}
    backup = config_path.with_suffix(".json.bak-pre-voice-probe")
    shutil.copy2(config_path, backup)

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    written: dict[str, Any] = {
        "voice_enabled": result.viable,
        "voice_engine": "faster_whisper",
        "voice_probe_done": True,
    }
    if result.viable:
        written["voice_model_size"] = result.recommended_size
    cfg.update(written)

    config_path.write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return written


def print_report(result: ProbeResult, *, write_path: Optional[Path] = None) -> None:
    """Pretty-print probe results + verdict + next-steps to stdout/stderr."""
    p = print

    p("🔍 larkhelm voice probe — local STT capability check")
    p("=" * 56)
    p(f"\n[1/4] System dependencies")
    p(f"  ffmpeg: {'✅ ' + result.ffmpeg_path if result.ffmpeg_present else '❌ not found (apt install ffmpeg)'}")

    p(f"\n[2/4] CPU capability")
    flags = []
    flags.append(f"AVX2 {'✓' if result.cpu_has_avx2 else '✗'}")
    flags.append(f"AVX {'✓' if result.cpu_has_avx else '✗'}")
    flags.append(f"SSE4 {'✓' if result.cpu_has_sse4 else '✗'}")
    flags.append(f"FMA {'✓' if result.cpu_has_fma else '✗'}")
    p(f"  flags: {'  '.join(flags)}")

    p(f"\n[3/4] Memory")
    p(f"  total:     {result.mem_total_mb} MB")
    p(f"  available: {result.mem_available_mb} MB")

    p(f"\n[4/4] Real benchmark")
    if result.benchmark_ran:
        p(f"  load:      {result.benchmark_load_sec:6.2f} s   (first run includes ~244 MB HF download)")
        p(f"  inference: {result.benchmark_infer_sec:6.2f} s   for {result.benchmark_audio_sec:.1f}s audio")
        p(f"  RTF:       {result.benchmark_rtf:6.2f}      (threshold < {RTF_THRESHOLD} for viability)")
    elif result.benchmark_error:
        p(f"  ❌ failed: {result.benchmark_error}")
    else:
        p(f"  ⏭️  skipped (--no-benchmark or ffmpeg missing)")

    p("\n" + "=" * 56)
    if result.viable:
        p(f"✅ Verdict: local model viable")
    else:
        p(f"❌ Verdict: local model NOT viable on this system")
    p(f"   Reason: {result.recommendation_reason}")
    for note in result.extra_notes:
        p(f"   Note:   {note}")

    if write_path is not None:
        p(f"\n📌 Action taken (config.json):")
        p(f"   - voice_enabled = {str(result.viable).lower()}")
        if result.viable:
            p(f"   - voice_engine = faster_whisper")
            p(f"   - voice_model_size = {result.recommended_size}")
        p(f"   - voice_probe_done = true")
        p(f"   backup at: {write_path.with_suffix('.json.bak-pre-voice-probe')}")
        if result.viable:
            p(f"\n   Next: sudo systemctl restart larkhelm  →  send a voice message in Feishu.")
        else:
            p(f"\n   Next (DashScope opt-in path):")
            p(f"     1) pipx runpip larkhelm install dashscope")
            p(f"     2) Edit {write_path}: set voice_enabled=true and voice_engine=\"dashscope\"")
            p(f"     3) Add Environment=\"DASHSCOPE_API_KEY=sk-...\" to systemd drop-in")
            p(f"     4) sudo systemctl restart larkhelm")
            p(f"     See README §\"语音功能\" for the full DashScope setup.")


def cli_main(argv: list[str]) -> int:
    """Entry point for ``larkhelm voice probe`` CLI subcommand."""
    import argparse
    p = argparse.ArgumentParser(prog="larkhelm voice probe")
    p.add_argument("--no-benchmark", action="store_true",
                   help="skip the real-inference benchmark; check CPU flags + RAM only "
                        "(faster but RTF is estimated, not measured)")
    p.add_argument("--no-write", action="store_true",
                   help="don't write results to config.json; just print the report")
    p.add_argument("--config", metavar="PATH",
                   help="path to config.json (default: auto-detect)")
    args = p.parse_args(argv)

    # Locate config (read-only initially) — same priority order as bridge
    if args.config:
        config_path = Path(args.config)
    else:
        candidates = [
            Path(os.environ.get("LARKHELM_CONFIG", "")) if os.environ.get("LARKHELM_CONFIG") else None,
            Path("/etc/larkhelm/config.json"),
            Path.home() / ".config/larkhelm/config.json",
        ]
        config_path = next((c for c in candidates if c and c.exists()), None)
        if config_path is None and not args.no_write:
            print("❌ config.json not found; pass --config PATH or use --no-write",
                  file=sys.stderr)
            return 2

    print(f"Probing system… (config: {config_path})\n", file=sys.stderr)
    result = run_full_probe(with_benchmark=not args.no_benchmark)

    write_path: Optional[Path] = None
    if not args.no_write and config_path is not None:
        try:
            apply_to_config(result, config_path)
            write_path = config_path
        except Exception as e:
            print(f"⚠️ config writeback failed: {e}", file=sys.stderr)

    print_report(result, write_path=write_path)
    return 0 if result.viable else 1
