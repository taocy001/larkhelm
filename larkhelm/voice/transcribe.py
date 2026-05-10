"""larkhelm voice · WhisperModel singleton + serial inference worker (M3.2).

Public surface (re-exported by ``larkhelm.voice``):

    transcribe(audio_path, *, lang="zh", beam_size=1) -> TranscribeResult
    is_ready() -> bool
    is_model_loaded() -> bool

Design contract (see ``.crew_workspace/design.md`` v1.0):

* ``faster_whisper`` is imported lazily inside ``_load_model``; merely
  importing this module must not pull in faster-whisper. Bridge boot and
  text-only chat paths therefore pay zero cost when the voice feature
  is disabled or the dependency is missing.
* ``_load_model`` uses double-checked locking — ``max_workers=1`` already
  serializes inference, but ``is_ready`` / ``is_model_loaded`` are read
  from caller threads and need the explicit memory barrier the lock
  provides.
* Failure is terminal for the process: any ImportError or constructor
  exception flips ``_LOAD_FAILED=True`` and writes ``_cfg.VOICE_ENABLED=False``
  via ``_disable_voice``. There is no retry / half-open / reset path
  (PRD §4.2 — "warn + 关 VOICE_ENABLED, 禁止重试").
* ``transcribe`` never raises; failures are encoded as
  ``TranscribeResult(ok=False, error=...)`` so callers can branch on
  ``r["ok"]`` instead of try/except.
"""
from __future__ import annotations

import atexit
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, TypedDict

import larkhelm.config as _cfg
from larkhelm import log as _log


class TranscribeResult(TypedDict):
    ok: bool
    text: str
    duration: float
    lang: str
    error: Optional[str]


# ── Module-level singleton state ──────────────────────────────────────────
# Tests reset these in setUp; do NOT wrap them in a class — the simple
# module-level form mirrors ``larkhelm/log.py:_min_level`` and avoids a
# setter/getter layer that mypy would have to chase.
_MODEL: Optional[object] = None        # WhisperModel instance once loaded
_LOAD_FAILED: bool = False              # terminal flag — set once, never cleared
_LOAD_LOGGED_SUCCESS: bool = False      # gate for "loaded in Xs" info message
_load_model_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-stt")


def _disable_voice(reason: str) -> None:
    """Flip the terminal DISABLED state. Idempotent — calling twice is a no-op
    aside from a duplicate warn line, which is acceptable for diagnostics."""
    global _LOAD_FAILED
    _LOAD_FAILED = True
    try:
        _cfg.VOICE_ENABLED = False
    except Exception:
        pass
    _log.warn(f"[Voice] disabled: {reason}")


def _load_model() -> Optional[object]:
    """Return the WhisperModel singleton, lazy-loading on first call.

    Double-checked locking: the unlocked check makes the hot path
    (model already loaded) a single read; the locked check inside the
    critical section guards against two threads racing to construct
    twice. ``max_workers=1`` makes the race window narrow but not
    impossible, since ``is_ready`` / ``is_model_loaded`` run on caller
    threads.

    Never raises — ImportError and any WhisperModel constructor
    exception is captured and routed through ``_disable_voice``.
    """
    global _MODEL, _LOAD_LOGGED_SUCCESS
    if _LOAD_FAILED:
        return None
    if _MODEL is not None:
        return _MODEL
    with _load_model_lock:
        if _LOAD_FAILED:
            return None
        if _MODEL is not None:
            return _MODEL
        t0 = time.monotonic()
        try:
            from faster_whisper import WhisperModel  # lazy import boundary
        except Exception as e:
            _disable_voice(f"import faster_whisper failed: {e}")
            return None
        try:
            model = WhisperModel(
                _cfg.VOICE_MODEL_SIZE,
                compute_type=_cfg.VOICE_COMPUTE_TYPE,
            )
        except Exception as e:
            _disable_voice(f"WhisperModel({_cfg.VOICE_MODEL_SIZE!r}) ctor failed: {e}")
            return None
        _MODEL = model
        if not _LOAD_LOGGED_SUCCESS:
            _LOAD_LOGGED_SUCCESS = True
            _log.info(
                f"[Voice] model {_cfg.VOICE_MODEL_SIZE!r} "
                f"({_cfg.VOICE_COMPUTE_TYPE}) loaded in {time.monotonic() - t0:.1f}s"
            )
        return _MODEL


def _run_inference(audio_path: str, lang: str, beam_size: int) -> TranscribeResult:
    """Worker-thread body: load model on first call, then run inference.

    Lives strictly inside ``_executor``'s single worker thread; never
    raises. Any exception (model load failure, file-not-found, decode
    error) collapses into a ``TranscribeResult`` with a non-empty
    ``error`` field.
    """
    model = _load_model()
    if model is None:
        return TranscribeResult(
            ok=False, text="", duration=0.0, lang=lang,
            error="load_failed",
        )
    try:
        segments_iter, info = model.transcribe(
            audio_path,
            language=None if lang == "auto" else lang,
            beam_size=beam_size,
        )
        text = "".join(seg.text for seg in segments_iter).strip()
        detected = getattr(info, "language", lang) or lang
        duration = float(getattr(info, "duration", 0.0) or 0.0)
        return TranscribeResult(
            ok=True, text=text, duration=duration, lang=detected,
            error=None,
        )
    except Exception as e:
        return TranscribeResult(
            ok=False, text="", duration=0.0, lang=lang,
            error=f"inference_failed:{e}",
        )


def transcribe(
    audio_path: "str | Path",
    *,
    lang: str = "zh",
    beam_size: int = 1,
) -> TranscribeResult:
    """Transcribe an audio file to text. Blocking; never raises.

    Parameters
    ----------
    audio_path : str | Path
        Path to a faster-whisper-readable audio file (ogg / wav / mp3 / ...).
    lang : str
        ``"zh"`` / ``"en"`` / ``"auto"`` (auto = let whisper detect).
        Other values are forwarded as-is; whitelisting lives in
        ``_cfg`` validation, not here.
    beam_size : int
        Whisper beam size; default 1 for speed. Callers wanting accuracy
        can bump to 5.

    Returns
    -------
    TranscribeResult
        Dict with stable keys ``ok / text / duration / lang / error``.
        On any failure ``ok`` is False and ``error`` carries a short tag:
        ``"disabled"`` (already in DISABLED state),
        ``"load_failed"`` (model load just failed this call),
        ``"inference_failed:<msg>"`` (whisper raised mid-decode).

    Notes for callers
    -----------------
    * Check ``_cfg.VOICE_ENABLED`` before calling — calling when the
      feature is off still returns a valid result but wastes one
      executor submission.
    * The handlers layer is responsible for any external timeout —
      this function does not wrap ``future.result(timeout=...)``.
    """
    if _LOAD_FAILED or not _cfg.VOICE_ENABLED:
        return TranscribeResult(
            ok=False, text="", duration=0.0, lang=lang,
            error="disabled",
        )
    # TODO(M3.2 next commit): enforce _cfg.VOICE_MAX_DURATION_MS here once
    # handlers commit wires the duration check; until then the caller
    # (lark_client._download_message_file) is the gatekeeper.
    fut = _executor.submit(_run_inference, str(audio_path), lang, beam_size)
    return fut.result()


def is_ready() -> bool:
    """True iff the feature is enabled and not in DISABLED state.

    Does not trigger a load. Cheap; safe to call from hot paths.
    """
    return bool(_cfg.VOICE_ENABLED) and not _LOAD_FAILED


def is_model_loaded() -> bool:
    """True iff the WhisperModel singleton has been instantiated.

    Does not trigger a load. Useful for ``/status`` introspection.
    """
    return _MODEL is not None


def _shutdown_executor() -> None:
    """atexit hook — drop pending futures and let the worker thread die.

    ``cancel_futures=True`` aborts queued submissions; the in-flight
    inference (if any) finishes naturally because faster-whisper does
    not honor interrupts. ``wait=False`` so process shutdown isn't
    held up by a stuck transcribe call.
    """
    try:
        _executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


atexit.register(_shutdown_executor)
