"""larkhelm voice · DashScope Paraformer engine adapter (M3.2 commit β).

Opt-in cloud STT path. Activated only when **all** of:

* ``voice_engine == "dashscope"`` in config.json
* ``voice_api_key`` resolves to a non-empty string (typically via the
  ``${DASHSCOPE_API_KEY}`` env-var placeholder injected through systemd
  drop-in — same pattern as the DeepSeek backend)
* ``dashscope`` SDK is installed —
  ``pipx runpip larkhelm install dashscope`` (it lives in the
  ``[voice-cloud]`` optional-extras, *not* main dependencies, so users
  who never opt in pay zero install cost)

The adapter exposes a single function ``transcribe_via_dashscope`` that
returns a ``TranscribeResult`` matching the faster-whisper engine — the
public ``transcribe()`` dispatcher in ``larkhelm.voice.transcribe`` is
the only caller.

Failure modes (all collapse into ``TranscribeResult(ok=False, error=…)``,
never raise):

* ``dashscope_no_api_key``    — engine selected but key empty
* ``dashscope_sdk_missing``   — ``import dashscope`` failed
* ``dashscope_call_failed:…`` — SDK or HTTP layer raised
* ``dashscope_status_<code>`` — non-200 HTTP from upstream
* ``dashscope_empty_result``  — call succeeded but no transcript text
* ``dashscope_result_parse_failed:…`` — SDK returned an unexpected shape

> NOTE: DashScope SDK shape varies between minor versions (1.20+ added
> ``Recognition.call(file=…)``). The parse path is **defensive** — falls
> back across multiple known result attribute names. If a future SDK
> version breaks compatibility, ``dashscope_result_parse_failed`` makes
> the diagnostic obvious; fix is one ``getattr`` chain in this file.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import larkhelm.config as _cfg
from larkhelm import log as _log

if TYPE_CHECKING:
    from larkhelm.voice.transcribe import TranscribeResult


def _make_result(
    ok: bool, *, text: str = "", duration: float = 0.0,
    lang: str = "zh", error: "str | None" = None,
) -> "TranscribeResult":
    """Construct a TranscribeResult dict. Local helper so the file stays
    importable without importing transcribe.py at module load time
    (avoids a circular reference if transcribe.py is the dispatcher).
    """
    from larkhelm.voice.transcribe import TranscribeResult
    return TranscribeResult(
        ok=ok, text=text, duration=duration, lang=lang, error=error,
    )


def transcribe_via_dashscope(
    audio_path: str,
    *,
    lang: str = "zh",
    _dashscope_module: "Any | None" = None,
    _recognition_cls: "type | None" = None,
) -> "TranscribeResult":
    """Run DashScope Paraformer ASR on a local audio file.

    Synchronous; blocks the calling thread until DashScope responds or
    the request raises. Caller (the ``transcribe.py`` dispatcher) runs
    this on the same ``ThreadPoolExecutor`` worker that the
    faster-whisper path uses, so concurrency limits stay identical
    across engines.

    ``_dashscope_module`` / ``_recognition_cls`` are test hooks: production
    callers leave them at ``None`` so the live ``import dashscope`` and
    ``from dashscope.audio.asr import Recognition`` paths run. Tests pass
    fakes to short-circuit the imports without touching ``sys.modules``.
    """
    if not _cfg.VOICE_API_KEY:
        return _make_result(
            False, lang=lang, error="dashscope_no_api_key",
        )

    # Lazy import — SDK is in [voice-cloud] extras, may not be installed.
    if _dashscope_module is None:
        try:
            import dashscope
        except ImportError as e:
            _log.warn(
                "[Voice] dashscope SDK missing; "
                "run: pipx runpip larkhelm install dashscope"
            )
            return _make_result(
                False, lang=lang, error=f"dashscope_sdk_missing:{e}",
            )
    else:
        dashscope = _dashscope_module

    if _recognition_cls is None:
        try:
            from dashscope.audio.asr import Recognition  # type: ignore[import-not-found]
        except ImportError as e:
            _ver = getattr(dashscope, "__version__", "?")
            return _make_result(
                False, lang=lang,
                error=f"dashscope_sdk_missing:Recognition not in {_ver}: {e}",
            )
    else:
        Recognition = _recognition_cls

    # SDK reads api_key from module-level attribute on each call. Setting
    # every call costs nothing and means a key rotation just needs a
    # bridge restart — no in-process key cache to invalidate.
    dashscope.api_key = _cfg.VOICE_API_KEY

    audio_p = Path(audio_path)
    fmt = (audio_p.suffix.lstrip('.').lower() or 'wav')
    # DashScope only knows certain format strings; map common Feishu/test ones.
    fmt = {"opus": "opus", "ogg": "ogg", "mp3": "mp3",
           "wav": "wav", "m4a": "m4a", "aac": "aac"}.get(fmt, "wav")

    t0 = time.monotonic()
    try:
        # Real DashScope SDK shape: Recognition is a class whose constructor
        # takes the streaming params, and `.call(file=...)` is an INSTANCE
        # method that runs the actual ASR. Calling Recognition.call(...) as
        # a classmethod blows up with "missing 1 required positional argument:
        # 'self'" against the real SDK (verified against dashscope 1.25.17).
        # Pre-1.20 SDKs that exposed only a module-level helper are not
        # supported — the optional-extras pin is `dashscope>=1.20.0`.
        recognition = Recognition(
            model='paraformer-realtime-v2',
            callback=None,             # sync mode — .call() returns the full transcript
            format=fmt,
            sample_rate=16000,
        )
        result = recognition.call(file=str(audio_p))
    except Exception as e:
        return _make_result(
            False, lang=lang,
            error=f"dashscope_call_failed:{type(e).__name__}:{e}",
        )
    elapsed = time.monotonic() - t0

    # Extract status_code if present
    status = getattr(result, "status_code", None) or getattr(result, "code", None)
    if status not in (None, 200, "200"):
        msg = getattr(result, "message", "") or getattr(result, "msg", "") or "(no message)"
        return _make_result(
            False, duration=elapsed, lang=lang,
            error=f"dashscope_status_{status}:{msg}",
        )

    # Extract transcript text — defensive across known SDK shapes
    text = ""
    try:
        out = getattr(result, "output", None)
        if out is not None:
            sentences = getattr(out, "sentence", None)
            if sentences is None and isinstance(out, dict):
                sentences = out.get("sentence")
            if sentences:
                # sentences is list[dict|obj] with `.text` or ['text']
                parts: list[str] = []
                for s in sentences:
                    if isinstance(s, dict):
                        parts.append(str(s.get("text", "")))
                    else:
                        parts.append(str(getattr(s, "text", "")))
                text = "".join(parts).strip()
        # Fallback: some SDK versions expose .get_sentence() helper
        if not text and hasattr(result, "get_sentence"):
            sentences = result.get_sentence() or []
            text = "".join(str(s.get("text", "")) for s in sentences if isinstance(s, dict)).strip()
    except Exception as e:
        return _make_result(
            False, duration=elapsed, lang=lang,
            error=f"dashscope_result_parse_failed:{type(e).__name__}:{e}",
        )

    if not text:
        return _make_result(
            False, duration=elapsed, lang=lang,
            error="dashscope_empty_result",
        )

    return _make_result(
        True, text=text, duration=elapsed, lang=lang, error=None,
    )
