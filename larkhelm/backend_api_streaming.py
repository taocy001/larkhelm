"""larkhelm · backend API streaming template (P2 REQ-09 / AC-04).

Extracts the common scaffolding around the three HTTP backends
(Anthropic, Google Gen-AI, OpenAI-compatible) into a single
``_run_streaming_api()`` template that:

  * spins up the SDK client via the adapter,
  * builds the request payload via the adapter,
  * iterates the streaming text chunks,
  * checks ``cancel_ev`` between chunks,
  * fans ``on_text`` updates to the caller,
  * always records the outcome (success/failure/cancel) to BackendRegistry.

Each ``StreamingAPIAdapter`` implementation is a small wrapper around the
SDK-specific glue, keeping the public ``run_*`` functions in
``backend_api.py`` ≤ 30 lines as required by PRD AC-04.

Public ``run_anthropic`` / ``run_google`` / ``run_openai_compat`` signatures
in ``backend_api.py`` are NOT changed — third-party plugins import those by
name. The template + adapter classes are internal but importable for tests.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Callable, Iterator, Protocol

from larkhelm.backend_registry import BackendSpec, BACKEND_REGISTRY
from larkhelm.log import _debug_log, safe_log
from larkhelm.token_budget import compute_api_max_tokens

# Process-wide memory of whether the Anthropic ``extended-cache-ttl`` beta
# header has been rejected by the API. Once set, subsequent calls in this
# process skip the 1h TTL request shape entirely so we don't pay a wasted
# handshake on every query. Restart clears the flag, allowing recovery once
# Anthropic enables the beta on the account.
_extended_cache_disabled: bool = False
_extended_cache_lock = threading.Lock()


def _is_extended_cache_rejection(exc: Exception) -> bool:
    """Match Anthropic SDK errors that indicate the beta header was rejected.

    Recognises ``anthropic.BadRequestError`` / ``anthropic.PermissionDeniedError``
    whose payload mentions ``extended-cache`` / ``extended_cache`` / ``ttl`` /
    ``beta``. Falls back to ``str(exc)`` substring match when the SDK class
    hierarchy is unavailable (test stub modules expose plain ``Exception``
    subclasses with the right names). Returns False on any other exception
    type so unrelated errors propagate.
    """
    # Substring-match the message first — this is the most reliable signal
    # across SDK / stub variants and avoids depending on the precise class
    # hierarchy being importable.
    try:
        msg = str(exc).lower()
    except Exception:
        return False
    if not any(token in msg for token in ("extended-cache", "extended_cache", "ttl", "beta")):
        return False

    # Constrain to the SDK error classes when possible so unrelated
    # ``ValueError("ttl ...")`` don't trigger fallback. When the import
    # fails (stub module / partial install), accept any exception whose
    # message already matched the keyword list — keyword + reachable code
    # path is signal enough.
    try:
        import anthropic  # type: ignore[import-not-found]
    except Exception:
        return True

    cls_names = ("BadRequestError", "PermissionDeniedError", "APIStatusError")
    for name in cls_names:
        cls = getattr(anthropic, name, None)
        if cls is not None and isinstance(exc, cls):
            return True
    # Fallback for tests/stubs: accept by class-name match.
    return type(exc).__name__ in cls_names


# ── Outcome recording (moved verbatim from backend_api.py) ────────────────


def _record_outcome(spec_id: str, exc: Exception | None) -> None:
    """Push call-outcome to BackendRegistry. Cancellation does NOT update health.

    Mirrors ``backend_cli._record_outcome``; kept as a sibling helper so the
    two dispatch families (CLI / API) stay parallel and either can be moved
    to a shared module later without one waiting on the other.
    """
    # Local import keeps the cycle out of module import time.
    from larkhelm.ai_runner import QueryCancelledError
    try:
        if exc is None:
            BACKEND_REGISTRY.record_call_success(spec_id)
            return
        if isinstance(exc, QueryCancelledError):
            return  # user-initiated, not a backend fault
        try:
            from larkhelm import config as _cfg
            window = float(getattr(_cfg, "BACKEND_TRANSIENT_WINDOW_SEC", 600.0))
            threshold = int(getattr(_cfg, "BACKEND_TRANSIENT_THRESHOLD", 3))
        except Exception:
            window, threshold = 600.0, 3
        BACKEND_REGISTRY.record_call_failure(
            spec_id, str(exc),
            transient_window_sec=window,
            transient_threshold=threshold,
        )
    except Exception:
        safe_log(f"[BackendRegistry] _record_outcome failed for {spec_id}")


# ── Adapter protocol ──────────────────────────────────────────────────────


class StreamingAPIAdapter(Protocol):
    """Four-hook contract used by :func:`_run_streaming_api`.

    Implementations sit in this module so the template can be inspected
    without crossing module boundaries. Each ``run_*`` function in
    ``backend_api.py`` instantiates an adapter and forwards to the
    template — no Protocol method is called by anything but the template.
    """
    provider_label: str

    def build_client(self, spec: BackendSpec) -> Any: ...

    def prepare_request(
        self, spec: BackendSpec, history: list[dict],
        message: str, extra_system: str,
    ) -> dict: ...

    def iter_text_chunks(self, client: Any, request: dict) -> Iterator[str]: ...

    def format_history(
        self, history: list[dict], message: str, response_text: str,
    ) -> list[dict]: ...

    def extract_usage(self) -> dict: ...


# ── Adapter implementations ───────────────────────────────────────────────


class AnthropicAdapter:
    """Streaming adapter for the official Anthropic Python SDK."""
    provider_label = "anthropic_api"

    def __init__(self, anthropic_module: Any | None = None) -> None:
        # Production callers leave the module ``None``; ``run_anthropic``'s
        # ``_anthropic_module`` test hook is forwarded here so the import
        # of ``anthropic`` can be short-circuited in unit tests.
        if anthropic_module is None:
            try:
                import anthropic  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "anthropic SDK not installed; run: pip install anthropic"
                ) from e
            self._anthropic = anthropic
        else:
            self._anthropic = anthropic_module
        self._usage_raw: Any = None

    def build_client(self, spec: BackendSpec) -> Any:
        api_key = spec.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        client_kwargs: dict = {"api_key": api_key}
        if spec.base_url:
            client_kwargs["base_url"] = spec.base_url
        return self._anthropic.Anthropic(**client_kwargs)

    def prepare_request(
        self, spec: BackendSpec, history: list[dict],
        message: str, extra_system: str,
    ) -> dict:
        # Anthropic disallows ``role=system`` inside ``messages``; lift it
        # into a separate ``system`` channel. Caller-supplied ``extra_system``
        # always wins (prepended) so per-query context injection survives.
        system_parts = [h["content"] for h in history if h["role"] == "system"]
        if extra_system:
            system_parts.insert(0, extra_system)
        messages = [h for h in history if h["role"] != "system"]
        messages.append({"role": "user", "content": message})

        kwargs: dict = dict(
            model=spec.model or "claude-sonnet-4-6",
            max_tokens=compute_api_max_tokens(spec),
            messages=messages,
        )
        if system_parts:
            # Prompt caching: ``cache_control: ephemeral`` enables ~10%
            # input pricing on the next call within the 5-min TTL. The
            # 4500-char three-tier memory_ctx is the typical prefix
            # (stable across turns); break-even is 2 reads. When the
            # operator opts into ``anthropic_extended_cache_enabled`` AND
            # the beta header has not yet been rejected this process,
            # upgrade the TTL to 1h via the
            # ``anthropic-beta: extended-cache-ttl-2025-04-11`` header.
            system_text = "\n\n".join(system_parts)
            cache_control: dict = {"type": "ephemeral"}
            use_extended = False
            try:
                from larkhelm import config as _cfg
                use_extended = bool(
                    getattr(_cfg, "ANTHROPIC_EXTENDED_CACHE_ENABLED", True)
                ) and not _extended_cache_disabled
            except Exception:
                use_extended = False
            if use_extended:
                cache_control = {"type": "ephemeral", "ttl": "1h"}
                kwargs["extra_headers"] = {
                    "anthropic-beta": "extended-cache-ttl-2025-04-11",
                }
            kwargs["system"] = [{
                "type":          "text",
                "text":          system_text,
                "cache_control": cache_control,
            }]
        return kwargs

    def _is_extended_cache_request(self, request: dict) -> bool:
        """True iff ``request`` carries the 1h TTL + anthropic-beta header."""
        headers = request.get("extra_headers") or {}
        if headers.get("anthropic-beta") != "extended-cache-ttl-2025-04-11":
            return False
        for sys_part in request.get("system") or ():
            cc = sys_part.get("cache_control") if isinstance(sys_part, dict) else None
            if isinstance(cc, dict) and cc.get("ttl") == "1h":
                return True
        return False

    def _strip_extended_cache(self, request: dict) -> dict:
        """Return a shallow-copied request with the 1h TTL + beta header removed.

        Never mutates the input; safe to retry. The retry shape matches the
        pre-extended-cache prompt-cache payload byte-for-byte.
        """
        new_req: dict = dict(request)
        new_req.pop("extra_headers", None)
        old_system = request.get("system") or []
        new_system = []
        for sys_part in old_system:
            if isinstance(sys_part, dict):
                copy = dict(sys_part)
                cc = copy.get("cache_control")
                if isinstance(cc, dict) and "ttl" in cc:
                    new_cc = {k: v for k, v in cc.items() if k != "ttl"}
                    copy["cache_control"] = new_cc
                new_system.append(copy)
            else:
                new_system.append(sys_part)
        if new_system:
            new_req["system"] = new_system
        return new_req

    def iter_text_chunks(self, client: Any, request: dict) -> Iterator[str]:
        global _extended_cache_disabled
        self._usage_raw = None
        # Anthropic SDK's ``client.messages.stream(**request)`` only constructs
        # a ``MessageStreamManager``; the HTTP request fires in
        # ``MessageStreamManager.__enter__()``. So beta-header rejections
        # (BadRequestError / PermissionDeniedError) surface from ``__enter__``,
        # not from ``stream()``. The try block therefore wraps both calls.
        try:
            stream_cm = client.messages.stream(**request)
            stream = stream_cm.__enter__()
        except Exception as e:
            if self._is_extended_cache_request(request) and _is_extended_cache_rejection(e):
                _debug_log(
                    f"[{self.provider_label}] extended-cache rejected, "
                    f"falling back to 5min ephemeral: {e}"
                )
                with _extended_cache_lock:
                    _extended_cache_disabled = True
                request = self._strip_extended_cache(request)
                stream_cm = client.messages.stream(**request)
                stream = stream_cm.__enter__()
            else:
                raise
        try:
            for chunk in stream.text_stream:
                yield chunk
            try:
                final = stream.get_final_message()
                self._usage_raw = final.usage
            except Exception:
                pass
        finally:
            try:
                stream_cm.__exit__(None, None, None)
            except Exception:
                pass

    def extract_usage(self) -> dict:
        try:
            u = self._usage_raw
            if u is None:
                return {}
            return {
                "input_tokens":  getattr(u, "input_tokens", 0) or 0,
                "output_tokens": getattr(u, "output_tokens", 0) or 0,
                "cache_read":    getattr(u, "cache_read_input_tokens", 0) or 0,
                "cache_create":  getattr(u, "cache_creation_input_tokens", 0) or 0,
            }
        except Exception:
            return {}

    def format_history(
        self, history: list[dict], message: str, response_text: str,
    ) -> list[dict]:
        return list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ]


class GoogleGenaiAdapter:
    """Streaming adapter for ``google-genai``."""
    provider_label = "google_api"

    def __init__(self, google_module: Any | None = None) -> None:
        # ``google_module`` is the legacy test hook: when provided, the
        # object exposes ``.genai`` and ``.genai_types`` attributes
        # (mirroring ``from google import genai`` / ``from google.genai
        # import types``). Live callers leave it None.
        if google_module is None:
            try:
                from google import genai  # type: ignore[import-not-found]
                from google.genai import types as genai_types  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "google-genai SDK not installed; run: pip install google-genai"
                ) from e
            self._genai = genai
            self._types = genai_types
        else:
            self._genai = google_module.genai
            self._types = google_module.genai_types
        self._usage_raw: Any = None

    def build_client(self, spec: BackendSpec) -> Any:
        api_key = spec.api_key or os.environ.get("GOOGLE_API_KEY", "")
        return self._genai.Client(api_key=api_key)

    def prepare_request(
        self, spec: BackendSpec, history: list[dict],
        message: str, extra_system: str,
    ) -> dict:
        types = self._types
        system_texts: list[str] = []
        if extra_system:
            system_texts.append(extra_system)
        contents = []
        for h in history:
            role = "user" if h["role"] == "user" else "model"
            contents.append(types.Content(role=role, parts=[types.Part(text=h["content"])]))
        contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

        gen_config = None
        if system_texts:
            gen_config = types.GenerateContentConfig(
                system_instruction="\n\n".join(system_texts)
            )
        return {
            "model":    spec.model or "gemini-2.0-flash",
            "contents": contents,
            "config":   gen_config,
        }

    def iter_text_chunks(self, client: Any, request: dict) -> Iterator[str]:
        self._usage_raw = None
        stream = client.models.generate_content_stream(
            model=request["model"], contents=request["contents"], config=request["config"],
        )
        for chunk in stream:
            # usage_metadata is populated on the last chunk from Google Gen-AI.
            if getattr(chunk, "usage_metadata", None):
                self._usage_raw = chunk.usage_metadata
            yield chunk.text or ""

    def extract_usage(self) -> dict:
        try:
            u = self._usage_raw
            if u is None:
                return {}
            return {
                "input_tokens":  getattr(u, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(u, "candidates_token_count", 0) or 0,
                "cache_read":    getattr(u, "cached_content_token_count", 0) or 0,
                "cache_create":  0,
            }
        except Exception:
            return {}

    def format_history(
        self, history: list[dict], message: str, response_text: str,
    ) -> list[dict]:
        return list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ]


class OpenAICompatAdapter:
    """Streaming adapter for OpenAI-style chat-completions APIs (DeepSeek etc.)."""
    provider_label = "openai_compat_api"

    def __init__(self, openai_module: Any | None = None) -> None:
        if openai_module is None:
            try:
                import openai  # type: ignore[import-not-found]
            except ImportError as e:
                raise RuntimeError(
                    "openai SDK not installed; run: pip install openai"
                ) from e
            self._openai = openai
        else:
            self._openai = openai_module
        self._usage_raw: Any = None

    def build_client(self, spec: BackendSpec) -> Any:
        api_key = spec.api_key or os.environ.get("OPENAI_API_KEY", "")
        kwargs: dict = {"api_key": api_key}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        return self._openai.OpenAI(**kwargs)

    def prepare_request(
        self, spec: BackendSpec, history: list[dict],
        message: str, extra_system: str,
    ) -> dict:
        messages = list(history)
        if extra_system:
            messages.insert(0, {"role": "system", "content": extra_system})
        messages.append({"role": "user", "content": message})
        return {
            "model":          spec.model or "gpt-4o",
            "messages":       messages,
            "stream":         True,
            "max_tokens":     compute_api_max_tokens(spec),
            # stream_options=include_usage causes the last chunk to carry
            # usage stats (TOKEN-C1); most OpenAI-compatible backends support
            # this extension; unsupported ones silently ignore the field.
            "stream_options": {"include_usage": True},
        }

    def iter_text_chunks(self, client: Any, request: dict) -> Iterator[str]:
        self._usage_raw = None
        with client.chat.completions.create(**request) as stream:
            for chunk in stream:
                # Streaming chunks may have empty ``choices`` (heartbeat
                # frames); skip silently.
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
                # Usage arrives in the final chunk when stream_options is set.
                if hasattr(chunk, "usage") and chunk.usage:
                    self._usage_raw = chunk.usage

    def extract_usage(self) -> dict:
        try:
            u = self._usage_raw
            if u is None:
                return {}
            cached = 0
            try:
                pd = getattr(u, "prompt_tokens_details", None)
                cached = getattr(pd, "cached_tokens", 0) or 0
            except Exception:
                pass
            return {
                "input_tokens":  getattr(u, "prompt_tokens", 0) or 0,
                "output_tokens": getattr(u, "completion_tokens", 0) or 0,
                "cache_read":    cached,
                "cache_create":  0,
            }
        except Exception:
            return {}

    def format_history(
        self, history: list[dict], message: str, response_text: str,
    ) -> list[dict]:
        return list(history) + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ]


# ── Common streaming template ─────────────────────────────────────────────


def _run_streaming_api(
    adapter: StreamingAPIAdapter,
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: Any | None = None,
    on_text: Callable | None = None,
    extra_system: str = "",
    suppress_token_recording: bool = False,
    usage_holder: dict | None = None,
) -> tuple[str, list[dict]]:
    """Run a streaming API turn through the supplied adapter.

    Common to all three providers. Always reports outcome to
    BackendRegistry — success on a clean return, failure on a raised
    exception (other than QueryCancelledError), and silently on cancel.
    """
    from larkhelm.ai_runner import QueryCancelledError

    client = adapter.build_client(spec)
    request = adapter.prepare_request(spec, history, message, extra_system)
    _debug_log(
        f"[{adapter.provider_label}] {spec.id} "
        f"model={request.get('model', spec.model)} chat={chat_id}"
    )

    result_text = ""
    try:
        for chunk in adapter.iter_text_chunks(client, request):
            if cancel_ev and cancel_ev.is_set():
                # Drop out of the loop; the post-loop cancel check below
                # raises QueryCancelledError uniformly across adapters.
                break
            if not chunk:
                continue
            result_text += chunk
            if on_text:
                try:
                    on_text(result_text, status="typing")
                except Exception as cb_err:
                    # Swallow on_text errors — a buggy UI callback must
                    # not break the streaming loop. The diagnostic still
                    # makes it to debug log for later investigation.
                    # Route through ``backend_api._debug_log`` (re-exported)
                    # so existing tests that monkey-patch the public
                    # ``backend_api`` module's name see the log entries.
                    try:
                        from larkhelm import backend_api as _bapi
                        _bapi._debug_log(
                            f"[{adapter.provider_label}] on_text callback failed: {cb_err}"
                        )
                    except Exception:
                        _debug_log(
                            f"[{adapter.provider_label}] on_text callback failed: {cb_err}"
                        )
    except Exception as e:
        _debug_log(f"[{adapter.provider_label}] {spec.id} error: {e}")
        _record_outcome(spec.id, e)
        raise

    if cancel_ev and cancel_ev.is_set():
        # Cancel is user-initiated; do NOT touch backend health.
        raise QueryCancelledError(
            f"Query cancelled during {adapter.provider_label} streaming"
        )

    # TOKEN-C1 / TOKEN-C2: record token usage via Protocol method.
    _adapter_usage = adapter.extract_usage()
    if _adapter_usage:
        if suppress_token_recording:
            if usage_holder is not None:
                usage_holder.update(_adapter_usage)
        else:
            try:
                from larkhelm.token_stats import record_token_usage, resolve_record_chat_id
                record_chat_id = resolve_record_chat_id(chat_id)
                record_token_usage(record_chat_id, spec.model or spec.id, _adapter_usage)
            except Exception as _ue:
                safe_log(f"[{adapter.provider_label}] usage recording failed: {_ue}")

    updated_history = adapter.format_history(history, message, result_text)
    _record_outcome(spec.id, None)
    return result_text.strip(), updated_history


__all__ = [
    "StreamingAPIAdapter",
    "AnthropicAdapter",
    "GoogleGenaiAdapter",
    "OpenAICompatAdapter",
    "_record_outcome",
    "_run_streaming_api",
    "_is_extended_cache_rejection",
]
