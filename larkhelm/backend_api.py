"""larkhelm · API key backends (Anthropic / Google / OpenAI-compat)

Each run_* function streams responses via on_text callback and returns
(response_text, updated_history). History is NOT saved here — callers
use api_session.save_history() after getting the result.

SDKs are optional: if not installed, health_check marks the backend unhealthy.
"""
from __future__ import annotations

import os
from typing import Callable

from larkhelm.backend_registry import BackendSpec, BACKEND_REGISTRY
from larkhelm.log import _debug_log, safe_log
# Direct top-level import — symmetric with backend_cli._record_outcome (which
# also imports QueryCancelledError at module level). The previous defensive
# try/except ImportError was dead code: by the time _record_outcome runs, every
# run_* function has already imported ai_runner at its own top so the module
# is fully loaded.
from larkhelm.ai_runner import QueryCancelledError


def _record_outcome(spec_id: str, exc: Exception | None) -> None:
    """Push call-outcome to BackendRegistry. Cancellation does NOT update health.

    Mirrors ``backend_cli._record_outcome``; kept as a sibling helper so the
    two dispatch families (CLI / API) stay parallel and either can be moved
    to a shared module later without one waiting on the other.
    """
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


def run_anthropic(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: "object | None" = None,
    on_text: Callable = None,
    extra_system: str = "",
) -> tuple[str, list[dict]]:
    """Stream via Anthropic SDK. Returns (response_text, updated_history)."""
    from larkhelm.ai_runner import QueryCancelledError
    try:
        import anthropic
    except ImportError:
        raise RuntimeError("anthropic SDK not installed; run: pip install anthropic")

    api_key = spec.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    client_kwargs: dict = {"api_key": api_key}
    if spec.base_url:
        client_kwargs["base_url"] = spec.base_url
    client = anthropic.Anthropic(**client_kwargs)

    # Anthropic only allows user/assistant roles in messages; extract system separately.
    # extra_system (per-query injection) takes highest priority → prepended before history systems.
    system_parts = [h["content"] for h in history if h["role"] == "system"]
    if extra_system:
        system_parts.insert(0, extra_system)
    messages = [h for h in history if h["role"] != "system"]
    messages.append({"role": "user", "content": message})

    result_text = ""
    kwargs: dict = dict(
        model=spec.model or "claude-sonnet-4-6",
        max_tokens=8192,
        messages=messages,
    )
    if system_parts:
        # Prompt caching: send the system channel as a single block with
        # ``cache_control: ephemeral`` so the same prefix on the next request
        # within ~5 min is read at ~10% input price. The dominant component
        # is the 4500-char three-tier memory_ctx — typically stable across
        # turns. Cache-write costs ~25% more than uncached input, so the
        # break-even is 2 reads within the TTL; any active chat with >2
        # turns / 5 min wins. Below the minimum cacheable size (~1024 tokens
        # for Sonnet, ~2048 for Haiku) Anthropic silently skips caching, so
        # the marker is a free hint in small-system cases.
        system_text = "\n\n".join(system_parts)
        kwargs["system"] = [{
            "type":          "text",
            "text":          system_text,
            "cache_control": {"type": "ephemeral"},
        }]

    _debug_log(f"[anthropic_api] {spec.id} model={kwargs['model']} chat={chat_id}")

    try:
        with client.messages.stream(**kwargs) as stream:
            for text_chunk in stream.text_stream:
                if cancel_ev and cancel_ev.is_set():
                    break
                result_text += text_chunk
                if on_text:
                    try:
                        on_text(result_text, status="typing")
                    except Exception as e:
                        _debug_log(f"[anthropic_api] on_text callback failed: {e}")
    except Exception as e:
        _debug_log(f"[anthropic_api] {spec.id} error: {e}")
        _record_outcome(spec.id, e)
        raise

    if cancel_ev and cancel_ev.is_set():
        # Cancel is user-initiated → don't update health
        raise QueryCancelledError("Query cancelled during anthropic_api streaming")

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result_text},
    ]
    _record_outcome(spec.id, None)
    return result_text.strip(), updated_history


def run_google(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: "object | None" = None,
    on_text: Callable = None,
    extra_system: str = "",
) -> tuple[str, list[dict]]:
    """Stream via google-genai SDK. Returns (response_text, updated_history)."""
    from larkhelm.ai_runner import QueryCancelledError
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        raise RuntimeError("google-genai SDK not installed; run: pip install google-genai")

    api_key = spec.api_key or os.environ.get("GOOGLE_API_KEY", "")
    client = genai.Client(api_key=api_key)

    # extra_system takes highest priority; history should not contain system-role messages
    # (api_session only stores user/assistant turns), but guard defensively.
    system_texts: list[str] = []
    if extra_system:
        system_texts.append(extra_system)
    contents = []
    for h in history:
        role = "user" if h["role"] == "user" else "model"
        contents.append(genai_types.Content(role=role, parts=[genai_types.Part(text=h["content"])]))
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part(text=message)]))

    model_name = spec.model or "gemini-2.0-flash"
    _debug_log(f"[google_api] {spec.id} model={model_name} chat={chat_id}")

    gen_config = None
    if system_texts:
        gen_config = genai_types.GenerateContentConfig(
            system_instruction="\n\n".join(system_texts)
        )

    result_text = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=model_name, contents=contents, config=gen_config
        ):
            if cancel_ev and cancel_ev.is_set():
                break
            chunk_text = chunk.text or ""
            result_text += chunk_text
            if on_text and chunk_text:
                try:
                    on_text(result_text, status="typing")
                except Exception as e:
                    _debug_log(f"[google_api] on_text callback failed: {e}")
    except Exception as e:
        _debug_log(f"[google_api] {spec.id} error: {e}")
        _record_outcome(spec.id, e)
        raise

    if cancel_ev and cancel_ev.is_set():
        raise QueryCancelledError("Query cancelled during google_api streaming")

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result_text},
    ]
    _record_outcome(spec.id, None)
    return result_text.strip(), updated_history


def run_openai_compat(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: "object | None" = None,
    on_text: Callable = None,
    extra_system: str = "",
) -> tuple[str, list[dict]]:
    """Stream via OpenAI-compat SDK (DeepSeek etc.). Returns (response_text, updated_history)."""
    from larkhelm.ai_runner import QueryCancelledError
    try:
        import openai
    except ImportError:
        raise RuntimeError("openai SDK not installed; run: pip install openai")

    api_key = spec.api_key or os.environ.get("OPENAI_API_KEY", "")
    kwargs = dict(api_key=api_key)
    if spec.base_url:
        kwargs["base_url"] = spec.base_url
    client = openai.OpenAI(**kwargs)

    messages = list(history)
    if extra_system:
        messages.insert(0, {"role": "system", "content": extra_system})
    messages.append({"role": "user", "content": message})

    model_name = spec.model or "gpt-4o"
    _debug_log(f"[openai_compat_api] {spec.id} model={model_name} chat={chat_id}")

    result_text = ""
    try:
        with client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=True,
        ) as stream:
            for chunk in stream:
                if cancel_ev and cancel_ev.is_set():
                    break
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    result_text += delta
                    if on_text:
                        try:
                            on_text(result_text, status="typing")
                        except Exception as e:
                            _debug_log(f"[openai_compat_api] on_text callback failed: {e}")
    except Exception as e:
        _debug_log(f"[openai_compat_api] {spec.id} error: {e}")
        _record_outcome(spec.id, e)
        raise

    if cancel_ev and cancel_ev.is_set():
        raise QueryCancelledError("Query cancelled during openai_compat_api streaming")

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result_text},
    ]
    # Record success last — matches placement in run_anthropic / run_google
    _record_outcome(spec.id, None)
    return result_text.strip(), updated_history
