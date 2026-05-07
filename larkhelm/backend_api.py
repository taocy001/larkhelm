"""larkhelm · API key backends (Anthropic / Google / OpenAI-compat)

Each run_* function streams responses via on_text callback and returns
(response_text, updated_history). History is NOT saved here — callers
use api_session.save_history() after getting the result.

SDKs are optional: if not installed, health_check marks the backend unhealthy.
"""
from __future__ import annotations

import os
from typing import Callable

from larkhelm.backend_registry import BackendSpec
from larkhelm.log import _debug_log


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
        kwargs["system"] = "\n\n".join(system_parts)

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
                    except Exception:
                        pass
    except Exception as e:
        _debug_log(f"[anthropic_api] {spec.id} error: {e}")
        raise

    if cancel_ev and cancel_ev.is_set():
        raise QueryCancelledError("Query cancelled during anthropic_api streaming")

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result_text},
    ]
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
                except Exception:
                    pass
    except Exception as e:
        _debug_log(f"[google_api] {spec.id} error: {e}")
        raise

    if cancel_ev and cancel_ev.is_set():
        raise QueryCancelledError("Query cancelled during google_api streaming")

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result_text},
    ]
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
                        except Exception:
                            pass
    except Exception as e:
        _debug_log(f"[openai_compat_api] {spec.id} error: {e}")
        raise

    if cancel_ev and cancel_ev.is_set():
        raise QueryCancelledError("Query cancelled during openai_compat_api streaming")

    updated_history = list(history) + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result_text},
    ]
    return result_text.strip(), updated_history
