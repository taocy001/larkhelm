"""larkhelm · API key backends (Anthropic / Google / OpenAI-compat).

Each ``run_*`` function streams responses via ``on_text`` and returns
``(response_text, updated_history)``. History is NOT saved here —
callers use ``api_session.save_history()`` after getting the result.

SDKs are optional: if not installed, the adapter raises RuntimeError and
``health_check`` marks the backend unhealthy.

P2 REQ-09: the streaming scaffolding now lives in
``backend_api_streaming.py`` (template + three Adapter classes). The
``run_*`` functions below are thin shims that build an Adapter and forward
to ``_run_streaming_api`` — keeping their public signatures unchanged so
third-party plugins importing them by name continue to work.
"""
from __future__ import annotations

from typing import Any, Callable

from larkhelm.backend_registry import BackendSpec
from larkhelm.backend_api_streaming import (
    AnthropicAdapter,
    GoogleGenaiAdapter,
    OpenAICompatAdapter,
    _record_outcome,
    _run_streaming_api,
)
# Re-exported for backward compatibility with any caller that patched
# ``backend_api._debug_log`` (tests/test_exception_handling.py does this).
# The streaming template hooks the on_text-callback log through the
# module-level binding here so monkey-patching this attribute redirects
# the debug output as expected.
from larkhelm.log import _debug_log  # noqa: F401  (re-exported)

# Re-exported for backward compatibility with any caller that imported
# ``backend_api._record_outcome`` directly (tests do this).
__all__ = [
    "_record_outcome", "_debug_log",
    "run_anthropic", "run_google", "run_openai_compat",
]


def run_anthropic(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: "Any | None" = None,
    on_text: Callable | None = None,
    extra_system: str = "",
    *,
    _anthropic_module: "Any | None" = None,
    suppress_token_recording: bool = False,
    usage_holder: "dict | None" = None,
) -> tuple[str, list[dict]]:
    """Stream via Anthropic SDK. See ``backend_api_streaming.AnthropicAdapter``.

    ``_anthropic_module`` is a test hook: production callers leave it
    ``None`` so the live ``import anthropic`` path runs; tests pass a fake
    module to short-circuit the import.
    """
    adapter = AnthropicAdapter(anthropic_module=_anthropic_module)
    return _run_streaming_api(
        adapter, spec, chat_id, message, history,
        cancel_ev=cancel_ev, on_text=on_text, extra_system=extra_system,
        suppress_token_recording=suppress_token_recording,
        usage_holder=usage_holder,
    )


def run_google(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: "Any | None" = None,
    on_text: Callable | None = None,
    extra_system: str = "",
    *,
    _google_module: "Any | None" = None,
    suppress_token_recording: bool = False,
    usage_holder: "dict | None" = None,
) -> tuple[str, list[dict]]:
    """Stream via google-genai SDK. See ``backend_api_streaming.GoogleGenaiAdapter``.

    ``_google_module`` test hook: provide an object exposing ``.genai`` and
    ``.genai_types`` to short-circuit the live import.
    """
    adapter = GoogleGenaiAdapter(google_module=_google_module)
    return _run_streaming_api(
        adapter, spec, chat_id, message, history,
        cancel_ev=cancel_ev, on_text=on_text, extra_system=extra_system,
        suppress_token_recording=suppress_token_recording,
        usage_holder=usage_holder,
    )


def run_openai_compat(
    spec: BackendSpec,
    chat_id: str,
    message: str,
    history: list[dict],
    cancel_ev: "Any | None" = None,
    on_text: Callable | None = None,
    extra_system: str = "",
    suppress_token_recording: bool = False,
    usage_holder: "dict | None" = None,
) -> tuple[str, list[dict]]:
    """Stream via OpenAI-compat SDK (DeepSeek etc.).

    See ``backend_api_streaming.OpenAICompatAdapter``.
    """
    adapter = OpenAICompatAdapter()
    return _run_streaming_api(
        adapter, spec, chat_id, message, history,
        cancel_ev=cancel_ev, on_text=on_text, extra_system=extra_system,
        suppress_token_recording=suppress_token_recording,
        usage_holder=usage_holder,
    )
