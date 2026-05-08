"""larkhelm · backend routing — resolve_backend()

Routing rules (priority high → low):
  0. locked_backend in chat state → raise LockedBackendUnavailableError if unhealthy
  1. has_images → get_by_tag(["vision"])
  2. has_doc_urls → get_by_tag(["tools"])
  3. enable_cheap_routing + short message + no images/docs → get_by_tag(["cheap", "fast"])
  4. user preference backend_id (chat_state) → registry.get(backend_id)
  5. config default_backend → orchestrator chain → first healthy enabled

Fallback: RuntimeError if no healthy backend found.
"""
from __future__ import annotations

import larkhelm.config as _cfg
from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
from larkhelm.chat_state import _get_chat_state
from larkhelm.log import _debug_log

_SHORT_MSG_THRESHOLD = 100


class LockedBackendUnavailableError(RuntimeError):
    """Raised when the chat's locked_backend is set but currently unhealthy.

    Callers must catch this before the broad Exception handler to show a
    user-facing error card rather than silently falling back to other backends.
    """
    def __init__(self, backend_id: str, last_error: str = ""):
        self.backend_id = backend_id
        detail = f"：{last_error}" if last_error else ""
        super().__init__(f"锁定的后端 {backend_id} 当前不可用{detail}")


def resolve_backend(
    chat_id: str,
    message: str,
    has_images: bool = False,
    has_doc_urls: bool = False,
    force_backend_id: str | None = None,
) -> BackendSpec:
    """Route the query to the best available BackendSpec."""
    try:
        enable_cheap = getattr(_cfg, "config", {}).get("enable_cheap_routing", False)
    except Exception:
        enable_cheap = False

    # Rule 0.5: per-message backend force (e.g. /c, /g, /k — override orchestrator routing)
    if force_backend_id:
        spec = BACKEND_REGISTRY.get(force_backend_id)
        if spec is None:
            # Provider fallback: legacy model names "claude"/"gemini"/"kimi" → provider lookup
            _provider_map: dict[str, tuple[str, ...]] = {
                "claude": ("claude_cli", "anthropic_api"),
                "gemini": ("gemini_cli", "google_api"),
                "kimi":   ("kimi_cli",),
            }
            _providers = _provider_map.get(force_backend_id, ())
            spec = next(
                (s for s in BACKEND_REGISTRY.all_enabled()
                 if s.healthy and s.provider in _providers),
                None
            )
        if spec and spec.enabled and spec.healthy:
            _debug_log(f"[router] {chat_id}: force → {spec.id}")
            return spec
        _debug_log(f"[router] {chat_id}: force {force_backend_id!r} unavailable, falling through")

    # Rule 0: locked_backend in chat state → fast-fail if unhealthy, else return spec
    locked_state = _get_chat_state(chat_id)
    locked_id = locked_state.get("locked_backend")
    if locked_id:
        spec = BACKEND_REGISTRY.get(locked_id)
        if spec is None or not spec.enabled:
            # Backend removed from config or explicitly disabled — treat as unavailable
            _id = locked_id if spec is None else spec.id
            _err = "" if spec is None else "已禁用"
            raise LockedBackendUnavailableError(_id, _err)
        if not spec.healthy:
            raise LockedBackendUnavailableError(spec.id, spec.last_error or "")
        _debug_log(f"[router] {chat_id}: locked_backend → {spec.id}")
        return spec

    # Rule 1: image → vision-capable backend (prefer orchestrator so delegation works)
    if has_images:
        spec = BACKEND_REGISTRY.get_by_tag(["vision"], prefer_role="orchestrator")
        if spec:
            _debug_log(f"[router] {chat_id}: image → {spec.id}")
            return spec
        _debug_log(f"[router] {chat_id}: no vision backend available, falling through")

    # Rule 2: doc URLs → tools-capable backend (prefer orchestrator so delegation works)
    if has_doc_urls:
        spec = BACKEND_REGISTRY.get_by_tag(["tools"], prefer_role="orchestrator")
        if spec:
            _debug_log(f"[router] {chat_id}: doc_url → {spec.id}")
            return spec
        _debug_log(f"[router] {chat_id}: no tools backend available, falling through")

    # Rule 3: cheap routing for short messages
    if enable_cheap and not has_images and not has_doc_urls and len(message) < _SHORT_MSG_THRESHOLD:
        spec = BACKEND_REGISTRY.get_by_tag(["cheap", "fast"])
        if spec:
            _debug_log(f"[router] {chat_id}: short+cheap → {spec.id}")
            return spec

    # Rule 4: user preference (backend_id or model set via /model command)
    # Note: legacy configs use "model" field (values: claude/gemini/kimi) which
    # happen to match the auto-migrated backend IDs. New configs should use backend_id.
    preferred_id = locked_state.get("backend_id") or locked_state.get("model")
    if preferred_id:
        spec = BACKEND_REGISTRY.get(preferred_id)
        if spec and spec.healthy and spec.enabled:
            _debug_log(f"[router] {chat_id}: user_pref → {spec.id}")
            return spec

    # Rule 5: config default_backend → orchestrator chain → first healthy enabled
    default_bid = getattr(_cfg, "config", {}).get("default_backend", "")
    if default_bid:
        spec = BACKEND_REGISTRY.get(default_bid)
        if spec and spec.healthy and spec.enabled:
            _debug_log(f"[router] {chat_id}: default_backend → {spec.id}")
            return spec

    orch_chain = BACKEND_REGISTRY.get_orchestrator_chain()
    if orch_chain:
        _debug_log(f"[router] {chat_id}: orchestrator → {orch_chain[0].id}")
        return orch_chain[0]

    raise RuntimeError("No healthy backend available — all backends are down or disabled")
