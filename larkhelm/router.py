"""larkhelm · backend routing — resolve_backend()

Routing rules (priority high → low):
  1. has_images → get_by_tag(["vision"])
  2. has_doc_urls → get_by_tag(["tools"])
  3. enable_cheap_routing + short message + no images/docs → get_by_tag(["cheap", "fast"])
  4. user preference backend_id (chat_state) → registry.get(backend_id)
  5. get_orchestrator() or all_enabled()[0]

Fallback: RuntimeError if no healthy backend found.
"""
from __future__ import annotations

import larkhelm.config as _cfg
from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
from larkhelm.chat_state import _get_chat_state
from larkhelm.log import _debug_log

_SHORT_MSG_THRESHOLD = 100


def resolve_backend(
    chat_id: str,
    message: str,
    has_images: bool = False,
    has_doc_urls: bool = False,
) -> BackendSpec:
    """Route the query to the best available BackendSpec."""
    try:
        enable_cheap = getattr(_cfg, "config", {}).get("enable_cheap_routing", False)
    except Exception:
        enable_cheap = False

    # Rule 1: image → vision-capable backend
    if has_images:
        spec = BACKEND_REGISTRY.get_by_tag(["vision"])
        if spec:
            _debug_log(f"[router] {chat_id}: image → {spec.id}")
            return spec

    # Rule 2: doc URLs → tools-capable backend
    if has_doc_urls:
        spec = BACKEND_REGISTRY.get_by_tag(["tools"])
        if spec:
            _debug_log(f"[router] {chat_id}: doc_url → {spec.id}")
            return spec

    # Rule 3: cheap routing for short messages
    if enable_cheap and not has_images and not has_doc_urls and len(message) < _SHORT_MSG_THRESHOLD:
        spec = BACKEND_REGISTRY.get_by_tag(["cheap", "fast"])
        if spec:
            _debug_log(f"[router] {chat_id}: short+cheap → {spec.id}")
            return spec

    # Rule 4: user preference (backend_id or model set via /model command)
    # Note: legacy configs use "model" field (values: claude/gemini/kimi) which
    # happen to match the auto-migrated backend IDs. New configs should use backend_id.
    state = _get_chat_state(chat_id)
    preferred_id = state.get("backend_id") or state.get("model")
    if preferred_id:
        spec = BACKEND_REGISTRY.get(preferred_id)
        if spec and spec.healthy and spec.enabled:
            _debug_log(f"[router] {chat_id}: user_pref → {spec.id}")
            return spec

    # Rule 5: default orchestrator or first enabled
    spec = BACKEND_REGISTRY.get_orchestrator()
    if spec:
        _debug_log(f"[router] {chat_id}: orchestrator → {spec.id}")
        return spec

    healthy = [s for s in BACKEND_REGISTRY.all_enabled() if s.healthy]
    if healthy:
        _debug_log(f"[router] {chat_id}: first_enabled → {healthy[0].id}")
        return healthy[0]

    raise RuntimeError("No healthy backend available — all backends are down or disabled")
