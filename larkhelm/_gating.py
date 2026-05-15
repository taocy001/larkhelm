"""larkhelm · shared traffic-split gating helper.

Used by both ``agent_hub.intent_router`` (phase 5) and
``memory_retriever`` (phase D) so that the same chat_id falls into the same
bucket under the same traffic %, satisfying NFR-DEPLOY-1 / AC-05.

Algorithm:
  - If the ``enabled_key`` config flag is falsy → False.
  - Else read ``traffic_key`` as a float in [0, 1].
  - traffic <= 0 → False; traffic >= 1 → True.
  - Otherwise: ``int(md5(chat_id)[:8], 16) % 10000 < traffic * 10000``.

Stdlib + ``larkhelm.config`` only — keep this module side-effect free."""
from __future__ import annotations

import hashlib


def hash_traffic_active(
    chat_id: str,
    enabled_key: str,
    traffic_key: str,
    *,
    default_enabled: bool = False,
    default_traffic: float = 0.0,
) -> bool:
    """Return True iff this chat_id should see the feature gated by
    ``enabled_key`` / ``traffic_key`` in ``larkhelm.config.config``.

    Fails closed (returns False) on any lookup or parse error so a config
    typo cannot accidentally turn a rollout on for 100% of users."""
    try:
        import larkhelm.config as _cfg
        cfg = getattr(_cfg, "config", {}) or {}
    except Exception:
        return False

    if not cfg.get(enabled_key, default_enabled):
        return False

    try:
        traffic = float(cfg.get(traffic_key, default_traffic) or 0.0)
    except (TypeError, ValueError):
        return False

    if traffic <= 0.0:
        return False
    if traffic >= 1.0:
        return True

    try:
        digest = hashlib.md5(chat_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
    except Exception:
        return False
    return bucket < int(traffic * 10000)


__all__ = ["hash_traffic_active"]
