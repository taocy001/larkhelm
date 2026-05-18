"""larkhelm · shared traffic-split gating helper.

Used by both ``agent_hub.intent_router`` (phase 5) and
``memory_retriever`` (phase D) so that the same chat_id falls into the same
bucket under the same traffic %, satisfying NFR-DEPLOY-1 / AC-05.

Algorithm:
  - If the ``enabled_key`` config flag is falsy → False.
  - Else read ``traffic_key`` as a float in [0, 1].
  - traffic <= 0 → False; traffic >= 1 → True.
  - Otherwise: ``int(md5(salt + ":" + chat_id)[:8], 16) % 10000 < traffic * 10000``.

The digest is **salted with the traffic_key name** so two independent
rollouts (e.g. Phase D Stage A memory_retriever_traffic and Stage B
embedding_traffic) sit in statistically independent buckets — review
SF-01 follow-up. Without the salt, the same chat_id had the same bucket
under both rollouts, which silently made Stage B nested inside Stage A
instead of orthogonal. Now Stage A=0.3 ∩ Stage B=0.7 ≈ 0.21 (as the
PRD's "orthogonal" wording promises) rather than 0.30.

Stdlib + ``larkhelm.config`` only — keep this module side-effect free."""
from __future__ import annotations

import hashlib


def hash_bucket_allows(chat_id: str, traffic: float) -> bool:
    """Deterministic per-chat bucketing for gated rollout.

    Sibling of :func:`hash_traffic_active`, but stripped to the bare
    decision: caller has already resolved the traffic float (e.g. from
    ``_cfg.QUERY_SESSION_V2_TRAFFIC``) and just wants the bool.

    Algorithm uses sha256 of the chat_id mod 10000 against
    ``traffic * 10000``. Same chat_id always lands in the same bucket
    for a given traffic value (stable, deterministic). 1000 random
    chat_ids at traffic=0.5 fall in [0.45, 0.55] (verified by
    ``tests/test_query_session_gating.py``).

    Note: this helper is **unsalted** on purpose — it is a generic
    per-chat dial and not meant to compose orthogonally with other
    rollouts the way :func:`hash_traffic_active` does.
    """
    try:
        t = float(traffic)
    except (TypeError, ValueError):
        return False
    if t <= 0.0:
        return False
    if t >= 1.0:
        return True
    try:
        digest = hashlib.sha256(chat_id.encode("utf-8")).hexdigest()
        bucket = int(digest[:8], 16) % 10000
    except Exception:
        return False
    return bucket < int(t * 10000)


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
    typo cannot accidentally turn a rollout on for 100% of users.

    Bucket independence: the md5 digest is salted with ``traffic_key`` so
    that distinct rollouts (e.g. memory_retriever_traffic vs
    embedding_traffic) fall into independent buckets. See module docstring.
    """
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
        salted = f"{traffic_key}:{chat_id}".encode("utf-8")
        digest = hashlib.md5(salted).hexdigest()
        bucket = int(digest[:8], 16) % 10000
    except Exception:
        return False
    return bucket < int(traffic * 10000)


__all__ = ["hash_traffic_active", "hash_bucket_allows"]
