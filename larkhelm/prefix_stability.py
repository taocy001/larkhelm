"""larkhelm · stable prefix tracker for Anthropic layered cache control (Week-4 REQ-07).

Tracks the SHA-256 hash of the stable system-prompt prefix per chat_id so
operators can detect when the global+project memory block changes unexpectedly
between turns (which would bust the Anthropic prompt-cache prefix and cause a
cache miss).

Thread-safe. No I/O. Never raises.
"""
from __future__ import annotations

import hashlib
import threading
from typing import ClassVar

from larkhelm.log import _debug_log

__all__ = ["StablePrefixTracker"]


class StablePrefixTracker:
    """Process-singleton (classmethod-accessed) stable-prefix hash registry.

    Callers invoke ``StablePrefixTracker.track(chat_id, stable_text, backend=...)``
    once per Anthropic API call made in layered-cache mode. The first call for a
    given ``chat_id`` seeds the hash and returns ``False``; subsequent calls
    return ``True`` when the hash has changed (prefix instability) and ``False``
    when it matches.
    """

    _hashes: ClassVar[dict[str, str]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def track(cls, chat_id: str, stable_text: str, *, backend: str = "") -> bool:
        """Record stable prefix hash for ``chat_id``; detect changes.

        Hashes ``stable_text[:4096]`` via SHA-256. On first call returns False.
        On hash change: logs ``[PrefixStability]`` debug line and bumps the
        ``larkhelm_prefix_stability_low_total`` Prometheus counter.

        Thread-safe. Never raises.

        Returns True iff a hash change was detected (prefix instability).
        """
        try:
            digest = hashlib.sha256(stable_text[:4096].encode("utf-8")).hexdigest()
            with cls._lock:
                prev = cls._hashes.get(chat_id)
                cls._hashes[chat_id] = digest
            if prev is None:
                return False
            if prev == digest:
                return False
            _debug_log(
                f"[PrefixStability] prefix changed for chat={chat_id[:8]} "
                f"backend={backend} (prev={prev[:8]}... new={digest[:8]}...)"
            )
            try:
                from larkhelm.metrics import inc_prefix_stability_low
                inc_prefix_stability_low(backend)
            except Exception:
                pass
            return True
        except Exception:
            return False
