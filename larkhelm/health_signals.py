"""larkhelm · health-signal classifier.

Maps a backend failure (exception text or stderr) to a coarse category so the
registry can decide *immediately* whether to flip ``healthy=False``:

* AUTH / QUOTA / MODEL_NOT_FOUND  → flip on first hit (won't self-heal soon)
* TRANSIENT                       → only flip after N hits within a window
                                    (single hiccups should not disable a backend)
* USER_CANCEL / TIMEOUT           → don't change health (user/time, not backend)

Patterns are case-insensitive, applied in priority order. Anything unmatched
defaults to TRANSIENT — conservative because we'd rather miss a flip than
mis-classify a real auth failure as a hiccup.
"""
from __future__ import annotations

import re

# ── Categories ────────────────────────────────────────────────────────────────

AUTH           = "AUTH"
QUOTA          = "QUOTA"
MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
TRANSIENT      = "TRANSIENT"
USER_CANCEL    = "USER_CANCEL"
TIMEOUT        = "TIMEOUT"

#: Categories that should NOT update healthy state (caller should early-return).
NO_HEALTH_UPDATE = frozenset({USER_CANCEL, TIMEOUT})

#: Categories that flip healthy=False on the very first occurrence.
INSTANT_UNHEALTHY = frozenset({AUTH, QUOTA, MODEL_NOT_FOUND})

#: Categories that must NOT be retried on a different backend.
#: OOM/TIMEOUT is a host resource issue — falling back to another backend is wrong:
#: it silently serves a different AI while the real problem (memory pressure) goes unnoticed.
#: USER_CANCEL is also non-retriable (user intent).
NON_RETRIABLE = frozenset({TIMEOUT, USER_CANCEL})

# ── Pattern table (applied top-down) ──────────────────────────────────────────

_PATTERNS: list[tuple[str, list[str]]] = [
    (USER_CANCEL, [
        r"\bquerycancellederror\b",
        r"\bcancel(?:l|ll)?ed\b",
        r"\buser cancel",
    ]),
    (TIMEOUT, [
        r"force[- ]killed",
        r"soft[- ]?timeout",
        r"hard[- ]?timeout",
        r"killed by os",       # OOM-killed subprocess (rc=-9); host memory pressure, not backend fault
        r"\brc=-9\b",
        r"cgroup oom",
    ]),
    (AUTH, [
        r"\b401\b",
        r"\b403\b",
        r"invalid authentication",
        r"unauthor",
        r"authentication.*fail",
        r"api.?key.*invalid",
        r"missing.*api.?key",
    ]),
    (QUOTA, [
        r"\b429\b",
        r"quota.*exhaust",
        r"quota.*exceed",
        r"rate.?limit",
        r"too many requests",
        r"insufficient.*quota",
        r"resource.exhausted",         # Google: RESOURCE_EXHAUSTED status
    ]),
    (MODEL_NOT_FOUND, [
        r"model.*not.found",
        r"modelnotfound",
        r"\b404\b.*model",
        r"unknown model",
    ]),
    # NOTE: Anthropic 529 ``overloaded_error`` is intentionally classified
    # TRANSIENT (server overload, retry later — not a quota / cost issue).
    # Listed explicitly here so it's clear it shouldn't be moved to AUTH.
    # Also catches Google ``UNAVAILABLE`` 503-equivalents.
    (TRANSIENT, [
        r"overloaded_error",
        r"\boverloaded\b",
        r"\bunavailable\b",
    ]),
]


def classify_error(err: str | Exception | None) -> str:
    """Return one of the AUTH / QUOTA / MODEL_NOT_FOUND / TRANSIENT / USER_CANCEL / TIMEOUT
    constants. Default ``TRANSIENT`` for unrecognized errors.
    """
    if err is None:
        return TRANSIENT
    text = str(err).lower()
    if not text:
        return TRANSIENT
    for category, patterns in _PATTERNS:
        for p in patterns:
            if re.search(p, text):
                return category
    return TRANSIENT


def is_no_op(category: str) -> bool:
    """True if this category should NOT touch health state (cancel/timeout)."""
    return category in NO_HEALTH_UPDATE


def is_instant_unhealthy(category: str) -> bool:
    """True if this category should flip ``healthy=False`` on first hit."""
    return category in INSTANT_UNHEALTHY
