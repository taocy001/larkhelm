"""larkhelm · pure-function helpers extracted from handlers/_message.py.

P2 REQ-03 / AC-02. These five helpers contain no Feishu SDK dependencies
and no I/O, so the message-router unit tests in
``tests/test_handlers_message_pure.py`` can drive them without spinning up
``lark_oapi`` or mocking ``P2ImMessageReceiveV1``.

Side-effecting routing (sending cards, mutating chat state, dispatching to
``_do_query``) still lives in ``handlers/_message.py``; this module only
encodes the decision logic.

Public API:
    classify_message_kind(msg_dict)         -> MessageKind
    extract_allowed_chat_decision(...)      -> AllowDecision
    should_skip_due_to_dedup(event_id, cache_keys) -> bool
    parse_doc_urls(text)                    -> list[str]
    route_to_command(text, command_names)   -> RouteDecision
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class MessageKind(Enum):
    """Subset of ``message_type`` values the bridge actually routes.

    The set is closed: any other value (sticker, system, audio_call …)
    maps to ``UNKNOWN`` and the message router drops the event silently.
    Keeping the enum small means a test for "did classification stay
    stable across refactors" doesn't have to enumerate the Feishu side.
    """
    TEXT = "text"
    IMAGE = "image"
    POST = "post"
    VOICE = "audio"      # Feishu calls it "audio"; "VOICE" is our name.
    FILE = "file"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class AllowDecision:
    """Result of ACL + group-mention checks combined."""
    allowed: bool
    reason: str   # human-readable diagnostic, e.g. "acl_reject" / "mention_missing"


@dataclass(frozen=True)
class RouteDecision:
    """Result of routing the (text, command_registry) pair.

    ``is_command=False`` means the text is not an explicit slash command and
    should flow through to AI dispatch (or the intent router gate).
    """
    handler_name: str   # canonical name (e.g. "/help"); "" for non-commands
    args: str           # text minus the matched command token, leading space stripped
    is_command: bool


# Mapping table is intentionally module-private; tests assert on enum
# values, not strings, so adding a synonym here doesn't break them.
_KIND_BY_VALUE: dict[str, MessageKind] = {k.value: k for k in MessageKind}


def classify_message_kind(msg: dict) -> MessageKind:
    """Map a raw Feishu message dict (``data.event.message.__dict__``-shaped)
    to a :class:`MessageKind`.

    Defensive: a non-dict input or a missing ``message_type`` returns
    ``UNKNOWN`` instead of raising — the caller already has a try/except
    around its routing block, but failing here would lose the chance to
    diagnose the bad input through normal observation.
    """
    if not isinstance(msg, dict):
        return MessageKind.UNKNOWN
    raw = msg.get("message_type") or msg.get("type") or ""
    return _KIND_BY_VALUE.get(str(raw).lower(), MessageKind.UNKNOWN)


def extract_allowed_chat_decision(
    chat_id: str,
    allowed: Iterable[str] | None,
    sender: str = "",
) -> AllowDecision:
    """Combine ``ALLOWED_CHATS`` whitelist + sender presence checks.

    The sender check is informational — empty allowed set means "all
    chats permitted". When the whitelist is non-empty and the chat isn't
    in it, the message is dropped silently (current bridge behaviour).
    ``sender`` is passed through so future ACL extensions (per-user
    whitelists) can reuse this helper without adding a fourth arg.
    """
    if not chat_id:
        return AllowDecision(allowed=False, reason="missing_chat_id")
    allow_set = set(allowed or ())
    if allow_set and chat_id not in allow_set:
        return AllowDecision(allowed=False, reason="acl_reject")
    return AllowDecision(allowed=True, reason="ok" if sender else "ok_no_sender")


def should_skip_due_to_dedup(event_id: str, cache_keys: Iterable[str]) -> bool:
    """Return True iff ``event_id`` is already in the dedup cache.

    The cache is passed in (not imported) so the function stays pure: a
    test can hand a literal ``set`` and verify the branch without
    touching the real ``dedup`` module. An empty / falsy ``event_id``
    returns False (the real dedup also short-circuits, but for a
    different reason — we mirror its observable behaviour).
    """
    if not event_id:
        return False
    try:
        return event_id in set(cache_keys or ())
    except TypeError:
        return False


# Feishu doc / wiki / sheets URLs. Keep the pattern liberal on the tenant
# subdomain so multi-tenant setups (xxx.feishu.cn, yyy.larksuite.com)
# both match; the path component is the discriminator.
_DOC_URL_RE = re.compile(
    r"https?://[^\s]+?(?:feishu\.cn|larksuite\.com)/(?:docx|wiki|sheets|docs|file)/[A-Za-z0-9_-]+"
)


def parse_doc_urls(text: str) -> list[str]:
    """Extract Feishu doc/wiki/sheet URLs from ``text``, preserving order.

    Duplicates are dropped (a user pasting the same link twice should not
    trigger a double-fetch). Returns an empty list when ``text`` is empty
    or contains no URLs.
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for m in _DOC_URL_RE.finditer(text):
        url = m.group(0)
        if url not in seen:
            seen.add(url)
            out.append(url)
    return out


def route_to_command(text: str, command_names: Iterable[str]) -> RouteDecision:
    """Decide whether ``text`` invokes one of the named slash commands.

    Matching rules:
      * exact (``tl == name``): ``args=""``;
      * prefix (``tl.startswith(name + " ")``): ``args`` = trailing remainder
        with the single separating space stripped.

    The ``command_names`` set is supplied by the caller (typically derived
    from ``larkhelm.command_registry.COMMAND_REGISTRY``) so this helper has
    no dependency on the registry itself.

    Longer command names are matched first so ``/memory diagnose`` wins
    over ``/memory`` when the registry contains both.
    """
    if not text or not text.startswith("/"):
        return RouteDecision(handler_name="", args="", is_command=False)
    tl = text.lower().strip()
    body = text.strip()
    # Iterate longest-first so a sub-command like ``/memory diagnose``
    # doesn't get pre-empted by its parent ``/memory``.
    names = sorted({str(n).lower() for n in (command_names or ()) if n}, key=len, reverse=True)
    for name in names:
        if tl == name:
            return RouteDecision(handler_name=name, args="", is_command=True)
        if tl.startswith(name + " "):
            # Strip the matched command token (preserving the original case
            # of the args by slicing on the raw text, not ``tl``).
            args = body[len(name):].lstrip()
            return RouteDecision(handler_name=name, args=args, is_command=True)
    return RouteDecision(handler_name="", args="", is_command=False)


__all__ = [
    "MessageKind",
    "AllowDecision",
    "RouteDecision",
    "classify_message_kind",
    "extract_allowed_chat_decision",
    "should_skip_due_to_dedup",
    "parse_doc_urls",
    "route_to_command",
]
