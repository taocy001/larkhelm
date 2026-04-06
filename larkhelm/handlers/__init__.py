"""
larkhelm.handlers — Feishu event handler sub-package

Backward-compatible re-export; external code can continue importing all
public symbols from larkhelm.handlers.
"""
from larkhelm.handlers._query import (
    _extract_feishu_urls,
    _inject_doc_context,
    _do_query,
)
from larkhelm.handlers._card_action import handle_card_action
from larkhelm.handlers._message import (
    handle_message,
    handle_reaction_created,
)

__all__ = [
    "handle_message",
    "handle_card_action",
    "handle_reaction_created",
    "_do_query",
    "_extract_feishu_urls",
    "_inject_doc_context",
]
