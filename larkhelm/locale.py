"""larkhelm · locale helpers — per-chat language support (zh / en).

Usage:
    from larkhelm.locale import _t
    lang = _get_lang(chat_id)          # "zh" | "en"
    title = _t(lang, "推理力度", "Reasoning Effort")
"""
from __future__ import annotations

SUPPORTED_LANGS = ("zh", "en")
DEFAULT_LANG = "zh"


def _t(lang: str, zh: str, en: str) -> str:
    """Return *en* when lang is 'en', otherwise *zh*."""
    return en if lang == "en" else zh
