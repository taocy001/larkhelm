"""larkhelm · pure helpers extracted from ``_do_query`` (P1-1 PR1).

Each function is a deterministic input→output transform with **no** I/O:
no Feishu API, no backend invocation, no chat lock, no shared mutable
state. They are the atomic pieces ``_do_query`` (and the upcoming
``QuerySession.run``) compose into a query pipeline.

Why this exists: ``_do_query`` was 938 lines of closures, the bulk of
which were string-shaping or list-building steps tangled into the
streaming-card state machine. Extracting them into a flat module makes
each step independently unit-testable without spinning up the chat
lock / backend resolution / Feishu API stack.
"""
from __future__ import annotations

import os as _os
import re
from typing import Any

import larkhelm.config as _cfg
from larkhelm.card_builder import _make_card, _split_md, _fmt_elapsed
from larkhelm.log import _debug_log


_FEISHU_URL_RE = re.compile(r'https://[a-zA-Z0-9-]+\.feishu\.cn/[^\s\]>）]+')


def build_init_card(m_name: str, cwd: str, chat_id: str) -> str:
    """Render the initial "connecting" card as a card-JSON string.

    Returned form mirrors the legacy in-place call (``_make_card(...)``)
    so callers can hand it straight to ``_reply_card_raw`` /
    ``_send_card_raw``.
    """
    return _make_card(
        f"⏳ {m_name} 连接中",
        f"> 正在启动...\n\n目录: `{cwd}`",
        color="grey",
        buttons=[("🛑 取消", f"cancel:{chat_id}")],
    )


def build_failover_chain(primary_spec, registry, force_direct: bool) -> list:
    """Compose the backend failover chain: primary first, then orchestrators.

    Mirrors the slice from ``_do_query`` so changes touch one place:
      * Start with ``registry.get_orchestrator_chain()``.
      * Prepend ``primary_spec`` when healthy and not already first.
      * When ``force_direct`` is True, collapse the chain to ``[primary_spec]``
        (e.g. user typed ``/c`` / ``/g`` / ``/k`` — no delegation).
      * Returns ``[]`` when no chain is available; caller falls back to
        legacy routing.
    """
    chain = list(registry.get_orchestrator_chain())
    if primary_spec is not None and getattr(primary_spec, "healthy", False):
        chain_ids = [s.id for s in chain]
        if primary_spec.id not in chain_ids:
            chain = [primary_spec] + chain
        elif chain and chain[0].id != primary_spec.id:
            chain = [primary_spec] + [s for s in chain if s.id != primary_spec.id]

    if force_direct and primary_spec is not None and getattr(primary_spec, "healthy", False):
        return [primary_spec]
    return chain


def inject_doc_and_memory(
    msg: str,
    chat_id: str,
    cwd: str,
    *,
    intent: Any = None,
    doc_auto_inject: bool,
    has_doc_urls: bool,
    sender_open_id: "str | None" = None,
) -> tuple[str, str, list[str]]:
    """Combined doc-injection + memory-context build (pure-ish wrapper).

    Despite "pure" in the module name, this helper does touch the
    Feishu doc-read API and disk (memory load) because those are exactly
    the side effects ``_do_query`` was performing inline. The split is
    purely structural: lift one big nested block into a named function
    so it can be replaced/stubbed in tests.

    Returns ``(enriched_msg, memory_ctx, deduped_recent_turns)``.
    On any failure each component degrades to its empty equivalent
    (``msg`` returned unchanged, empty ``memory_ctx``, empty list).
    """
    enriched = msg
    memory_ctx = ""
    deduped_recent: list[str] = []

    if doc_auto_inject:
        try:
            from larkhelm.handlers._query import _inject_doc_context
            enriched = _inject_doc_context(enriched, chat_id)
        except Exception as e:
            _debug_log(f"[QueryPure] doc inject error: {e}")
            enriched = msg

    try:
        from larkhelm.log import _get_recent_turns
        from larkhelm.memory import load_memory
        from larkhelm.memory_context import extract_work_context

        dedup_prefix: str | None = None
        try:
            session_raw = load_memory(chat_id)
            wc = extract_work_context(session_raw)
            dedup_prefix = wc or None
        except Exception as e:
            _debug_log(f"[QueryPure] work_context extract error: {e}")
            dedup_prefix = None

        try:
            raw_recent = _get_recent_turns(chat_id, dedup_prefix=dedup_prefix) or ""
        except Exception as e:
            _debug_log(f"[QueryPure] dedup recent_turns error: {e}, retrying without prefix")
            raw_recent = _get_recent_turns(chat_id) or ""

        recent_list = [ln for ln in raw_recent.splitlines() if ln.strip()] if raw_recent else []

        try:
            from larkhelm.memory import get_memory_context_v2
            memory_ctx, deduped_recent = get_memory_context_v2(
                chat_id,
                cwd=cwd,
                query=msg,
                recent_turns=recent_list,
                has_doc_urls=has_doc_urls,
                intent=intent,
                sender_open_id=sender_open_id,
            )
        except Exception as e:
            _debug_log(f"[QueryPure] memory context error: {e}")
            memory_ctx = ""
            deduped_recent = recent_list
    except Exception as e:
        _debug_log(f"[QueryPure] memory pipeline outer error: {e}")
        memory_ctx = ""
        deduped_recent = []

    return enriched, memory_ctx, deduped_recent


def select_legacy_runner(model: str):
    """Return the legacy single-backend runner callable for ``model``.

    Mirrors the ``elif`` chain in ``_do_query`` that fires when the
    BackendRegistry chain is empty (no probe results yet). Returns the
    function reference — caller binds positional args.
    """
    from larkhelm.ai_runner import (
        query_claude, query_gemini, query_kimi, query_deepseek,
    )
    if model == "gemini":
        return query_gemini
    if model == "kimi":
        return query_kimi
    if model == "deepseek":
        return query_deepseek
    return query_claude


def format_completion_card(
    m_name: str,
    output: str,
    n_tools: int,
    elapsed: str,
    final_tools: list,
    max_card_len: int,
) -> tuple[list[str], str, "list | None"]:
    """Compose the final card chunks + note + safe tools payload.

    Splits ``output`` via ``_split_md`` (Feishu card-length limit),
    builds the ``"使用了 N 次工具 · 耗时 X"`` note, and decides whether
    the tool-list payload is safe to attach to the first card. When the
    serialised tool list exceeds 20 000 bytes, returns ``None`` so the
    caller drops detailed results (Feishu silently truncates oversized
    payloads, otherwise the first card disappears entirely).

    Returns ``(chunks, note, tools_payload_or_None)``.
    """
    del max_card_len  # threshold lives in card_builder; signature kept for clarity
    chunks = _split_md(output.strip() if output else "")
    note = (f"使用了 {n_tools} 次工具 · " if n_tools else "") + f"耗时 {elapsed}"

    tools_payload: list | None = final_tools if final_tools else None
    if tools_payload:
        try:
            import json as _json
            if len(_json.dumps(tools_payload, ensure_ascii=False)) > 20_000:
                tools_payload = None
        except Exception as e:
            _debug_log(f"[QueryPure] tools payload serialize failed: {e}")
            tools_payload = None
    return chunks, note, tools_payload


def cleanup_temp_images(images: "list[str] | None") -> None:
    """Unlink any temp image paths under ``/tmp/`` created for the query.

    Safe no-op on missing files / non-tmp paths. Matches the in-place
    cleanup block at the bottom of ``_do_query``'s ``finally``.
    """
    if not images:
        return
    for img in images:
        try:
            if img and str(img).startswith("/tmp/"):
                _os.unlink(img)
        except FileNotFoundError:
            pass
        except Exception as e:
            _debug_log(f"[QueryPure] temp image cleanup failed: {e}")


def extract_feishu_urls(text: str) -> list[str]:
    """Re-export of the URL extractor so callers needn't reach into _query."""
    return _FEISHU_URL_RE.findall(text)


__all__ = [
    "build_init_card",
    "build_failover_chain",
    "inject_doc_and_memory",
    "select_legacy_runner",
    "format_completion_card",
    "cleanup_temp_images",
    "extract_feishu_urls",
]
