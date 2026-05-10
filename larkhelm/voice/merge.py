"""larkhelm voice · per-chat STT merge buffer (M3.2 commit 4).

Sits between ``larkhelm.voice.transcribe`` and
``larkhelm.handlers._query._do_query``: callers (commit 5 wires this up
in ``_message.py``) push transcribed fragments via :func:`add_voice`;
this module batches them per chat and flushes the joined prompt to
``_do_query`` in a daemon thread.

Concurrency model
-----------------
* A single module-level ``threading.Lock`` (``_meta``) covers
  ``_buffers`` / ``_timers`` and Timer start/cancel.  PRD §4 explicitly
  forbids a second lock — single-lock keeps the deadlock proof trivial:
  the only operations holding ``_meta`` are pure-Python list mutations
  plus ``Timer.cancel`` (which itself never re-enters this module).
* :func:`_on_timer` (the Timer callback) re-acquires ``_meta`` from the
  Timer thread before delegating to :func:`_flush_locked`.  Races where
  a cap-flush already drained the buffer are absorbed by the empty-check
  at the top of :func:`_flush_locked`.
* The actual ``_do_query`` call happens in a fresh ``daemon=True``
  thread — never on the caller's thread, never on the Timer thread.
  This satisfies AC-06 and keeps a slow LLM call from holding ``_meta``.

Commit 5 integration
--------------------
Commit 5 will add a single line in ``_message.py`` after a successful
``transcribe`` call::

    from larkhelm.voice.merge import add_voice
    add_voice(chat_id, result["text"], model,
              user_msg_id=msg_id, parent_id=root_id)

The signature is locked here; this module is intentionally **not**
re-exported via ``larkhelm.voice.__init__`` so commit 4 landing alone
cannot accidentally reroute production traffic.

Public surface
--------------
Only :func:`add_voice` is public.  ``_VoiceItem`` / ``_flush_locked`` /
``_on_timer`` / ``_dispatch`` are private but visible to tests via
attribute access (mocked in ``tests/test_voice_merge.py``).
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import larkhelm.config as _cfg
from larkhelm import log as _log

__all__ = ["add_voice"]


@dataclass(slots=True)
class _VoiceItem:
    text: str
    model: str
    user_msg_id: Optional[str]
    parent_id: Optional[str]
    ts: float


# ── Module-level state ─────────────────────────────────────────────────────
# Reset by tests in setUp via direct attribute access (mirrors transcribe.py).
_buffers: "OrderedDict[str, list[_VoiceItem]]" = OrderedDict()
_timers: "dict[str, threading.Timer]" = {}
_meta: threading.Lock = threading.Lock()
# Monkey-patch seam — tests swap this for a FakeTimer that records calls
# without actually scheduling a thread.  Patching the attribute on this
# module avoids touching ``threading.Timer`` globally and bleeding into
# unrelated background threads.
_Timer = threading.Timer


def add_voice(
    chat_id: str,
    text: str,
    model: str,
    user_msg_id: Optional[str] = None,
    parent_id: Optional[str] = None,
) -> None:
    """Push one transcribed voice fragment into ``chat_id``'s merge buffer.

    Behavior contract (PRD §3 P0):

    * ``text.strip() == ""`` → silently dropped; buffer & timer untouched.
    * ``len(buffer) >= VOICE_MAX_MERGE`` after append → flush synchronously
      (still inside ``_meta`` lock); do NOT start a Timer.
    * ``VOICE_MERGE_WINDOW_SEC == 0`` → flush synchronously after append;
      do NOT start a Timer.
    * Else → cancel any prior Timer for ``chat_id``, start a new one.

    Never raises.  Returns immediately after scheduling — the actual LLM
    call runs in a daemon thread.
    """
    if not text or not text.strip():
        return

    item = _VoiceItem(
        text=text.strip(),
        model=model,
        user_msg_id=user_msg_id,
        parent_id=parent_id,
        ts=time.monotonic(),
    )

    window = int(getattr(_cfg, "VOICE_MERGE_WINDOW_SEC", 0))
    cap = max(1, int(getattr(_cfg, "VOICE_MAX_MERGE", 5)))

    with _meta:
        buf = _buffers.setdefault(chat_id, [])
        buf.append(item)

        # Cap-flush takes priority over window-flush so that a buffer
        # which crosses the cap exactly at window=0 still flushes once.
        if len(buf) >= cap:
            # Cancel any pending Timer first — cap reached means we
            # don't want a stale Timer firing on an empty buffer later.
            t = _timers.pop(chat_id, None)
            if t is not None:
                try:
                    t.cancel()
                except Exception:
                    pass
            _flush_locked(chat_id)
            return

        if window <= 0:
            _flush_locked(chat_id)
            return

        # Window > 0 and below cap: (re)arm the Timer.
        prev = _timers.pop(chat_id, None)
        if prev is not None:
            try:
                prev.cancel()
            except Exception:
                pass
        timer = _Timer(window, _on_timer, args=(chat_id,))
        timer.daemon = True
        _timers[chat_id] = timer
        timer.start()


def _flush_locked(chat_id: str) -> None:
    """Drain ``chat_id``'s buffer and dispatch.  Caller MUST hold ``_meta``.

    Pops both the buffer and any outstanding Timer for the chat (so a
    racing Timer callback that re-enters via :func:`_on_timer` finds an
    empty buffer and short-circuits).  No-op if the buffer is empty.
    """
    items = _buffers.pop(chat_id, None)
    timer = _timers.pop(chat_id, None)
    if timer is not None:
        try:
            timer.cancel()
        except Exception:
            pass
    if not items:
        return

    prompt = "\n\n".join(it.text for it in items)
    head = items[0]
    threading.Thread(
        target=_dispatch,
        args=(prompt, chat_id, head.model, head.user_msg_id, head.parent_id),
        daemon=True,
        name=f"voice-merge-dispatch-{chat_id[:8]}",
    ).start()


def _on_timer(chat_id: str) -> None:
    """Timer callback.  Runs on the Timer thread; re-acquires ``_meta``.

    Lives outside ``add_voice``'s call stack — without re-acquiring the
    lock here a Timer that fires concurrently with a cap-flush would
    double-dispatch.
    """
    with _meta:
        _flush_locked(chat_id)


def _dispatch(
    prompt: str,
    chat_id: str,
    model: str,
    user_msg_id: Optional[str],
    parent_id: Optional[str],
) -> None:
    """Daemon-thread body.  Lazy-imports ``_do_query`` and calls it.

    The lazy import is the single critical decoupling that keeps
    ``import larkhelm.voice.merge`` zero-side-effect: ``_do_query``
    transitively pulls in ``chat_state`` / ``lark_client`` / the rest
    of the bridge, so importing it at module top would create an
    import cycle (commit 5 imports this module from ``_message.py``,
    which is in turn imported by handlers that ``_do_query`` reaches).

    Wraps the call in ``try/except`` so a daemon-thread death does not
    surface as ``unraisable exception`` on stderr — instead routes to
    ``log.warn`` with a ``[VoiceMerge]`` prefix.
    """
    try:
        from larkhelm.handlers._query import _do_query  # lazy import
        _do_query(
            chat_id,
            prompt,
            model,
            user_msg_id=user_msg_id,
            parent_id=parent_id,
        )
    except Exception as e:
        try:
            _log.warn(f"[VoiceMerge] dispatch failed for chat={chat_id}: {e}")
        except Exception:
            pass
