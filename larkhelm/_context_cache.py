"""larkhelm · context-injection cache primitives

Three call sites in ``_do_query`` reload the same data on every turn:

  * ``log._get_recent_turns`` re-reads the tail 100 KB of ``all.jsonl``.
  * ``memory_context._layer_global / _layer_project / _layer_session``
    re-open up to four ``.md`` files.
  * ``handlers._query._inject_doc_context`` re-issues a Feishu RPC for
    each URL in the user message.

The first two are **idempotent for a given file mtime**; the third is
idempotent for ~60 seconds (Feishu has no mtime channel we can read).
This module owns the three caches that make those calls effectively free
on repeat invocations, plus a small set of helpers callers use to wrap
their loaders.

Module boundaries (PRD §6 + design.md §2): this file MUST NOT import any
business module (``memory_context``, ``lark_client``, ``log``, etc.).
The two cache primitives are generic; the three high-level helpers
receive a ``loader`` callback that the caller supplies, so the cache
never knows what the loader actually does.

Known backlog items (round-1 reviewer kimi 2026-05-20, **non-blocking**):
  * ``TTLCache`` has no ``maxsize`` cap — relies on the 60s TTL +
    lazy eviction on read + ``invalidate_chat`` for sweep. Current
    worst case (50 chats × 5 docs each × ~few KB) is bounded by
    natural usage, but a long-running large-scale instance might
    want either an LRU eviction layer on top, or a periodic
    background sweeper. Track as a future feature, not a defect.
  * ``MemoryContextBuilder._layer_*`` fallback path re-attempts
    ``import larkhelm._context_cache`` on every call when the
    cache module is missing. Python doesn't cache failed imports,
    so an absent module costs a fresh ImportError per request.
    The cache module being absent is a deployment defect (it ships
    with the package), so this path realistically never fires —
    documenting for completeness only.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Generic, Hashable, Optional, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


# ── Cache key dataclasses ──────────────────────────────────────────────────


@dataclass(frozen=True)
class RecentTurnsKey:
    chat_id: str
    max_turns: int
    max_chars: int
    dedup_prefix_hash: str
    jsonl_mtime_ns: int
    jsonl_size: int


@dataclass(frozen=True)
class MemoryLayerKey:
    layer: str
    file_path: str
    mtime_ns: int


@dataclass(frozen=True)
class DocKey:
    chat_id: str
    doc_type: str
    token: str
    max_chars: int


@dataclass(frozen=True)
class DocCachedEntry:
    """Value side of the doc TTL cache.

    Reviewer round-1 nit #2: previously this carried a ``fetched_at:
    float`` field, but ``TTLCache`` already stamps each entry with a
    monotonic ``fetched_at`` in its own ``(fetched_at, value)`` tuple
    (see ``TTLCache.put``). The dataclass field was never consumed by
    anyone — the expiry check reads the tuple stamp, not the field.
    Slimmed to ``payload`` only; ``__init__`` keeps the keyword for
    backwards-compat with any third-party constructor calls but
    ignores it. Default = no caller has to migrate.
    """
    payload: Any

    def __init__(self, *, payload: Any, fetched_at: float | None = None) -> None:
        # frozen=True forces us to use object.__setattr__ here; this is
        # the documented pattern for back-compat keyword arguments on a
        # frozen dataclass. ``fetched_at`` is accepted purely for
        # backwards compatibility with the pre-cleanup constructor and
        # is intentionally discarded (TTLCache stamps its own monotonic
        # timestamp). Reviewer round-1 nit follow-up: kimi flagged the
        # earlier ``del fetched_at`` as semantically misleading (``del``
        # implies a lifetime release the name doesn't actually have),
        # so we omit it; the project's ruff config doesn't enable
        # ARG002 (unused-argument) today, so no lint suppression needed.
        object.__setattr__(self, "payload", payload)


# ── Generic primitives ─────────────────────────────────────────────────────


class LRUCache(Generic[K, V]):
    """Thread-safe LRU on top of ``OrderedDict``.

    The read path takes the lock twice (once for the hit-check, once to
    write back on miss) rather than once to keep the I/O-doing loader OUT
    of the critical section. Holding the lock for the full loader would
    serialise ``_get_recent_turns`` across chats — defeating the cache.
    """

    def __init__(self, name: str, maxsize: int) -> None:
        if maxsize <= 0:
            raise ValueError("maxsize must be > 0")
        self.name = name
        self._maxsize = int(maxsize)
        self._data: "OrderedDict[K, V]" = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0
        self._evicts = 0

    def get(self, key: K) -> tuple[bool, Optional[V]]:
        """Return ``(hit, value)``. On ``hit=False`` the value is undefined."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._hits += 1
                return True, self._data[key]
            self._misses += 1
            return False, None

    def put(self, key: K, value: V) -> Optional[K]:
        """Insert or promote ``key`` → ``value``. Returns the evicted key
        when the LRU capacity is exceeded, else ``None``."""
        evicted: Optional[K] = None
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self._data[key] = value
                return None
            self._data[key] = value
            if len(self._data) > self._maxsize:
                evicted, _ = self._data.popitem(last=False)
                self._evicts += 1
        return evicted

    def invalidate(self, key: K) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0
            self._evicts = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "maxsize": self._maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "evicts": self._evicts,
            }


class TTLCache(Generic[K, V]):
    """Thread-safe TTL cache keyed on ``Hashable``.

    Expired entries are removed in-place on the read that observed the
    expiry — no separate sweeper thread. Suitable for the Feishu doc
    cache where natural usage rotates the working set.
    """

    def __init__(self, name: str, ttl_sec: float) -> None:
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be > 0")
        self.name = name
        self._ttl = float(ttl_sec)
        self._data: dict[K, tuple[float, V]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    @property
    def ttl_sec(self) -> float:
        return self._ttl

    def set_ttl(self, ttl_sec: float) -> None:
        """Hot-update the TTL value.

        Reviewer round-1 nit #3: previously callers (``cached_doc_read``
        + ``reset_for_tests``) wrote ``self._ttl`` directly to honour the
        ``doc_inject_cache_ttl_sec`` config hot-reload contract. This
        encapsulated method replaces the private-attr-poke. CPython
        float assignment is already atomic at the bytecode level, so we
        don't grab the lock — just validate and store. ``ttl_sec`` must
        be positive (same contract as ``__init__``).
        """
        if ttl_sec <= 0:
            raise ValueError("ttl_sec must be > 0")
        self._ttl = float(ttl_sec)

    def get(self, key: K) -> Optional[V]:
        now = time.monotonic()
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self._misses += 1
                return None
            fetched_at, payload = entry
            if now - fetched_at > self._ttl:
                # Expired — drop in place, count as miss.
                self._data.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return payload

    def put(self, key: K, value: V) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)

    def invalidate_chat(self, chat_id: str) -> None:
        """Drop every entry whose key has a matching ``chat_id`` attribute.

        Used by ``/reset`` so a flushed chat doesn't keep serving stale
        doc bodies after the user explicitly asked for a clean slate.
        Keys without a ``chat_id`` attribute are left untouched.
        """
        with self._lock:
            doomed = [k for k in self._data
                      if getattr(k, "chat_id", None) == chat_id]
            for k in doomed:
                self._data.pop(k, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "size": len(self._data),
                "ttl_sec": int(self._ttl),
                "hits": self._hits,
                "misses": self._misses,
            }


# ── Module-level singletons ────────────────────────────────────────────────
#
# Capacity numbers come from design.md §1.1 + §3.2. The doc TTL is
# resolved lazily from config on every call so an operator flipping
# ``doc_inject_cache_ttl_sec`` doesn't need a bridge restart.

_RECENT_TURNS_MAX = 64
_MEMORY_LAYER_MAX = 128
_DOC_DEFAULT_TTL = 60.0

_recent_turns_cache: LRUCache[RecentTurnsKey, str] = LRUCache(
    "recent_turns", _RECENT_TURNS_MAX
)
_memory_layer_cache: LRUCache[MemoryLayerKey, Optional[str]] = LRUCache(
    "memory_layer", _MEMORY_LAYER_MAX
)
_doc_cache: TTLCache[DocKey, DocCachedEntry] = TTLCache(
    "doc_inject", _DOC_DEFAULT_TTL
)


# ── Internal utilities ─────────────────────────────────────────────────────


def _stat_file(path: Optional[Path]) -> tuple[int, int]:
    """Return ``(mtime_ns, size)`` for ``path`` or ``(0, 0)`` on miss / error.

    Never raises — a stat failure must collapse to "file missing" so the
    cache lookup still has a stable key (and an unrelated miss doesn't
    crash a hot path).
    """
    if path is None:
        return 0, 0
    try:
        st = path.stat()
        return int(st.st_mtime_ns), int(st.st_size)
    except (FileNotFoundError, OSError):
        return 0, 0


def _dedup_hash(prefix: Optional[str]) -> str:
    """Stable 16-hex digest of ``prefix`` for embedding in a cache key.

    Empty / None → empty hash so the key still compares cleanly. blake2b
    with ``digest_size=8`` gives 16 hex chars — enough collision space
    for a 64-entry cache without bloating each key.
    """
    raw = (prefix or "").encode("utf-8", errors="ignore")
    return hashlib.blake2b(raw, digest_size=8).hexdigest()


def _doc_ttl_sec() -> float:
    """Resolve the live doc TTL. Reads ``_cfg.DOC_INJECT_CACHE_TTL_SEC``
    on every call so an operator flipping the value doesn't need a
    bridge restart. Falls back to the design-doc default on any access
    failure (config not yet loaded / module not yet imported).
    """
    try:
        import larkhelm.config as _cfg  # local import — break the cycle
        v = float(getattr(_cfg, "DOC_INJECT_CACHE_TTL_SEC", _DOC_DEFAULT_TTL)
                  or _DOC_DEFAULT_TTL)
        return v if v > 0 else _DOC_DEFAULT_TTL
    except Exception:
        return _DOC_DEFAULT_TTL


def _config_flag(name: str, default: bool = True) -> bool:
    """Read a config flag without forcing a hard import dependency."""
    try:
        import larkhelm.config as _cfg
        return bool(getattr(_cfg, name, default))
    except Exception:
        return default


def _inc_outcome(cache_name: str, outcome: str, layer: str | None = None) -> None:
    """Best-effort Prometheus counter bridge. Never raises — a metrics
    failure must not break the cache hot path."""
    try:
        from larkhelm import metrics as _metrics
        if cache_name == "recent_turns":
            _metrics.inc_recent_turns_cache(outcome)
        elif cache_name == "memory_layer":
            _metrics.inc_memory_layer_cache(layer or "unknown", outcome)
        elif cache_name == "doc_inject":
            _metrics.inc_doc_inject_cache(outcome)
    except Exception:
        pass


def _log_event(cache_name: str, outcome: str, key_repr: str) -> None:
    """Best-effort ``[Cache]`` debug log. Mirrors the prefix style used by
    ``log.py`` so operators can grep both halves of the cache path."""
    try:
        from larkhelm.log import _debug_log
        _debug_log(f"[Cache] {cache_name} {outcome} {key_repr}")
    except Exception:
        pass


# ── High-level helpers ─────────────────────────────────────────────────────


def cached_recent_turns(
    chat_id: str,
    max_turns: int,
    max_chars: int,
    dedup_prefix: Optional[str],
    *,
    loader: Callable[[], str],
) -> str:
    """LRU-cached wrapper for ``log._get_recent_turns_uncached``.

    Key components:

      * ``chat_id`` — turns are per-chat by construction.
      * ``max_turns`` / ``max_chars`` — caller-controlled limits.
      * ``dedup_prefix_hash`` — 8-byte blake2b so a 400-char Work Context
        slot doesn't bloat the key.
      * ``jsonl_mtime_ns`` + ``jsonl_size`` — the only signal larkhelm
        sees when a new turn arrives. ext4/xfs preserve both on each
        ``log_entry`` append, so any new write invalidates the entry.

    Cache disabled (config flag off) → directly returns ``loader()``;
    no metric, no debug log.
    """
    if not _config_flag("RECENT_TURNS_CACHE_ENABLED", True):
        return loader()

    try:
        import larkhelm.config as _cfg
        jsonl_path = Path(_cfg.LOG_DIR) / "all.jsonl"
    except Exception:
        jsonl_path = None
    mtime_ns, size = _stat_file(jsonl_path)

    key = RecentTurnsKey(
        chat_id=chat_id,
        max_turns=int(max_turns),
        max_chars=int(max_chars),
        dedup_prefix_hash=_dedup_hash(dedup_prefix),
        jsonl_mtime_ns=mtime_ns,
        jsonl_size=size,
    )

    hit, value = _recent_turns_cache.get(key)
    if hit:
        _inc_outcome("recent_turns", "hit")
        _log_event("recent_turns", "hit", f"chat={chat_id[:8]}")
        return value or ""

    value = loader()
    evicted = _recent_turns_cache.put(key, value)
    _inc_outcome("recent_turns", "miss")
    _log_event("recent_turns", "miss", f"chat={chat_id[:8]} size={size}")
    if evicted is not None:
        _inc_outcome("recent_turns", "evict")
        _log_event("recent_turns", "evict", f"chat={evicted.chat_id[:8]}")
    return value


def cached_memory_layer(
    layer: str,
    file_path: Optional[Path],
    *,
    loader: Callable[[], Optional[str]],
) -> Optional[str]:
    """LRU-cached wrapper for a single memory layer load.

    ``layer`` ∈ {"global", "project", "session", "global_slots",
    "project_sections"}. ``file_path=None`` (e.g. global memory when the
    sender open_id isn't known yet) bypasses the cache and falls through
    to the loader — there's no stable key in that case and the loader
    will itself return ``None``.
    """
    if not _config_flag("MEMORY_LEGACY_CACHE_ENABLED", True):
        return loader()

    if file_path is None:
        # No stable key — call the loader directly. We still record the
        # miss so /metrics shows the bypass volume.
        _inc_outcome("memory_layer", "bypass", layer=layer)
        return loader()

    mtime_ns, _size = _stat_file(file_path)
    key = MemoryLayerKey(layer=layer, file_path=str(file_path), mtime_ns=mtime_ns)

    hit, value = _memory_layer_cache.get(key)
    if hit:
        _inc_outcome("memory_layer", "hit", layer=layer)
        _log_event("memory_layer", "hit", f"layer={layer}")
        return value

    value = loader()
    evicted = _memory_layer_cache.put(key, value)
    _inc_outcome("memory_layer", "miss", layer=layer)
    _log_event("memory_layer", "miss", f"layer={layer}")
    if evicted is not None:
        _inc_outcome("memory_layer", "evict", layer=evicted.layer)
        _log_event("memory_layer", "evict", f"layer={evicted.layer}")
    return value


def cached_doc_read(
    chat_id: str,
    ref: Any,
    max_chars: int,
    *,
    loader: Callable[[], Any],
) -> Any:
    """TTL-cached wrapper for ``FeishuDocClient.read``.

    ``ref`` is duck-typed against ``DocRef`` — we only read ``doc_type``
    and ``token`` so the cache module stays decoupled from
    ``lark_client``. Errors raised by ``loader()`` propagate unchanged
    (callers — ``_inject_doc_context`` — already have ``except
    DocPermissionError / DocError`` handlers that must keep firing).
    Only successful results enter the cache.
    """
    if not _config_flag("DOC_INJECT_CACHE_ENABLED", True):
        return loader()

    # Refresh the TTL lazily so an operator flipping
    # ``doc_inject_cache_ttl_sec`` in config.json takes effect without a
    # bridge restart. Use ``set_ttl`` (reviewer round-1 nit #3) so we
    # don't poke the private ``_ttl`` attribute; keeps the singleton
    # stable for tests that reach for the underlying dict.
    new_ttl = _doc_ttl_sec()
    if new_ttl != _doc_cache.ttl_sec:
        _doc_cache.set_ttl(new_ttl)

    doc_type = getattr(ref, "doc_type", "")
    if doc_type and not isinstance(doc_type, str):
        # ``DocType`` is an Enum; coerce to its string name so the key
        # stays hashable and stable across imports.
        doc_type = getattr(doc_type, "name", "") or str(doc_type)
    token = getattr(ref, "token", "") or ""
    key = DocKey(
        chat_id=chat_id,
        doc_type=str(doc_type),
        token=str(token),
        max_chars=int(max_chars),
    )

    entry = _doc_cache.get(key)
    if entry is not None:
        _inc_outcome("doc_inject", "hit")
        _log_event("doc_inject", "hit", f"chat={chat_id[:8]} doc={token[:8]}")
        return entry.payload

    payload = loader()
    # ``fetched_at`` is stamped by TTLCache itself (tuple slot); the
    # DocCachedEntry no longer carries a separate copy (reviewer
    # round-1 nit #2).
    _doc_cache.put(key, DocCachedEntry(payload=payload))
    _inc_outcome("doc_inject", "miss")
    _log_event("doc_inject", "miss", f"chat={chat_id[:8]} doc={token[:8]}")
    return payload


def reset_for_tests() -> None:
    """Test-only: clear every cache + counter. Production must not call.

    The metrics-counter side is owned by ``larkhelm.metrics``; this
    function does NOT touch those because Prometheus counters are by
    design monotonic. Tests that need fresh counters should call
    ``larkhelm.metrics._reset_for_tests`` themselves.
    """
    _recent_turns_cache.clear()
    _memory_layer_cache.clear()
    _doc_cache.clear()
    # Pin TTL back to the design default in case a test mutated it.
    # Reviewer round-1 nit #3: use ``set_ttl`` instead of poking ``_ttl``.
    _doc_cache.set_ttl(_DOC_DEFAULT_TTL)


__all__ = [
    "LRUCache", "TTLCache",
    "RecentTurnsKey", "MemoryLayerKey", "DocKey", "DocCachedEntry",
    "cached_recent_turns", "cached_memory_layer", "cached_doc_read",
    "reset_for_tests",
]
