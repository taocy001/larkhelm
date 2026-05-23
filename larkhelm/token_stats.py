"""larkhelm · in-memory token usage statistics and persistent read/write"""
import json
import threading
from collections import OrderedDict
from datetime import datetime

import larkhelm.config as _cfg
from larkhelm.concurrency import _jsonl_lock
from larkhelm.log import _debug_log
from larkhelm.secure_io import secure_open

__all__ = [
    "_token_stats", "_token_stats_lock", "_jsonl_lock",
    "resolve_record_chat_id",
    "record_token_usage", "get_token_stats", "get_token_stats_persistent",
    "record_crew_agent_tokens", "get_crew_agent_tokens", "evict_crew_agent_tokens",
    "summarize_crew_agent_tokens_for_chat",
    "summarize_crew_agent_tokens_by_type",
]

# P5 REQ-08 / design.md §3.2: static agent_id → bucket name table.
# Covers the 6 `/dev` pipeline IDs plus the 5 Phase 5 agent_hub types.
# Unknown agent_ids fall back to `_AGENT_TYPE_FALLBACK` (Chinese label
# matches existing /stats UI). Keep this table aligned with
# crew/_pipeline.py + agent_hub/builtin/.
_AGENT_TYPE_MAP: "dict[str, str]" = {
    # /dev pipeline agent IDs → task_profile name
    "pm":          "planner",
    "architect":   "planner",
    "implementer": "engineer",
    "fixer":       "engineer",
    "qa":          "qa",
    "reviewer":    "reviewer",
    # Phase 5 agent_hub agent_type → itself (identity)
    "chat":  "chat",
    "dev":   "dev",
    "crew":  "crew",
    "plan":  "plan",
    "doc":   "doc",
}
_AGENT_TYPE_FALLBACK = "其它"

# In-memory statistics (reset on restart)
# Structure: {chat_id: {model: {input_tokens, output_tokens, cache_read, cache_create, cost_usd, calls}}}
# LRU-capped to prevent unbounded growth when serving many distinct chat_ids over a long uptime.
_TOKEN_STATS_MAX = 5000
_token_stats: OrderedDict[str, dict[str, dict]] = OrderedDict()
_token_stats_lock = threading.Lock()

# Per crew-agent token stats (in-memory only, keyed by crew_ns = "chat_id__crew_X_agent_Y").
# Each crew run with N agents creates N entries; LRU cap prevents unbounded growth across many runs.
_CREW_AGENT_TOKENS_MAX = 2000
_crew_agent_tokens: OrderedDict[str, dict] = OrderedDict()
_crew_agent_lock = threading.Lock()


def resolve_record_chat_id(chat_id: str, record_under: str | None = None) -> str:
    """Return the chat_id under which token usage should be recorded.

    Handles namespace stripping for crew-agent and memory-cascade namespaces:
      * "abc__crew_X"         → "abc"
      * "abc__memory_session" → "abc"
      * "abc__memory_project" → "abc"
      * "abc__memory_global"  → "abc"

    ``record_under`` takes highest priority (used when a runner is explicitly
    told to record under a different chat_id, e.g. plan-replay).
    """
    if record_under is not None:
        return record_under
    if "__crew_" in chat_id:
        return chat_id.split("__crew_")[0]
    if "__memory_" in chat_id:
        return chat_id.split("__memory_")[0]
    return chat_id


def record_crew_agent_tokens(crew_ns: str, model: str, usage: dict) -> None:
    """Record token usage for a specific crew agent namespace (in-memory only)."""
    with _crew_agent_lock:
        if crew_ns in _crew_agent_tokens:
            _crew_agent_tokens.move_to_end(crew_ns)
            entry = _crew_agent_tokens[crew_ns]
        else:
            entry = {
                "input_tokens": 0, "output_tokens": 0,
                "cache_read": 0, "cache_create": 0,
                "cost_usd": 0.0,
            }
            _crew_agent_tokens[crew_ns] = entry
            if len(_crew_agent_tokens) > _CREW_AGENT_TOKENS_MAX:
                _crew_agent_tokens.popitem(last=False)
        entry["input_tokens"]  += usage.get("input_tokens", 0)
        entry["output_tokens"] += usage.get("output_tokens", 0)
        entry["cache_read"]    += usage.get("cache_read", 0)
        entry["cache_create"]  += usage.get("cache_create", 0)
        entry["cost_usd"]      += usage.get("cost_usd", 0.0)


def get_crew_agent_tokens(crew_ns: str) -> dict:
    """Get token stats for a crew agent namespace."""
    with _crew_agent_lock:
        return dict(_crew_agent_tokens.get(crew_ns, {}))


def summarize_crew_agent_tokens_for_chat(chat_id: str) -> dict:
    """Aggregate the in-memory crew-agent token counters for one chat.

    Round-4 audit P1 (R4-1e): the `_crew_agent_tokens` dict was populated
    by every `/crew` / `/dev` / `/plan` run but had no user-facing entry
    point — CLAUDE.md promised "crew agent token independent tracking"
    but `/stats` rendered nothing about it. This helper sums all entries
    keyed by `chat_id__crew_*` so `/stats` can show "本进程 crew agents
    消耗：N tokens / $X" at the card bottom. Process-local: values reset
    on bridge restart (no JSONL persistence, by design — these are
    rolled into the parent chat's `role=token` records by
    `runner_base._record_tokens`, so we only surface them here as an
    in-process drill-down).

    Returns ``{}`` when no entries match. Otherwise returns
    ``{"input_tokens", "output_tokens", "cache_read", "cache_create",
       "cost_usd", "agents"}`` where ``agents`` is the count of distinct
    crew_ns entries that contributed.
    """
    prefix = f"{chat_id}__crew_"
    with _crew_agent_lock:
        matched = [v for k, v in _crew_agent_tokens.items() if k.startswith(prefix)]
    if not matched:
        return {}
    out = {
        "input_tokens":  0, "output_tokens": 0,
        "cache_read":    0, "cache_create":  0,
        "cost_usd":      0.0,
        "agents":        len(matched),
    }
    for entry in matched:
        out["input_tokens"]  += entry.get("input_tokens", 0)
        out["output_tokens"] += entry.get("output_tokens", 0)
        out["cache_read"]    += entry.get("cache_read", 0)
        out["cache_create"]  += entry.get("cache_create", 0)
        out["cost_usd"]      += entry.get("cost_usd", 0.0)
    return out


def _classify_agent_type(crew_ns: str, chat_id: str) -> str:
    """Return the bucket name for ``crew_ns`` under ``chat_id``.

    Parses agent_id from ``crew_ns = "{chat_id}__crew_{crew_id}_{agent_id}"``
    (design.md §4.1 contract #6) and looks it up in ``_AGENT_TYPE_MAP``.
    Unknown agent_ids and any unparseable prefix collapse to
    ``_AGENT_TYPE_FALLBACK``.
    """
    prefix = f"{chat_id}__crew_"
    if not crew_ns.startswith(prefix):
        return _AGENT_TYPE_FALLBACK
    # Strip the chat_id prefix, leaving "<crew_id>_<agent_id>". crew_id
    # is an 8-hex slug (no underscore), so partition on the first "_"
    # gives us agent_id on the right.
    remainder = crew_ns[len(prefix):]
    _crew_id, sep, agent_id = remainder.partition("_")
    if not sep or not agent_id:
        return _AGENT_TYPE_FALLBACK
    return _AGENT_TYPE_MAP.get(agent_id, _AGENT_TYPE_FALLBACK)


def summarize_crew_agent_tokens_by_type(chat_id: str) -> dict[str, dict]:
    """Aggregate ``_crew_agent_tokens`` for ``chat_id`` bucketed by agent_type.

    Iterates the in-memory ``_crew_agent_tokens`` LRU, keeps entries whose
    key starts with ``f"{chat_id}__crew_"`` (strict double-underscore
    anchor — prevents "chatA" swallowing "chatAB__crew_*" entries), then
    classifies each via :func:`_classify_agent_type` and sums the 5 raw
    counters per bucket.

    Returns ``{}`` when no entry matches. Otherwise
    ``{agent_type: {"agents", "input_tokens", "output_tokens",
                    "cache_read", "cache_create", "cost_usd"}}``.

    Process-local (no JSONL persistence; resets on bridge restart).
    """
    prefix = f"{chat_id}__crew_"
    with _crew_agent_lock:
        matched = [
            (k, dict(v)) for k, v in _crew_agent_tokens.items()
            if k.startswith(prefix)
        ]
    if not matched:
        return {}
    out: dict[str, dict] = {}
    for crew_ns, entry in matched:
        bucket = _classify_agent_type(crew_ns, chat_id)
        agg = out.setdefault(bucket, {
            "agents":        0,
            "input_tokens":  0,
            "output_tokens": 0,
            "cache_read":    0,
            "cache_create":  0,
            "cost_usd":      0.0,
        })
        agg["agents"]        += 1
        agg["input_tokens"]  += entry.get("input_tokens", 0)
        agg["output_tokens"] += entry.get("output_tokens", 0)
        agg["cache_read"]    += entry.get("cache_read", 0)
        agg["cache_create"]  += entry.get("cache_create", 0)
        agg["cost_usd"]      += entry.get("cost_usd", 0.0)
    return out


def evict_crew_agent_tokens(crew_id_prefix: str) -> None:
    """Remove all crew agent token entries whose key starts with crew_id_prefix.
    Called after a crew run completes to free memory eagerly rather than waiting for LRU eviction.

    Round-4 audit P1: previously used substring matching (``crew_id_prefix in k``),
    violating the function's prefix contract. crew_id is currently a UUID so
    collisions were unlikely, but a future move to semantic IDs (e.g. "abc"
    vs "xx-abc-yy") would silently evict unrelated entries. Pinned by
    test_evict_crew_agent_tokens_strict_prefix.
    """
    with _crew_agent_lock:
        stale = [k for k in _crew_agent_tokens if k.startswith(crew_id_prefix)]
        for k in stale:
            _crew_agent_tokens.pop(k, None)


def record_token_usage(chat_id: str, model: str, usage: dict) -> None:
    """Accumulate token usage for one query and persist it to all.jsonl.
    usage keys: input_tokens, output_tokens, cache_read, cache_create, cost_usd
    """
    with _token_stats_lock:
        if chat_id in _token_stats:
            _token_stats.move_to_end(chat_id)
            chat = _token_stats[chat_id]
        else:
            chat = {}
            _token_stats[chat_id] = chat
            if len(_token_stats) > _TOKEN_STATS_MAX:
                _token_stats.popitem(last=False)
        m = chat.setdefault(model, {
            "input_tokens": 0, "output_tokens": 0,
            "cache_read": 0, "cache_create": 0,
            "cost_usd": 0.0, "calls": 0,
        })
        m["input_tokens"]  += usage.get("input_tokens", 0)
        m["output_tokens"] += usage.get("output_tokens", 0)
        m["cache_read"]    += usage.get("cache_read", 0)
        m["cache_create"]  += usage.get("cache_create", 0)
        m["cost_usd"]      += usage.get("cost_usd", 0.0)
        m["calls"]         += 1

    record = {
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "chat_id":       chat_id,
        "role":          "token",
        "model":         model,
        "input_tokens":  usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "cache_read":    usage.get("cache_read", 0),
        "cache_create":  usage.get("cache_create", 0),
        "cost_usd":      usage.get("cost_usd", 0.0),
    }
    # Preserve the ``estimated`` flag (currently only Kimi sets it — the
    # CLI emits no usage envelope so we char-count / 4 in cleanup_extra).
    # Persisting the flag lets future audit / cost-rollup tooling exclude
    # estimated rows when computing precise SDK-derived totals. Don't
    # write the field when it's the default False to keep the JSONL
    # rows backwards-compatible.
    if usage.get("estimated"):
        record["estimated"] = True
    with _jsonl_lock:
        try:
            _cfg.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with secure_open(_cfg.LOG_DIR / "all.jsonl", "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[token_stats] JSONL write failed: {e}", file=__import__("sys").stderr)

    try:
        from larkhelm.metrics import inc_tokens
        inc_tokens(model, usage)
    except Exception as e:
        _debug_log(f"[TokenStats] inc_tokens failed (model={model}): {e}")

    # P0: Claude session auto-reset hook. Lazy import avoids a config →
    # token_stats → claude_session_guard → memory → … cycle during boot;
    # guard module itself swallows exceptions, but we wrap defensively so
    # any import-time error here can't break record_token_usage.
    try:
        from larkhelm.claude_session_guard import maybe_auto_reset_session
        maybe_auto_reset_session(chat_id, model, usage)
    except Exception as e:
        _debug_log(
            f"[TokenStats] maybe_auto_reset_session failed (model={model}): {e}"
        )


def get_token_stats(chat_id: str | None = None) -> dict:
    """Return token statistics for a specific chat, or aggregated across all chats.
    Note: in-memory stats are capped at the last _TOKEN_STATS_MAX chat_ids; use
    get_token_stats_persistent() for complete historical data.
    """
    with _token_stats_lock:
        if chat_id:
            return {k: dict(v) for k, v in _token_stats.get(chat_id, {}).items()}
        total: dict[str, dict] = {}
        for chat in _token_stats.values():
            for model, m in chat.items():
                t = total.setdefault(model, {
                    "input_tokens": 0, "output_tokens": 0,
                    "cache_read": 0, "cache_create": 0,
                    "cost_usd": 0.0, "calls": 0,
                })
                for k, v in m.items():
                    t[k] += v
        return total


def get_token_stats_persistent(chat_id: str, date_prefix: str | None = None) -> dict:
    """Read persistent token statistics from all.jsonl (survives restarts).
    date_prefix: e.g. "2026-04" to filter by month, "2026-04-06" by day, None for all.
    Returns {model: {input_tokens, output_tokens, cache_read, cache_create, cost_usd, calls}}

    Reads BOTH ``all.jsonl`` and ``all.jsonl.1`` (the single rotation
    backup ``log.py`` keeps when the live file exceeds 100 MiB). The
    earlier version read only the live file, so once a rotation happened
    every token record in the backup permanently disappeared from
    ``/stats`` — the "累计（全部）" window could legitimately become
    SMALLER than the "本月" window. Independent stats audit (P0).
    """
    with _jsonl_lock:
        live = _cfg.LOG_DIR / "all.jsonl"
        backup = _cfg.LOG_DIR / "all.jsonl.1"
    totals: dict[str, dict] = {}

    def _scan(path) -> None:
        try:
            with path.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("role") != "token":
                        continue
                    # Strict per-chat match. The previous fuzzy
                    # ``startswith(chat_id + "__")`` branch was dead code
                    # at write-time (``runner_base._record_tokens`` strips
                    # the ``__crew_*`` suffix before writing the bare
                    # parent chat_id) AND a cross-chat-leak risk: any
                    # future record with ``chat_id="abc__leak"`` would
                    # have been silently aggregated into chat ``abc``.
                    if r.get("chat_id", "") != chat_id:
                        continue
                    if date_prefix and not r.get("ts", "").startswith(date_prefix):
                        continue
                    mdl = r.get("model", "claude")
                    t = totals.setdefault(mdl, {
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_read": 0, "cache_create": 0,
                        "cost_usd": 0.0, "calls": 0,
                    })
                    # Defensive ``max(0, …)``: a misbehaving backend
                    # returning a negative count would otherwise silently
                    # pull the running total below truth.
                    t["input_tokens"]  += max(0, int(r.get("input_tokens", 0) or 0))
                    t["output_tokens"] += max(0, int(r.get("output_tokens", 0) or 0))
                    t["cache_read"]    += max(0, int(r.get("cache_read", 0) or 0))
                    t["cache_create"]  += max(0, int(r.get("cache_create", 0) or 0))
                    t["cost_usd"]      += max(0.0, float(r.get("cost_usd", 0.0) or 0.0))
                    t["calls"]         += 1
        except FileNotFoundError:
            return
        except Exception as e:
            _debug_log(f"[token_stats] get_token_stats_persistent scan {path.name} failed: {e}")

    # Read backup first then live so the natural file order (older →
    # newer) is preserved; the aggregation itself is order-independent
    # but reading older first keeps log-tail debugging intuitive.
    _scan(backup)
    _scan(live)
    return totals
