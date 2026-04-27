"""larkhelm · in-memory token usage statistics and persistent read/write

Note: writing to all.jsonl uses _log_lock imported from log.py, sharing the same lock as log.py
to prevent interleaved writes from the two modules.
"""
import json
import threading
from collections import OrderedDict
from datetime import datetime

import larkhelm.config as _cfg
from larkhelm.log import _log_lock

__all__ = [
    "_token_stats", "_token_stats_lock", "_jsonl_lock",
    "record_token_usage", "get_token_stats", "get_token_stats_persistent",
    "record_crew_agent_tokens", "get_crew_agent_tokens", "evict_crew_agent_tokens",
]

# In-memory statistics (reset on restart)
# Structure: {chat_id: {model: {input_tokens, output_tokens, cache_read, cache_create, cost_usd, calls}}}
# LRU-capped to prevent unbounded growth when serving many distinct chat_ids over a long uptime.
_TOKEN_STATS_MAX = 5000
_token_stats: OrderedDict[str, dict[str, dict]] = OrderedDict()
_token_stats_lock = threading.Lock()
_jsonl_lock = _log_lock  # share the same lock as log.py to avoid interleaved writes to all.jsonl

# Per crew-agent token stats (in-memory only, keyed by crew_ns = "chat_id__crew_X_agent_Y").
# Each crew run with N agents creates N entries; LRU cap prevents unbounded growth across many runs.
_CREW_AGENT_TOKENS_MAX = 2000
_crew_agent_tokens: OrderedDict[str, dict] = OrderedDict()
_crew_agent_lock = threading.Lock()


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


def evict_crew_agent_tokens(crew_id_prefix: str) -> None:
    """Remove all crew agent token entries whose key starts with crew_id_prefix.
    Called after a crew run completes to free memory eagerly rather than waiting for LRU eviction.
    """
    with _crew_agent_lock:
        stale = [k for k in _crew_agent_tokens if crew_id_prefix in k]
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
    with _jsonl_lock:
        try:
            with (_cfg.LOG_DIR / "all.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def get_token_stats(chat_id: str = None) -> dict:
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


def get_token_stats_persistent(chat_id: str, date_prefix: str = None) -> dict:
    """Read persistent token statistics from all.jsonl (survives restarts).
    date_prefix: e.g. "2026-04" to filter by month, "2026-04-06" by day, None for all.
    Returns {model: {input_tokens, output_tokens, cache_read, cache_create, cost_usd, calls}}
    """
    with _jsonl_lock:
        log_path = _cfg.LOG_DIR / "all.jsonl"
    totals: dict[str, dict] = {}
    try:
        with log_path.open(encoding="utf-8") as f:
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
                # Match normal queries (exact) and crew agent namespaces (prefixed with chat_id + "__")
                rec_chat = r.get("chat_id", "")
                if rec_chat != chat_id and not rec_chat.startswith(chat_id + "__"):
                    continue
                if date_prefix and not r.get("ts", "").startswith(date_prefix):
                    continue
                mdl = r.get("model", "claude")
                t = totals.setdefault(mdl, {
                    "input_tokens": 0, "output_tokens": 0,
                    "cache_read": 0, "cache_create": 0,
                    "cost_usd": 0.0, "calls": 0,
                })
                t["input_tokens"]  += r.get("input_tokens", 0)
                t["output_tokens"] += r.get("output_tokens", 0)
                t["cache_read"]    += r.get("cache_read", 0)
                t["cache_create"]  += r.get("cache_create", 0)
                t["cost_usd"]      += r.get("cost_usd", 0.0)
                t["calls"]         += 1
    except Exception:
        pass
    return totals
