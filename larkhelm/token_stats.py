"""larkhelm · in-memory token usage statistics and persistent read/write"""
import json
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.concurrency import _jsonl_lock
from larkhelm.log import _debug_log, warn
from larkhelm.secure_io import secure_open

__all__ = [
    "_token_stats", "_token_stats_lock", "_jsonl_lock",
    "resolve_record_chat_id",
    "record_token_usage", "get_token_stats", "get_token_stats_persistent",
    "record_crew_agent_tokens", "get_crew_agent_tokens", "evict_crew_agent_tokens",
    "estimate_cache_savings",
    "get_cache_savings_summary",
    "get_cache_hit_rate_summary",
]

# Per-million cache_read tokens: (input_price - cache_read_price) per model.
# Models not listed return 0.0 (no known cache pricing or not applicable).
_CACHE_SAVINGS_PER_M: dict[str, float] = {
    "claude":   2.70,   # Sonnet 4.x: $3.00/M input − $0.30/M cache_read
    "gemini":   1.125,  # Pro 1.5: $1.25/M input − $0.125/M cache_read
    "deepseek": 0.20,   # V3: $0.27/M − $0.07/M cache_hit (approximate)
    "kimi":     0.0,    # no public cache pricing
}


def estimate_cache_savings(model: str, usage: dict) -> float:
    """Estimate USD saved by prompt cache for one query.

    Uses per-model (input_price - cache_read_price) per-million-tokens rate.
    Returns 0.0 for unknown models or if cache_read == 0. Never raises.
    """
    try:
        rate = _CACHE_SAVINGS_PER_M.get(str(model), 0.0)
        if not rate:
            return 0.0
        cache_read = max(0, int((usage or {}).get("cache_read", 0) or 0))
        if not cache_read:
            return 0.0
        return cache_read * rate / 1_000_000
    except Exception:
        return 0.0


def get_cache_savings_summary() -> dict[str, float]:
    """Return per-model estimated cache savings (USD) accumulated since process start.

    Computed from _cache_totals_by_model["read"] × _CACHE_SAVINGS_PER_M.
    Returns {} for models with no cache reads or no pricing data.
    Thread-safe (acquires _token_stats_lock). Never raises.
    """
    try:
        with _token_stats_lock:
            snapshot = {m: dict(v) for m, v in _cache_totals_by_model.items()}
        result: dict[str, float] = {}
        for model, totals in snapshot.items():
            rate = _CACHE_SAVINGS_PER_M.get(str(model), 0.0)
            if not rate:
                continue
            read_tokens = max(0, int(totals.get("read", 0) or 0))
            if not read_tokens:
                continue
            result[model] = read_tokens * rate / 1_000_000
        return result
    except Exception:
        return {}

def get_cache_hit_rate_summary() -> dict[str, dict]:
    """Return per-model cache read and input token totals accumulated since process start.

    Used by _cmd_stats_cache to compute hit_rate = read / (read + input).
    Returns {} for models with no cache reads. Thread-safe (acquires _token_stats_lock).
    Never raises.
    """
    try:
        with _token_stats_lock:
            snapshot = {m: dict(v) for m, v in _cache_totals_by_model.items()}
        result: dict[str, dict] = {}
        for model, totals in snapshot.items():
            read_tokens = max(0, int(totals.get("read", 0) or 0))
            if not read_tokens:
                continue
            result[model] = {
                "read": read_tokens,
                "input": max(0, int(totals.get("input", 0) or 0)),
            }
        return result
    except Exception:
        return {}


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

# Accumulated cache token totals per model — protected by _token_stats_lock.
# Used to compute the cache_hit_ratio Gauge without re-scanning the LRU.
# Key = model string; value = {"write": int, "read": int}.
_cache_totals_by_model: dict[str, dict[str, int]] = {}


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
        _ct = _cache_totals_by_model.setdefault(model, {"write": 0, "read": 0, "input": 0})
        _ct["write"] += max(0, int(usage.get("cache_create", 0) or 0))
        _ct["read"]  += max(0, int(usage.get("cache_read", 0) or 0))
        _ct["input"] += max(0, int(usage.get("input_tokens", 0) or 0))
        _snap_write, _snap_read = _ct["write"], _ct["read"]

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

    try:
        cache_create_val = max(0, int(usage.get("cache_create", 0) or 0))
        cache_read_val   = max(0, int(usage.get("cache_read", 0) or 0))
        from larkhelm.metrics import inc_cache_write_tokens, inc_cache_read_tokens, set_cache_hit_ratio, observe_cache_hit_rate
        if cache_create_val > 0:
            inc_cache_write_tokens(model, cache_create_val)
        if cache_read_val > 0:
            inc_cache_read_tokens(model, cache_read_val)
            _input_tokens = max(0, int(usage.get("input_tokens", 0) or 0))
            _hit_rate = cache_read_val / (cache_read_val + _input_tokens + 1e-9)
            observe_cache_hit_rate(model, _hit_rate)
            try:
                _alert_threshold = float(getattr(_cfg, "CACHE_HIT_RATE_ALERT_THRESHOLD", 0.5))
                if cache_read_val > 0 and _hit_rate < _alert_threshold:
                    _debug_log(
                        f"[TokenStats] low cache hit rate for {model}: {_hit_rate:.0%}"
                    )
            except Exception:
                pass
        denom = _snap_write + _snap_read
        ratio = _snap_read / denom if denom > 0 else 0.0
        set_cache_hit_ratio(model, ratio)
        if ratio < 0.70 and denom > 0:
            warn(f"[TokenStats] cache_hit_ratio for {model} = {ratio:.2f} < 0.70")
    except Exception as e:
        _debug_log(f"[TokenStats] cache metric update failed (model={model}): {e}")

    # Session guard hook (all backends). Lazy import avoids circular import
    # during boot; guard module itself swallows exceptions, but we wrap
    # defensively so any import-time error can't break record_token_usage.
    try:
        from larkhelm.session_guard import maybe_auto_reset
        maybe_auto_reset(chat_id, model, usage)
    except Exception as e:
        _debug_log(
            f"[TokenStats] maybe_auto_reset failed (model={model}): {e}"
        )

    try:
        savings = estimate_cache_savings(model, usage)
        if savings > 0:
            from larkhelm.metrics import inc_cache_savings
            inc_cache_savings(model, savings)
    except Exception as e:
        _debug_log(f"[TokenStats] inc_cache_savings failed (model={model}): {e}")


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

    def _scan(path: Path) -> None:
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
