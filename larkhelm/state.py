"""larkhelm · backward-compatibility shim.

This module re-exports all previously public symbols from the five sub-modules
that replaced the monolithic state.py.  Import from the sub-modules directly
for new code; this file exists only for callers that have not been updated.
"""
from larkhelm.chat_state import (
    _state_lock, _chat_state_store,
    _load_global_state, _save_state, _get_chat_state, _set_chat_field,
    _sid_file, _load_sid, _save_sid, _clear_sid,
    _get_cwd, _set_cwd, _get_chat_model, _set_chat_model,
    _register_btw_msg, _is_btw_reply,
    set_pending_doc_write, pop_pending_doc_write,
)
from larkhelm.concurrency import (
    BTW_TIMEOUT,
    _chat_locks, _get_chat_lock,
    _btw_locks, _get_btw_lock,
    _cancel_events, _cancel_events_ts,
    _get_cancel_event, _trigger_cancel, _reset_cancel, _replace_cancel_event,
    _shutting_down, set_shutting_down, is_shutting_down, wait_for_idle,
    _pending_msg, _set_pending, _pop_pending, _update_pending_card_mid,
    _cron_lock,
    get_busy_chat_ids,
)
from larkhelm.dedup import (
    DEDUP_CAP, _is_duplicate,
    _seen_event_ids, _seen_msg_ids, _seen_lock,
)
from larkhelm.log import (
    _log_lock, log_entry, _read_logs, _debug_log,
)
from larkhelm.token_stats import (
    _token_stats, _token_stats_lock, _jsonl_lock,
    record_token_usage, get_token_stats, get_token_stats_persistent,
    record_crew_agent_tokens, get_crew_agent_tokens, evict_crew_agent_tokens,
)

__all__ = [
    "_state_lock", "_chat_state_store",
    "_load_global_state", "_save_state", "_get_chat_state", "_set_chat_field",
    "_sid_file", "_load_sid", "_save_sid", "_clear_sid",
    "_get_cwd", "_set_cwd", "_get_chat_model", "_set_chat_model",
    "_register_btw_msg", "_is_btw_reply",
    "set_pending_doc_write", "pop_pending_doc_write",
    "BTW_TIMEOUT",
    "_chat_locks", "_get_chat_lock", "_btw_locks", "_get_btw_lock",
    "_cancel_events", "_get_cancel_event", "_trigger_cancel", "_reset_cancel",
    "_replace_cancel_event", "_shutting_down", "set_shutting_down",
    "is_shutting_down", "wait_for_idle",
    "_pending_msg", "_set_pending", "_pop_pending", "_update_pending_card_mid", "_cron_lock",
    "get_busy_chat_ids",
    "DEDUP_CAP", "_is_duplicate",
    "_seen_event_ids", "_seen_msg_ids", "_seen_lock",
    "_log_lock", "log_entry", "_read_logs", "_debug_log",
    "_token_stats", "_token_stats_lock", "_jsonl_lock",
    "record_token_usage", "get_token_stats", "get_token_stats_persistent",
    "record_crew_agent_tokens", "get_crew_agent_tokens", "evict_crew_agent_tokens",
    "_cancel_events_ts",
]
