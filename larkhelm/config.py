"""
larkhelm · runtime configuration

_init_runtime() is called before main() starts and initialises all global config variables.
Other modules import the config items they need from this module.

Caller convention: import larkhelm.config as _cfg → _cfg.CLAUDE_CMD etc.
Module-level globals are kept unchanged; _RuntimeConfig serves as a typed internal
storage container for mypy.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import threading
from pathlib import Path

from larkhelm.secure_io import secure_atomic_write


@dataclasses.dataclass
class _RuntimeConfig:
    """Typed runtime configuration container (internal use); external access is via module-level globals."""
    CONFIG_PATH:  Path
    DATA_DIR:     Path
    SESSION_DIR:  Path
    LOG_DIR:      Path
    STATE_FILE:   Path
    DEBUG_LOG:    Path
    config:       dict
    APP_ID:       str
    APP_SECRET:   str
    CLAUDE_CMD:   str
    GEMINI_CMD:   str
    KIMI_CMD:     str
    DEEPSEEK_API_KEY:  str
    DEEPSEEK_BASE_URL: str
    DEEPSEEK_MODEL:    str
    DEFAULT_MODEL:    str
    SKIP_PERMISSIONS: bool
    RESPONSE_TIMEOUT: int
    HARD_TIMEOUT:     int
    SHELL_TIMEOUT:    int
    MAX_CARD_LEN:     int
    ALLOWED_CHATS:    set
    GEMINI_IDLE_TTL:  int
    MAX_AI_PROCS_CONFIG: "int | None"
    MAX_AI_PROCS:        int
    DEFAULT_CWD:      str
    CRON_TIMEZONE:    str
    DOC_AUTO_INJECT:        bool
    DOC_INJECT_MAX_CHARS:   int
    DOC_INJECT_MAX_DOCS:    int
    DOC_READ_MAX_CHARS:     int
    DEFAULT_DRIVE_FOLDER:   str
    DOC_WRITE_CONFIRM:      bool
    DOC_WRITE_BACKEND:      str
    DEFAULT_WIKI_SPACE_ID:     str
    DEFAULT_WIKI_PARENT_TOKEN: str
    DEFAULT_OWNER_OPEN_ID:     str
    # OAuth user_access_token (allows creating Feishu docs as the user directly,
    # avoiding the "transfer ownership" notification chain — see oauth_user.py)
    USER_TOKEN_PATH:    Path
    OAUTH_REDIRECT_PORT: int
    LOGGED_IN_OPEN_ID:   str
    PERM_HOOK_SCRIPT: str
    PERM_SOCKET_PATH: str
    SOURCE_DIR:       Path
    # ── Voice (M3.2) ──────────────────────────────────────
    VOICE_ENABLED:           bool
    VOICE_ENGINE:            str    # one of: faster_whisper / dashscope
    VOICE_API_KEY:           str    # only used by dashscope engine
    VOICE_MODEL_SIZE:        str
    VOICE_COMPUTE_TYPE:      str
    VOICE_MAX_DURATION_MS:   int
    VOICE_DEFAULT_LANG:      str
    VOICE_MERGE_WINDOW_SEC:  int
    VOICE_MAX_MERGE:         int
    VOICE_KEEP_AUDIO:        bool
    MEMORY_LIMIT_MB:         int
    CREW_BREAKPOINT_TIMEOUT_SEC: int
    # P2 toggles
    METRICS_TEXT_LEGACY:                bool = False
    ANTHROPIC_EXTENDED_CACHE_ENABLED:   bool = True
    MEMORY_EXTRACT_BUFFER_WINDOW_SEC:   int = 0
    MEMORY_SESSION_SMART_COMPRESS:      bool = False
    MEMORY_GLOBAL_PROFILE_SLOT_ENABLED: bool = False
    MEMORY_PROJECT_SECTION_ENABLED:     bool = False
    # Context-injection cache toggles (REQ-01..04)
    RECENT_TURNS_CACHE_ENABLED:        bool = True
    MEMORY_LEGACY_CACHE_ENABLED:       bool = True
    DOC_INJECT_CACHE_ENABLED:          bool = True
    DOC_INJECT_CACHE_TTL_SEC:          int = 600
    CLI_SKIP_RECENT_TURNS_WHEN_SID:    bool = True
    # Workspace-hint / stats-breakdown toggles (P3/P5)
    WORKSPACE_HINT_KEYWORD_GATE:        bool = False
    STATS_AGENT_TYPE_BREAKDOWN_ENABLED: bool = True
    # File processing (M4.1)
    FILE_ENABLED:          bool = True
    MAX_FILE_SIZE_BYTES:   int = 10 * 1024 * 1024
    FILE_TEXT_EXTENSIONS:  "frozenset[str]" = frozenset()
    FILE_PDF_ENABLED:      bool = True
    FILE_PDF_LIB:          str = "PyPDF2"

# ── Runtime config (assigned by _init_runtime()) ────────────────────────
CONFIG_PATH: Path
DATA_DIR:    Path
SESSION_DIR: Path
LOG_DIR:     Path
STATE_FILE:  Path
DEBUG_LOG:   Path

config:           dict
APP_ID:           str
APP_SECRET:       str
CLAUDE_CMD:       str
GEMINI_CMD:       str
DEEPSEEK_API_KEY:  str
DEEPSEEK_BASE_URL: str
DEEPSEEK_MODEL:    str
DEFAULT_MODEL:    str
SKIP_PERMISSIONS: bool
RESPONSE_TIMEOUT: int
HARD_TIMEOUT:     int
SHELL_TIMEOUT:    int
MAX_CARD_LEN:     int
ALLOWED_CHATS:    set
GEMINI_IDLE_TTL:  int
# Concurrent AI subprocess cap. ``MAX_AI_PROCS_CONFIG`` reflects the *raw*
# operator preference: a positive int means "honour this", ``None`` means
# "auto-detect from cgroup / RAM". ``MAX_AI_PROCS`` (set after
# ``runner_base._init_ai_sem()`` resolves the value) holds the *effective*
# cap used by the running bridge. New code should read the effective value
# via ``runner_base.get_max_ai_procs()`` — these globals are kept for
# back-compat and observability (e.g. /status output).
MAX_AI_PROCS_CONFIG: "int | None"
MAX_AI_PROCS:        int
DEFAULT_CWD:      str
CRON_TIMEZONE:    str

# Document feature configuration
DOC_AUTO_INJECT:        bool
DOC_INJECT_MAX_CHARS:   int
DOC_INJECT_MAX_DOCS:    int
DOC_READ_MAX_CHARS:     int
DEFAULT_DRIVE_FOLDER:   str
DOC_WRITE_CONFIRM:      bool
DOC_WRITE_BACKEND:      str   # "auto" | "feishu" | "local"

# Wiki / knowledge-base configuration
DEFAULT_WIKI_SPACE_ID:     str
DEFAULT_WIKI_PARENT_TOKEN: str
DEFAULT_OWNER_OPEN_ID:     str

# OAuth user_access_token (see oauth_user.py)
# When LOGGED_IN_OPEN_ID is set and matches the requested owner, doc-create
# requests use a user_access_token and skip the transfer_owner notification.
USER_TOKEN_PATH:    "Path"
OAUTH_REDIRECT_PORT: int   # 0 = OS-assigned, used only for `larkhelm user-login` loopback
LOGGED_IN_OPEN_ID:   str   # open_id of the user who completed `larkhelm user-login`

# Permission approval socket path (derived from PID; fixed once _init_runtime sets it)
PERM_HOOK_SCRIPT: str
PERM_SOCKET_PATH: str
SOURCE_DIR: "Path"  # source repo root directory (auto-detected for editable installs)
BACKEND_REGISTRY: "object"  # BackendRegistry singleton (set by _init_runtime)

# Voice configuration (M3.2)
VOICE_ENABLED:           bool
VOICE_ENGINE:            str   # one of: faster_whisper (default, local) / dashscope (cloud, opt-in)
VOICE_API_KEY:           str   # resolved from ${DASHSCOPE_API_KEY} env (only used when VOICE_ENGINE=dashscope)
VOICE_MODEL_SIZE:        str   # faster-whisper only: tiny / base / small / medium / large-v3
VOICE_COMPUTE_TYPE:      str   # faster-whisper only: int8 / float16
VOICE_MAX_DURATION_MS:   int
VOICE_DEFAULT_LANG:      str   # one of: zh / en / auto
VOICE_MERGE_WINDOW_SEC:  int   # 0 = disable merge
VOICE_MAX_MERGE:         int
VOICE_KEEP_AUDIO:        bool
MEMORY_LIMIT_MB:         int   # RSS hard limit in MB; auto-detected on first run
CREW_BREAKPOINT_TIMEOUT_SEC: int   # Phase C: max wait for human confirmation in /dev (default 1800s)

# ── P1-3 / P1-5 / P1-6 globals ──────────────────────────────────────────────
# All default to "feature off" so byte-compat with master holds when the
# operator hasn't opted in. Reads come through ``getattr(_cfg, NAME, default)``
# so an unmigrated process (e.g. a worker spawned before _init_runtime) sees
# the safe fallback rather than an AttributeError.
HEALTH_ENDPOINT_PORT: int = 0
HEALTH_BIND_ADDR: str = "127.0.0.1"
SESSION_LAYER_BUDGETS: dict = {
    "work_context": 1200,
    "decisions":     800,
    "history":       600,
}
MEMORY_CASCADE_MIDFLIGHT_CANCEL: bool = True
QUERY_SESSION_V2_ENABLED: bool = False
MEMORY_SESSION_LAYER_SMART: bool = True

# ── P2 globals (REQ-01 / 05 / 06 / 07) ─────────────────────────────────────
# All P2 toggles default to "feature off" so byte-compatibility with P1 holds
# when the operator hasn't opted in. Reads through ``getattr(_cfg, NAME, default)``
# so an unmigrated process (worker spawned before _init_runtime) sees the safe
# fallback rather than an AttributeError.
METRICS_TEXT_LEGACY: bool = False
ANTHROPIC_EXTENDED_CACHE_ENABLED: bool = True
MEMORY_EXTRACT_BUFFER_WINDOW_SEC: int = 0
MEMORY_SESSION_SMART_COMPRESS: bool = False
MEMORY_GLOBAL_PROFILE_SLOT_ENABLED: bool = False
MEMORY_PROJECT_SECTION_ENABLED: bool = False

# ── P3 globals (REQ-02 / 03 / 04 / 05 / 06 / 07 / 08 / 09 / 10) ────────────
# All eleven default to "feature off / status quo" so byte-compat with P2 holds.
# Operators flip them in config.json; setdefault preserves their override.
QUERY_SESSION_V2_TRAFFIC: float = 0.0
INTENT_EMBEDDING_THRESHOLD: float = 0.30
# Phase D intent-router quality knobs (May 2026). L1 keyword score must
# exceed the threshold to short-circuit; below it the router defers to
# L2 (embedding or LLM JSON). Setting intent_l1_enabled=false bisects
# the entire keyword tier without restart.
INTENT_L1_ENABLED: bool = True
INTENT_L1_PROMOTION_THRESHOLD: float = 0.70
INTENT_MICROLEARN_ENABLED: bool = False
INTENT_MICROLEARN_MIN_CONFIDENCE: float = 0.65
# Phase D follow-up (May 2026): extended user-behaviour signals piped into
# intent_feedback.jsonl beyond the force_chat button. Master switch +
# tunables; setting EXTENDED_SIGNALS=False reverts to the legacy
# force_chat-only behaviour without restart.
INTENT_FEEDBACK_EXTENDED_SIGNALS: bool = True
INTENT_FEEDBACK_CANCEL_WINDOW_SEC: float = 60.0
INTENT_FEEDBACK_SIGNAL_TEXT_MAX: int = 800
INTENT_FEEDBACK_L1_GRAY_BAND: float = 0.10
LLM_ROUTER_CIRCUIT_FAILURES: int = 5
LLM_ROUTER_CIRCUIT_COOLDOWN_SEC: float = 30.0
CASCADE_BACKOFF_MAX_ATTEMPTS: int = 3
PLAN_RETRY_STRATEGY: str = "off"             # "now" | "manual" | "off"
PLUGIN_REPORT_CARD_ENABLED: bool = False
FAILURE_REPORT_CARD_ENABLED: bool = False
ADMIN_CHAT_ID: str = ""
MEMORY_GC_INTERVAL_HOURS: float = 6.0
CREW_CHECKPOINT_TTL_DAYS: float = 7.0
DEV_STAGE_TIMEOUTS: "dict[str, int]" = {}

# ── Context-injection cache flags (REQ-01..04) ────────────────────────────
# All default to safe values: REQ-01..03 caches default ON because the
# byte-compat guarantee already holds (loaders run identically; the cache
# is purely a memoization layer). The CLI sid skip in REQ-04 also defaults
# ON because the new behaviour mirrors what API backends already do via
# load_history. Operators flip any of these off in config.json to bisect a
# regression without redeploying.
RECENT_TURNS_CACHE_ENABLED: bool = True
MEMORY_LEGACY_CACHE_ENABLED: bool = True
DOC_INJECT_CACHE_ENABLED: bool = True
DOC_INJECT_CACHE_TTL_SEC: int = 600
CLI_SKIP_RECENT_TURNS_WHEN_SID: bool = True

# ── P3 workspace-hint / P5 stats-breakdown knobs ──────────────────────────
# WORKSPACE_HINT_KEYWORD_GATE (default False): when True, the workspace
# hint segment is suppressed unless the user message matches the keyword
# regex (see _message._WORKSPACE_KEYWORD_RE). REQ-02.
# STATS_AGENT_TYPE_BREAKDOWN_ENABLED (default True): when True, /stats
# renders crew agents bucketed by agent_type; when False, falls back to
# the P2 single-line summary for cards approaching MAX_CARD_LEN. REQ-09.
WORKSPACE_HINT_KEYWORD_GATE: bool = False
STATS_AGENT_TYPE_BREAKDOWN_ENABLED: bool = True

# ── P0/P1/P2 cache-bleed knobs (.crew_workspace/design.md §3.3) ───────────
# P0: Claude session auto-reset
CLAUDE_SESSION_AUTO_RESET_ENABLED: bool = True
CLAUDE_SESSION_RESET_CACHE_TOKENS: int = 5_000_000
CLAUDE_SESSION_RESET_TURNS: int = 50
# P1: ChatAgent cheap routing
CHAT_AGENT_CHEAP_ROUTING_ENABLED: bool = True
# P2: Sticky crew context tuning
RECENT_CREW_STICKY_TTL_SEC: int = 1800
RECENT_CREW_STICKY_MAX_INJECTIONS: int = 5

# ── File handling configuration ───────────────────────────────────────────
FILE_ENABLED: bool = True
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024   # 10 MB
FILE_TEXT_EXTENSIONS: "frozenset[str]" = frozenset({
    "txt", "md", "py", "js", "json", "yaml", "yml", "csv", "log",
    "sh", "go", "rs", "java", "c", "cpp", "h", "ts", "tsx", "jsx",
    "css", "html", "xml", "sql", "dockerfile", "toml", "ini", "cfg", "conf",
})
FILE_PDF_ENABLED: bool = True
FILE_PDF_LIB: str = "PyPDF2"
# Single source of truth for accepted voice languages — referenced by both
# config validation (_init_runtime) and the /voice command handler.
_VOICE_LANG_WHITELIST: "frozenset[str]" = frozenset({"zh", "en", "auto"})

# ── Card UX parameters (module-level constants) ────────────────────────────────────
TOOL_HISTORY_CAP   = 20      # max number of completed tool pairs kept in history
CARD_PUSH_INTERVAL = 5.0     # heartbeat push interval (seconds)
CURSOR_INTERVAL    = 0.3     # local cursor-blink interval (seconds)
STALL_THRESHOLD    = 30.0    # threshold for detecting a stalled tool (seconds)
CURSOR_FRAMES      = ["▌", "▍", "▎", "▏"]


def _auto_discover_cli() -> list[dict]:
    """Probe known CLIs in PATH: claude, gemini, kimi, kimi-code.

    Returns list of BackendSpec dicts (without 'role' — inferred by BackendRegistry.load()).
    Only includes CLIs found via shutil.which().

    ``capability_scores`` keys MUST match the keys used by
    ``TASK_PROFILES`` in ``crew/_backend_resolver.py``
    (``reasoning`` / ``coding`` / ``tools`` / ``long_context`` / ``chat``).
    Without these, ``rank_for_task`` falls back to a tag-intersection
    score that ties every backend at 0 or 1, then breaks the tie on
    ``spec.id`` alphabetical order — which means every profile would
    resolve to ``claude`` regardless of capability fit.
    """
    known = [
        {"id": "claude",    "provider": "claude_cli", "display_name": "Claude",    "tags": ["vision", "tools"], "command": "claude",
         "capability_scores": {"reasoning": 0.95, "coding": 0.95, "tools": 0.95, "long_context": 0.90, "chat": 0.85}},
        {"id": "gemini",    "provider": "gemini_cli", "display_name": "Gemini",    "tags": ["tools"],           "command": "gemini",
         "capability_scores": {"reasoning": 0.90, "coding": 0.85, "tools": 0.85, "long_context": 0.95, "chat": 0.80}},
        {"id": "kimi",      "provider": "kimi_cli",   "display_name": "Kimi",      "tags": ["vision", "tools"], "command": "kimi",
         "capability_scores": {"reasoning": 0.80, "coding": 0.85, "tools": 0.85, "long_context": 0.85, "chat": 0.95}},
        {"id": "kimi-code", "provider": "kimi_cli",   "display_name": "Kimi-Code", "tags": ["tools"],           "command": "kimi-code",
         "capability_scores": {"reasoning": 0.75, "coding": 0.95, "tools": 0.90, "long_context": 0.80, "chat": 0.65}},
    ]
    return [spec for spec in known if shutil.which(spec["command"])]


def _auto_discover_http() -> list[dict]:
    """Probe HTTP backends whose API keys are present in config or env.

    Currently DeepSeek only. Mirrors ``_auto_discover_cli`` shape so callers can
    treat both the same way.

    Tags intentionally omit ``tools`` — the HTTP runner does not implement
    function calling, so ``rank_for_task`` must filter DeepSeek out of any
    profile with ``require_tools=True`` (planner stays unaffected; engineer /
    qa stay tool-bound; chat / reviewer remain eligible).
    """
    out: list[dict] = []

    raw_key = config.get("deepseek_api_key", "") or ""
    if raw_key.startswith("${") and raw_key.endswith("}"):
        ds_key = os.environ.get(raw_key[2:-1], "")
    else:
        ds_key = raw_key or os.environ.get("DEEPSEEK_API_KEY", "")
    if ds_key:
        out.append({
            "id":           "deepseek",
            "provider":     "deepseek_api",
            "display_name": "DeepSeek",
            "role":         "worker",
            "tags":         ["cheap", "fast"],
            "model":        config.get("deepseek_model", "deepseek-chat") or "deepseek-chat",
            "api_key":      ds_key,
            "base_url":     config.get("deepseek_base_url", "https://api.deepseek.com") or "https://api.deepseek.com",
            "capability_scores": {"reasoning": 0.75, "coding": 0.80, "long_context": 0.65, "chat": 0.85},
            "latency_tier":      "fast",
            "cost_per_1k_input":  0.00014,
            "cost_per_1k_output": 0.00220,
        })
    return out


def _migrate_legacy_backends(config: dict) -> list[dict]:
    """Three-layer merge: config explicit > auto-discover (CLI + HTTP) > empty.

    Layer 1: config 'backends' (highest priority).
    Layer 2: auto-discover CLIs in PATH and HTTP keys in env supplement IDs not in Layer 1.
    Layer 3: if all empty, return [].
    """
    explicit: list[dict] = config.get("probe_models", config.get("backends", []))
    auto_discovered = _auto_discover_cli() + _auto_discover_http()

    explicit_ids = {b["id"] for b in explicit}
    supplement = [b for b in auto_discovered if b["id"] not in explicit_ids]
    return explicit + supplement


_recover_thread_started = False


def _start_recover_thread() -> None:
    """Start the unified backend health-tick daemon thread.

    On every ``BACKEND_HEALTH_TICK_SEC`` tick (default 60s), walks each enabled
    backend and decides whether to fire a fresh probe based on its current
    health and last-activity timestamps:

    * unhealthy → re-probe if ``last_probed_at`` was ≥ ``BACKEND_RECOVER_INTERVAL_SEC`` ago
    * healthy   → re-probe if no real traffic for ≥ ``BACKEND_STALE_INTERVAL_SEC`` (idle re-validation)

    This replaces the legacy 300s ``recover_check()`` loop with a single
    decision point that respects ``last_used_at`` set by real traffic — a
    backend actively in use will never trigger an idle probe.

    Guard against duplicate starts when _init_runtime() is called more than
    once (e.g. during tests or if both bridge and MCP server share a process).
    """
    global _recover_thread_started
    if _recover_thread_started:
        return
    _recover_thread_started = True

    # Captured at thread start; used by _health_tick to skip specs whose
    # startup probe hasn't completed yet (avoids first-tick double-probe race).
    _startup_mono = [0.0]  # list-as-mutable-cell so closure can write
    _STARTUP_GRACE_SEC = 120  # how long after start to defer "never probed" specs
    # Persistent pool — created once at thread start to avoid spawn/teardown every tick.
    from concurrent.futures import ThreadPoolExecutor as _TPE, as_completed as _as_completed
    _probe_pool = _TPE(max_workers=4, thread_name_prefix="health-tick")

    def _health_tick():
        import time as _time
        from larkhelm.model_probe import probe_spec
        from larkhelm.log import _debug_log as _dlog
        # Use monotonic for staleness decisions — wall-clock jumps (NTP, manual
        # `date -s`) would otherwise poison the cadence (backwards jump → inf age
        # → probe storm; forwards jump → tiny age → stale specs go un-probed).
        now_mono = _time.monotonic()
        in_grace = (now_mono - _startup_mono[0]) < _STARTUP_GRACE_SEC

        # Phase 1 — decide which specs need probing (no I/O)
        to_probe: list[tuple[object, str]] = []
        for spec in BACKEND_REGISTRY.all_enabled():
            try:
                last_activity = max(spec.last_used_mono, spec.last_probed_mono, 0.0)
                # First-tick storm guard: if a spec has NEVER been touched
                # (probe still in flight from startup, or startup probe failed
                # silently and never ran), defer for STARTUP_GRACE_SEC. After
                # grace expires, a still-untouched spec gets probed normally.
                if last_activity == 0.0 and in_grace:
                    continue
                age = (now_mono - last_activity) if last_activity > 0 else float("inf")
                if not spec.healthy:
                    if age >= BACKEND_RECOVER_INTERVAL_SEC:
                        to_probe.append((spec, "retry"))
                else:
                    if age >= BACKEND_STALE_INTERVAL_SEC:
                        to_probe.append((spec, "idle"))
            except Exception as e:
                _dlog(f"[BackendRegistry] tick decision error for {getattr(spec, 'id', '?')}: {e}")

        if not to_probe:
            return

        # Phase 2 — run probes concurrently, mirroring run_probes_async pattern.
        # Same MAX_WORKERS=4 so a 5-backend setup with 12s timeouts completes
        # in ~24s worst case instead of the 60s sequential ceiling that would
        # equal the tick interval and cause the loop to fall behind.
        # _probe_pool is created once at thread start; reused here to avoid
        # 4-thread spawn/teardown on every 60 s tick (ARCH-H3).
        future_to_target = {_probe_pool.submit(probe_spec, s): (s, reason) for s, reason in to_probe}
        for fut in _as_completed(future_to_target):
            spec, reason = future_to_target[fut]
            try:
                ok, err = fut.result()
            except Exception as e:
                ok, err = False, str(e)[:200]
            try:
                BACKEND_REGISTRY.set_probe_result(spec.id, ok, err)
            except Exception as e:
                _dlog(f"[BackendRegistry] tick set_probe_result failed for {spec.id}: {e}")
            # Three-state icon: ✓ confirmed reachable, ✗ confirmed failed,
            # ? indeterminate (ok=None — e.g. subprocess timeout, no
            # healthy mutation, real-traffic record_call_failure decides).
            # Round-1 review #1: original ``"✓" if ok else "✗"`` flagged
            # None as failure visually, misleading operators reading
            # ``/status`` or the journal.
            if ok is True:
                icon = "✓"
            elif ok is False:
                icon = "✗"
            else:
                icon = "?"
            # Recompute reason post-hoc: a "retry" that succeeded is a "recover"
            if reason == "retry" and ok is True:
                reason = "recover"
            suffix = f" ({err})" if err else ""
            _dlog(f"[BackendRegistry] tick {reason} {icon} {spec.id}{suffix}")

    def _recover_loop():
        import time as _time
        # Mark startup time the moment the loop spins up, so the first tick
        # ~60s later can compute the grace window correctly.
        _startup_mono[0] = _time.monotonic()
        while True:
            try:
                _time.sleep(max(BACKEND_HEALTH_TICK_SEC, 1))
            except Exception:
                _time.sleep(60)
            try:
                _health_tick()
            except Exception as e:
                # Daemon must never die — its absence creates silent staleness
                from larkhelm.log import lazy_debug_log
                lazy_debug_log(f"[BackendRegistry] health_tick loop error: {e}")

    t = threading.Thread(target=_recover_loop, daemon=True, name="backend-health-tick")
    t.start()


def _init_paths(config_path: "str | None", data_dir: "str | None") -> None:
    """P1-2: Phase 1 of bootstrap — resolve config / data paths and mkdir.

    Sets the module globals CONFIG_PATH, DATA_DIR, SESSION_DIR, LOG_DIR,
    STATE_FILE, DEBUG_LOG. Safe to call multiple times; later calls
    overwrite earlier values. Has NO side effect on config.json content.
    """
    global CONFIG_PATH, DATA_DIR, SESSION_DIR, LOG_DIR, STATE_FILE, DEBUG_LOG

    _sys_cfg  = Path("/etc/larkhelm/config.json")
    _sys_data = Path("/var/lib/larkhelm")
    _xdg_cfg  = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    _xdg_data = Path(os.environ.get("XDG_DATA_HOME",   Path.home() / ".local" / "share"))

    if config_path:
        CONFIG_PATH = Path(config_path)
    elif "LARKHELM_CONFIG" in os.environ:
        CONFIG_PATH = Path(os.environ["LARKHELM_CONFIG"])
    elif _sys_cfg.exists():
        CONFIG_PATH = _sys_cfg
    else:
        CONFIG_PATH = _xdg_cfg / "larkhelm" / "config.json"

    if data_dir:
        DATA_DIR = Path(data_dir)
    elif "LARKHELM_DATA_DIR" in os.environ:
        DATA_DIR = Path(os.environ["LARKHELM_DATA_DIR"])
    elif _sys_data.exists():
        DATA_DIR = _sys_data
    else:
        DATA_DIR = _xdg_data / "larkhelm"

    SESSION_DIR = DATA_DIR / "sessions"
    LOG_DIR     = DATA_DIR / "logs"
    STATE_FILE  = DATA_DIR / "state.json"
    DEBUG_LOG   = Path(os.environ.get("LARKHELM_LOG",
                       str(Path("/var/log/larkhelm/larkhelm.log")
                           if Path("/var/log/larkhelm").exists()
                           else DATA_DIR / "larkhelm.log")))

    for _d in (SESSION_DIR, LOG_DIR):
        _d.mkdir(parents=True, exist_ok=True)


def _init_app_config() -> None:
    """P1-2: Phase 2 of bootstrap — load CONFIG_PATH and set all app globals.

    Pre-conditions: ``_init_paths`` has populated CONFIG_PATH / DATA_DIR.
    Side effects: loads config.json, validates fields, populates every
    module-level scalar global used by handlers / runners.
    """
    global config, APP_ID, APP_SECRET, CLAUDE_CMD, GEMINI_CMD, KIMI_CMD
    global DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEFAULT_MODEL
    global SKIP_PERMISSIONS, RESPONSE_TIMEOUT, HARD_TIMEOUT, SHELL_TIMEOUT, MAX_CARD_LEN
    global ALLOWED_CHATS, GEMINI_IDLE_TTL, DEFAULT_CWD, CRON_TIMEZONE
    global MAX_AI_PROCS_CONFIG, MAX_AI_PROCS
    global PERM_HOOK_SCRIPT, PERM_SOCKET_PATH
    global DOC_AUTO_INJECT, DOC_INJECT_MAX_CHARS, DOC_INJECT_MAX_DOCS
    global DOC_READ_MAX_CHARS, DEFAULT_DRIVE_FOLDER, DOC_WRITE_CONFIRM, DOC_WRITE_BACKEND
    global DEFAULT_WIKI_SPACE_ID, DEFAULT_WIKI_PARENT_TOKEN, DEFAULT_OWNER_OPEN_ID
    global VOICE_ENABLED, VOICE_ENGINE, VOICE_API_KEY
    global VOICE_MODEL_SIZE, VOICE_COMPUTE_TYPE, VOICE_MAX_DURATION_MS
    global VOICE_DEFAULT_LANG, VOICE_MERGE_WINDOW_SEC, VOICE_MAX_MERGE, VOICE_KEEP_AUDIO
    global MEMORY_LIMIT_MB
    global HEALTH_ENDPOINT_PORT, HEALTH_BIND_ADDR
    global SESSION_LAYER_BUDGETS, MEMORY_CASCADE_MIDFLIGHT_CANCEL
    global QUERY_SESSION_V2_ENABLED, MEMORY_SESSION_LAYER_SMART
    global METRICS_TEXT_LEGACY, ANTHROPIC_EXTENDED_CACHE_ENABLED
    global MEMORY_EXTRACT_BUFFER_WINDOW_SEC
    global MEMORY_SESSION_SMART_COMPRESS
    global MEMORY_GLOBAL_PROFILE_SLOT_ENABLED, MEMORY_PROJECT_SECTION_ENABLED
    global QUERY_SESSION_V2_TRAFFIC, INTENT_EMBEDDING_THRESHOLD
    global LLM_ROUTER_CIRCUIT_FAILURES, LLM_ROUTER_CIRCUIT_COOLDOWN_SEC
    global CASCADE_BACKOFF_MAX_ATTEMPTS, PLAN_RETRY_STRATEGY
    global PLUGIN_REPORT_CARD_ENABLED, FAILURE_REPORT_CARD_ENABLED, ADMIN_CHAT_ID
    global MEMORY_GC_INTERVAL_HOURS, CREW_CHECKPOINT_TTL_DAYS, DEV_STAGE_TIMEOUTS
    global RECENT_TURNS_CACHE_ENABLED, MEMORY_LEGACY_CACHE_ENABLED
    global DOC_INJECT_CACHE_ENABLED, DOC_INJECT_CACHE_TTL_SEC
    global CLI_SKIP_RECENT_TURNS_WHEN_SID
    global FILE_ENABLED, MAX_FILE_SIZE_BYTES, FILE_TEXT_EXTENSIONS
    global FILE_PDF_ENABLED, FILE_PDF_LIB

    try:
        # SEC-H1: warn when config file is world-readable (contains APP_SECRET).
        try:
            _mode = CONFIG_PATH.stat().st_mode & 0o777
            if _mode & 0o077:  # group or world bits set
                _perm_msg = (
                    f"[Config] config file {CONFIG_PATH} has permissions {oct(_mode)}, "
                    "recommend chmod 600 to protect APP_SECRET"
                )
                print(f"⚠️  安全警告: {_perm_msg}")
                try:
                    from larkhelm.log import warn as _log_warn
                    _log_warn(_perm_msg)
                except Exception:
                    pass
        except OSError:
            pass
        config = json.loads(CONFIG_PATH.read_text())
        APP_ID     = config["APP_ID"]
        APP_SECRET = config["APP_SECRET"]
        if not APP_ID or not APP_SECRET:
            print("❌ 配置加载失败: APP_ID/APP_SECRET 不能为空字符串")
            sys.exit(1)
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    CLAUDE_CMD       = config.get("claude_command", "claude") or "claude"
    GEMINI_CMD       = config.get("gemini_command", "gemini") or "gemini"
    KIMI_CMD         = config.get("kimi_command",   "kimi")   or "kimi"

    # DeepSeek HTTP backend (no CLI). API key falls back to env var so secrets
    # don't have to live in config.json on shared hosts.
    _raw_ds_key = config.get("deepseek_api_key", "") or ""
    if _raw_ds_key.startswith("${") and _raw_ds_key.endswith("}"):
        DEEPSEEK_API_KEY = os.environ.get(_raw_ds_key[2:-1], "")
    else:
        DEEPSEEK_API_KEY = _raw_ds_key or os.environ.get("DEEPSEEK_API_KEY", "")
    DEEPSEEK_BASE_URL = config.get("deepseek_base_url", "https://api.deepseek.com") or "https://api.deepseek.com"
    DEEPSEEK_MODEL    = config.get("deepseek_model",    "deepseek-chat")            or "deepseek-chat"

    DEFAULT_MODEL    = config.get("default_model", "claude")
    if DEFAULT_MODEL not in ("claude", "gemini", "kimi", "deepseek"):
        print(
            f"⚠️  default_model 值 '{DEFAULT_MODEL}' 无效（允许: claude/gemini/kimi/deepseek），已回退为 'claude'",
            file=sys.stderr,
        )
        DEFAULT_MODEL = "claude"
    if DEFAULT_MODEL == "deepseek" and not DEEPSEEK_API_KEY:
        print(
            "⚠️  default_model='deepseek' 但 deepseek_api_key 未配置，已回退为 'claude'",
            file=sys.stderr,
        )
        DEFAULT_MODEL = "claude"
    SKIP_PERMISSIONS = bool(config.get("skip_permissions", True))
    RESPONSE_TIMEOUT = int(config.get("response_timeout", 300))   # soft timeout: release lock but don't kill process
    HARD_TIMEOUT     = int(config.get("hard_timeout", 21600))      # hard timeout: force kill (default 6 hours)
    # P3-a (W17): /run shell timeout — was hardcoded 30s; floor at 1s.
    _raw_shell_to    = config.get("shell_timeout_sec", 30)
    try:
        SHELL_TIMEOUT = max(1, int(_raw_shell_to))
    except (TypeError, ValueError):
        SHELL_TIMEOUT = 30
    MAX_CARD_LEN     = int(config.get("max_card_len", 3000))
    ALLOWED_CHATS    = set(config.get("allowed_chat_ids", []))
    GEMINI_IDLE_TTL  = int(config.get("gemini_idle_ttl", 1800))

    # max_ai_procs: positive int → honour; "auto" / None / 0 / non-int → probe.
    # The probe (runner_base._compute_max_procs) walks cgroup MemoryMax →
    # physical RAM and computes a memory-budget-safe value so 4-GB hosts
    # under tight cgroup quotas don't OOM (the bug round-3 of OOM defense
    # is fixing). See runner_base for the formula and HARD_CEILING.
    _raw_max_procs = config.get("max_ai_procs")
    if _raw_max_procs is None or (isinstance(_raw_max_procs, str) and _raw_max_procs.lower() == "auto"):
        MAX_AI_PROCS_CONFIG = None
    elif (isinstance(_raw_max_procs, int)
          and not isinstance(_raw_max_procs, bool)
          and _raw_max_procs > 0):
        # bool ⊆ int in Python — must reject ``True``/``False`` here
        # explicitly, otherwise ``max_ai_procs: true`` becomes a silent
        # cap=1 and the user gets no warning that their config is wrong.
        MAX_AI_PROCS_CONFIG = _raw_max_procs
    else:
        print(
            f"⚠️  max_ai_procs={_raw_max_procs!r} 无效（允许：正整数 / \"auto\" / 缺省），已回退为 auto-detect",
            file=sys.stderr,
        )
        MAX_AI_PROCS_CONFIG = None
    # Trigger sem rebuild now that MAX_AI_PROCS_CONFIG is set; runner_base
    # logs the effective value via larkhelm.log.info. The mirror global
    # MAX_AI_PROCS is populated by _init_ai_sem in runner_base; we read it
    # back here so /status and the _RuntimeConfig snapshot see the same value.
    try:
        from larkhelm.runner_base import _init_ai_sem as _rb_init_ai_sem, get_max_ai_procs as _rb_get_max
        _rb_init_ai_sem()
        MAX_AI_PROCS = _rb_get_max()
    except Exception as _e:
        # Bootstrap-safe fallback: if runner_base import fails for any reason
        # (test mocks, missing deps), don't crash the whole bridge — fall back
        # to the legacy default of 2 so concurrency is at least bounded.
        print(f"[config] _init_ai_sem failed (defaulting MAX_AI_PROCS=2): {_e}", file=sys.stderr)
        MAX_AI_PROCS = 2
    DEFAULT_CWD      = config.get("default_cwd", str(Path.home() / "code"))
    CRON_TIMEZONE    = config.get("timezone", "Asia/Shanghai")
    if HARD_TIMEOUT <= RESPONSE_TIMEOUT:
        print(
            f"⚠️  hard_timeout ({HARD_TIMEOUT}s) ≤ response_timeout ({RESPONSE_TIMEOUT}s); "
            f"adjusting hard_timeout to response_timeout + 60",
            file=sys.stderr,
        )
        HARD_TIMEOUT = RESPONSE_TIMEOUT + 60

    # ── Backend health-tracking knobs ──────────────────────────────────────
    # All times in seconds. The unified health-tick loop in
    # _start_recover_thread reads these to decide when to fire a probe.
    global BACKEND_HEALTH_TICK_SEC, BACKEND_RECOVER_INTERVAL_SEC
    global BACKEND_STALE_INTERVAL_SEC, BACKEND_TRANSIENT_WINDOW_SEC
    global BACKEND_TRANSIENT_THRESHOLD, BACKEND_PROBE_API_REAL_CALL
    # Sane floors to prevent footguns (e.g. setting stale_interval=1 → infinite
    # paid API calls/sec). Floor < default by ≥1 order of magnitude so config
    # tuning still works for legitimate testing.
    def _read_clamped(key: str, default: int, floor: int, kind=int):
        raw = kind(config.get(key, default))
        if raw < floor:
            print(
                f"⚠️  config.{key}={raw} below floor {floor}; clamping to {floor} "
                "(too low would cause excessive probe traffic / instability)",
                file=sys.stderr,
            )
            return floor
        return raw

    BACKEND_HEALTH_TICK_SEC      = _read_clamped("backend_health_tick_sec",      60,    10)        # tick cadence
    BACKEND_RECOVER_INTERVAL_SEC = _read_clamped("backend_recover_interval_sec", 300,   30)        # unhealthy → re-probe at most every N s
    BACKEND_STALE_INTERVAL_SEC   = _read_clamped("backend_stale_interval_sec",   86400, 600)       # healthy + idle this long → idle re-probe (default 24h, floor 10min)
    BACKEND_TRANSIENT_WINDOW_SEC = float(_read_clamped("backend_transient_window_sec", 600, 10, kind=float))  # sliding window
    BACKEND_TRANSIENT_THRESHOLD  = _read_clamped("backend_transient_threshold",  3,     1)        # TRANSIENT failures within window → unhealthy
    BACKEND_PROBE_API_REAL_CALL  = bool(config.get("backend_probe_api_real_call", True))           # API probes hit the network (1-token call) vs key-presence only

    # ── Voice config (M3.2) ────────────────────────────────────────────────
    VOICE_ENABLED      = bool(config.get("voice_enabled", False))
    VOICE_ENGINE       = str(config.get("voice_engine",       "faster_whisper") or "faster_whisper")
    VOICE_MODEL_SIZE   = str(config.get("voice_model_size",   "small") or "small")
    VOICE_COMPUTE_TYPE = str(config.get("voice_compute_type", "int8")  or "int8")
    VOICE_DEFAULT_LANG = str(config.get("voice_default_lang", "zh")    or "zh")
    VOICE_KEEP_AUDIO   = bool(config.get("voice_keep_audio", False))

    # voice_api_key: ``${ENV_VAR}`` placeholder resolution mirrors DEEPSEEK_API_KEY.
    # Only relevant when VOICE_ENGINE=="dashscope"; for faster-whisper this stays empty.
    _raw_voice_key = config.get("voice_api_key", "") or ""
    if _raw_voice_key.startswith("${") and _raw_voice_key.endswith("}"):
        VOICE_API_KEY = os.environ.get(_raw_voice_key[2:-1], "")
    else:
        VOICE_API_KEY = _raw_voice_key or os.environ.get("DASHSCOPE_API_KEY", "")

    _VOICE_ENGINE_WHITELIST = {"faster_whisper", "dashscope"}
    if VOICE_ENGINE not in _VOICE_ENGINE_WHITELIST:
        print(
            f"⚠️  voice_engine '{VOICE_ENGINE}' 无效"
            f"（允许: faster_whisper/dashscope），已回退为 'faster_whisper'",
            file=sys.stderr,
        )
        VOICE_ENGINE = "faster_whisper"

    # If user picked dashscope but didn't set an API key, warn and disable voice
    # rather than crashing later mid-transcribe. Mirrors DeepSeek's "warn + degrade"
    # pattern (line 365-370 above).
    if VOICE_ENABLED and VOICE_ENGINE == "dashscope" and not VOICE_API_KEY:
        print(
            "⚠️  voice_engine='dashscope' 但 voice_api_key / $DASHSCOPE_API_KEY 未设置，"
            "已临时关闭 voice_enabled。请配置 systemd drop-in 注入 DASHSCOPE_API_KEY。",
            file=sys.stderr,
        )
        VOICE_ENABLED = False

    _VOICE_MODEL_WHITELIST = {"tiny", "base", "small", "medium", "large-v3"}
    if VOICE_MODEL_SIZE not in _VOICE_MODEL_WHITELIST:
        print(
            f"⚠️  voice_model_size '{VOICE_MODEL_SIZE}' 无效"
            f"（允许: tiny/base/small/medium/large-v3），已回退为 'small'",
            file=sys.stderr,
        )
        VOICE_MODEL_SIZE = "small"

    if VOICE_DEFAULT_LANG not in _VOICE_LANG_WHITELIST:
        print(
            f"⚠️  voice_default_lang '{VOICE_DEFAULT_LANG}' 无效"
            f"（允许: zh/en/auto），已回退为 'zh'",
            file=sys.stderr,
        )
        VOICE_DEFAULT_LANG = "zh"

    VOICE_MAX_DURATION_MS  = _read_clamped("voice_max_duration_ms",  180000, 1000)
    VOICE_MERGE_WINDOW_SEC = _read_clamped("voice_merge_window_sec", 0,      0)
    VOICE_MAX_MERGE        = _read_clamped("voice_max_merge",        5,      1)

    # ── Memory watchdog limit ─────────────────────────────────────────────────
    # Priority: config.json["memory_limit_mb"] > auto-detected value.
    # On first run (key absent) the auto-detected value is persisted to config.json
    # so the user can inspect and tune it; subsequent starts honour that persisted value.
    from larkhelm.memory_watchdog import detect_memory_limit_mb as _detect_mem
    if "memory_limit_mb" in config:
        MEMORY_LIMIT_MB = max(128, int(config["memory_limit_mb"]))
    else:
        MEMORY_LIMIT_MB = _detect_mem()
        # Persist auto-detected value (first-run write-back)
        try:
            config["memory_limit_mb"] = MEMORY_LIMIT_MB
            secure_atomic_write(CONFIG_PATH, json.dumps(config, ensure_ascii=False, indent=2))
            print(f"[config] memory_limit_mb 自动探测并写入: {MEMORY_LIMIT_MB} MB", file=sys.stderr)
        except Exception as _e:
            print(f"[config] memory_limit_mb 写回失败（不影响运行）: {_e}", file=sys.stderr)

    DOC_AUTO_INJECT      = config.get("doc_auto_inject",      True)
    DOC_INJECT_MAX_CHARS = int(config.get("doc_inject_max_chars", 2000))
    DOC_INJECT_MAX_DOCS  = int(config.get("doc_inject_max_docs",  2))
    DOC_READ_MAX_CHARS   = int(config.get("doc_read_max_chars",   6000))
    DEFAULT_DRIVE_FOLDER = config.get("default_drive_folder", "")
    DOC_WRITE_CONFIRM    = config.get("doc_write_confirm",    True)
    DOC_WRITE_BACKEND    = config.get("doc_write_backend",   "auto")
    if DOC_WRITE_BACKEND not in ("auto", "feishu", "local"):
        print(
            f"⚠️  doc_write_backend 值 '{DOC_WRITE_BACKEND}' 无效（允许: auto/feishu/local），已回退为 'auto'",
            file=sys.stderr,
        )
        DOC_WRITE_BACKEND = "auto"

    DEFAULT_WIKI_SPACE_ID     = config.get("default_wiki_space_id",     "")
    DEFAULT_WIKI_PARENT_TOKEN = config.get("default_wiki_parent_token", "")
    DEFAULT_OWNER_OPEN_ID     = config.get("default_owner_open_id",     "")

    # Phase 5: intent router / agent_hub flags (all optional). P1-4 flipped
    # the default to a 10% gray rollout (intent_router_enabled=True,
    # intent_router_traffic=0.1); set traffic to 0.0 or enabled to False to
    # disable. Explicit user-provided values in config.json are preserved
    # via setdefault (legacy `false` deployments stay unaffected).
    config.setdefault("intent_router_enabled", True)
    config.setdefault("intent_router_traffic", 0.1)
    # Default flipped from "llm" → "embedding" (Phase D-C, May 2026).
    # Embedding L2 is faster, deterministic, and cheaper than the cheap
    # LLM JSON path. When the ONNX runtime / model isn't available the
    # embedding backend returns None and the router auto-falls-back to
    # the LLM path — no operator action needed for the legacy code path.
    config.setdefault("intent_layer2_strategy", "embedding")
    # L1 keyword-tier overrides (Phase D-A, May 2026). intent_l1_enabled
    # =false skips the keyword classifier entirely; threshold controls
    # how high a keyword score must be to short-circuit to L1
    # (otherwise abstain → fall through to L2).
    config.setdefault("intent_l1_enabled", True)
    config.setdefault("intent_l1_promotion_threshold", 0.70)
    # Phase D-D: opt-in micro-learn classifier trained from
    # intent_feedback.jsonl. Off by default; flip on once a checkpoint
    # exists at ``data_dir/intent_microlearn.pkl``. See
    # ``larkhelm/agent_hub/intent_microlearn.py`` for the inference API
    # and ``scripts/train_intent_classifier.py`` for training.
    config.setdefault("intent_microlearn_enabled", False)
    config.setdefault("intent_microlearn_min_confidence", 0.65)
    # Phase D follow-up: extended-signal collection for intent_feedback.jsonl
    # (see ``larkhelm/agent_hub/intent_feedback.py``). Defaults to ON so the
    # L1 keyword tuner has real-world data to train against; flip
    # ``intent_feedback_extended_signals=false`` for instant rollback.
    config.setdefault("intent_feedback_extended_signals", True)
    config.setdefault("intent_feedback_cancel_window_sec", 60.0)
    config.setdefault("intent_feedback_signal_text_max", 800)
    config.setdefault("intent_feedback_l1_gray_band", 0.10)
    config.setdefault("agent_plugins", [])
    config.setdefault("agent_acl", {})
    config.setdefault("intent_feedback_path", "")
    config.setdefault("intent_audit_path", "")

    # Phase D: on-demand memory retriever knobs (default off — flag-gated
    # roll-out, same model as intent_router_*). Algorithm: see
    # ``larkhelm/_gating.py`` and ``larkhelm/memory_retriever.py``.
    config.setdefault("memory_retriever_enabled", False)
    config.setdefault("memory_retriever_traffic", 0.0)
    # Phase D / Phase 2 — was "keyword"; now "auto" follows POLICY_TABLE,
    # which already encodes the three "hybrid" entries (dev/crew/plan).
    # Operators wanting the Phase 1 behaviour explicitly can still set this
    # to "keyword".
    config.setdefault("memory_retriever_mode", "auto")
    config.setdefault("memory_retriever_top_k_default", 6)
    config.setdefault("memory_retriever_alpha_recency", 0.3)
    config.setdefault("memory_retriever_alpha_importance", 0.3)
    config.setdefault("memory_retriever_alpha_relevance", 0.4)
    config.setdefault("memory_retriever_debug_card", False)
    config.setdefault("memory_retriever_audit_path", "")

    # Phase D / Phase 2 — hybrid recall + stale lifecycle knobs (REQ-48).
    # All default to "off / neutral" so the bridge runs as Phase 1 unless an
    # operator explicitly opens the gate. See design.md §6.3 for the table.
    config.setdefault("embedding_backend", "none")          # "local" | "http" | "none"
    config.setdefault("embedding_http_endpoint", "")
    config.setdefault("embedding_model_path", "~/.larkhelm/models/bge-small-zh-v1.5.onnx")
    config.setdefault("embedding_dim", 512)
    config.setdefault("embedding_http_timeout_sec", 5.0)
    config.setdefault("embedding_traffic", 0.0)             # Stage B gradual rollout, orthogonal to memory_retriever_traffic
    config.setdefault("embedding_enabled", False)           # Stage B master switch
    # MEM-C2: stale window must not exceed audit retention (30d) — audit records
    # older than retain_days are deleted, so slices can't be validated as "hit"
    # beyond that horizon and would be wrongly marked stale.
    config.setdefault("memory_stale_window_days", 30)       # how far back to look for "never hit"
    config.setdefault("memory_stale_decay", 0.5)            # stale relevance multiplier
    config.setdefault("memory_audit_rotate_max_mb", 32)     # per-file rollover threshold
    config.setdefault("memory_audit_retain_days", 30)       # unlink rotated files older than this

    # Phase D · Phase 3 — LLM memory router (Stage C). Decorator over the
    # underlying keyword/hybrid retriever; cheap LLM picks the most
    # relevant slices from the top-N candidate pool. Gated by all four
    # flags + scope: agent_type ∈ {crew, dev} AND complexity == "high".
    # Defaults conservative — feature off, 3 calls/chat/min, 5 min cache.
    config.setdefault("memory_llm_router_enabled", False)             # Stage C master switch
    config.setdefault("memory_llm_router_traffic", 0.0)               # Stage C gradual rollout, orthogonal to A/B
    config.setdefault("memory_llm_router_max_per_chat_per_min", 3)    # per-chat rate ceiling
    config.setdefault("memory_llm_router_cache_ttl_sec", 300)         # (query, candidate_set) verdict TTL

    # Phase B: memory token-optimization flags (S49–S53 + S3 GC).
    # All default to "true" so the new code paths run by default; toggle
    # individually to bisect token-regression reports without redeploying.
    config.setdefault("memory_lazy_global", True)
    config.setdefault("memory_project_conditional", True)
    config.setdefault("memory_session_layered", True)
    config.setdefault("memory_recent_turns_dedup", True)
    config.setdefault("memory_cascade_shortcircuit", True)
    config.setdefault("memory_cascade_max_concurrent", 4)
    config.setdefault("memory_session_gc_enabled", True)
    config.setdefault("memory_session_gc_max_age_days", 7)

    # Phase C: max seconds to wait for the user to click 继续/取消 at a
    # crew breakpoint. Default 1800s (30 min); see crew/_runner.py
    # _wait_for_breakpoint and _failure_card.emit_breakpoint_timeout.
    global CREW_BREAKPOINT_TIMEOUT_SEC
    try:
        CREW_BREAKPOINT_TIMEOUT_SEC = max(60, int(config.get("crew_breakpoint_timeout_sec", 1800)))
    except (TypeError, ValueError):
        CREW_BREAKPOINT_TIMEOUT_SEC = 1800

    # ── OAuth user_access_token (see oauth_user.py) ──────────────────────
    # The token file is loaded lazily by oauth_user.get_user_token(); this
    # block only reads the cached ``open_id`` to populate LOGGED_IN_OPEN_ID
    # so callers (lark_client.create_doc) can branch without I/O.
    global USER_TOKEN_PATH, OAUTH_REDIRECT_PORT, LOGGED_IN_OPEN_ID
    USER_TOKEN_PATH     = DATA_DIR / "user_token.json"
    OAUTH_REDIRECT_PORT = int(config.get("oauth_redirect_port", 0))
    LOGGED_IN_OPEN_ID   = ""
    try:
        if USER_TOKEN_PATH.exists():
            _td = json.loads(USER_TOKEN_PATH.read_text())
            LOGGED_IN_OPEN_ID = (_td.get("open_id") or "").strip()
    except Exception as _e:
        print(f"[config] user_token load failed (ignored): {_e}", file=sys.stderr)

    PERM_HOOK_SCRIPT  = str(Path(__file__).parent / "perm_hook.py")
    PERM_SOCKET_PATH  = str(DATA_DIR / "perm.sock")

    global SOURCE_DIR
    # For editable installs __file__ = <repo>/larkhelm/config.py; two levels up is the repo root
    _candidate = Path(__file__).parent.parent
    if (_candidate / ".git").exists():
        SOURCE_DIR = _candidate          # git repo root (editable install)
    else:
        SOURCE_DIR = Path(__file__).parent  # fall back to package directory for non-editable installs

    # ── P1-3 / P1-5 / P1-6 / P1-8 new keys ─────────────────────────────────
    # All default to "feature off" so byte-compat with master holds when the
    # operator hasn't opted in. setdefault here keeps overridden values
    # untouched; reads later use _cfg.HEALTH_* etc. as well as config[...].
    config.setdefault("health_endpoint_port", 0)
    config.setdefault("health_bind_addr", "127.0.0.1")
    config.setdefault("query_session_v2_enabled", False)
    config.setdefault("memory_session_layer_smart", True)
    config.setdefault("memory_session_layer_budgets", {
        "work_context": 1200, "decisions": 800, "history": 600,
    })
    config.setdefault("memory_cascade_midflight_cancel", True)

    try:
        HEALTH_ENDPOINT_PORT = int(config.get("health_endpoint_port", 0) or 0)
    except (TypeError, ValueError):
        HEALTH_ENDPOINT_PORT = 0
    HEALTH_BIND_ADDR = str(config.get("health_bind_addr", "127.0.0.1") or "127.0.0.1")
    QUERY_SESSION_V2_ENABLED = bool(config.get("query_session_v2_enabled", False))
    MEMORY_SESSION_LAYER_SMART = bool(config.get("memory_session_layer_smart", True))
    MEMORY_CASCADE_MIDFLIGHT_CANCEL = bool(config.get("memory_cascade_midflight_cancel", True))

    # ── P2 new keys (REQ-01 / 05 / 06 / 07) ────────────────────────────────
    # All five default to "feature off" so the bridge stays byte-compatible
    # with P1 unless an operator explicitly opens the gate. ``setdefault``
    # preserves any operator override already in config.json.
    config.setdefault("metrics_text_legacy", False)
    config.setdefault("anthropic_extended_cache_enabled", True)
    config.setdefault("memory_extract_buffer_window_sec", 0)
    config.setdefault("memory_session_smart_compress", False)
    config.setdefault("memory_global_profile_slot_enabled", False)
    config.setdefault("memory_project_section_enabled", False)

    METRICS_TEXT_LEGACY = bool(config.get("metrics_text_legacy", False))
    ANTHROPIC_EXTENDED_CACHE_ENABLED = bool(
        config.get("anthropic_extended_cache_enabled", True)
    )
    try:
        MEMORY_EXTRACT_BUFFER_WINDOW_SEC = max(
            0, int(config.get("memory_extract_buffer_window_sec", 0) or 0),
        )
    except (TypeError, ValueError):
        MEMORY_EXTRACT_BUFFER_WINDOW_SEC = 0
    MEMORY_SESSION_SMART_COMPRESS = bool(
        config.get("memory_session_smart_compress", False)
    )
    MEMORY_GLOBAL_PROFILE_SLOT_ENABLED = bool(
        config.get("memory_global_profile_slot_enabled", False)
    )
    MEMORY_PROJECT_SECTION_ENABLED = bool(
        config.get("memory_project_section_enabled", False)
    )

    _budgets = config.get("memory_session_layer_budgets") or {}
    try:
        SESSION_LAYER_BUDGETS = {
            "work_context": int(_budgets.get("work_context", 1200) or 1200),
            "decisions":    int(_budgets.get("decisions",     800) or 800),
            "history":      int(_budgets.get("history",       600) or 600),
        }
    except (TypeError, ValueError):
        SESSION_LAYER_BUDGETS = {"work_context": 1200, "decisions": 800, "history": 600}

    # ── P3 new keys (REQ-02 / 03 / 04 / 05 / 06 / 07 / 08 / 09 / 10) ──────
    # All eleven flags default to "feature off / status quo" so prod operators
    # have to opt in. setdefault preserves any operator override.
    config.setdefault("query_session_v2_traffic", 0.0)
    config.setdefault("intent_embedding_top_k_threshold", 0.30)
    config.setdefault("llm_router_circuit_failures", 5)
    config.setdefault("llm_router_circuit_cooldown_sec", 30.0)
    config.setdefault("cascade_backoff_max_attempts", 3)
    config.setdefault("plan_retry_strategy", "off")
    config.setdefault("plugin_report_card_enabled", False)
    config.setdefault("failure_report_card_enabled", False)
    config.setdefault("admin_chat_id", "")
    config.setdefault("memory_gc_interval_hours", 6.0)
    config.setdefault("crew_checkpoint_ttl_days", 7.0)
    config.setdefault("dev_stage_timeouts", {})

    try:
        QUERY_SESSION_V2_TRAFFIC = max(
            0.0, min(1.0, float(config.get("query_session_v2_traffic", 0.0) or 0.0)),
        )
    except (TypeError, ValueError):
        QUERY_SESSION_V2_TRAFFIC = 0.0

    try:
        INTENT_EMBEDDING_THRESHOLD = max(
            0.0, min(1.0, float(config.get("intent_embedding_top_k_threshold", 0.30) or 0.30)),
        )
    except (TypeError, ValueError):
        INTENT_EMBEDDING_THRESHOLD = 0.30

    global INTENT_L1_ENABLED, INTENT_L1_PROMOTION_THRESHOLD
    global INTENT_MICROLEARN_ENABLED, INTENT_MICROLEARN_MIN_CONFIDENCE
    INTENT_L1_ENABLED = bool(config.get("intent_l1_enabled", True))
    try:
        INTENT_L1_PROMOTION_THRESHOLD = max(
            0.05, min(1.0, float(config.get("intent_l1_promotion_threshold", 0.70) or 0.70)),
        )
    except (TypeError, ValueError):
        INTENT_L1_PROMOTION_THRESHOLD = 0.70
    INTENT_MICROLEARN_ENABLED = bool(config.get("intent_microlearn_enabled", False))
    try:
        INTENT_MICROLEARN_MIN_CONFIDENCE = max(
            0.0, min(1.0, float(config.get("intent_microlearn_min_confidence", 0.65) or 0.65)),
        )
    except (TypeError, ValueError):
        INTENT_MICROLEARN_MIN_CONFIDENCE = 0.65

    global INTENT_FEEDBACK_EXTENDED_SIGNALS, INTENT_FEEDBACK_CANCEL_WINDOW_SEC
    global INTENT_FEEDBACK_SIGNAL_TEXT_MAX, INTENT_FEEDBACK_L1_GRAY_BAND
    INTENT_FEEDBACK_EXTENDED_SIGNALS = bool(
        config.get("intent_feedback_extended_signals", True)
    )
    try:
        INTENT_FEEDBACK_CANCEL_WINDOW_SEC = max(
            0.0, float(config.get("intent_feedback_cancel_window_sec", 60.0) or 60.0),
        )
    except (TypeError, ValueError):
        INTENT_FEEDBACK_CANCEL_WINDOW_SEC = 60.0
    try:
        INTENT_FEEDBACK_SIGNAL_TEXT_MAX = max(
            0, int(config.get("intent_feedback_signal_text_max", 800) or 800),
        )
    except (TypeError, ValueError):
        INTENT_FEEDBACK_SIGNAL_TEXT_MAX = 800
    try:
        INTENT_FEEDBACK_L1_GRAY_BAND = max(
            0.0, min(0.5, float(config.get("intent_feedback_l1_gray_band", 0.10) or 0.10)),
        )
    except (TypeError, ValueError):
        INTENT_FEEDBACK_L1_GRAY_BAND = 0.10

    try:
        LLM_ROUTER_CIRCUIT_FAILURES = max(
            1, int(config.get("llm_router_circuit_failures", 5) or 5),
        )
    except (TypeError, ValueError):
        LLM_ROUTER_CIRCUIT_FAILURES = 5

    try:
        LLM_ROUTER_CIRCUIT_COOLDOWN_SEC = max(
            1.0, float(config.get("llm_router_circuit_cooldown_sec", 30.0) or 30.0),
        )
    except (TypeError, ValueError):
        LLM_ROUTER_CIRCUIT_COOLDOWN_SEC = 30.0

    try:
        CASCADE_BACKOFF_MAX_ATTEMPTS = max(
            1, int(config.get("cascade_backoff_max_attempts", 3) or 3),
        )
    except (TypeError, ValueError):
        CASCADE_BACKOFF_MAX_ATTEMPTS = 3

    _strategy = str(config.get("plan_retry_strategy", "off") or "off").lower()
    if _strategy not in ("now", "manual", "off"):
        _strategy = "off"
    PLAN_RETRY_STRATEGY = _strategy

    PLUGIN_REPORT_CARD_ENABLED = bool(config.get("plugin_report_card_enabled", False))
    FAILURE_REPORT_CARD_ENABLED = bool(config.get("failure_report_card_enabled", False))
    ADMIN_CHAT_ID = str(config.get("admin_chat_id", "") or "")

    try:
        MEMORY_GC_INTERVAL_HOURS = max(
            0.0, float(config.get("memory_gc_interval_hours", 6.0) or 6.0),
        )
    except (TypeError, ValueError):
        MEMORY_GC_INTERVAL_HOURS = 6.0

    try:
        CREW_CHECKPOINT_TTL_DAYS = max(
            0.0, float(config.get("crew_checkpoint_ttl_days", 7.0) or 7.0),
        )
    except (TypeError, ValueError):
        CREW_CHECKPOINT_TTL_DAYS = 7.0

    # Context-injection cache flags (REQ-01..04). Defaults preserve PR-prior
    # behaviour: caches on (loaders byte-identical, just memoized) and the
    # CLI sid skip on (mirrors what API backends already do).
    config.setdefault("recent_turns_cache_enabled", True)
    config.setdefault("memory_legacy_cache_enabled", True)
    config.setdefault("doc_inject_cache_enabled", True)
    config.setdefault("doc_inject_cache_ttl_sec", 600)
    config.setdefault("cli_skip_recent_turns_when_sid", True)
    # P3 REQ-02 / P5 REQ-09. Defaults preserve P2 byte-compat for the gate
    # (false = inject as before) and the new "by type" rendering is opt-out
    # (true) so operators only flip false when card overflow triggers.
    config.setdefault("workspace_hint_keyword_gate", False)
    config.setdefault("stats_agent_type_breakdown_enabled", True)

    RECENT_TURNS_CACHE_ENABLED = bool(
        config.get("recent_turns_cache_enabled", True)
    )
    MEMORY_LEGACY_CACHE_ENABLED = bool(
        config.get("memory_legacy_cache_enabled", True)
    )
    DOC_INJECT_CACHE_ENABLED = bool(
        config.get("doc_inject_cache_enabled", True)
    )
    try:
        DOC_INJECT_CACHE_TTL_SEC = max(
            1, int(config.get("doc_inject_cache_ttl_sec", 600) or 600),
        )
    except (TypeError, ValueError):
        DOC_INJECT_CACHE_TTL_SEC = 600
    CLI_SKIP_RECENT_TURNS_WHEN_SID = bool(
        config.get("cli_skip_recent_turns_when_sid", True)
    )

    global WORKSPACE_HINT_KEYWORD_GATE, STATS_AGENT_TYPE_BREAKDOWN_ENABLED
    WORKSPACE_HINT_KEYWORD_GATE = bool(
        config.get("workspace_hint_keyword_gate", False)
    )
    STATS_AGENT_TYPE_BREAKDOWN_ENABLED = bool(
        config.get("stats_agent_type_breakdown_enabled", True)
    )

    # ── P0/P1/P2 cache-bleed knobs (design.md §3.3) ───────────────────────
    # All three groups default to "feature on" — the new behaviours are the
    # explicit P0/P1/P2 design intent. Operators flip individual flags in
    # config.json to bisect a regression without redeploying.
    config.setdefault("claude_session_auto_reset_enabled", True)
    config.setdefault("claude_session_reset_cache_tokens", 5_000_000)
    config.setdefault("claude_session_reset_turns", 50)
    config.setdefault("chat_agent_cheap_routing_enabled", True)
    config.setdefault("recent_crew_sticky_ttl_sec", 1800)
    config.setdefault("recent_crew_sticky_max_injections", 5)

    global CLAUDE_SESSION_AUTO_RESET_ENABLED, CLAUDE_SESSION_RESET_CACHE_TOKENS
    global CLAUDE_SESSION_RESET_TURNS, CHAT_AGENT_CHEAP_ROUTING_ENABLED
    global RECENT_CREW_STICKY_TTL_SEC, RECENT_CREW_STICKY_MAX_INJECTIONS

    CLAUDE_SESSION_AUTO_RESET_ENABLED = bool(
        config.get("claude_session_auto_reset_enabled", True)
    )
    try:
        CLAUDE_SESSION_RESET_CACHE_TOKENS = max(
            1, int(config.get("claude_session_reset_cache_tokens", 5_000_000)
                   or 5_000_000),
        )
    except (TypeError, ValueError):
        CLAUDE_SESSION_RESET_CACHE_TOKENS = 5_000_000
    try:
        CLAUDE_SESSION_RESET_TURNS = max(
            1, int(config.get("claude_session_reset_turns", 50) or 50),
        )
    except (TypeError, ValueError):
        CLAUDE_SESSION_RESET_TURNS = 50
    CHAT_AGENT_CHEAP_ROUTING_ENABLED = bool(
        config.get("chat_agent_cheap_routing_enabled", True)
    )
    try:
        # Floor 60s: anything shorter would drop a freshly-completed crew
        # context before the user could send a reply.
        RECENT_CREW_STICKY_TTL_SEC = max(
            60, int(config.get("recent_crew_sticky_ttl_sec", 1800) or 1800),
        )
    except (TypeError, ValueError):
        RECENT_CREW_STICKY_TTL_SEC = 1800
    try:
        RECENT_CREW_STICKY_MAX_INJECTIONS = max(
            0, int(config.get("recent_crew_sticky_max_injections", 5) or 5),
        )
    except (TypeError, ValueError):
        RECENT_CREW_STICKY_MAX_INJECTIONS = 5

    # ── File handling configuration ────────────────────────────────────────
    config.setdefault("file_enabled", True)
    config.setdefault("max_file_size_bytes", 10 * 1024 * 1024)
    config.setdefault("file_pdf_enabled", True)
    config.setdefault("file_pdf_lib", "PyPDF2")

    FILE_ENABLED = bool(config.get("file_enabled", True))
    try:
        MAX_FILE_SIZE_BYTES = max(1, int(config.get("max_file_size_bytes", 10 * 1024 * 1024) or 10 * 1024 * 1024))
    except (TypeError, ValueError):
        MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
    _raw_file_exts = config.get("file_text_extensions")
    if isinstance(_raw_file_exts, list):
        FILE_TEXT_EXTENSIONS = frozenset(
            str(e).lower().strip().lstrip(".") for e in _raw_file_exts if e
        )
    else:
        FILE_TEXT_EXTENSIONS = frozenset({
            "txt", "md", "py", "js", "json", "yaml", "yml", "csv", "log",
            "sh", "go", "rs", "java", "c", "cpp", "h", "ts", "tsx", "jsx",
            "css", "html", "xml", "sql", "dockerfile", "toml", "ini", "cfg", "conf",
        })
    FILE_PDF_ENABLED = bool(config.get("file_pdf_enabled", True))
    FILE_PDF_LIB = str(config.get("file_pdf_lib", "PyPDF2") or "PyPDF2")

    _raw_timeouts = config.get("dev_stage_timeouts") or {}
    _clean_timeouts: "dict[str, int]" = {}
    if isinstance(_raw_timeouts, dict):
        for _k, _v in _raw_timeouts.items():
            try:
                _iv = int(_v)
            except (TypeError, ValueError):
                continue
            if _iv > 0 and isinstance(_k, str) and _k.strip():
                _clean_timeouts[_k.strip()] = _iv
    DEV_STAGE_TIMEOUTS = _clean_timeouts


def _init_backends() -> None:
    """P1-2: Phase 3 of bootstrap — BackendRegistry + health checks / probes."""
    global BACKEND_REGISTRY
    from larkhelm.backend_registry import BackendRegistry
    BACKEND_REGISTRY = BackendRegistry()
    _backends_list = _migrate_legacy_backends(config)
    BACKEND_REGISTRY.load(_backends_list)
    # Also update the module-level singleton in backend_registry so imports of it stay fresh
    import larkhelm.backend_registry as _br_mod
    _br_mod.BACKEND_REGISTRY = BACKEND_REGISTRY

    _test_mode = bool(os.environ.get("LARKHELM_TEST_MODE", "").strip())
    if _test_mode:
        return

    BACKEND_REGISTRY.health_check()
    from larkhelm.model_probe import run_probes_async
    run_probes_async(BACKEND_REGISTRY.all_enabled(), BACKEND_REGISTRY)
    _start_recover_thread()


def _init_plugins() -> None:
    """P1-2: Phase 4 of bootstrap — agent plugins + memory_gc daemon."""
    _test_mode = bool(os.environ.get("LARKHELM_TEST_MODE", "").strip())
    if _test_mode:
        return

    try:
        from larkhelm.agent_hub.plugin_loader import load_plugins
        load_plugins(config)
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[Config] agent plugin load failed: {e}")

    try:
        from larkhelm.memory_gc import start_memory_gc_thread
        start_memory_gc_thread()
    except Exception as e:
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[Config] memory_gc start failed: {e}")


def _init_runtime(config_path: str = None, data_dir: str = None) -> None:
    """Initialise paths and configuration (P1-2 facade).

    Delegates in order to ``_init_paths`` → ``_init_app_config`` →
    ``_init_backends`` → ``_init_plugins``, then builds the typed
    :class:`_RuntimeConfig` snapshot.

    Priority: CLI argument > environment variable > system paths (/etc / /var) > XDG user paths.
    """
    _init_paths(config_path, data_dir)
    _init_app_config()
    _init_backends()
    _init_plugins()

    # Build the typed config object (for mypy etc.; module-level globals remain for backward compat)
    global _runtime
    _runtime = _RuntimeConfig(
        CONFIG_PATH=CONFIG_PATH, DATA_DIR=DATA_DIR, SESSION_DIR=SESSION_DIR,
        LOG_DIR=LOG_DIR, STATE_FILE=STATE_FILE, DEBUG_LOG=DEBUG_LOG,
        config=config, APP_ID=APP_ID, APP_SECRET=APP_SECRET,
        CLAUDE_CMD=CLAUDE_CMD, GEMINI_CMD=GEMINI_CMD, KIMI_CMD=KIMI_CMD,
        DEEPSEEK_API_KEY=DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL=DEEPSEEK_BASE_URL,
        DEEPSEEK_MODEL=DEEPSEEK_MODEL,
        DEFAULT_MODEL=DEFAULT_MODEL,
        SKIP_PERMISSIONS=SKIP_PERMISSIONS, RESPONSE_TIMEOUT=RESPONSE_TIMEOUT,
        HARD_TIMEOUT=HARD_TIMEOUT, SHELL_TIMEOUT=SHELL_TIMEOUT,
        MAX_CARD_LEN=MAX_CARD_LEN,
        ALLOWED_CHATS=ALLOWED_CHATS, GEMINI_IDLE_TTL=GEMINI_IDLE_TTL,
        MAX_AI_PROCS_CONFIG=MAX_AI_PROCS_CONFIG, MAX_AI_PROCS=MAX_AI_PROCS,
        DEFAULT_CWD=DEFAULT_CWD, CRON_TIMEZONE=CRON_TIMEZONE,
        DOC_AUTO_INJECT=DOC_AUTO_INJECT, DOC_INJECT_MAX_CHARS=DOC_INJECT_MAX_CHARS,
        DOC_INJECT_MAX_DOCS=DOC_INJECT_MAX_DOCS, DOC_READ_MAX_CHARS=DOC_READ_MAX_CHARS,
        DEFAULT_DRIVE_FOLDER=DEFAULT_DRIVE_FOLDER, DOC_WRITE_CONFIRM=DOC_WRITE_CONFIRM,
        DOC_WRITE_BACKEND=DOC_WRITE_BACKEND,
        DEFAULT_WIKI_SPACE_ID=DEFAULT_WIKI_SPACE_ID,
        DEFAULT_WIKI_PARENT_TOKEN=DEFAULT_WIKI_PARENT_TOKEN,
        DEFAULT_OWNER_OPEN_ID=DEFAULT_OWNER_OPEN_ID,
        USER_TOKEN_PATH=USER_TOKEN_PATH,
        OAUTH_REDIRECT_PORT=OAUTH_REDIRECT_PORT,
        LOGGED_IN_OPEN_ID=LOGGED_IN_OPEN_ID,
        PERM_HOOK_SCRIPT=PERM_HOOK_SCRIPT, PERM_SOCKET_PATH=PERM_SOCKET_PATH,
        SOURCE_DIR=SOURCE_DIR,
        VOICE_ENABLED=VOICE_ENABLED, VOICE_ENGINE=VOICE_ENGINE,
        VOICE_API_KEY=VOICE_API_KEY,
        VOICE_MODEL_SIZE=VOICE_MODEL_SIZE,
        VOICE_COMPUTE_TYPE=VOICE_COMPUTE_TYPE, VOICE_MAX_DURATION_MS=VOICE_MAX_DURATION_MS,
        VOICE_DEFAULT_LANG=VOICE_DEFAULT_LANG, VOICE_MERGE_WINDOW_SEC=VOICE_MERGE_WINDOW_SEC,
        VOICE_MAX_MERGE=VOICE_MAX_MERGE, VOICE_KEEP_AUDIO=VOICE_KEEP_AUDIO,
        MEMORY_LIMIT_MB=MEMORY_LIMIT_MB,
        CREW_BREAKPOINT_TIMEOUT_SEC=CREW_BREAKPOINT_TIMEOUT_SEC,
        METRICS_TEXT_LEGACY=METRICS_TEXT_LEGACY,
        ANTHROPIC_EXTENDED_CACHE_ENABLED=ANTHROPIC_EXTENDED_CACHE_ENABLED,
        MEMORY_EXTRACT_BUFFER_WINDOW_SEC=MEMORY_EXTRACT_BUFFER_WINDOW_SEC,
        MEMORY_SESSION_SMART_COMPRESS=MEMORY_SESSION_SMART_COMPRESS,
        MEMORY_GLOBAL_PROFILE_SLOT_ENABLED=MEMORY_GLOBAL_PROFILE_SLOT_ENABLED,
        MEMORY_PROJECT_SECTION_ENABLED=MEMORY_PROJECT_SECTION_ENABLED,
        RECENT_TURNS_CACHE_ENABLED=RECENT_TURNS_CACHE_ENABLED,
        MEMORY_LEGACY_CACHE_ENABLED=MEMORY_LEGACY_CACHE_ENABLED,
        DOC_INJECT_CACHE_ENABLED=DOC_INJECT_CACHE_ENABLED,
        DOC_INJECT_CACHE_TTL_SEC=DOC_INJECT_CACHE_TTL_SEC,
        CLI_SKIP_RECENT_TURNS_WHEN_SID=CLI_SKIP_RECENT_TURNS_WHEN_SID,
        WORKSPACE_HINT_KEYWORD_GATE=WORKSPACE_HINT_KEYWORD_GATE,
        STATS_AGENT_TYPE_BREAKDOWN_ENABLED=STATS_AGENT_TYPE_BREAKDOWN_ENABLED,
        FILE_ENABLED=FILE_ENABLED,
        MAX_FILE_SIZE_BYTES=MAX_FILE_SIZE_BYTES,
        FILE_TEXT_EXTENSIONS=FILE_TEXT_EXTENSIONS,
        FILE_PDF_ENABLED=FILE_PDF_ENABLED,
        FILE_PDF_LIB=FILE_PDF_LIB,
    )


_runtime: _RuntimeConfig = None  # None means _init_runtime has not been called yet


_config_write_lock = threading.Lock()


def save_config_field(key: str, value) -> None:
    """Persist a single config field to config.json and update the in-memory config dict and corresponding global."""
    global config, DEFAULT_DRIVE_FOLDER, DEFAULT_WIKI_SPACE_ID, DEFAULT_WIKI_PARENT_TOKEN
    with _config_write_lock:
        config[key] = value
        # Update the corresponding global variable
        if key == "default_drive_folder":
            DEFAULT_DRIVE_FOLDER = value
        elif key == "default_wiki_space_id":
            DEFAULT_WIKI_SPACE_ID = value
        elif key == "default_wiki_parent_token":
            DEFAULT_WIKI_PARENT_TOKEN = value
        try:
            secure_atomic_write(CONFIG_PATH, json.dumps(config, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"[config] 写入 config.json 失败: {e}")
