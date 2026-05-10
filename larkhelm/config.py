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
    MAX_CARD_LEN:     int
    ALLOWED_CHATS:    set
    GEMINI_IDLE_TTL:  int
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
MAX_CARD_LEN:     int
ALLOWED_CHATS:    set
GEMINI_IDLE_TTL:  int
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
    """
    known = [
        {"id": "claude",    "provider": "claude_cli", "display_name": "Claude",    "tags": ["vision", "tools"], "command": "claude"},
        {"id": "gemini",    "provider": "gemini_cli", "display_name": "Gemini",    "tags": ["tools"],           "command": "gemini"},
        {"id": "kimi",      "provider": "kimi_cli",   "display_name": "Kimi",      "tags": ["vision", "tools"], "command": "kimi"},
        {"id": "kimi-code", "provider": "kimi_cli",   "display_name": "Kimi-Code", "tags": ["tools"],           "command": "kimi-code"},
    ]
    return [spec for spec in known if shutil.which(spec["command"])]


def _auto_discover_http() -> list[dict]:
    """Probe HTTP backends whose API keys are present in config or env.

    Currently DeepSeek only. Mirrors ``_auto_discover_cli`` shape so callers can
    treat both the same way.
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

    def _health_tick():
        import time as _time
        from concurrent.futures import ThreadPoolExecutor, as_completed
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
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="health-tick") as pool:
            future_to_target = {pool.submit(probe_spec, s): (s, reason) for s, reason in to_probe}
            for fut in as_completed(future_to_target):
                spec, reason = future_to_target[fut]
                try:
                    ok, err = fut.result()
                except Exception as e:
                    ok, err = False, str(e)[:200]
                try:
                    BACKEND_REGISTRY.set_probe_result(spec.id, ok, err)
                except Exception as e:
                    _dlog(f"[BackendRegistry] tick set_probe_result failed for {spec.id}: {e}")
                icon = "✓" if ok else "✗"
                # Recompute reason post-hoc: a "retry" that succeeded is a "recover"
                if reason == "retry" and ok:
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


def _init_runtime(config_path: str = None, data_dir: str = None) -> None:
    """Initialise paths and configuration.

    Priority: CLI argument > environment variable > system paths (/etc / /var) > XDG user paths
    """
    global CONFIG_PATH, DATA_DIR, SESSION_DIR, LOG_DIR, STATE_FILE, DEBUG_LOG
    global config, APP_ID, APP_SECRET, CLAUDE_CMD, GEMINI_CMD, KIMI_CMD
    global DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEFAULT_MODEL
    global SKIP_PERMISSIONS, RESPONSE_TIMEOUT, HARD_TIMEOUT, MAX_CARD_LEN
    global ALLOWED_CHATS, GEMINI_IDLE_TTL, DEFAULT_CWD, CRON_TIMEZONE
    global PERM_HOOK_SCRIPT, PERM_SOCKET_PATH
    global DOC_AUTO_INJECT, DOC_INJECT_MAX_CHARS, DOC_INJECT_MAX_DOCS
    global DOC_READ_MAX_CHARS, DEFAULT_DRIVE_FOLDER, DOC_WRITE_CONFIRM, DOC_WRITE_BACKEND
    global DEFAULT_WIKI_SPACE_ID, DEFAULT_WIKI_PARENT_TOKEN, DEFAULT_OWNER_OPEN_ID
    global BACKEND_REGISTRY
    global VOICE_ENABLED, VOICE_ENGINE, VOICE_API_KEY
    global VOICE_MODEL_SIZE, VOICE_COMPUTE_TYPE, VOICE_MAX_DURATION_MS
    global VOICE_DEFAULT_LANG, VOICE_MERGE_WINDOW_SEC, VOICE_MAX_MERGE, VOICE_KEEP_AUDIO

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

    try:
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
    SKIP_PERMISSIONS = bool(config.get("skip_permissions", False))
    RESPONSE_TIMEOUT = int(config.get("response_timeout", 300))   # soft timeout: release lock but don't kill process
    HARD_TIMEOUT     = int(config.get("hard_timeout", 21600))      # hard timeout: force kill (default 6 hours)
    MAX_CARD_LEN     = int(config.get("max_card_len", 3000))
    ALLOWED_CHATS    = set(config.get("allowed_chat_ids", []))
    GEMINI_IDLE_TTL  = int(config.get("gemini_idle_ttl", 1800))
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

    # Phase 5: intent router / agent_hub flags (all optional, default off so
    # phase4 behavior is unchanged when these keys are missing).
    config.setdefault("intent_router_enabled", False)
    config.setdefault("intent_router_traffic", 0.0)
    config.setdefault("intent_layer2_strategy", "llm")
    config.setdefault("agent_plugins", [])
    config.setdefault("agent_acl", {})
    config.setdefault("intent_feedback_path", "")
    config.setdefault("intent_audit_path", "")

    PERM_HOOK_SCRIPT  = str(Path(__file__).parent / "perm_hook.py")
    PERM_SOCKET_PATH  = str(DATA_DIR / "perm.sock")

    global SOURCE_DIR
    # For editable installs __file__ = <repo>/larkhelm/config.py; two levels up is the repo root
    _candidate = Path(__file__).parent.parent
    if (_candidate / ".git").exists():
        SOURCE_DIR = _candidate          # git repo root (editable install)
    else:
        SOURCE_DIR = Path(__file__).parent  # fall back to package directory for non-editable installs

    # Initialize BackendRegistry from config (must happen after all globals are set)
    from larkhelm.backend_registry import BackendRegistry
    BACKEND_REGISTRY = BackendRegistry()
    _backends_list = _migrate_legacy_backends(config)
    BACKEND_REGISTRY.load(_backends_list)
    BACKEND_REGISTRY.health_check()
    # Also update the module-level singleton in backend_registry so imports of it stay fresh
    import larkhelm.backend_registry as _br_mod
    _br_mod.BACKEND_REGISTRY = BACKEND_REGISTRY

    from larkhelm.model_probe import run_probes_async
    run_probes_async(BACKEND_REGISTRY.all_enabled(), BACKEND_REGISTRY)

    _start_recover_thread()

    # Phase 5: load third-party agent plugins. Built-in agents register
    # themselves via ``larkhelm.agent_hub.builtin`` import side-effects.
    try:
        from larkhelm.agent_hub.plugin_loader import load_plugins
        load_plugins(config)
    except Exception as e:
        # Plugin discovery runs during startup, when ``larkhelm.log`` may not
        # be wired up yet — ``lazy_debug_log`` is the bootstrap-safe variant.
        from larkhelm.log import lazy_debug_log
        lazy_debug_log(f"[Config] agent plugin load failed: {e}")

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
        HARD_TIMEOUT=HARD_TIMEOUT, MAX_CARD_LEN=MAX_CARD_LEN,
        ALLOWED_CHATS=ALLOWED_CHATS, GEMINI_IDLE_TTL=GEMINI_IDLE_TTL,
        DEFAULT_CWD=DEFAULT_CWD, CRON_TIMEZONE=CRON_TIMEZONE,
        DOC_AUTO_INJECT=DOC_AUTO_INJECT, DOC_INJECT_MAX_CHARS=DOC_INJECT_MAX_CHARS,
        DOC_INJECT_MAX_DOCS=DOC_INJECT_MAX_DOCS, DOC_READ_MAX_CHARS=DOC_READ_MAX_CHARS,
        DEFAULT_DRIVE_FOLDER=DEFAULT_DRIVE_FOLDER, DOC_WRITE_CONFIRM=DOC_WRITE_CONFIRM,
        DOC_WRITE_BACKEND=DOC_WRITE_BACKEND,
        DEFAULT_WIKI_SPACE_ID=DEFAULT_WIKI_SPACE_ID,
        DEFAULT_WIKI_PARENT_TOKEN=DEFAULT_WIKI_PARENT_TOKEN,
        DEFAULT_OWNER_OPEN_ID=DEFAULT_OWNER_OPEN_ID,
        PERM_HOOK_SCRIPT=PERM_HOOK_SCRIPT, PERM_SOCKET_PATH=PERM_SOCKET_PATH,
        SOURCE_DIR=SOURCE_DIR,
        VOICE_ENABLED=VOICE_ENABLED, VOICE_ENGINE=VOICE_ENGINE,
        VOICE_API_KEY=VOICE_API_KEY,
        VOICE_MODEL_SIZE=VOICE_MODEL_SIZE,
        VOICE_COMPUTE_TYPE=VOICE_COMPUTE_TYPE, VOICE_MAX_DURATION_MS=VOICE_MAX_DURATION_MS,
        VOICE_DEFAULT_LANG=VOICE_DEFAULT_LANG, VOICE_MERGE_WINDOW_SEC=VOICE_MERGE_WINDOW_SEC,
        VOICE_MAX_MERGE=VOICE_MAX_MERGE, VOICE_KEEP_AUDIO=VOICE_KEEP_AUDIO,
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
            _tmp = CONFIG_PATH.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(_tmp, CONFIG_PATH)
        except Exception as e:
            print(f"[config] 写入 config.json 失败: {e}")
