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
import sys
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
    PERM_HOOK_SCRIPT: str
    PERM_SOCKET_PATH: str
    SOURCE_DIR:       Path

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

# Permission approval socket path (derived from PID; fixed once _init_runtime sets it)
PERM_HOOK_SCRIPT: str
PERM_SOCKET_PATH: str
SOURCE_DIR: "Path"  # source repo root directory (auto-detected for editable installs)

# ── Card UX parameters (module-level constants) ────────────────────────────────────
TOOL_HISTORY_CAP   = 20      # max number of completed tool pairs kept in history
CARD_PUSH_INTERVAL = 5.0     # heartbeat push interval (seconds)
CURSOR_INTERVAL    = 0.3     # local cursor-blink interval (seconds)
STALL_THRESHOLD    = 30.0    # threshold for detecting a stalled tool (seconds)
CURSOR_FRAMES      = ["▌", "▍", "▎", "▏"]


def _init_runtime(config_path: str | None = None, data_dir: str | None = None) -> None:
    """Initialise paths and configuration.

    Priority: CLI argument > environment variable > system paths (/etc / /var) > XDG user paths
    """
    global CONFIG_PATH, DATA_DIR, SESSION_DIR, LOG_DIR, STATE_FILE, DEBUG_LOG
    global config, APP_ID, APP_SECRET, CLAUDE_CMD, GEMINI_CMD, KIMI_CMD, DEFAULT_MODEL
    global SKIP_PERMISSIONS, RESPONSE_TIMEOUT, HARD_TIMEOUT, MAX_CARD_LEN
    global ALLOWED_CHATS, GEMINI_IDLE_TTL, DEFAULT_CWD, CRON_TIMEZONE
    global PERM_HOOK_SCRIPT, PERM_SOCKET_PATH
    global DOC_AUTO_INJECT, DOC_INJECT_MAX_CHARS, DOC_INJECT_MAX_DOCS
    global DOC_READ_MAX_CHARS, DEFAULT_DRIVE_FOLDER, DOC_WRITE_CONFIRM, DOC_WRITE_BACKEND
    global DEFAULT_WIKI_SPACE_ID, DEFAULT_WIKI_PARENT_TOKEN

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
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    CLAUDE_CMD       = config.get("claude_command", "claude")
    GEMINI_CMD       = config.get("gemini_command", "gemini")
    KIMI_CMD         = config.get("kimi_command", "kimi")
    DEFAULT_MODEL    = config.get("default_model", "claude")
    SKIP_PERMISSIONS = config.get("skip_permissions", False)
    RESPONSE_TIMEOUT = config.get("response_timeout", 300)   # soft timeout: release lock but don't kill process
    HARD_TIMEOUT     = config.get("hard_timeout", 21600)      # hard timeout: force kill (default 6 hours)
    MAX_CARD_LEN     = config.get("max_card_len", 3000)
    ALLOWED_CHATS    = set(config.get("allowed_chat_ids", []))
    GEMINI_IDLE_TTL  = config.get("gemini_idle_ttl", 1800)
    DEFAULT_CWD      = config.get("default_cwd", str(Path.home() / "code"))
    CRON_TIMEZONE    = config.get("timezone", "Asia/Shanghai")

    DOC_AUTO_INJECT      = config.get("doc_auto_inject",      True)
    DOC_INJECT_MAX_CHARS = config.get("doc_inject_max_chars", 2000)
    DOC_INJECT_MAX_DOCS  = config.get("doc_inject_max_docs",  2)
    DOC_READ_MAX_CHARS   = config.get("doc_read_max_chars",   6000)
    DEFAULT_DRIVE_FOLDER = config.get("default_drive_folder", "")
    DOC_WRITE_CONFIRM    = config.get("doc_write_confirm",    True)
    DOC_WRITE_BACKEND    = config.get("doc_write_backend",   "auto")

    DEFAULT_WIKI_SPACE_ID     = config.get("default_wiki_space_id",     "")
    DEFAULT_WIKI_PARENT_TOKEN = config.get("default_wiki_parent_token", "")

    PERM_HOOK_SCRIPT  = str(Path(__file__).parent / "perm_hook.py")
    PERM_SOCKET_PATH  = f"/tmp/feishu_perm_{os.getpid()}.sock"

    global SOURCE_DIR
    # For editable installs __file__ = <repo>/larkhelm/config.py; two levels up is the repo root
    _candidate = Path(__file__).parent.parent
    if (_candidate / ".git").exists():
        SOURCE_DIR = _candidate          # git repo root (editable install)
    else:
        SOURCE_DIR = Path(__file__).parent  # fall back to package directory for non-editable installs

    # Build the typed config object (for mypy etc.; module-level globals remain for backward compat)
    global _runtime
    _runtime = _RuntimeConfig(
        CONFIG_PATH=CONFIG_PATH, DATA_DIR=DATA_DIR, SESSION_DIR=SESSION_DIR,
        LOG_DIR=LOG_DIR, STATE_FILE=STATE_FILE, DEBUG_LOG=DEBUG_LOG,
        config=config, APP_ID=APP_ID, APP_SECRET=APP_SECRET,
        CLAUDE_CMD=CLAUDE_CMD, GEMINI_CMD=GEMINI_CMD, KIMI_CMD=KIMI_CMD, DEFAULT_MODEL=DEFAULT_MODEL,
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
        PERM_HOOK_SCRIPT=PERM_HOOK_SCRIPT, PERM_SOCKET_PATH=PERM_SOCKET_PATH,
        SOURCE_DIR=SOURCE_DIR,
    )


_runtime: _RuntimeConfig | None = None  # None means _init_runtime has not been called yet


_config_write_lock = __import__("threading").Lock()


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
            import os as _os
            _tmp = CONFIG_PATH.with_suffix(".json.tmp")
            _tmp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            _os.replace(_tmp, CONFIG_PATH)
        except Exception as e:
            print(f"[config] 写入 config.json 失败: {e}")
