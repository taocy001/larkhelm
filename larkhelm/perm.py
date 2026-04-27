"""
larkhelm · permission approval service

Contains:
  - _perm_yolo       Set of session namespaces that have been granted Allow All
  - grant_yolo() / revoke_yolo()  External API for mutating _perm_yolo
  - _bash_needs_approval()        Determine whether a Bash command needs approval
  - _send_perm_card()             Send a permission confirmation card
  - _handle_perm_conn()           Handle a single permission-request connection
  - _start_perm_server()          Start the Unix Socket permission approval server
"""
import json
import re
import threading
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _get_cwd
from larkhelm.lark_client import _send_card_raw, _patch_card_raw
from larkhelm.card_builder import _btn_type

# ── Permission state ─────────────────────────────────────────────────────
_perm_lock        = threading.Lock()
_perm_pending:  dict[str, threading.Event] = {}   # tool_use_id → Event
_perm_decision: dict[str, str]             = {}   # tool_use_id → "allow"|"deny"|"yolo"
_perm_card_mid: dict[str, str]             = {}   # tool_use_id → permission card message_id
_perm_tool_name: dict[str, str]            = {}   # tool_use_id → tool_name
_perm_tool_input: dict[str, dict]          = {}   # tool_use_id → tool_input

_perm_yolo: dict[str, float] = {}   # namespace → expiry timestamp
_YOLO_TTL = 3 * 3600                # Allow-All grants expire after 3 hours


def grant_yolo(ns: str) -> None:
    """Grant Allow All permission to the given namespace (expires after _YOLO_TTL seconds)."""
    _perm_yolo[ns] = time.time() + _YOLO_TTL


def revoke_yolo(ns: str) -> None:
    """Revoke Allow All permission from the given namespace."""
    _perm_yolo.pop(ns, None)


def is_yolo(ns: str) -> bool:
    """Return True if Allow All is currently granted and not expired for this namespace."""
    return _perm_yolo.get(ns, 0.0) > time.time()


# ── Dangerous command patterns (pre-compiled) ────────────────────────────────────────
_DANGEROUS_PATTERNS = [
    re.compile(pat) for pat in [
        r'(?<![a-zA-Z0-9_/])rm(\s|$)',
        r'\brmdir\b',
        r'\bdd\b',
        r'\bmkfs\b',
        r'\bfdisk\b',
        r'\bparted\b',
        r'\bshred\b',
        r'\btruncate\b',
        r'\bsudo\b',
        r'\bchmod\b',
        r'\bchown\b',
        r'\bkill\b',
        r'\bpkill\b',
        r'\bkillall\b',
        r'\bsystemctl\s+(stop|disable|mask|delete)\b',
        r'\bservice\s+\S+\s+stop\b',
        r'\bapt(-get)?\s+(remove|purge|autoremove)\b',
        r'\bpip3?\s+uninstall\b',
        r'\byum\s+(remove|erase)\b',
        r'\bdnf\s+remove\b',
        r'[|;`]\s*(bash|sh|zsh|python3?|ruby|perl|node)\s',
        r'>\s*/(?!tmp/)',
    ]
]


def _is_dangerous_cmd(command: str, cwd: str) -> bool:
    """Check only for dangerous command keyword patterns, ignoring paths (for direct use in tests)."""
    cmd = command.strip()
    for pat in _DANGEROUS_PATTERNS:
        if pat.search(cmd):
            return True
    return False


def get_safe_prefixes(cwd: str) -> list:
    """Return the list of safe path prefixes for the given cwd (injectable for tests)."""
    prefixes = ["/tmp/", "/tmp"]
    if cwd:
        prefixes.append(cwd.rstrip("/") + "/")
        prefixes.append(cwd.rstrip("/"))
    return prefixes


def _bash_needs_approval(command: str, cwd: str) -> bool:
    """
    Determine whether a Bash command requires interactive approval.
    Rules:
    1. Contains dangerous operation keywords → requires approval
    2. Contains absolute paths outside cwd and /tmp → requires approval
    3. Everything else → auto-allowed
    """
    cmd = command.strip()
    if _is_dangerous_cmd(cmd, cwd):
        return True

    safe_prefixes = get_safe_prefixes(cwd)
    abs_paths = re.findall(r'(?<!\w)(\/[^\s\'\";,|&>]+)', cmd)
    for path in abs_paths:
        path = path.rstrip(".,;\"')")
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = path
        if not any(resolved.startswith(p) for p in safe_prefixes):
            return True

    return False


def _send_perm_card(chat_id: str, tool_name: str, tool_input: dict, tool_use_id: str) -> str:
    """Send a permission confirmation card (JSON 1.0; uses multiple div blocks to ensure content visibility)."""
    elements: list = []

    def md(content: str):
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": content}})

    if tool_name == "Bash":
        cmd_text = tool_input.get("command", "").strip()
        desc_field = tool_input.get("description", "").strip()
        md(f"**工具：** `Bash`")
        if desc_field:
            md(f"_{desc_field}_")
        md("**命令：**\n" + (cmd_text[:800] or "(空)"))
    elif tool_name in ("Write", "Edit", "NotebookEdit"):
        path = tool_input.get("file_path", tool_input.get("notebook_path", "?"))
        md(f"**工具：** `{tool_name}`")
        md(f"**文件：** `{path}`")
        old = tool_input.get("old_string", "")
        new = tool_input.get("new_string", tool_input.get("new_string", ""))
        if old:
            md(f"**修改内容：** {len(old.splitlines())} 行 → {len(new.splitlines())} 行")
    elif tool_name == "Read":
        path = tool_input.get("file_path", "?")
        offset = tool_input.get("offset", "")
        limit  = tool_input.get("limit", "")
        extra  = f"  行 {offset}–{int(offset)+int(limit)}" if offset and limit else ""
        md(f"**工具：** `Read`")
        md(f"**文件：** `{path}`{extra}")
    elif tool_name == "Glob":
        md(f"**工具：** `Glob`")
        md(f"**模式：** `{tool_input.get('pattern','?')}`")
    elif tool_name == "Grep":
        md(f"**工具：** `Grep`")
        md(f"**模式：** `{tool_input.get('pattern','?')}`")
    else:
        md(f"**工具：** `{tool_name}`")
        for k, v in list(tool_input.items())[:6]:
            md(f"**{k}：** `{str(v)[:120]}`")

    elements.append({"tag": "hr"})
    elements.append({
        "tag": "action",
        "actions": [
            {"tag": "button", "text": {"tag": "plain_text", "content": label},
             "type": _btn_type(label), "value": {"cmd": f"perm:{act}:{tool_use_id}"}}
            for label, act in [("✅ 允许", "allow"), ("❌ 拒绝", "deny"), ("🚀 允许所有", "yolo")]
        ],
    })

    card_json = json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {"template": "orange", "title": {"tag": "plain_text", "content": "🔐 权限请求"}},
        "elements": elements,
    }, ensure_ascii=False)
    return _send_card_raw(chat_id, card_json)


_MAX_PERM_CONN_BUF = 65536  # 64 KB — prevent DoS via oversized payloads from rogue hook scripts


def _handle_perm_conn(conn):
    """Handle a single permission-request connection from the hook script."""
    try:
        buf = b""
        while b"\n" not in buf:
            chunk = conn.recv(1024)
            if not chunk:
                break
            buf += chunk
            if len(buf) > _MAX_PERM_CONN_BUF:
                _debug_log("[Perm] oversized hook payload, closing connection")
                conn.close()
                return
        if not buf.strip():
            conn.close()
            return

        req         = json.loads(buf.strip())
        chat_id     = req.get("chat_id", "")
        tool_name   = req.get("tool_name", "?")
        tool_input  = req.get("tool_input", {})
        tool_use_id = req.get("tool_use_id", "") or str(time.time())
        _debug_log(f"[Perm] received request tool={tool_name} id={tool_use_id[:16]}")

        # Fast-path: Allow All already granted
        if is_yolo(chat_id):
            conn.sendall((json.dumps({"decision": "allow"}) + "\n").encode())
            conn.close()
            return

        # Fast-path: non-dangerous command, auto-allow
        if tool_name == "Bash":
            cwd = _get_cwd(chat_id)
            command = tool_input.get("command", "")
            if not _bash_needs_approval(command, cwd):
                _debug_log(f"[Perm] auto-allowed Bash: {command[:80]}")
                conn.sendall((json.dumps({"decision": "allow"}) + "\n").encode())
                conn.close()
                return

        elif tool_name in ("Read", "Glob", "Grep", "TodoWrite", "TodoRead", "ToolSearch"):
            _debug_log(f"[Perm] auto-allowed {tool_name}")
            conn.sendall((json.dumps({"decision": "allow"}) + "\n").encode())
            conn.close()
            return

        ev = threading.Event()
        with _perm_lock:
            _perm_pending[tool_use_id] = ev
            _perm_tool_name[tool_use_id] = tool_name
            _perm_tool_input[tool_use_id] = tool_input

        mid = _send_perm_card(chat_id, tool_name, tool_input, tool_use_id)
        _debug_log(f"[Perm] card sent {'ok mid=' + mid if mid else 'failed'}")
        if mid:
            with _perm_lock:
                _perm_card_mid[tool_use_id] = mid

        # Block waiting for user click (up to 5 minutes; auto-deny on timeout)
        clicked = ev.wait(timeout=300)

        with _perm_lock:
            decision = _perm_decision.pop(tool_use_id, "deny")
            _perm_pending.pop(tool_use_id, None)
            card_mid = _perm_card_mid.pop(tool_use_id, None)
            _perm_tool_name.pop(tool_use_id, None)
            _perm_tool_input.pop(tool_use_id, None)

        if not clicked and decision == "deny":
            decision = "timeout"

        if card_mid:
            labels = {"allow": "✅ 已允许", "deny": "❌ 已拒绝", "yolo": "🚀 已允许全部", "timeout": "⏰ 已超时"}
            result_title = labels.get(decision, "已处理")
            result_color = "green" if decision != "deny" else "red"
            result_card = json.dumps({
                "config": {"wide_screen_mode": True},
                "header": {"template": result_color,
                           "title": {"tag": "plain_text", "content": result_title}},
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md",
                                            "content": f"**工具：** `{tool_name}`"}},
                ],
            }, ensure_ascii=False)
            ok = _patch_card_raw(card_mid, result_card)
            _debug_log(f"[Perm] card update {'ok' if ok else 'failed'} decision={decision}")

        if decision in ("allow", "yolo"):
            conn.sendall((json.dumps({"decision": "allow"}) + "\n").encode())
        else:
            conn.sendall((json.dumps({"decision": "deny", "reason": "用户拒绝了此操作"}) + "\n").encode())
        conn.close()

    except Exception as e:
        _debug_log(f"[Perm] connection handling exception: {e}")
        try:
            conn.sendall((json.dumps({"decision": "deny", "reason": str(e)}) + "\n").encode())
            conn.close()
        except Exception:
            pass


def _start_perm_server():
    """Start the Unix Socket permission approval server (called only when skip_permissions=False)."""
    import socket as _sock
    import os as _os
    path = _cfg.PERM_SOCKET_PATH
    # Remove stale socket file first
    try:
        _os.unlink(path)
    except FileNotFoundError:
        pass
    # Set umask before bind so the socket file is created with 0o600 permissions atomically,
    # eliminating the TOCTTOU window that exists between bind() and a subsequent chmod().
    _old_umask = _os.umask(0o077)
    try:
        server = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        server.bind(path)
    finally:
        _os.umask(_old_umask)
    server.listen(32)
    _debug_log(f"[Perm] permission approval server started: {path}")

    def _serve():
        while True:
            try:
                conn, _ = server.accept()
                threading.Thread(target=_handle_perm_conn, args=(conn,),
                                 daemon=True, name="perm-conn").start()
            except Exception:
                break

    threading.Thread(target=_serve, daemon=True, name="perm-server").start()
