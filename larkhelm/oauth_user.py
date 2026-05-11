"""
larkhelm · OAuth 2.0 user_access_token flow (Feishu / Lark)

Why this exists
---------------
``FeishuDocClient.create_doc()`` historically created documents under the
**app**'s identity (tenant_access_token), then immediately called
``transfer_doc_owner`` to hand the file to the user. That transfer is a real
Feishu API call that triggers a "X wants to transfer ownership of Y to you"
system notification — there is no query parameter that suppresses it.

With a **user_access_token**, the create call happens *as the user*, so the
document is owned by the user from the start and no transfer (or notification)
is needed. This module owns the OAuth handshake and persistent token store
that makes that possible.

Scope of this v1
----------------
* **CLI-only login** (`larkhelm user-login`): a loopback HTTP server on
  127.0.0.1 captures the authorization ``code``. This requires running the
  command on a host with a browser — service operators typically SSH-forward
  the port or run on their own workstation.
* **No Feishu bot ``/auth`` command** (yet): the Feishu mobile webview path
  has too many unknowns (whether the address bar is reachable after consent,
  whether deep links work). A bot flow is planned as v2 once those are
  validated end-to-end.
* **Single user**: one token file at ``DATA_DIR/user_token.json``. Multi-user
  is not a goal — larkhelm is operator-personal software.

Failure modes are designed to be **non-fatal**: every consumer
(``get_user_token`` / ``is_token_valid``) returns ``None`` / ``False`` on
any error rather than raising, so ``FeishuDocClient`` can transparently fall
back to the tenant-token + transfer path.
"""
from __future__ import annotations

import http.server
import json
import os
import queue
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Optional

import larkhelm.config as _cfg

# ── Constants ───────────────────────────────────────────────────────────────

AUTHORIZE_URL    = "https://accounts.feishu.cn/open-apis/authen/v1/authorize"
ACCESS_TOKEN_URL = "https://open.feishu.cn/open-apis/authen/v1/access_token"
REFRESH_URL      = "https://open.feishu.cn/open-apis/authen/v1/refresh_access_token"
APP_TOKEN_URL    = "https://open.feishu.cn/open-apis/auth/v3/app_access_token/internal"

# Scopes required to create + write Feishu documents on the user's behalf.
# Trimmed deliberately — broader scopes would prompt the user with permissions
# they don't actually need.
DEFAULT_SCOPES = " ".join([
    "docx:document",                   # read + write docx content
    "docx:document:create",            # POST /docx/v1/documents
    "drive:drive",                     # drive folder / file metadata
    "wiki:wiki",                       # wiki node create/read
])

# Margin before ``expires_at`` at which we proactively refresh. 10 min is well
# above any reasonable clock skew or call latency.
_REFRESH_MARGIN_SEC = 600

# Loopback callback path. Fixed because it has to match the URL registered
# in the Feishu developer console.
_CALLBACK_PATH = "/callback"

# Maximum time we wait for the user to complete the consent flow before
# giving up and tearing down the loopback server. 5 minutes matches a
# reasonable human pace for OAuth flows.
_LOGIN_TIMEOUT_SEC = 300


# ── Token persistence ───────────────────────────────────────────────────────

_token_lock = threading.Lock()      # serializes refresh + save
_token_cache: Optional[dict] = None  # in-process cache; None == not loaded


def _load_token() -> Optional[dict]:
    """Read and cache the token file. Returns ``None`` on any failure."""
    global _token_cache
    if _token_cache is not None:
        return _token_cache
    path = _cfg.USER_TOKEN_PATH
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        # Minimum shape check — anything else means a corrupted file, which
        # we treat as "no token" so the caller falls back to tenant path.
        if not isinstance(data, dict) or "access_token" not in data:
            return None
        _token_cache = data
        return data
    except Exception as e:
        print(f"[oauth_user] load token failed (ignored): {e}", file=sys.stderr)
        return None


def _save_token(data: dict) -> None:
    """Atomically persist the token file with 0600 permissions."""
    global _token_cache
    path = _cfg.USER_TOKEN_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    # Write + chmod the tmp file *before* the rename, so the destination never
    # exists in a world-readable state — even briefly. ``os.replace`` is atomic.
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass  # Best-effort on non-POSIX; the data is already secret-grade.
    os.replace(tmp, path)
    _token_cache = data


def _clear_cache() -> None:
    """Forget the in-process token cache. Used after logout / refresh failures."""
    global _token_cache
    _token_cache = None


# ── HTTP helpers ────────────────────────────────────────────────────────────

def _http_post_json(url: str, body: dict, *, headers: Optional[dict] = None,
                    timeout: float = 15.0) -> dict:
    """POST JSON, return parsed JSON dict. Raises on transport or HTTP error."""
    raw_headers = {"Content-Type": "application/json; charset=utf-8"}
    if headers:
        raw_headers.update(headers)
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=raw_headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_app_access_token() -> str:
    """Fetch an ``app_access_token``. Required by the OAuth exchange endpoints."""
    resp = _http_post_json(APP_TOKEN_URL,
                           {"app_id": _cfg.APP_ID, "app_secret": _cfg.APP_SECRET})
    tok = resp.get("app_access_token", "")
    if not tok:
        raise RuntimeError(f"app_access_token fetch failed: {resp}")
    return tok


# ── Token flow primitives ───────────────────────────────────────────────────

def _exchange_code(code: str, redirect_uri: str) -> dict:
    """Exchange an authorization ``code`` for an access_token + refresh_token.

    Returns the normalized token dict that ``_save_token`` will persist.
    Raises on failure.
    """
    app_tok = _get_app_access_token()
    resp = _http_post_json(
        ACCESS_TOKEN_URL,
        {
            "grant_type":   "authorization_code",
            "code":         code,
            "redirect_uri": redirect_uri,
        },
        headers={"Authorization": f"Bearer {app_tok}"},
    )
    if resp.get("code", 0) != 0:
        raise RuntimeError(f"access_token exchange failed: {resp}")
    return _normalize_token_response(resp)


def _refresh_token_call(refresh_token: str) -> dict:
    """Refresh the access_token; returns a normalized token dict on success."""
    app_tok = _get_app_access_token()
    resp = _http_post_json(
        REFRESH_URL,
        {"grant_type": "refresh_token", "refresh_token": refresh_token},
        headers={"Authorization": f"Bearer {app_tok}"},
    )
    if resp.get("code", 0) != 0:
        raise RuntimeError(f"refresh_token call failed: {resp}")
    return _normalize_token_response(resp)


def _normalize_token_response(resp: dict) -> dict:
    """Convert Feishu's response shape to our on-disk schema.

    Feishu returns absolute lifetimes in ``expires_in`` / ``refresh_expires_in``;
    we store absolute Unix timestamps so the consumer can do a single
    comparison without re-deriving "saved_at".
    """
    data = resp.get("data", {}) or {}
    now = int(time.time())
    return {
        "access_token":        data.get("access_token", ""),
        "token_type":          data.get("token_type", "Bearer"),
        "refresh_token":       data.get("refresh_token", ""),
        "expires_at":          now + int(data.get("expires_in", 0)),
        "refresh_expires_at":  now + int(data.get("refresh_expires_in", 0)),
        "open_id":             data.get("open_id", ""),
        "scope":               data.get("scope", ""),
        "saved_at":            now,
    }


# ── Public API consumed by lark_client ──────────────────────────────────────

def is_token_valid() -> bool:
    """Cheap check — does *not* hit the network.

    Returns True iff a token is on disk and its ``expires_at`` is in the
    future. A token within the refresh margin is still "valid" — the actual
    refresh happens lazily inside ``get_user_token``.
    """
    data = _load_token()
    if not data:
        return False
    try:
        return time.time() < float(data.get("expires_at", 0))
    except Exception:
        return False


def get_user_token() -> Optional[str]:
    """Return a usable access_token, refreshing if necessary.

    Returns ``None`` on any failure — callers must fall back. The function
    never raises so ``create_doc`` can be wrapped in a single try-free if-branch.
    """
    data = _load_token()
    if not data:
        return None
    now = time.time()
    expires_at = float(data.get("expires_at", 0))
    if now < expires_at - _REFRESH_MARGIN_SEC:
        # Still well within validity — fast path, no refresh.
        return data.get("access_token") or None

    # Need refresh. Serialize: many concurrent ``create_doc`` calls would
    # otherwise stampede the refresh endpoint and waste rate budget.
    with _token_lock:
        # Re-read inside the lock in case another thread already refreshed.
        data = _load_token()
        if not data:
            return None
        now = time.time()
        expires_at = float(data.get("expires_at", 0))
        if now < expires_at - _REFRESH_MARGIN_SEC:
            return data.get("access_token") or None

        refresh_token = data.get("refresh_token", "")
        refresh_expires_at = float(data.get("refresh_expires_at", 0))
        if not refresh_token or now >= refresh_expires_at:
            # Refresh token itself dead — drop the file and force re-login.
            print("[oauth_user] refresh_token expired; clearing user token",
                  file=sys.stderr)
            try:
                _cfg.USER_TOKEN_PATH.unlink(missing_ok=True)
            except Exception:
                pass
            _clear_cache()
            return None
        try:
            new_data = _refresh_token_call(refresh_token)
        except Exception as e:
            print(f"[oauth_user] refresh failed: {e}", file=sys.stderr)
            return None
        _save_token(new_data)
        return new_data.get("access_token") or None


def clear_token() -> None:
    """Erase the on-disk token (used by ``larkhelm user-logout``)."""
    try:
        _cfg.USER_TOKEN_PATH.unlink(missing_ok=True)
    except Exception as e:
        print(f"[oauth_user] clear_token: {e}", file=sys.stderr)
    _clear_cache()


def get_status() -> dict:
    """Summary for ``larkhelm user-status``. Never hits the network."""
    data = _load_token()
    if not data:
        return {"logged_in": False}
    now = time.time()
    exp = float(data.get("expires_at", 0))
    rexp = float(data.get("refresh_expires_at", 0))
    return {
        "logged_in":         True,
        "open_id":           data.get("open_id", ""),
        "scope":             data.get("scope", ""),
        "expires_in_sec":    max(0, int(exp - now)),
        "refresh_expires_in_sec": max(0, int(rexp - now)),
    }


# ── CLI: `larkhelm user-login` (loopback flow) ──────────────────────────────

def _build_authorize_url(redirect_uri: str, state: str,
                         scope: str = DEFAULT_SCOPES) -> str:
    """Build the Feishu authorize URL with the given redirect + state."""
    q = urllib.parse.urlencode({
        "app_id":       _cfg.APP_ID,
        "redirect_uri": redirect_uri,
        "state":        state,
        "response_type": "code",
        "scope":        scope,
    })
    return f"{AUTHORIZE_URL}?{q}"


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """Single-shot handler that captures the OAuth ``code`` query parameter.

    Result is communicated to the parent thread via the queue stored on the
    server instance — see ``_run_loopback_server`` below.
    """

    # Quiet — default BaseHTTPRequestHandler logs every request to stderr,
    # which is noisy during interactive login.
    def log_message(self, format, *args):  # noqa: A003 — overriding stdlib
        pass

    def do_GET(self):  # noqa: N802 — http.server convention
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != _CALLBACK_PATH:
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
        err = (params.get("error") or [""])[0]
        # Acknowledge to the user's browser before signaling the main thread,
        # otherwise the browser sees a connection reset when the server tears
        # down (the main thread shuts the server down on signal).
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        if err:
            html = f"<h2>Authorization failed: {err}</h2><p>You can close this window.</p>"
        elif not code:
            html = "<h2>Missing code parameter.</h2><p>You can close this window.</p>"
        else:
            html = ("<h2>Authorization complete.</h2>"
                    "<p>You can close this window and return to the terminal.</p>")
        self.wfile.write(html.encode("utf-8"))
        try:
            self.server.result_queue.put((code, state, err))  # type: ignore[attr-defined]
        except Exception:
            pass


def _pick_port() -> int:
    """If config requested an explicit port use it; otherwise let the OS pick."""
    port = int(getattr(_cfg, "OAUTH_REDIRECT_PORT", 0))
    if port > 0:
        return port
    # Ask the OS for a free port by binding to 0, then close so the real
    # server can take it. There's a TOCTOU window but it's harmless for
    # interactive login — collisions just retry.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run_loopback_login() -> int:
    """Run the full interactive login. Returns a process exit code."""
    if not _cfg.APP_ID or not _cfg.APP_SECRET:
        print("[user-login] APP_ID / APP_SECRET 未配置，无法启动 OAuth", file=sys.stderr)
        return 2

    port = _pick_port()
    redirect_uri = f"http://127.0.0.1:{port}{_CALLBACK_PATH}"
    state = secrets.token_urlsafe(32)
    url = _build_authorize_url(redirect_uri, state)

    print(f"[user-login] redirect_uri = {redirect_uri}", file=sys.stderr)
    print("[user-login] 注意：飞书后台 → 应用 → 安全设置 → 重定向 URL 白名单",
          file=sys.stderr)
    print(f"[user-login] 必须把 {redirect_uri} 加入白名单后授权才会成功", file=sys.stderr)
    print(f"\n请在浏览器中打开（如未自动打开）：\n{url}\n", file=sys.stderr)

    q: "queue.Queue[tuple[str, str, str]]" = queue.Queue(maxsize=1)
    server = http.server.HTTPServer(("127.0.0.1", port), _CallbackHandler)
    server.result_queue = q  # type: ignore[attr-defined]
    server_thread = threading.Thread(target=server.serve_forever,
                                     name="oauth-loopback", daemon=True)
    server_thread.start()
    try:
        try:
            webbrowser.open(url)
        except Exception:
            pass  # User can copy-paste; not critical.
        try:
            code, recv_state, err = q.get(timeout=_LOGIN_TIMEOUT_SEC)
        except queue.Empty:
            print("[user-login] 超时（5 min），登录取消。", file=sys.stderr)
            return 3
        if err:
            print(f"[user-login] 授权被拒绝: {err}", file=sys.stderr)
            return 4
        if recv_state != state:
            print("[user-login] state 不匹配，疑似 CSRF，拒绝。", file=sys.stderr)
            return 5
        if not code:
            print("[user-login] 回调无 code。", file=sys.stderr)
            return 6
    finally:
        server.shutdown()
        server.server_close()

    try:
        token_data = _exchange_code(code, redirect_uri)
    except Exception as e:
        print(f"[user-login] code → token 失败: {e}", file=sys.stderr)
        return 7

    if not token_data.get("access_token"):
        print(f"[user-login] 飞书返回无 access_token: {token_data}", file=sys.stderr)
        return 8
    _save_token(token_data)
    print(f"\n✅ 已登录 open_id={token_data.get('open_id', '?')}", file=sys.stderr)
    print(f"   scope={token_data.get('scope', '?')}", file=sys.stderr)
    print(f"   expires in {token_data['expires_at'] - int(time.time())}s",
          file=sys.stderr)
    print(f"   token saved to {_cfg.USER_TOKEN_PATH}", file=sys.stderr)
    print("   重启 larkhelm 以让 bridge 进程感知到新 token：", file=sys.stderr)
    print("       sudo systemctl restart larkhelm", file=sys.stderr)
    return 0


def cli_login() -> int:
    """Entry point for ``larkhelm user-login``."""
    return _run_loopback_login()


def cli_logout() -> int:
    """Entry point for ``larkhelm user-logout``."""
    if not _cfg.USER_TOKEN_PATH.exists():
        print("[user-logout] 当前未登录。", file=sys.stderr)
        return 0
    clear_token()
    print("✅ 已登出，user_token 已清除。", file=sys.stderr)
    print("   重启 larkhelm 让 bridge 感知：sudo systemctl restart larkhelm",
          file=sys.stderr)
    return 0


def cli_status() -> int:
    """Entry point for ``larkhelm user-status``."""
    st = get_status()
    if not st["logged_in"]:
        print("🔒 未登录。运行 `larkhelm user-login` 开始授权。", file=sys.stderr)
        return 1
    print("🔓 已登录")
    print(f"   open_id            : {st['open_id']}")
    print(f"   scope              : {st['scope']}")
    print(f"   access_token  剩余 : {st['expires_in_sec']} 秒")
    print(f"   refresh_token 剩余 : {st['refresh_expires_in_sec']} 秒")
    return 0
