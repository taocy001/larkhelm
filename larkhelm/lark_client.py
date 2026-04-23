"""larkhelm · Feishu client wrapper (client, send_card, update_card, reply_card, and other Feishu API wrappers)."""
import json as _json_mod
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import lark_oapi as lark
from lark_oapi import BaseRequest, HttpMethod, AccessTokenType
from lark_oapi.api.im.v1 import (
    CreateMessageRequestBuilder, CreateMessageRequestBodyBuilder,
    PatchMessageRequestBuilder, PatchMessageRequestBodyBuilder,
    CreateMessageReactionRequestBuilder, CreateMessageReactionRequestBodyBuilder,
    DeleteMessageReactionRequestBuilder,
    EmojiBuilder,
    GetMessageResourceRequestBuilder,
    GetMessageRequest,
    CreatePinRequest, CreatePinRequestBody,
    DeletePinRequest,
)
from lark_oapi.api.im.v1.model.reply_message_request import ReplyMessageRequest
from lark_oapi.api.im.v1.model.reply_message_request_body import (
    ReplyMessageRequestBody, ReplyMessageRequestBodyBuilder)

import larkhelm.config as _cfg
from larkhelm.card_builder import _make_card, _split_md
from larkhelm.log import _debug_log

# ── Global client (assigned by main()) ─────────────────────────────
client: lark.Client  # noqa: F821
BOT_OPEN_ID: str = ""  # This bot's open_id, fetched at startup, used to filter group @mentions


def _fetch_bot_open_id() -> None:
    """Call /open-apis/bot/v3/info to fetch this bot's open_id and store it in the global variable."""
    global BOT_OPEN_ID
    try:
        req = BaseRequest()
        req.http_method = HttpMethod.GET
        req.uri = "/open-apis/bot/v3/info"
        req.token_types = {AccessTokenType.TENANT}
        req.body = None
        resp = client.request(req)
        if resp.raw:
            data = _json_mod.loads(resp.raw.content)
            bot_open_id = data.get("bot", {}).get("open_id", "")
            if bot_open_id:
                BOT_OPEN_ID = bot_open_id
                _debug_log(f"[Bot] 获取 open_id={bot_open_id}")
            else:
                _debug_log(f"[Bot] 未能获取 open_id，响应: {data}")
        else:
            _debug_log(f"[Bot] 获取 open_id 失败 code={resp.code} msg={resp.msg}")
    except Exception as e:
        _debug_log(f"[Bot] 获取 open_id 异常: {e}")


# ── Pin: track the currently pinned message_id per chat ─────────────
_pinned_mid: dict[str, str] = {}
_pinned_lock = threading.Lock()

# ── Retry index: last N bot messages mapped to (chat_id, prompt, model) ──
_REPLY_INDEX_MAX = 50
_reply_index: OrderedDict[str, tuple[str, str, str]] = OrderedDict()
_reply_index_lock = threading.Lock()

# ── Emoji constants ───────────────────────────────────────────────────
EMOJI_PROCESSING = "THINKING"
EMOJI_DONE       = "DONE"
EMOJI_ERROR      = "ERROR"

REACTION_ACTIONS = {
    "THUMBSUP": "positive",
    "LGTM":     "positive",
    "DONE":     "positive",
    "ERROR":    "retry",
    "MUSCLE":   "retry",
}


# ═══════════════════════════════════════════════════
#  Raw API operations
# ═══════════════════════════════════════════════════

def _send_text_raw(chat_id: str, text: str) -> str | None:
    """Send a plain-text message; used only as a fallback when card delivery fails."""
    try:
        resp = client.im.v1.message.create(
            CreateMessageRequestBuilder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBodyBuilder()
                .receive_id(chat_id)
                .msg_type("text")
                .content(_json_mod.dumps({"text": text[:4000]}, ensure_ascii=False))
                .build()
            ).build()
        )
        return resp.data.message_id if resp.success() else None
    except Exception:
        return None


def _send_card_raw(chat_id: str, card_json: str, _fallback_text: str | None = None) -> str | None:
    """Send a message using a pre-built card JSON; returns the message_id.
    _fallback_text: plain text to send as fallback if card delivery fails (None = no fallback).
    """
    try:
        resp = client.im.v1.message.create(
            CreateMessageRequestBuilder()
            .receive_id_type("chat_id")
            .request_body(
                CreateMessageRequestBodyBuilder()
                .receive_id(chat_id)
                .msg_type("interactive")
                .content(card_json)
                .build()
            ).build()
        )
        if not resp.success():
            _debug_log(f"[SendCard] 失败 code={resp.code}: {resp.msg}")
            if _fallback_text:
                _debug_log(f"[SendCard] 降级发送文本")
                return _send_text_raw(chat_id, _fallback_text)
            return None
        return resp.data.message_id
    except Exception as e:
        _debug_log(f"[SendCard] 异常: {e}")
        if _fallback_text:
            return _send_text_raw(chat_id, _fallback_text)
        return None


def _patch_card_raw(message_id: str | None, card_json: str) -> bool:
    """Update an existing message using a pre-built card JSON."""
    if not message_id:
        return False
    try:
        resp = client.im.v1.message.patch(
            PatchMessageRequestBuilder()
            .message_id(message_id)
            .request_body(
                PatchMessageRequestBodyBuilder()
                .content(card_json)
                .build()
            ).build()
        )
        return resp.success()
    except Exception as e:
        _debug_log(f"[PatchCard] 异常: {e}")
        return False


def _reply_card_raw(message_id: str, card_json: str, in_thread: bool = True) -> str | None:
    """Reply to a specified message with a card; returns the new message_id.
    in_thread=True: reply inside a thread; in_thread=False: quote-reply in the main chat stream.
    """
    try:
        body = (ReplyMessageRequestBodyBuilder()
                .msg_type("interactive")
                .content(card_json)
                .reply_in_thread(in_thread)
                .build())
        req = (ReplyMessageRequest.builder()
               .message_id(message_id)
               .request_body(body)
               .build())
        resp = client.im.v1.message.reply(req)
        if not resp.success():
            _debug_log(f"[ReplyCard] 失败 code={resp.code}: {resp.msg}")
            return None
        return resp.data.message_id
    except Exception as e:
        _debug_log(f"[ReplyCard] 异常: {e}")
        return None


# ═══════════════════════════════════════════════════
#  High-level card operations
# ═══════════════════════════════════════════════════

def send_card(chat_id: str, title: str, body: str,
              color: str = "blue", note: str = "",
              buttons: list[tuple[str, str]] | None = None,
              normalize: bool = True) -> str | None:
    chunk = _split_md(body.strip())[0]
    fallback = f"[{title}]\n{chunk.strip()[:500]}" if chunk.strip() else title
    return _send_card_raw(chat_id, _make_card(title, chunk.strip(), color, note, buttons, normalize=normalize),
                          _fallback_text=fallback)


def send_card_reply(chat_id: str, msg_id: str | None, title: str, body: str,
                    color: str = "blue", note: str = "",
                    buttons: list[tuple[str, str]] | None = None,
                    normalize: bool = True) -> str | None:
    """Send a card; if msg_id is set, send as a quote-reply to that message, otherwise send directly to the chat."""
    chunk = _split_md(body.strip())[0]
    card_json = _make_card(title, chunk.strip(), color, note, buttons, normalize=normalize)
    if msg_id:
        mid = _reply_card_raw(msg_id, card_json, in_thread=False)
        if mid:
            return mid
    fallback = f"[{title}]\n{chunk.strip()[:500]}" if chunk.strip() else title
    return _send_card_raw(chat_id, card_json, _fallback_text=fallback)


def update_card(message_id: str | None, title: str, body: str,
                color: str = "blue", note: str = "",
                buttons: list[tuple[str, str]] | None = None) -> bool:
    if not message_id:
        return False
    chunk = _split_md(body.strip())[0]
    return _patch_card_raw(message_id, _make_card(title, chunk.strip(), color, note, buttons))


def reply_card(chat_id: str, message_id: str | None,
               title: str, body: str, color: str = "blue", note: str = ""):
    chunks = _split_md(body.strip())
    first, rest = chunks[0], chunks[1:]
    final_note = note if not rest else ""
    card_json = _make_card(title, first.strip(), color, final_note)
    if message_id:
        if not _patch_card_raw(message_id, card_json):
            _send_card_raw(chat_id, card_json)
    else:
        _send_card_raw(chat_id, card_json)
    for i, chunk in enumerate(rest, 2):
        n = note if i == len(chunks) else ""
        send_card(chat_id, f"{title} ({i}/{len(chunks)})", chunk, color, n)


# ═══════════════════════════════════════════════════
#  Pin operations
# ═══════════════════════════════════════════════════

def _pin_message(message_id: str) -> bool:
    """Pin a message; returns True on success."""
    try:
        req = (CreatePinRequest.builder()
               .request_body(CreatePinRequestBody.builder().message_id(message_id).build())
               .build())
        resp = client.im.v1.pin.create(req)
        if not resp.success():
            _debug_log(f"[Pin] 失败 code={resp.code}: {resp.msg}")
            return False
        return True
    except Exception as e:
        _debug_log(f"[Pin] 异常: {e}")
        return False


def _unpin_message(message_id: str) -> bool:
    """Unpin a message; returns True on success."""
    try:
        req = DeletePinRequest.builder().message_id(message_id).build()
        resp = client.im.v1.pin.delete(req)
        if not resp.success():
            _debug_log(f"[Unpin] 失败 code={resp.code}: {resp.msg}")
            return False
        return True
    except Exception as e:
        _debug_log(f"[Unpin] 异常: {e}")
        return False


def _pin_task_card(chat_id: str, message_id: str):
    """Replace the chat's pinned message: unpin the previous one and pin the new one."""
    with _pinned_lock:
        old = _pinned_mid.get(chat_id)
        _pinned_mid[chat_id] = message_id
    if old and old != message_id:
        threading.Thread(target=_unpin_message, args=(old,), daemon=True).start()
    threading.Thread(target=_pin_message, args=(message_id,), daemon=True).start()


# ═══════════════════════════════════════════════════
#  Emoji reactions
# ═══════════════════════════════════════════════════

def react_to_message(message_id: str, emoji_type: str) -> str | None:
    """Add an emoji reaction to a message; returns the reaction_id or None on failure."""
    try:
        resp = client.im.v1.message_reaction.create(
            CreateMessageReactionRequestBuilder()
            .message_id(message_id)
            .request_body(
                CreateMessageReactionRequestBodyBuilder()
                .reaction_type(EmojiBuilder().emoji_type(emoji_type).build())
                .build()
            ).build()
        )
        if resp.success():
            return resp.data.reaction_id if resp.data else None
        _debug_log(f"[React] 失败 code={resp.code}: {resp.msg}")
        return None
    except Exception as e:
        _debug_log(f"[React] 异常: {e}")
        return None


def delete_reaction(message_id: str, reaction_id: str):
    """Remove an emoji reaction from a message."""
    try:
        resp = client.im.v1.message_reaction.delete(
            DeleteMessageReactionRequestBuilder()
            .message_id(message_id)
            .reaction_id(reaction_id)
            .build()
        )
        if not resp.success():
            _debug_log(f"[DeleteReact] 失败 code={resp.code}: {resp.msg}")
    except Exception as e:
        _debug_log(f"[DeleteReact] 异常: {e}")


# ═══════════════════════════════════════════════════
#  Image download
# ═══════════════════════════════════════════════════

def _download_image(image_key: str, chat_id: str, message_id: str) -> str | None:
    """Download a message image from Feishu, save to SESSION_DIR/chat_id/imgs/, return local path or None."""
    try:
        imgs_dir = _cfg.SESSION_DIR / chat_id / "imgs"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        safe_key = re.sub(r"[^a-zA-Z0-9_\-.]", "_", image_key)
        out_path = imgs_dir / f"{safe_key}.jpg"
        if out_path.exists():
            return str(out_path)
        resp = client.im.v1.message_resource.get(
            GetMessageResourceRequestBuilder()
            .message_id(message_id)
            .file_key(image_key)
            .type("image")
            .build()
        )
        if not resp.success():
            _debug_log(f"[Image] 下载失败 image_key={image_key} code={resp.code}: {resp.msg}")
            return None
        file_data = resp.file
        if hasattr(file_data, "read"):
            file_data = file_data.read()
        with open(out_path, "wb") as f:
            f.write(file_data)
        _debug_log(f"[Image] 下载成功 -> {out_path}")
        return str(out_path)
    except Exception as e:
        _debug_log(f"[Image] 下载异常: {e}")
        return None


# ═══════════════════════════════════════════════════
#  Quoted message fetching
# ═══════════════════════════════════════════════════

# Local cache to avoid re-fetching the same message (invalidated on restart, correctness unaffected)
_parent_msg_cache: OrderedDict[str, str] = OrderedDict()
_parent_msg_cache_lock = threading.Lock()
_PARENT_MSG_CACHE_MAX = 200


def _fetch_parent_message_text(message_id: str) -> str:
    """Fetch a Feishu message's plain-text content by message_id (up to 800 chars).
    Supports text / post / interactive (card) message types; returns empty string silently on failure.
    """
    with _parent_msg_cache_lock:
        cached = _parent_msg_cache.get(message_id)
    if cached is not None:
        return cached

    try:
        resp = client.im.v1.message.get(
            GetMessageRequest.builder().message_id(message_id).build()
        )
        if not resp.success() or not resp.data or not resp.data.items:
            return ""
        item = resp.data.items[0]
        msg_type = item.msg_type or ""
        raw_content = item.body.content if item.body else ""
        if not raw_content:
            return ""

        text = ""
        if msg_type == "text":
            text = _json_mod.loads(raw_content).get("text", "").strip()
        elif msg_type == "post":
            data = _json_mod.loads(raw_content)
            content = data.get("content", {})
            # post format: {"content": {"zh_cn": {"title": ..., "content": [[...]]}}}
            # or new format: {"content": [[...]]}
            if isinstance(content, dict):
                lang_data = next(iter(content.values()), {})
                paragraphs = lang_data.get("content", [])
                title = lang_data.get("title", "")
            else:
                paragraphs = content
                title = data.get("title", "")
            parts = [title] if title else []
            for para in paragraphs:
                for elem in para:
                    if elem.get("tag") == "text":
                        parts.append(elem.get("text", ""))
            text = " ".join(parts).strip()
        elif msg_type == "interactive":
            # Card message: attempt to extract markdown text content
            data = _json_mod.loads(raw_content)
            parts: list[str] = []
            # JSON 1.0
            for el in data.get("elements", []):
                if el.get("tag") == "div":
                    t = el.get("text", {})
                    if isinstance(t, dict):
                        parts.append(t.get("content", ""))
            # JSON 2.0
            for el in data.get("body", {}).get("elements", []):
                if el.get("tag") == "markdown":
                    parts.append(el.get("content", ""))
                elif el.get("tag") == "collapsible_panel":
                    for sub in el.get("elements", []):
                        if sub.get("tag") == "markdown":
                            parts.append(sub.get("content", ""))
            text = "\n".join(p for p in parts if p).strip()

        # Truncate to avoid overly long context
        result = text[:800] + ("…" if len(text) > 800 else "")

        with _parent_msg_cache_lock:
            _parent_msg_cache[message_id] = result
            while len(_parent_msg_cache) > _PARENT_MSG_CACHE_MAX:
                _parent_msg_cache.popitem(last=False)
        return result

    except Exception as e:
        _debug_log(f"[ParentMsg] 拉取 {message_id[:12]} 失败: {e}")
        return ""


# ═══════════════════════════════════════════════════
#  Retry index
# ═══════════════════════════════════════════════════

def _index_reply(msg_id: str, chat_id: str, prompt: str, model: str):
    """Record a bot reply message so users can trigger a retry via emoji reaction."""
    with _reply_index_lock:
        _reply_index[msg_id] = (chat_id, prompt, model)
        while len(_reply_index) > _REPLY_INDEX_MAX:
            _reply_index.popitem(last=False)


# ═══════════════════════════════════════════════════
#  Feishu document read/write client
# ═══════════════════════════════════════════════════

DocType = Literal["docx", "docs", "wiki", "sheets", "folder", "unknown"]

_DOC_URL_PATTERNS = [
    (re.compile(r'feishu\.cn/docx/([A-Za-z0-9_-]+)'),          "docx"),
    (re.compile(r'feishu\.cn/docs/([A-Za-z0-9_-]+)'),          "docs"),
    (re.compile(r'feishu\.cn/wiki/([A-Za-z0-9_-]+)'),          "wiki"),
    (re.compile(r'feishu\.cn/sheets/([A-Za-z0-9_-]+)'),        "sheets"),
    (re.compile(r'feishu\.cn/drive/folder/([A-Za-z0-9_-]+)'),  "folder"),
]


@dataclass
class DocRef:
    doc_type: DocType
    token: str
    raw_url: str
    title: str = ""


@dataclass
class DocReadResult:
    title: str
    content: str
    truncated: bool
    doc_type: DocType
    token: str


@dataclass
class FolderItem:
    name: str
    token: str
    type: str
    url: str
    modified_time: str


class DocError(Exception):                 pass
class DocNotFoundError(DocError):          pass
class DocWriteNotSupportedError(DocError): pass


class DocPermissionError(DocError):
    def __init__(self, msg: str = "", code: int = 0):
        super().__init__(msg)
        self.code = code


class DocAPIError(DocError):
    def __init__(self, code: int, msg: str):
        super().__init__(f"API 错误 code={code}: {msg}")
        self.code = code
        self.msg  = msg


def parse_doc_url(url: str) -> "DocRef | None":
    """Extract a DocRef from a Feishu URL; returns None if unrecognised."""
    for pattern, doc_type in _DOC_URL_PATTERNS:
        m = pattern.search(url)
        if m:
            return DocRef(doc_type=doc_type, token=m.group(1), raw_url=url)
    return None


class FeishuDocClient:
    """Feishu document read/write client. Reuses the module-level global client."""

    # ── URL parsing ─────────────────────────────────────────────

    @staticmethod
    def parse_url(url: str) -> "DocRef | None":
        return parse_doc_url(url)

    def resolve_wiki(self, node_token: str) -> "DocRef":
        """Resolve a wiki node_token to a DocRef pointing at the underlying document."""
        data  = self._call_api("GET", f"/open-apis/wiki/v2/spaces/get_node?token={node_token}")
        node  = data.get("data", {}).get("node", {})
        obj_token = node.get("obj_token", "")
        obj_type  = node.get("obj_type", "")
        if not obj_token:
            raise DocNotFoundError(f"wiki node {node_token!r} 未找到 obj_token")
        type_map = {"doc": "docs", "docx": "docx", "sheet": "sheets"}
        mapped   = type_map.get(obj_type, "unknown")
        if mapped == "unknown":
            raise DocAPIError(0, f"不支持的 wiki obj_type: {obj_type!r}")
        return DocRef(doc_type=mapped, token=obj_token, raw_url=f"wiki/{node_token}")

    # ── Read ─────────────────────────────────────────────────────

    def read(self, ref: "DocRef", max_chars: int = 8000) -> "DocReadResult":
        """Read document plain-text content; truncates at max_chars and sets truncated=True."""
        if ref.doc_type == "wiki":
            ref = self.resolve_wiki(ref.token)
        if ref.doc_type == "docx":
            title, content = self._docx_raw_content(ref.token)
        elif ref.doc_type == "docs":
            title, content = self._docs_content(ref.token)
        elif ref.doc_type == "sheets":
            title, content = self._sheets_summary(ref.token)
        elif ref.doc_type == "folder":
            items   = self.list_folder(ref.token)
            title   = "云盘文件夹"
            content = "\n".join(f"- [{it.name}]({it.url})  _{it.type}_" for it in items)
        else:
            raise DocAPIError(0, f"不支持的文档类型: {ref.doc_type!r}")
        truncated = len(content) > max_chars
        return DocReadResult(
            title=title, content=content[:max_chars] if truncated else content,
            truncated=truncated, doc_type=ref.doc_type, token=ref.token,
        )

    def list_folder(self, folder_token: str, max_items: int = 50) -> "list[FolderItem]":
        """List the contents of a Drive folder."""
        data  = self._call_api("GET",
            f"/open-apis/drive/v1/files?folder_token={folder_token}&page_size={max_items}")
        items = data.get("data", {}).get("files", [])
        return [
            FolderItem(
                name=it.get("name", ""),
                token=it.get("token", ""),
                type=it.get("type", ""),
                url=it.get("url", ""),
                modified_time=it.get("modified_time", ""),
            )
            for it in items[:max_items]
        ]

    # ── Write (docx only) ────────────────────────────────────────

    def append(self, ref: "DocRef", content: str) -> bool:
        """Append Markdown text to the end of a document. Only supports docx type."""
        if ref.doc_type == "wiki":
            ref = self.resolve_wiki(ref.token)
        if ref.doc_type != "docx":
            raise DocWriteNotSupportedError(
                f"不支持写入类型 {ref.doc_type!r}，仅支持 docx")
        doc_id = ref.token
        blocks = self._md_to_blocks(content)
        if not blocks:
            return True
        # Note: SDK client.request(raw) double-serialises bytes bodies (causes error 9499).
        # Use direct HTTP requests to work around this.
        self._append_blocks_http(doc_id, blocks)
        return True

    def _append_blocks_http(self, doc_id: str, blocks: list) -> None:
        """Append blocks to the end of a document via direct HTTP calls (batched, max 50 per batch)."""
        import urllib.request as _urllib_req
        token = self._get_tenant_token()
        url = f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{doc_id}/children"
        batch_size = 50
        for i in range(0, len(blocks), batch_size):
            batch = blocks[i:i + batch_size]
            body  = _json_mod.dumps({"children": batch, "index": -1}, ensure_ascii=False).encode()
            req   = _urllib_req.Request(url, data=body, headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json; charset=utf-8",
            })
            try:
                with _urllib_req.urlopen(req, timeout=30) as resp:
                    data = _json_mod.loads(resp.read())
                    if data.get("code", 0) != 0:
                        raise DocAPIError(data["code"], data.get("msg", ""))
            except DocAPIError:
                raise
            except Exception as e:
                raise DocAPIError(0, f"HTTP append 失败: {e}")

    def _get_tenant_token(self) -> str:
        """Fetch a tenant_access_token (fetched live each time; TTL is controlled by Feishu)."""
        import urllib.request as _urllib_req
        import larkhelm.config as _cfg_mod
        body = _json_mod.dumps({
            "app_id": _cfg_mod.APP_ID,
            "app_secret": _cfg_mod.APP_SECRET,
        }).encode()
        req = _urllib_req.Request(
            "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with _urllib_req.urlopen(req, timeout=10) as resp:
            data = _json_mod.loads(resp.read())
        token = data.get("tenant_access_token", "")
        if not token:
            raise DocAPIError(0, "获取 tenant_access_token 失败")
        return token

    def replace_all(self, ref: "DocRef", content: str) -> bool:
        """Delete all blocks in the document then rewrite. Destructive operation — must be confirmed by user before calling."""
        if ref.doc_type == "wiki":
            ref = self.resolve_wiki(ref.token)
        if ref.doc_type != "docx":
            raise DocWriteNotSupportedError(
                f"不支持写入类型 {ref.doc_type!r}，仅支持 docx")
        doc_id   = ref.token
        children = self._docx_list_children(doc_id, doc_id)
        if children:
            # Same as append: SDK client.request double-serialises bytes bodies (9499), use direct HTTP instead
            self._http_request("DELETE",
                f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{doc_id}/children/batch_delete",
                {"start_index": 0, "end_index": len(children)})
        blocks = self._md_to_blocks(content)
        if blocks:
            self._http_request("POST",
                f"https://open.feishu.cn/open-apis/docx/v1/documents/{doc_id}/blocks/{doc_id}/children",
                {"children": blocks, "index": 0})
        return True

    def _http_request(self, method: str, url: str, body: dict) -> dict:
        """Direct HTTP request (bypasses SDK double-encode issue); returns the response dict."""
        import urllib.request as _urllib_req
        token     = self._get_tenant_token()
        data_bytes = _json_mod.dumps(body, ensure_ascii=False).encode()
        req = _urllib_req.Request(url, data=data_bytes, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json; charset=utf-8",
        }, method=method)
        try:
            with _urllib_req.urlopen(req, timeout=30) as resp:
                data = _json_mod.loads(resp.read())
        except Exception as e:
            raise DocAPIError(0, f"HTTP {method} 失败: {e}")
        code = data.get("code", 0)
        if code != 0:
            raise DocAPIError(code, data.get("msg", ""))
        return data

    # ── Internal helpers ─────────────────────────────────────────

    def _docx_raw_content(self, doc_id: str) -> "tuple[str, str]":
        """Return (title, plain_text) for a docx document."""
        doc_data     = self._call_api("GET", f"/open-apis/docx/v1/documents/{doc_id}")
        title        = doc_data.get("data", {}).get("document", {}).get("title", "")
        content_data = self._call_api("GET", f"/open-apis/docx/v1/documents/{doc_id}/raw_content")
        content      = content_data.get("data", {}).get("content", "")
        return title, content

    def _docs_content(self, doc_id: str) -> "tuple[str, str]":
        """Fetch legacy (v1) document content (read-only)."""
        data  = self._call_api("GET", f"/open-apis/docs/v1/documents/{doc_id}/content")
        doc   = data.get("data", {})
        title = doc.get("title", "")
        raw   = doc.get("content", "")
        try:
            obj     = _json_mod.loads(raw) if isinstance(raw, str) else raw
            content = self._extract_docs_text(obj)
        except Exception:
            content = str(raw)
        return title, content

    def _extract_docs_text(self, node) -> str:
        """Recursively extract plain text from a legacy document JSON node."""
        if isinstance(node, str):
            return node
        if isinstance(node, dict):
            if "text" in node and isinstance(node["text"], str):
                return node["text"]
            return "\n".join(filter(None, (self._extract_docs_text(v) for v in node.values())))
        if isinstance(node, list):
            return "\n".join(filter(None, (self._extract_docs_text(i) for i in node)))
        return ""

    def _sheets_summary(self, spreadsheet_token: str) -> "tuple[str, str]":
        """Read spreadsheet metadata and a summary of the first 20 rows of each sheet."""
        meta  = self._call_api("GET",
            f"/open-apis/sheets/v3/spreadsheets/{spreadsheet_token}")
        title = meta.get("data", {}).get("spreadsheet", {}).get("title", "表格")
        smeta = self._call_api("GET",
            f"/open-apis/sheets/v2/spreadsheets/{spreadsheet_token}/metainfo")
        sheets = smeta.get("data", {}).get("sheets", [])
        lines  = [f"# {title}", ""]
        for sh in sheets[:5]:
            sh_title = sh.get("title", "")
            sh_id    = sh.get("sheetId", "")
            lines.append(f"## Sheet: {sh_title}")
            try:
                rng  = f"{sh_id}!A1:Z20"
                rdata = self._call_api("GET",
                    f"/open-apis/sheets/v2/spreadsheets/{spreadsheet_token}/values/{rng}")
                rows = rdata.get("data", {}).get("valueRange", {}).get("values", [])
                for row in rows[:20]:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
            except Exception:
                lines.append("_（无法读取数据）_")
            lines.append("")
        return title, "\n".join(lines)

    def _docx_list_children(self, doc_id: str, block_id: str) -> list:
        """Return the list of direct child block IDs for a given block."""
        data = self._call_api("GET",
            f"/open-apis/docx/v1/documents/{doc_id}/blocks/{block_id}/children")
        return [c.get("block_id", "")
                for c in data.get("data", {}).get("items", [])
                if c.get("block_id")]

    def _md_to_blocks(self, content: str) -> list:
        """Convert Markdown text to a list of Feishu docx Block JSON objects.
        Supports: paragraphs, H1-H3 headings, code blocks, unordered lists, ordered lists.
        """
        blocks = []
        lines  = content.split("\n")
        i = 0
        while i < len(lines):
            line = lines[i]
            # Code block
            if line.startswith("```"):
                lang       = line[3:].strip()
                code_lines = []
                i += 1
                while i < len(lines) and not lines[i].startswith("```"):
                    code_lines.append(lines[i])
                    i += 1
                blocks.append(self._make_code_block("\n".join(code_lines), lang))
                i += 1
                continue
            # Heading
            if line.startswith("### "):
                blocks.append(self._make_heading_block(line[4:], 3))
            elif line.startswith("## "):
                blocks.append(self._make_heading_block(line[3:], 2))
            elif line.startswith("# "):
                blocks.append(self._make_heading_block(line[2:], 1))
            # Unordered list
            elif re.match(r'^[*\-] ', line):
                blocks.append(self._make_bullet_block(line[2:]))
            # Ordered list
            elif re.match(r'^\d+\. ', line):
                blocks.append(self._make_ordered_block(re.sub(r'^\d+\. ', '', line)))
            # Blank line — skip
            elif not line.strip():
                pass
            # Plain paragraph
            else:
                blocks.append(self._make_paragraph_block(line))
            i += 1
        return blocks

    @staticmethod
    def _text_elem(text: str) -> dict:
        return {"type": "text_run", "text_run": {"content": text, "text_element_style": {}}}

    def _make_paragraph_block(self, text: str) -> dict:
        return {"block_type": 2, "text": {
            "elements": [self._text_elem(text)], "style": {}}}

    def _make_heading_block(self, text: str, level: int) -> dict:
        # H1=block_type 3, H2=4, H3=5 (Feishu block_type mapping)
        bt  = 2 + level
        key = {3: "heading1", 4: "heading2", 5: "heading3"}[bt]
        return {"block_type": bt, key: {
            "elements": [self._text_elem(text)], "style": {}}}

    def _make_bullet_block(self, text: str) -> dict:
        return {"block_type": 12, "bullet": {
            "elements": [self._text_elem(text)], "style": {}}}

    def _make_ordered_block(self, text: str) -> dict:
        return {"block_type": 13, "ordered": {
            "elements": [self._text_elem(text)], "style": {}}}

    def _make_code_block(self, code: str, lang: str = "") -> dict:
        _LANG_MAP = {
            "python": 49, "py": 49, "javascript": 22, "js": 22,
            "typescript": 50, "ts": 50, "go": 16, "rust": 51,
            "java": 21, "c": 9, "cpp": 10, "c++": 10,
            "bash": 7, "sh": 7, "shell": 7,
            "json": 25, "yaml": 53, "markdown": 29, "md": 29,
            "sql": 47, "html": 18, "css": 12, "xml": 52,
        }
        return {"block_type": 14, "code": {
            "elements": [self._text_elem(code)],
            "style": {"language": _LANG_MAP.get(lang.lower(), 1), "wrap": False},
        }}

    def add_doc_member(self, doc_token: str, open_id: str, doc_type: str = "docx") -> None:
        """Grant full_access to a user (by open_id) on a Drive document."""
        import urllib.request as _urllib_req
        token = self._get_tenant_token()
        url   = (f"https://open.feishu.cn/open-apis/drive/v1/permissions"
                 f"/{doc_token}/members?type={doc_type}&need_notification=false")
        body  = _json_mod.dumps({
            "member_type": "openid",
            "member_id":   open_id,
            "perm":        "full_access",
        }, ensure_ascii=False).encode()
        req = _urllib_req.Request(url, data=body, headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json; charset=utf-8",
        })
        with _urllib_req.urlopen(req, timeout=10) as resp:
            data = _json_mod.loads(resp.read())
        code = data.get("code", 0)
        if code != 0:
            raise DocAPIError(code, data.get("msg", "add_doc_member failed"))

    # ── Create document / wiki node ──────────────────────────────

    def create_doc(self, title: str, folder_token: str = "") -> "DocRef":
        """Create a blank docx document in the specified Drive folder (or root if empty).
        Uses the SDK native interface (raw request has a double-encode issue for docx create).
        """
        from lark_oapi.api.docx.v1 import (
            CreateDocumentRequest, CreateDocumentRequestBody,
        )
        body_builder = CreateDocumentRequestBody.builder().title(title)
        if folder_token:
            body_builder = body_builder.folder_token(folder_token)
        resp = client.docx.v1.document.create(
            CreateDocumentRequest.builder()
            .request_body(body_builder.build())
            .build()
        )
        if not resp.success():
            # 1770040 = no write permission on the folder
            if resp.code in (1770040, 99991663, 99991661, 230001, 403):
                raise DocPermissionError(f"文件夹权限不足 code={resp.code}: {resp.msg}", code=resp.code)
            raise DocAPIError(resp.code, resp.msg or "create_doc failed")
        doc_id = resp.data.document.document_id if resp.data and resp.data.document else ""
        if not doc_id:
            raise DocAPIError(0, "创建文档响应缺少 document_id")
        return DocRef(doc_type="docx", token=doc_id, raw_url="", title=title)

    def create_wiki_node(
        self,
        space_id: str,
        title: str,
        parent_node_token: str = "",
        obj_type: str = "docx",
    ) -> "DocRef":
        """Create a new node in a wiki space, also creating the underlying document."""
        body: dict = {
            "obj_type":  obj_type,
            "node_type": "origin",
            "title":     title,
        }
        if parent_node_token:
            body["parent_node_token"] = parent_node_token
        data = self._call_api(
            "POST", f"/open-apis/wiki/v2/spaces/{space_id}/nodes", body
        )
        node        = data.get("data", {}).get("node", {})
        node_token  = node.get("node_token", "")
        obj_token   = node.get("obj_token",  "")
        if not obj_token:
            raise DocAPIError(0, f"创建 wiki 节点响应缺少 obj_token (node={node_token})")
        return DocRef(
            doc_type="docx",
            token=obj_token,
            raw_url=f"wiki/{node_token}",
            title=title,
        )

    def list_wiki_nodes(
        self,
        space_id: str,
        parent_node_token: str = "",
        max_items: int = 50,
    ) -> list:
        """List nodes in a wiki space (auto-paginated, up to max_items).
        Returns: list[dict], each item has title / node_token / obj_token / obj_type / has_child.
        """
        all_nodes: list = []
        page_token: str = ""
        while len(all_nodes) < max_items:
            page_size = min(50, max_items - len(all_nodes))
            params    = f"page_size={page_size}"
            if parent_node_token:
                params += f"&parent_node_token={parent_node_token}"
            if page_token:
                params += f"&page_token={page_token}"
            data  = self._call_api("GET", f"/open-apis/wiki/v2/spaces/{space_id}/nodes?{params}")
            items = data.get("data", {}).get("items") or []
            all_nodes.extend(items)
            if not data.get("data", {}).get("has_more"):
                break
            page_token = data.get("data", {}).get("page_token", "")
            if not page_token:
                break
        return all_nodes[:max_items]

    def _call_api(self, method: str, uri: str, body: dict | None = None) -> dict:
        """Generic lark_oapi raw request wrapper; raises the appropriate exception when code != 0."""
        req = BaseRequest()
        req.http_method  = getattr(HttpMethod, method.upper())
        req.uri          = uri
        req.token_types  = {AccessTokenType.TENANT}
        req.body         = _json_mod.dumps(body, ensure_ascii=False).encode() if body is not None else None
        resp = client.request(req)
        if not resp.raw:
            raise DocAPIError(0, f"空响应 uri={uri}")
        data = _json_mod.loads(resp.raw.content)
        code = data.get("code", 0)
        if code == 0:
            return data
        msg = data.get("msg", "")
        # Permission error
        if code in (99991663, 99991661, 230001, 99991401, 403):
            raise DocPermissionError(f"权限不足 code={code}: {msg}", code=code)
        # Resource not found
        if code in (99991664, 1069901, 1003, 404):
            raise DocNotFoundError(f"文档不存在 code={code}: {msg}")
        raise DocAPIError(code, msg)


# ── Permission guide card ──────────────────────────────────────────────────────

# Error code classification: scope missing (requires developer console approval) vs resource not shared (share with the app)
_SCOPE_MISSING_CODES  = {99991663, 99991661, 230001, 99991401}
_RESOURCE_ACCESS_CODES = {1770040, 403}

# operation → (card_title, resource_fix_step, scope_name) lookup table
_PERM_GUIDES: "dict[str, tuple[str, str, str]]" = {
    "read_doc": (
        "🔒 文档读取权限不足",
        "打开飞书文档 → 右上角**分享** → 邀请协作者 → 搜索并添加本应用（**可查看**）",
        "docx:document:readonly（文档只读）",
    ),
    "write_doc": (
        "🔒 文档写入权限不足",
        "打开飞书文档 → 右上角**分享** → 邀请协作者 → 搜索并添加本应用（**可编辑**）",
        "docx:document（文档读写）",
    ),
    "create_folder": (
        "🔒 云盘文件夹写入权限不足",
        "打开飞书云盘 → 右键目标文件夹 → **共享** → 邀请协作者 → 添加本应用（**可编辑**）",
        "drive:drive（云盘文件读写）",
    ),
    "list_folder": (
        "🔒 云盘文件夹读取权限不足",
        "打开飞书云盘 → 右键目标文件夹 → **共享** → 邀请协作者 → 添加本应用（**可查看**）",
        "drive:drive:readonly（云盘文件只读）",
    ),
    "wiki_read": (
        "🔒 知识库读取权限不足",
        "进入知识库 → **设置** → **成员** → 添加本应用为成员（**阅读者**）",
        "wiki:wiki:readonly（知识库只读）",
    ),
    "wiki_write": (
        "🔒 知识库写入权限不足",
        "进入知识库 → **设置** → **成员** → 添加本应用为成员（**编辑者**）",
        "wiki:wiki:write（知识库读写）",
    ),
}


def send_permission_guide(chat_id: str, operation: str, code: int = 0) -> None:
    """Send a permission-insufficient guide card to a Feishu chat, explaining which permission is missing and the exact fix steps."""
    import larkhelm.config as _cfg_mod

    title, resource_fix, scope_name = _PERM_GUIDES.get(operation, (
        "🔒 飞书权限不足",
        "请将目标资源（文档/文件夹/知识库）共享给本应用",
        "对应操作所需的 API 权限",
    ))

    app_id      = _cfg_mod.APP_ID
    console_url = f"https://open.feishu.cn/app/{app_id}/permission/api"

    # Determine the primary cause from the error code to decide which path to show
    is_scope    = code in _SCOPE_MISSING_CODES
    is_resource = code in _RESOURCE_ACCESS_CODES or (code == 0 and not is_scope)

    parts: list[str] = []

    if is_scope:
        parts += [
            "**原因：** 应用尚未获得该 API 权限，需管理员在开发者后台审批。",
            "",
            "**修复步骤（开发者后台）**",
            f"1. 打开[应用权限管理]({console_url})",
            f"2. 搜索并开启 `{scope_name}`",
            "3. 联系企业管理员审批",
        ]
    else:
        parts += [
            "**原因：** 目标资源未对本应用开放访问权限。",
            "",
            "**修复步骤（共享资源）**",
            resource_fix,
            "",
            f"若持续失败，也可在[开发者后台]({console_url})检查 `{scope_name}` 是否已启用。",
        ]

    if code:
        parts += ["", f"_错误码：{code}_"]

    send_card(chat_id, title, "\n".join(parts), color="red")
