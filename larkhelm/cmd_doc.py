"""
larkhelm · Feishu document command implementations

Contains all implementation functions for the /doc and /doc wiki sub-command families.
"""
import threading

import larkhelm.config as _cfg
from larkhelm.chat_state import set_pending_doc_write, pop_pending_doc_write
from larkhelm.lark_client import send_card, send_permission_guide


# ═══════════════════════════════════════════════════
#  Help text
# ═══════════════════════════════════════════════════

_DOC_HELP = (
    "`/doc read <url>` — 读取文档内容\n"
    "`/doc append <url> <内容>` — 追加文本到文档末尾\n"
    "`/doc write <url> <内容>` — 替换文档全部内容（需确认）\n"
    "`/doc ls [url]` — 列举云盘文件夹内容\n"
    "`/doc create <标题> [url]` — 新建文档（可指定云盘文件夹或 wiki 父节点 URL）\n"
    "`/doc wiki list [space_id]` — 列出知识库节点\n"
    "`/doc wiki create <space_id> <标题> [parent_token]` — 在知识库创建新页面\n"
    "`/doc setfolder <url>` — 设置 crew/dev 文档默认存储位置（云盘文件夹链接）\n\n"
    "支持 docx / docs / wiki / sheets / drive/folder 链接。"
)


# ═══════════════════════════════════════════════════
#  /doc command family
# ═══════════════════════════════════════════════════

def _cmd_doc(chat_id: str, args: str):
    """Dispatch /doc sub-commands. args is the raw text after '/doc '."""
    parts = args.split(maxsplit=1)
    sub   = parts[0].lower() if parts else ""
    rest  = parts[1] if len(parts) > 1 else ""
    if sub == "read":
        threading.Thread(target=_cmd_doc_read,   args=(chat_id, rest.strip()), daemon=True).start()
    elif sub == "append":
        threading.Thread(target=_cmd_doc_append, args=(chat_id, rest.strip()), daemon=True).start()
    elif sub == "write":
        threading.Thread(target=_cmd_doc_write,  args=(chat_id, rest.strip()), daemon=True).start()
    elif sub == "ls":
        threading.Thread(target=_cmd_doc_ls,     args=(chat_id, rest.strip()), daemon=True).start()
    elif sub == "create":
        threading.Thread(target=_cmd_doc_create, args=(chat_id, rest.strip()), daemon=True).start()
    elif sub == "wiki":
        threading.Thread(target=_cmd_doc_wiki,   args=(chat_id, rest.strip()), daemon=True).start()
    elif sub == "setfolder":
        threading.Thread(target=_cmd_doc_setfolder, args=(chat_id, rest.strip()), daemon=True).start()
    else:
        send_card(chat_id, "📄 文档命令", _DOC_HELP, color="blue")


def _cmd_doc_setfolder(chat_id: str, url: str):
    """/doc setfolder <url> — Set the default storage folder for crew/dev documents."""
    from larkhelm.lark_client import parse_doc_url, DocError
    if not url:
        send_card(chat_id, "❌ 缺少参数",
                  "用法：`/doc setfolder <飞书云盘文件夹链接>`\n\n"
                  "在飞书云盘中打开目标文件夹，复制浏览器地址栏链接粘贴到此处。",
                  color="orange")
        return
    if url.lower() == "clear":
        _cfg.save_config_field("default_drive_folder", "")
        send_card(chat_id, "✅ 已清空", "默认文件夹设置已清除。", color="green")
        return
    try:
        ref = parse_doc_url(url)
    except (DocError, Exception) as e:
        send_card(chat_id, "❌ 链接解析失败", str(e), color="red")
        return

    if ref.doc_type != "folder":
        send_card(chat_id, "❌ 不是文件夹链接",
                  f"识别到的类型为 `{ref.doc_type}`，请粘贴云盘**文件夹**的链接\n"
                  f"（URL 形如 `https://xxx.feishu.cn/drive/folder/xxxxx`）",
                  color="orange")
        return

    _cfg.save_config_field("default_drive_folder", ref.token)
    send_card(chat_id, "✅ 默认文件夹已设置",
              f"后续 crew/dev 任务的中间文档将自动上传到该文件夹。\n\n"
              f"文件夹 token：`{ref.token}`\n\n"
              f"如需清空设置，发送：`/doc setfolder clear`",
              color="green")


def _cmd_doc_read(chat_id: str, url: str):
    """/doc read <url>"""
    from larkhelm.lark_client import (
        FeishuDocClient, parse_doc_url,
        DocError, DocNotFoundError, DocPermissionError, DocAPIError,
    )
    if not url:
        send_card(chat_id, "⚠️ 用法", "`/doc read <url>`", color="orange")
        return
    ref = parse_doc_url(url)
    if ref is None:
        send_card(chat_id, "⚠️ 无法识别", "不是有效的飞书文档链接。\n\n" + _DOC_HELP, color="orange")
        return
    try:
        result = FeishuDocClient().read(ref, max_chars=_cfg.DOC_READ_MAX_CHARS)
    except DocPermissionError as e:
        send_permission_guide(chat_id, "read_doc", code=e.code)
        return
    except DocNotFoundError:
        send_card(chat_id, "❌ 文档不存在", "文档已删除或链接失效。", color="red")
        return
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"飞书 API 返回错误 code={e.code}：{e.msg}", color="red")
        return
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")
        return
    note = f"（内容已截断至 {_cfg.DOC_READ_MAX_CHARS} 字，完整内容请访问文档）" if result.truncated else ""
    send_card(chat_id, f"📄 {result.title or '文档内容'}", result.content, color="blue", note=note)


def _cmd_doc_append(chat_id: str, rest: str):
    """/doc append <url> <content>"""
    from larkhelm.lark_client import (
        FeishuDocClient, parse_doc_url,
        DocError, DocNotFoundError, DocPermissionError,
        DocWriteNotSupportedError, DocAPIError,
    )
    # Split URL and content: everything before the first space is the URL
    parts = rest.split(maxsplit=1)
    if len(parts) < 2:
        send_card(chat_id, "⚠️ 用法", "`/doc append <url> <内容>`", color="orange")
        return
    url, content = parts[0], parts[1]
    ref = parse_doc_url(url)
    if ref is None:
        send_card(chat_id, "⚠️ 无法识别", "不是有效的飞书文档链接。", color="orange")
        return
    doc_client = FeishuDocClient()
    try:
        # Fetch title first for user feedback (docx reads directly; wiki/others parsed inside append)
        title = ref.title or url
        try:
            r = doc_client.read(ref, max_chars=1)
            title = r.title or url
        except Exception:
            pass
        doc_client.append(ref, content)
    except DocPermissionError as e:
        send_permission_guide(chat_id, "write_doc", code=e.code)
        return
    except DocWriteNotSupportedError as e:
        send_card(chat_id, "⚠️ 不支持写入", str(e), color="orange")
        return
    except DocNotFoundError:
        send_card(chat_id, "❌ 文档不存在", "文档已删除或链接失效。", color="red")
        return
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"飞书 API 返回错误 code={e.code}：{e.msg}", color="red")
        return
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")
        return
    send_card(chat_id, "✅ 追加成功",
              f"已追加 {len(content)} 字到《{title}》", color="green")


def _cmd_doc_write(chat_id: str, rest: str):
    """/doc write <url> <content> — Send a confirmation card first and stash the content."""
    from larkhelm.lark_client import parse_doc_url
    parts = rest.split(maxsplit=1)
    if len(parts) < 2:
        send_card(chat_id, "⚠️ 用法", "`/doc write <url> <内容>`", color="orange")
        return
    url, content = parts[0], parts[1]
    ref = parse_doc_url(url)
    if ref is None:
        send_card(chat_id, "⚠️ 无法识别", "不是有效的飞书文档链接。", color="orange")
        return
    if _cfg.DOC_WRITE_CONFIRM:
        set_pending_doc_write(chat_id, url, content, ref)
        preview = content[:200] + ("…" if len(content) > 200 else "")
        send_card(
            chat_id,
            "⚠️ 确认替换文档内容",
            f"**目标文档：** {url}\n\n"
            f"**新内容预览（前 200 字）：**\n```\n{preview}\n```\n\n"
            "**此操作将删除文档全部现有内容**，确认继续？",
            color="orange",
            buttons=[("确认替换", "doc_write_confirm"), ("取消", "doc_write_cancel")],
        )
    else:
        # Execute immediately (no confirmation required)
        set_pending_doc_write(chat_id, url, content, ref)
        _cmd_doc_write_do(chat_id)


def _cmd_doc_write_do(chat_id: str):
    """Execute the stashed document write operation (triggered by the confirm button)."""
    from larkhelm.lark_client import (
        FeishuDocClient,
        DocError, DocNotFoundError, DocPermissionError,
        DocWriteNotSupportedError, DocAPIError,
    )
    entry = pop_pending_doc_write(chat_id)
    if entry is None:
        send_card(chat_id, "⏰ 操作超时", "写入确认已过期，请重新执行 `/doc write`。", color="orange")
        return
    ref     = entry["ref"]
    content = entry["content"]
    url     = entry["url"]
    try:
        FeishuDocClient().replace_all(ref, content)
    except DocPermissionError as e:
        send_permission_guide(chat_id, "write_doc", code=e.code)
        return
    except DocWriteNotSupportedError as e:
        send_card(chat_id, "⚠️ 不支持写入", str(e), color="orange")
        return
    except DocNotFoundError:
        send_card(chat_id, "❌ 文档不存在", "文档已删除或链接失效。", color="red")
        return
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"飞书 API 返回错误 code={e.code}：{e.msg}", color="red")
        return
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")
        return
    send_card(chat_id, "✅ 写入成功",
              f"已替换《{url}》的全部内容（{len(content)} 字）", color="green")


def _cmd_doc_ls(chat_id: str, url: str):
    """/doc ls [url]"""
    from larkhelm.lark_client import (
        FeishuDocClient, parse_doc_url,
        DocError, DocPermissionError, DocAPIError,
    )
    if url:
        ref = parse_doc_url(url)
        if ref is None or ref.doc_type != "folder":
            send_card(chat_id, "⚠️ 无法识别",
                      "请提供云盘文件夹链接（`feishu.cn/drive/folder/...`）。", color="orange")
            return
        folder_token = ref.token
    else:
        folder_token = _cfg.DEFAULT_DRIVE_FOLDER
        if not folder_token:
            send_card(chat_id, "⚠️ 未配置",
                      "请提供文件夹 URL，或在配置中设置 `default_drive_folder`。", color="orange")
            return
    try:
        items = FeishuDocClient().list_folder(folder_token)
    except DocPermissionError as e:
        send_permission_guide(chat_id, "list_folder", code=e.code)
        return
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"飞书 API 返回错误 code={e.code}：{e.msg}", color="red")
        return
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")
        return
    if not items:
        send_card(chat_id, "📂 文件夹为空", "该文件夹中没有文件。", color="blue")
        return
    _TYPE_ICON = {"docx": "📄", "docs": "📃", "sheet": "📊", "folder": "📁", "file": "📎"}
    lines = [f"共 {len(items)} 项\n"]
    for it in items:
        icon = _TYPE_ICON.get(it.type, "📎")
        link = f"[{it.name}]({it.url})" if it.url else it.name
        lines.append(f"{icon} {link}  _({it.type})_")
    send_card(chat_id, "📂 文件夹内容", "\n".join(lines), color="blue")


def _cmd_doc_create(chat_id: str, rest: str):
    """/doc create <title> [folder_or_wiki_url]"""
    from larkhelm.lark_client import (
        FeishuDocClient, parse_doc_url,
        DocError, DocNotFoundError, DocPermissionError, DocAPIError,
    )
    parts = rest.split(maxsplit=1)
    if not parts or not parts[0]:
        send_card(chat_id, "⚠️ 用法",
                  "`/doc create <标题> [folder_url 或 wiki_url]`\n\n"
                  "- folder_url：云盘文件夹链接，创建普通 docx\n"
                  "- wiki_url：知识库页面链接，作为父节点在同一空间创建\n"
                  "- 不填：使用配置中的默认位置",
                  color="orange")
        return

    title = parts[0]
    url   = parts[1].strip() if len(parts) > 1 else ""

    doc_client        = FeishuDocClient()
    folder_token      = ""
    wiki_space_id     = ""
    wiki_parent_token = ""

    if url:
        ref = parse_doc_url(url)
        if ref is None:
            send_card(chat_id, "⚠️ 无法识别", "不是有效的飞书链接。", color="orange")
            return
        if ref.doc_type == "folder":
            folder_token = ref.token
        elif ref.doc_type == "wiki":
            try:
                node_info = doc_client._call_api(
                    "GET", f"/open-apis/wiki/v2/spaces/get_node?token={ref.token}"
                )
                node              = node_info.get("data", {}).get("node", {})
                wiki_space_id     = node.get("space_id", "")
                wiki_parent_token = ref.token
            except DocError as e:
                send_card(chat_id, "❌ 无法解析 wiki 节点", str(e), color="red")
                return
        else:
            send_card(chat_id, "⚠️ 不支持",
                      "目标位置仅支持云盘文件夹或知识库页面链接。", color="orange")
            return
    else:
        wiki_space_id     = _cfg.DEFAULT_WIKI_SPACE_ID
        wiki_parent_token = _cfg.DEFAULT_WIKI_PARENT_TOKEN
        folder_token      = _cfg.DEFAULT_DRIVE_FOLDER

    from larkhelm.chat_state import _get_chat_state
    sender_open_id = _get_chat_state(chat_id).get("sender_open_id", "")

    try:
        if wiki_space_id:
            doc_ref      = doc_client.create_wiki_node(wiki_space_id, title, wiki_parent_token)
            node_token   = doc_ref.raw_url.split("/")[-1]
            wiki_url     = f"https://feishu.cn/wiki/{node_token}"
            if sender_open_id:
                try:
                    doc_client.add_doc_member(doc_ref.token, sender_open_id)
                except Exception:
                    pass
            send_card(chat_id, "✅ 已创建知识库页面",
                      f"**标题：** {title}\n"
                      f"**文档 ID：** `{doc_ref.token}`\n"
                      f"**链接：** {wiki_url}",
                      color="green")
        else:
            doc_ref  = doc_client.create_doc(title, folder_token)
            doc_url  = f"https://feishu.cn/docx/{doc_ref.token}"
            if sender_open_id:
                try:
                    doc_client.add_doc_member(doc_ref.token, sender_open_id)
                except Exception:
                    pass
            send_card(chat_id, "✅ 已创建文档",
                      f"**标题：** {title}\n"
                      f"**文档 ID：** `{doc_ref.token}`\n"
                      f"**链接：** {doc_url}",
                      color="green")
    except DocPermissionError as e:
        send_permission_guide(chat_id, "create_folder", code=e.code)
    except DocNotFoundError:
        send_card(chat_id, "❌ 目标不存在", "文件夹或知识库空间不存在。", color="red")
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"code={e.code}：{e.msg}", color="red")
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")


def _cmd_doc_wiki(chat_id: str, rest: str):
    """/doc wiki <list|create> ..."""
    parts = rest.split(maxsplit=1)
    sub   = parts[0].lower() if parts else ""
    rest2 = parts[1].strip() if len(parts) > 1 else ""

    if sub == "list":
        threading.Thread(target=_cmd_doc_wiki_list,
                         args=(chat_id, rest2), daemon=True).start()
    elif sub == "create":
        threading.Thread(target=_cmd_doc_wiki_create,
                         args=(chat_id, rest2), daemon=True).start()
    else:
        send_card(chat_id, "📚 Wiki 命令",
                  "`/doc wiki list [space_id]` — 列出根节点\n"
                  "`/doc wiki create <space_id> <标题> [parent_token]` — 创建新页面",
                  color="blue")


def _cmd_doc_wiki_list(chat_id: str, rest: str):
    """/doc wiki list [space_id]"""
    from larkhelm.lark_client import (
        FeishuDocClient, DocError, DocPermissionError, DocAPIError,
    )
    space_id = rest.strip() or _cfg.DEFAULT_WIKI_SPACE_ID
    if not space_id:
        send_card(chat_id, "⚠️ 未配置",
                  "请提供 space_id，或在配置中设置 `default_wiki_space_id`。",
                  color="orange")
        return
    try:
        nodes = FeishuDocClient().list_wiki_nodes(space_id)
    except DocPermissionError as e:
        send_permission_guide(chat_id, "wiki_read", code=e.code)
        return
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"code={e.code}：{e.msg}", color="red")
        return
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")
        return

    if not nodes:
        send_card(chat_id, "📚 知识库为空", "该空间中没有节点。", color="blue")
        return

    _ICON = {"docx": "📄", "sheet": "📊", "mindnote": "🧠", "bitable": "🗃️"}
    lines = [f"共 {len(nodes)} 个节点（space_id={space_id}）\n"]
    for n in nodes:
        icon   = _ICON.get(n.get("obj_type", ""), "📎")
        title  = n.get("title", "(无标题)")
        token  = n.get("node_token", "")
        suffix = " ▶" if n.get("has_child") else ""
        lines.append(f"{icon} {title}  `{token}`{suffix}")
    send_card(chat_id, "📚 知识库节点", "\n".join(lines), color="blue")


def _cmd_doc_wiki_create(chat_id: str, rest: str):
    """/doc wiki create <space_id> <title> [parent_token]"""
    from larkhelm.lark_client import (
        FeishuDocClient, DocError, DocNotFoundError, DocPermissionError, DocAPIError,
    )
    parts = rest.split(maxsplit=2)
    if len(parts) < 2:
        send_card(chat_id, "⚠️ 用法",
                  "`/doc wiki create <space_id> <标题> [parent_token]`",
                  color="orange")
        return
    space_id     = parts[0]
    title        = parts[1]
    parent_token = parts[2].strip() if len(parts) > 2 else ""
    from larkhelm.chat_state import _get_chat_state
    sender_open_id = _get_chat_state(chat_id).get("sender_open_id", "")
    try:
        doc_client = FeishuDocClient()
        doc_ref    = doc_client.create_wiki_node(space_id, title, parent_token)
        node_token = doc_ref.raw_url.split("/")[-1]
        wiki_url   = f"https://feishu.cn/wiki/{node_token}"
        if sender_open_id:
            try:
                doc_client.add_doc_member(doc_ref.token, sender_open_id)
            except Exception:
                pass
        send_card(chat_id, "✅ 已创建知识库页面",
                  f"**标题：** {title}\n"
                  f"**文档 ID：** `{doc_ref.token}`\n"
                  f"**链接：** {wiki_url}",
                  color="green")
    except DocPermissionError as e:
        send_permission_guide(chat_id, "wiki_write", code=e.code)
    except DocNotFoundError:
        send_card(chat_id, "❌ 节点不存在", "父节点或空间不存在。", color="red")
    except DocAPIError as e:
        send_card(chat_id, "❌ API 错误", f"code={e.code}：{e.msg}", color="red")
    except DocError as e:
        send_card(chat_id, "❌ 错误", str(e), color="red")
