"""
larkhelm · Feishu card builder utilities

Contains:
  - _make_card()          Build card JSON (JSON 1.0 with buttons / JSON 2.0 without)
  - _split_md()           Split Markdown at MAX_CARD_LEN boundaries
  - _normalize_newlines() Insert blank lines outside code blocks for Feishu paragraph rendering
  - _btn_type()           Determine button style from its label
  - _fmt_elapsed()        Format elapsed seconds as a human-readable string
"""
import json

import larkhelm.config as _cfg


# ═══════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════

def _fmt_elapsed(seconds: float) -> str:
    return f"{seconds:.0f}s" if seconds < 60 else f"{seconds/60:.1f}m"


def _btn_type(label: str) -> str:
    if any(k in label for k in ("允许", "确认", "✅", "同意", "Yes", "OK")):
        return "primary"
    if any(k in label for k in ("拒绝", "取消", "删除", "❌", "No", "Deny")):
        return "danger"
    return "default"


def _normalize_newlines(text: str) -> str:
    """Convert single newlines outside code blocks to double newlines (Feishu card Markdown requires double newlines for paragraph breaks)."""
    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    for line in lines:
        # Use stripped form so indented code fences (e.g. inside lists) are detected correctly
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if in_code:
            result.append(line)
        else:
            if result and result[-1] != "" and line != "":
                result.append("")
            result.append(line)
    return "\n".join(result)


def _split_md(text: str) -> list[str]:
    if len(text) <= _cfg.MAX_CARD_LEN:
        return [text]
    chunks, buf, buf_len, in_code = [], [], 0, False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        line_len = len(line) + 1
        # Single line longer than the limit: flush buffer then hard-split the line
        if line_len > _cfg.MAX_CARD_LEN:
            if buf:
                chunks.append("\n".join(buf))
                buf, buf_len = [], 0
            remainder = line
            while len(remainder) > _cfg.MAX_CARD_LEN:
                chunks.append(remainder[:_cfg.MAX_CARD_LEN])
                remainder = remainder[_cfg.MAX_CARD_LEN:]
            if remainder:
                buf.append(remainder)
                buf_len = len(remainder) + 1
            continue
        if buf_len + line_len > _cfg.MAX_CARD_LEN and buf and not in_code:
            chunks.append("\n".join(buf))
            buf, buf_len = [], 0
        buf.append(line)
        buf_len += line_len
    if buf:
        chunks.append("\n".join(buf))
    return chunks or [text]


# ═══════════════════════════════════════════════════
#  Card builder
# ═══════════════════════════════════════════════════

def _make_card(title: str, body: str, color: str = "blue", note: str = "",
               buttons: list[tuple[str, str]] = None,
               subtitle: str = "",
               tools_md: str = None,
               tools_expanded: bool = False,
               tools_list: list = None,
               normalize: bool = True) -> str:
    """
    With buttons  → JSON 1.0 (Feishu schema 2.0 body.elements does not support action tags)
    Without buttons → JSON 2.0 (supports code blocks, collapsible panels, quotes, etc.)
    """
    if buttons:
        # ── JSON 1.0: supports interactive buttons ──────────────────────────
        elements: list = []
        if tools_md:
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": tools_md}})
            elements.append({"tag": "hr"})
        if body.strip():
            elements.append({"tag": "div", "text": {"tag": "lark_md", "content": body}})
        elements.append({"tag": "hr"})
        elements.append({
            "tag": "action",
            "actions": [
                {"tag": "button", "text": {"tag": "plain_text", "content": label},
                 "type": _btn_type(label), "value": {"cmd": cmd}}
                for label, cmd in buttons
            ],
        })
        if note:
            elements.append({"tag": "hr"})
            elements.append({"tag": "note", "elements": [{"tag": "plain_text", "content": note}]})
        header: dict = {"template": color, "title": {"tag": "plain_text", "content": title}}
        return json.dumps({
            "config": {"wide_screen_mode": True},
            "header": header,
            "elements": elements,
        }, ensure_ascii=False)

    # ── JSON 2.0: supports markdown code blocks / headings / quotes / collapsible panels ──
    body_elements: list = []

    if tools_list:
        # Structured tool list: nested collapsibles to expand full output
        inner_elements: list = []
        for t in tools_list:
            icon = "✗" if t["is_error"] else "✓"
            desc_str = f" `{t['desc']}`" if t.get("desc") else ""
            header_md = f"{icon} **{t['name']}** ({_fmt_elapsed(t['elapsed'])}){desc_str}"
            full = t.get("full_result", "")
            if full:
                char_count = len(full)
                inner_elements.append({
                    "tag": "collapsible_panel",
                    "header": {"title": {"tag": "markdown", "content": header_md}},
                    "expanded": False,
                    "elements": [{"tag": "markdown",
                                  "content": f"```\n{full[:4800]}\n```"
                                             + (f"\n\n_（截断，共 {char_count} 字符）_"
                                                if char_count > 4800 else "")}],
                })
            else:
                inner_elements.append({"tag": "markdown", "content": header_md})
        body_elements.append({
            "tag": "collapsible_panel",
            "header": {"title": {"tag": "markdown", "content": "**🔧 工具调用**"}},
            "expanded": tools_expanded,
            "elements": inner_elements,
        })
        if body.strip():
            body_elements.append({"tag": "hr"})
    elif tools_md:
        body_elements.append({
            "tag": "collapsible_panel",
            "header": {"title": {"tag": "markdown", "content": "**🔧 工具调用**"}},
            "expanded": tools_expanded,
            "elements": [{"tag": "markdown", "content": tools_md}],
        })
        if body.strip():
            body_elements.append({"tag": "hr"})

    content = _normalize_newlines(body.strip()) if normalize else body.strip()
    if note:
        content = (content + "\n\n" if content else "") + f"---\n\n_{note}_"
    if content:
        body_elements.append({"tag": "markdown", "content": content})

    v2_header: dict = {"template": color, "title": {"tag": "plain_text", "content": title}}
    if subtitle:
        v2_header["subtitle"] = {"tag": "plain_text", "content": subtitle}

    return json.dumps({
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": v2_header,
        "body": {"elements": body_elements},
    }, ensure_ascii=False)
