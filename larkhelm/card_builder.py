"""
larkhelm · Feishu card builder utilities

Contains:
  - _make_card()           With buttons → JSON 1.0; without buttons → JSON 2.0
  - _make_simple_v1_card() Minimal JSON 1.0 card for patching existing V1 streaming cards
  - _split_md()            Split Markdown at MAX_CARD_LEN boundaries
  - _normalize_newlines()  Insert blank lines outside code blocks for Feishu paragraph rendering
  - _md_to_body_elements() Split markdown into body elements, converting tables to Feishu table elements
  - _btn_type()            Determine button style from its label
  - _fmt_elapsed()         Format elapsed seconds as a human-readable string

Schema strategy:
  JSON 1.0 (elements[])     — required for interactive buttons (action tag)
  JSON 2.0 (body.elements[])— used when no buttons; supports code blocks, tables, collapsible panels
"""
import json
import re

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


def _is_table_line(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 2


def _normalize_newlines(text: str) -> str:
    """Convert single newlines outside code blocks to double newlines.
    Table rows are kept together without extra blank lines so _md_to_body_elements
    can detect and convert them as contiguous blocks."""
    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if in_code:
            result.append(line)
        else:
            curr_table = _is_table_line(line)
            prev_table = bool(result) and _is_table_line(result[-1])
            if not curr_table and not prev_table:
                if result and result[-1] != "" and line != "":
                    result.append("")
            result.append(line)
    return "\n".join(result)


def _parse_md_table(table_lines: list[str]) -> dict | None:
    """Convert a list of markdown table lines into a Feishu JSON 2.0 table element."""
    def split_cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def is_separator(line: str) -> bool:
        return bool(re.match(r"^\|[-| :]+\|$", line.strip()))

    non_sep = [l for l in table_lines if not is_separator(l)]
    if not non_sep:
        return None

    headers = split_cells(non_sep[0])
    data_rows_cells = [split_cells(l) for l in non_sep[1:]]

    if not headers:
        return None

    col_keys = [f"c{i}" for i in range(len(headers))]
    columns = [
        {"data_source_column": col_keys[i], "width": "auto",
         "horizontal_align": "left", "name": headers[i]}
        for i in range(len(headers))
    ]
    rows = [
        {col_keys[i]: (cells[i] if i < len(cells) else "") for i in range(len(col_keys))}
        for cells in data_rows_cells
    ]

    if not rows:
        return None

    return {
        "tag": "table",
        "page_size": len(rows),
        "row_height": "low",
        "header_style": {"text_align": "left", "bold": True, "lines": 1},
        "columns": columns,
        "rows": rows,
    }


def _md_to_body_elements(text: str) -> list[dict]:
    """Split normalized markdown into body elements.
    Markdown tables are converted to Feishu native table elements;
    everything else becomes markdown elements."""
    elements: list[dict] = []
    lines = text.split("\n")
    buf: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if _is_table_line(line):
            content = "\n".join(buf).strip()
            if content:
                elements.append({"tag": "markdown", "content": content})
            buf = []
            table_lines: list[str] = []
            while i < len(lines) and _is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            table_elem = _parse_md_table(table_lines)
            if table_elem:
                elements.append(table_elem)
            else:
                elements.append({"tag": "markdown", "content": "\n".join(table_lines)})
        else:
            buf.append(line)
            i += 1

    content = "\n".join(buf).strip()
    if content:
        elements.append({"tag": "markdown", "content": content})

    return elements or [{"tag": "markdown", "content": text}]


def _split_md(text: str) -> list[str]:
    if len(text) <= _cfg.MAX_CARD_LEN:
        return [text]
    chunks, buf, buf_len, in_code = [], [], 0, False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        line_len = len(line) + 1
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
#  Card builders
# ═══════════════════════════════════════════════════

def _make_simple_v1_card(title: str, body: str, color: str = "blue") -> str:
    """Minimal JSON 1.0 card. Used to clean up streaming cards (which are JSON 1.0)
    before sending a new JSON 2.0 final card, avoiding the V1→V2 schema-change error."""
    elements = []
    if body.strip():
        elements.append({"tag": "div", "text": {"tag": "lark_md", "content": body}})
    return json.dumps({
        "config": {"wide_screen_mode": True},
        "header": {"template": color, "title": {"tag": "plain_text", "content": title}},
        "elements": elements,
    }, ensure_ascii=False)


def _make_card(title: str, body: str, color: str = "blue", note: str = "",
               buttons: list[tuple[str, str]] = None,
               subtitle: str = "",
               tools_md: str = None,
               tools_expanded: bool = False,
               tools_list: list = None,
               normalize: bool = True) -> str:
    """
    With buttons  → JSON 1.0 (only schema that supports the action tag for interactive buttons)
    Without buttons → JSON 2.0 (supports code blocks, tables, collapsible panels, etc.)

    When patching a JSON 1.0 streaming card with a JSON 2.0 final card, Feishu rejects the
    schema change (ErrCode 200830). Use _make_simple_v1_card() to first clean up the
    streaming card, then send the JSON 2.0 card as a new message.
    """
    if buttons:
        # ── JSON 1.0: required for interactive buttons (action tag) ──────────
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

    # ── JSON 2.0: code blocks, tables, collapsible panels ────────────────────
    body_elements: list = []

    if tools_list:
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
        body_elements.extend(_md_to_body_elements(content))

    v2_header: dict = {"template": color, "title": {"tag": "plain_text", "content": title}}
    if subtitle:
        v2_header["subtitle"] = {"tag": "plain_text", "content": subtitle}

    return json.dumps({
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": v2_header,
        "body": {"elements": body_elements},
    }, ensure_ascii=False)
