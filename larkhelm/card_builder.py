"""
larkhelm · Feishu card builder utilities

Public API
----------
_make_card_dict(...)  → dict   core builder; callers that need a dict (e.g. CallBackCard.data)
_make_card(...)       → str    json.dumps(_make_card_dict(...)); callers that need a JSON string
_split_md(text)       → list[str]
_fmt_elapsed(s)       → str
_normalize_newlines(t)→ str

body parameter accepts str or list[str].
  str       — treated as a single markdown section
  list[str] — each element becomes an independent markdown block (useful when content
               has multiple sections that should not be merged, e.g. perm cards)

Schema: JSON 2.0 by default; JSON 1.0 when buttons are present (required for action/callback support).
  Buttons — cards with buttons use JSON 1.0 format (_make_card_json10_dict); cards without buttons
            use JSON 2.0 format with button elements via column_set / behaviors/callback.
  Tables  — markdown pipe tables auto-converted to Feishu native table elements.
"""
import json
import re

import larkhelm.config as _cfg


# ── Utilities ───────────────────────────────────────────────────────────────

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
    """Double newlines outside code blocks for Feishu paragraph breaks.
    Table rows are kept contiguous so _md_to_body_elements can detect them."""
    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    for line in lines:
        was_in_code = in_code
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if was_in_code:
            # Inside a code block (including the closing fence): append verbatim.
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
    """Convert contiguous markdown table lines → Feishu JSON 2.0 table element."""
    def split_cells(line: str) -> list[str]:
        return [c.strip() for c in line.strip().strip("|").split("|")]

    def is_sep(line: str) -> bool:
        return bool(re.match(r"^\|[-| :]+\|$", line.strip()))

    non_sep = [l for l in table_lines if not is_sep(l)]
    if not non_sep:
        return None

    headers = split_cells(non_sep[0])
    if not headers:
        return None

    col_keys = [f"c{i}" for i in range(len(headers))]
    columns = [{"name": col_keys[i], "display_name": headers[i],
                "data_type": "text", "width": "auto", "horizontal_align": "left"}
               for i in range(len(headers))]
    rows = [{col_keys[i]: cells[i] if i < len(cells) else ""
             for i in range(len(col_keys))}
            for cells in (split_cells(l) for l in non_sep[1:])]

    if not rows:
        return None

    return {
        "tag": "table",
        "page_size": min(len(rows), 100),
        "header_style": {"text_align": "left", "bold": True, "lines": 1},
        "columns": columns,
        "rows": rows,
    }


def _md_to_body_elements(text: str) -> list[dict]:
    """Split a normalized markdown string into body elements,
    converting pipe tables to Feishu native table elements."""
    elements: list[dict] = []
    lines = text.split("\n")
    buf: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        if _is_table_line(line):
            if content := "\n".join(buf).strip():
                elements.append({"tag": "markdown", "content": content})
            buf = []
            table_lines: list[str] = []
            while i < len(lines) and _is_table_line(lines[i]):
                table_lines.append(lines[i])
                i += 1
            elem = _parse_md_table(table_lines)
            elements.append(elem if elem else
                            {"tag": "markdown", "content": "\n".join(table_lines)})
        else:
            buf.append(line)
            i += 1

    if content := "\n".join(buf).strip():
        elements.append({"tag": "markdown", "content": content})

    return elements or [{"tag": "markdown", "content": text}]


def _section_elements(section: str, normalize: bool) -> list[dict]:
    """Normalize one body section and expand tables."""
    content = _normalize_newlines(section.strip()) if normalize else section.strip()
    return _md_to_body_elements(content) if content else []


def _split_md(text: str) -> list[str]:
    if len(text) <= _cfg.MAX_CARD_LEN:
        return [text]
    chunks, buf, buf_len, in_code = [], [], 0, False
    for line in text.split("\n"):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        line_len = len(line) + 1
        if line_len > _cfg.MAX_CARD_LEN:
            if in_code:
                # Don't split inside a code block; accumulate and let Feishu truncate
                buf.append(line)
                buf_len += line_len
                continue
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


# ── JSON 1.0 builder (used when buttons are present) ─────────────────────────

def _make_card_json10_dict(
    title: str,
    body: "str | list[str]",
    color: str,
    note: str,
    buttons: list,
    subtitle: str,
    normalize: bool,
) -> dict:
    """Build a JSON 1.0 Feishu card (required format for action/button callbacks)."""
    elements: list = []
    sections: list[str] = [body] if isinstance(body, str) else list(body)
    for sec in sections:
        if sec:
            content = _normalize_newlines(sec.strip()) if normalize else sec.strip()
            if content:
                elements.append({"tag": "div", "text": {"tag": "lark_md", "content": content}})

    actions = [
        {
            "tag": "button",
            "text": {"tag": "lark_md", "content": label},
            "type": _btn_type(label),
            "value": {"cmd": cmd},
        }
        for label, cmd in buttons
    ]
    elements.append({"tag": "action", "actions": actions})

    if note:
        elements.append({"tag": "note", "elements": [{"tag": "lark_md", "content": note}]})

    header: dict = {"template": color, "title": {"tag": "lark_md", "content": title}}
    if subtitle:
        header["subtitle"] = {"tag": "lark_md", "content": subtitle}

    return {
        "config": {"wide_screen_mode": True},
        "header": header,
        "elements": elements,
    }


# ── Core card builder ────────────────────────────────────────────────────────

def _make_card_dict(
    title: str,
    body: "str | list[str]" = "",
    color: str = "blue",
    note: str = "",
    buttons: "list[tuple[str, str]] | None" = None,
    subtitle: str = "",
    tools_md: str = None,
    tools_expanded: bool = False,
    tools_list: list = None,
    normalize: bool = True,
    raw_elements: "list[dict] | None" = None,
) -> dict:
    """
    Build a Feishu card and return it as a dict (JSON 2.0).

    body — str or list[str].
      str:        single markdown section (normalized + table-converted).
      list[str]:  multiple independent markdown sections; each rendered as its own
                  block, useful when content has distinct visual paragraphs
                  (e.g. tool name line, then a code block for the command).
    raw_elements — pre-built list[dict] body elements; when provided, body/note/buttons/
                   tools_* are ignored and the list is used as-is. For complex cards
                   (e.g. crew_card.py terminal phase) that build their own element tree.
    """
    # Cards with buttons use JSON 1.0 format (required for action/callback support)
    if buttons and raw_elements is None:
        return _make_card_json10_dict(title, body, color, note, buttons, subtitle, normalize)

    # Fast path: caller supplies pre-built elements (e.g. crew_card terminal phase)
    if raw_elements is not None:
        header: dict = {"template": color, "title": {"tag": "plain_text", "content": title}}
        if subtitle:
            header["subtitle"] = {"tag": "plain_text", "content": subtitle}
        return {
            "schema": "2.0",
            "config": {"wide_screen_mode": True},
            "header": header,
            "body": {"elements": raw_elements},
        }

    body_elements: list = []

    # ── Tool call panels ──────────────────────────────────────────────────
    if tools_list:
        # Flat markdown elements inside one collapsible panel — JSON 2.0 does not
        # support nesting collapsible_panel inside collapsible_panel.
        inner: list = []
        for t in tools_list:
            icon = "✗" if t["is_error"] else "✓"
            desc_str = f" `{t['desc']}`" if t.get("desc") else ""
            hdr = f"{icon} **{t['name']}** ({_fmt_elapsed(t['elapsed'])}){desc_str}"
            inner.append({"tag": "markdown", "content": hdr})
            full = t.get("full_result", "")
            if full:
                n = len(full)
                suffix = f"\n\n_（截断，共 {n} 字符）_" if n > 4800 else ""
                inner.append({"tag": "markdown",
                               "content": f"```\n{full[:4800]}\n```{suffix}"})
        body_elements.append({
            "tag": "collapsible_panel",
            "header": {"title": {"tag": "plain_text", "content": "🔧 工具调用"}},
            "expanded": tools_expanded,
            "elements": inner,
        })
    elif tools_md:
        body_elements.append({
            "tag": "collapsible_panel",
            "header": {"title": {"tag": "plain_text", "content": "🔧 工具调用"}},
            "expanded": tools_expanded,
            "elements": [{"tag": "markdown", "content": tools_md}],
        })

    # ── Body content ──────────────────────────────────────────────────────
    tools_present = bool(body_elements)  # true if tools panel was added above
    sections: list[str] = [body] if isinstance(body, str) else list(body)
    body_has_content = False
    for sec in sections:
        elems = _section_elements(sec, normalize)
        if elems:
            if tools_present and not body_has_content:
                body_elements.append({"tag": "hr"})  # separator between tools and body
            body_elements.extend(elems)
            body_has_content = True

    # ── Note (appended to last markdown element or as new element) ────────
    if note:
        note_md = f"---\n\n_{note}_"
        # Try to append to last markdown element for compact rendering
        if body_elements and body_elements[-1].get("tag") == "markdown":
            body_elements[-1]["content"] += "\n\n" + note_md
        else:
            body_elements.append({"tag": "markdown", "content": note_md})

    # ── Header ───────────────────────────────────────────────────────────
    header: dict = {"template": color, "title": {"tag": "plain_text", "content": title}}
    if subtitle:
        header["subtitle"] = {"tag": "plain_text", "content": subtitle}

    return {
        "schema": "2.0",
        "config": {"wide_screen_mode": True},
        "header": header,
        "body": {"elements": body_elements},
    }


def _make_card(
    title: str,
    body: "str | list[str]" = "",
    color: str = "blue",
    note: str = "",
    buttons: "list[tuple[str, str]] | None" = None,
    subtitle: str = "",
    tools_md: str = None,
    tools_expanded: bool = False,
    tools_list: list = None,
    normalize: bool = True,
    raw_elements: "list[dict] | None" = None,
) -> str:
    """JSON string wrapper around _make_card_dict."""
    return json.dumps(
        _make_card_dict(title, body, color, note, buttons, subtitle,
                        tools_md, tools_expanded, tools_list, normalize,
                        raw_elements),
        ensure_ascii=False,
    )


