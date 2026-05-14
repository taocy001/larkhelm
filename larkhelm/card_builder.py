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

Schema: JSON 2.0 unconditionally. Cards with buttons use JSON 2.0's native
        button element (``tag:"button"``) wrapped in a ``column_set`` for
        multi-button rows; the callback ``behaviors`` carry the same
        ``value:{"cmd":...}`` payload that ``handlers/_card_action.py``
        already parses, so the migration is transparent to the dispatcher.
  Buttons — single button → bare element in body.elements; multiple buttons →
            column_set with width:"auto" columns. Color via ``type``
            (primary/danger/default).
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
    """Convert contiguous markdown table lines → Feishu JSON 2.0 table element.

    Width / wrap policy (vs. the previous "all columns width=auto" rendering
    which collapsed to even-split + no wrap on real content):

      * **Column widths** are sized from the longest visual cell. The default
        ``width: "auto"`` makes Feishu split available width evenly across
        columns, which clips content in long columns and wastes space in
        short ones. We emit per-column ``"{N}px"`` strings sized to the
        longest cell (capped at 360 px so one wide column can't push others
        off-screen).

      * **Cell rendering** uses ``data_type: "markdown"`` so multi-line
        content (and bold/code/links) wrap naturally. With ``data_type:
        "text"`` Feishu renders each cell as a single line, truncating
        anything past the column width.
    """
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

    body_cells = [split_cells(l) for l in non_sep[1:]]
    if not body_cells:
        return None

    n_cols = len(headers)
    # Width estimation looks at the header row too — a long header on a
    # short data column still needs room.
    widths_px = _estimate_table_col_widths_px([headers] + body_cells, n_cols)

    col_keys = [f"c{i}" for i in range(n_cols)]
    columns = [{
        "name":             col_keys[i],
        "display_name":     headers[i] if i < len(headers) else "",
        "data_type":        "markdown",     # enables wrap + inline markdown
        "width":            f"{widths_px[i]}px",
        "horizontal_align": "left",
        "vertical_align":   "top",
    } for i in range(n_cols)]

    rows = [{col_keys[i]: cells[i] if i < len(cells) else ""
             for i in range(n_cols)}
            for cells in body_cells]

    return {
        "tag": "table",
        "page_size": min(len(rows), 100),
        "header_style": {"text_align": "left", "bold": True, "lines": 1},
        "columns": columns,
        "rows": rows,
    }


# ── Column-width estimator (shared with lark_client docx tables) ─────────
#
# Same constants + visual-width heuristic as ``lark_client._estimate_col_widths_px``;
# kept duplicated here to avoid card_builder → lark_client import cycle (card_builder
# is imported by lark_client). Drift risk: tune both together.

_TBL_COL_MIN_PX = 120
_TBL_COL_MAX_PX = 360
_TBL_COL_PX_PER_CHAR = 12.0
_TBL_COL_PADDING_PX = 24


def _tbl_visual_width(s: str) -> int:
    if not s:
        return 0
    return sum(2 if ord(c) > 0x2E80 else 1 for c in s)


def _estimate_table_col_widths_px(rows: list[list[str]], n_cols: int) -> list[int]:
    widths: list[int] = []
    for c in range(n_cols):
        max_w = max(
            (_tbl_visual_width(r[c]) for r in rows if c < len(r)),
            default=4,
        )
        px = max_w * _TBL_COL_PX_PER_CHAR + _TBL_COL_PADDING_PX
        widths.append(int(max(_TBL_COL_MIN_PX, min(_TBL_COL_MAX_PX, px))))
    return widths


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


def _split_long_line(line: str, max_len: int) -> list[str]:
    """Best-effort soft-wrap of a single oversized line.

    Guarantee: every character of ``line`` appears in the returned list (in
    order, concatenated). The current implementation returns ``[line]``
    unchanged — Feishu's card renderer handles long lines via its own wrap
    logic, so cutting mid-line tends to make things worse (it breaks inline
    markdown syntax). The hook is kept as an extension point.
    """
    return [line]


def _split_md(text: str) -> list[str]:
    """Split markdown into chunks ≤ ``MAX_CARD_LEN`` where possible.

    Guarantees:
      - All characters of ``text`` are preserved across the returned list.
      - Fenced code blocks (``` … ```) are kept intact — no chunk boundary
        falls inside a block. An entire code block exceeding ``MAX_CARD_LEN``
        is emitted as one oversized-but-valid chunk.
      - A single line exceeding ``MAX_CARD_LEN`` is emitted as its own chunk
        (no mid-line split).
      - For multi-line input, chunks try to stay within the limit by flushing
        on line boundaries.
    """
    max_len = _cfg.MAX_CARD_LEN
    if len(text) <= max_len:
        return [text]

    lines = text.split("\n")
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    in_code = False

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append("\n".join(buf))
            buf = []
            buf_len = 0

    for line in lines:
        is_fence = line.lstrip().startswith("```")
        line_len = len(line) + 1  # +1 for the newline join

        if is_fence and not in_code:
            # Opening fence: flush pending normal lines so the code block
            # starts at a clean chunk boundary, then enter code mode.
            flush()
            buf.append(line)
            buf_len += line_len
            in_code = True
            continue

        if is_fence and in_code:
            # Closing fence: include in the current chunk, then optionally
            # flush — closing here keeps the fence pair on the same chunk.
            buf.append(line)
            buf_len += line_len
            in_code = False
            if buf_len > max_len:
                flush()
            continue

        if in_code:
            # Inside a code block: never split, even if buf overruns.
            buf.append(line)
            buf_len += line_len
            continue

        # Normal line outside a code block.
        if len(line) > max_len:
            # Oversized single line: emit on its own (no mid-line break).
            flush()
            chunks.extend(_split_long_line(line, max_len))
            continue

        if buf_len + line_len > max_len and buf:
            flush()
        buf.append(line)
        buf_len += line_len

    flush()
    return chunks or [text]


# ── JSON 2.0 button block builder ────────────────────────────────────────────
# Replaces the old JSON 1.0 ``{"tag":"action","actions":[...]}`` container.
#
# Why we migrated (commit history):
#   1. Visual bug: JSON 1.0's ``{"tag":"div","text":{"tag":"lark_md",...}}``
#      body element renders at a different default font size than JSON 2.0's
#      ``{"tag":"markdown",...}``. Streaming card (with cancel button) →
#      buttons → JSON 1.0 → big font; final card (no buttons) → JSON 2.0 →
#      normal font. Same content read at two sizes was the visible defect.
#   2. Markdown subset gap: ``lark_md`` doesn't render bullet lists, fenced
#      code blocks, or block quotes — they pass through as raw text.
#      The ``markdown`` element renders all three correctly.
#   3. Schema cleanup: maintaining two schema branches in the builder was
#      load-bearing complexity for no benefit once 2.0 supports buttons.
#
# JSON 2.0 button schema (per Feishu official docs):
#   - Single button: bare ``{"tag":"button", "text":{...}, "type":..., "behaviors":[...]}``
#     directly inside ``body.elements[]``.
#   - Multiple buttons in one row: wrap in ``column_set`` with ``width:"auto"``
#     columns (one button per column).
#   - Label uses ``plain_text`` (not ``lark_md`` — JSON 2.0 buttons don't
#     accept lark_md formatting in labels).
#   - Callback: ``"behaviors":[{"type":"callback","value":{"cmd":...}}]``.
#     The ``value`` dict surfaces unchanged on ``CallBackAction.value`` —
#     ``handlers/_card_action.py:27`` ``(action.value or {}).get("cmd",...)``
#     parses both schemas identically.


def _make_button_element(label: str, cmd: str) -> dict:
    """Build one JSON 2.0 button element with a callback ``cmd`` payload.

    Color (``type``) is auto-derived from ``label`` via ``_btn_type``
    (primary for OK/继续/✅; danger for cancel/取消/❌; default otherwise).
    The ``cmd`` lands in ``CallBackAction.value`` unchanged on the bot
    side — see ``handlers/_card_action.py:27``.
    """
    return {
        "tag": "button",
        "text": {"tag": "plain_text", "content": label},
        "type": _btn_type(label),
        "behaviors": [{"type": "callback", "value": {"cmd": cmd}}],
    }


def _make_buttons_block(buttons: "list[tuple[str, str]]") -> dict:
    """Return a single button element OR a ``column_set`` wrapping multiple.

    JSON 2.0 doesn't support a multi-button "action row" container; the
    canonical pattern is to use ``column_set`` with ``width:"auto"`` columns
    so buttons sit side-by-side without consuming the full row width each.
    """
    if len(buttons) == 1:
        return _make_button_element(*buttons[0])
    return {
        "tag": "column_set",
        "horizontal_spacing": "small",
        "columns": [
            {
                "tag": "column",
                "width": "auto",
                "elements": [_make_button_element(label, cmd)],
            }
            for (label, cmd) in buttons
        ],
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

    # ── Buttons (JSON 2.0 callback style) ──────────────────────────────
    # Single button → bare element; multiple → column_set wrapper. The
    # callback ``value`` dict surfaces unchanged on the card-action event,
    # so ``handlers/_card_action.py:27`` doesn't need to change.
    if buttons:
        body_elements.append(_make_buttons_block(buttons))

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


