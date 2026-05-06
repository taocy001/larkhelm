"""
larkhelm MCP server — exposes Feishu capabilities to Claude Code as native tools.

Spawned by Claude Code via --mcp-config; receives current session context through
environment variables set by ai_runner._spawn_claude_proc:
  FEISHU_CHAT_ID  — the active chat / crew-agent namespace
  LARKHELM_CONFIG — path to config file (set explicitly in MCP args)
  LARKHELM_DATA_DIR — path to data dir (set explicitly in MCP args)
"""
import json
import os
import sys
from pathlib import Path


def _init(config_path: str | None, data_dir: str | None) -> None:
    """Initialize larkhelm config only. Feishu client is built lazily on first use."""
    import larkhelm.config as _cfg
    _cfg._init_runtime(config_path=config_path, data_dir=data_dir)


def _ensure_client() -> None:
    """Build the Feishu client if it hasn't been built yet (lazy init)."""
    import lark_oapi as lark
    import larkhelm.config as _cfg
    import larkhelm.lark_client as _lc

    if getattr(_lc, "client", None) is None:
        _lc.client = (
            lark.Client.builder()
            .app_id(_cfg.APP_ID)
            .app_secret(_cfg.APP_SECRET)
            .build()
        )


def _get_per_chat_cwd(chat_id: str, data_dir: Path) -> str:
    """Read per-chat cwd from the persisted state file (read-only, best-effort)."""
    state_file = data_dir / ".feishu_state.json"
    try:
        data = json.loads(state_file.read_text())
        return data.get(chat_id, {}).get("cwd", "")
    except Exception:
        return ""


def run(config_path: str | None, data_dir: str | None) -> None:
    """Start the MCP stdio server. Blocks until the client disconnects."""
    from mcp.server.fastmcp import FastMCP
    import larkhelm
    import larkhelm.config as _cfg
    from larkhelm.lark_client import FeishuDocClient, parse_doc_url

    _init(config_path, data_dir)

    mcp = FastMCP("larkhelm", log_level="WARNING")

    # ── Tool: get_context ────────────────────────────────────────────────────

    @mcp.tool()
    def get_context() -> dict:
        """
        Return the current larkhelm session context.

        Includes: chat_id (Feishu chat this query originates from),
        cwd (per-chat working directory), default model, larkhelm version.
        """
        chat_id = os.environ.get("FEISHU_CHAT_ID", "")
        cwd = _get_per_chat_cwd(chat_id, _cfg.DATA_DIR) or _cfg.DEFAULT_CWD
        return {
            "chat_id": chat_id,
            "cwd": cwd,
            "model": _cfg.DEFAULT_MODEL,
            "version": larkhelm.__version__,
        }

    # ── Tool: doc_read ───────────────────────────────────────────────────────

    @mcp.tool()
    def doc_read(url: str) -> str:
        """
        Read a Feishu document, wiki page, or spreadsheet by URL.

        Returns the document title and plain-text content. Content is capped
        at ~50 000 characters; a truncation notice is appended if needed.
        """
        _ensure_client()
        ref = parse_doc_url(url)
        if ref is None:
            return f"Error: unrecognized Feishu URL: {url}"
        try:
            result = FeishuDocClient().read(ref, max_chars=50_000)
            header = f"# {result.title}\n\n" if result.title else ""
            suffix = "\n\n[Content truncated]" if result.truncated else ""
            return header + result.content + suffix
        except Exception as e:
            return f"Error reading {url}: {e}"

    # ── Tool: doc_create ─────────────────────────────────────────────────────

    @mcp.tool()
    def doc_create(title: str, content: str) -> str:
        """
        Create a new Feishu document with the given title and Markdown content.

        Returns the URL of the newly created document.
        """
        _ensure_client()
        try:
            client = FeishuDocClient()
            doc_ref = client.create_doc(title, owner_open_id=_cfg.DEFAULT_OWNER_OPEN_ID)
            if content.strip():
                client.append(doc_ref, content)
            return f"https://feishu.cn/docx/{doc_ref.token}"
        except Exception as e:
            return f"Error creating document: {e}"

    # ── Tool: doc_append ─────────────────────────────────────────────────────

    @mcp.tool()
    def doc_append(url: str, content: str) -> str:
        """
        Append Markdown content to an existing Feishu document.

        Returns a confirmation message with the number of characters appended.
        """
        _ensure_client()
        ref = parse_doc_url(url)
        if ref is None:
            return f"Error: unrecognized Feishu URL: {url}"
        try:
            FeishuDocClient().append(ref, content)
            return f"Appended {len(content)} characters to {url}"
        except Exception as e:
            return f"Error appending to {url}: {e}"

    mcp.run("stdio")
