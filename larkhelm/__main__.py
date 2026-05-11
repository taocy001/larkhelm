#!/usr/bin/env python3
"""larkhelm CLI entry point"""
import argparse
import sys


def _cmd_memory(args, memory_parser):
    """Handler for `larkhelm memory` subcommands."""
    from larkhelm.memory_io import export_memory, import_memory, _resolve_data_dir, _MEMORY_HOME

    if args.memory_command == "status":
        import json
        from pathlib import Path

        data_dir = _resolve_data_dir(getattr(args, "data_dir", None) or None)
        state_file = data_dir / "state.json"
        sessions_dir = data_dir / "sessions"
        logs_dir = data_dir / "logs"

        def _dir_size_kb(p: Path) -> str:
            if not p.exists():
                return "0 KB"
            total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            return f"{total / 1024:.1f} KB"

        chats = {}
        if state_file.exists():
            try:
                chats = json.loads(state_file.read_text(encoding="utf-8"))
            except Exception:
                pass

        n_sessions = len(list(sessions_dir.glob("*.sid"))) if sessions_dir.exists() else 0
        mem_files = list(_MEMORY_HOME.glob("*.md")) if _MEMORY_HOME.exists() else []

        print(f"Data dir  : {data_dir}")
        print(f"Chats     : {len(chats)}")
        print(f"Sessions  : {n_sessions} .sid files")
        print(f"Logs      : {_dir_size_kb(logs_dir)}")
        print(f"Memory    : {len(mem_files)} files  ({_dir_size_kb(_MEMORY_HOME)})")
        if chats:
            print("\nChats:")
            for cid, s in sorted(chats.items()):
                turns = s.get("turn_count", 0)
                model = s.get("model", "?")
                cwd = s.get("cwd", "?")
                short = cid[:20] + "…" if len(cid) > 20 else cid
                print(f"  {short}  model={model}  turns={turns}  cwd={cwd}")

    elif args.memory_command == "export":
        ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
        output = args.output or f"larkhelm-export-{ts}.zip"
        out = export_memory(
            output,
            chat_ids=args.chat_ids or None,
            data_dir=args.data_dir or None,
            include_debug_log=args.include_debug_log,
        )
        with __import__("zipfile").ZipFile(out, "r") as zf:
            n = len([n for n in zf.namelist() if n != "manifest.json"])
        print(f"Exported {n} file(s) → {out}")

    elif args.memory_command == "import":
        result = import_memory(
            args.archive,
            merge=not args.replace,
            dry_run=args.dry_run,
            data_dir=args.data_dir or None,
        )
        if args.dry_run:
            print(f"[dry-run] Would write {len(result['written'])} file(s):")
            for p in result["written"]:
                print(f"  {p}")
        else:
            print(f"Written: {len(result['written'])} file(s)")
        if result["skipped"]:
            print(f"Skipped: {len(result['skipped'])} file(s)")
            for ident, reason in result["skipped"]:
                print(f"  {ident}: {reason}")
        for w in result["warnings"]:
            print(f"Warning: {w}", file=sys.stderr)

    else:
        memory_parser.print_help()
        sys.exit(0)


def _cmd_doc(args):
    """Handler for `larkhelm doc` subcommands."""
    import larkhelm.config as _cfg
    import lark_oapi as lark
    import larkhelm.lark_client as _lc
    from larkhelm.lark_client import FeishuDocClient, parse_doc_url

    _cfg._init_runtime()
    _lc.client = lark.Client.builder().app_id(_cfg.APP_ID).app_secret(_cfg.APP_SECRET).build()
    doc_client = FeishuDocClient()

    content = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()

    if args.doc_command == "create":
        owner_open_id = getattr(args, "owner", None) or _cfg.DEFAULT_OWNER_OPEN_ID
        doc_ref = doc_client.create_doc(args.title, owner_open_id=owner_open_id)
        doc_client.append(doc_ref, content)
        print(f"https://feishu.cn/docx/{doc_ref.token}")

    elif args.doc_command == "append":
        ref = parse_doc_url(args.url)
        if ref is None:
            print(f"无法识别的飞书链接：{args.url}", file=sys.stderr)
            sys.exit(1)
        doc_client.append(ref, content)
        print("已追加内容。")

    elif args.doc_command == "write":
        ref = parse_doc_url(args.url)
        if ref is None:
            print(f"无法识别的飞书链接：{args.url}", file=sys.stderr)
            sys.exit(1)
        doc_client.replace_all(ref, content)
        print("已覆盖写入内容。")


def cli():
    from larkhelm import __version__

    parser = argparse.ArgumentParser(
        prog="larkhelm",
        description="Feishu (Lark) AI assistant platform powered by Claude CLI & Gemini CLI",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"larkhelm {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    start_parser = sub.add_parser("start", help="start the service (runs in the foreground)")
    start_parser.add_argument("--config", metavar="PATH",
                              help="path to config file (default: auto-detect)")
    start_parser.add_argument("--data-dir", metavar="DIR",
                              help="path to data directory (default: auto-detect)")

    # `larkhelm doc` — Feishu document operations for use by Claude Code / scripts
    doc_parser = sub.add_parser("doc", help="read/write Feishu documents from the CLI")
    doc_sub = doc_parser.add_subparsers(dest="doc_command")

    # larkhelm doc create <title> [--file <path|-]
    p_create = doc_sub.add_parser("create", help="create a new doc and write content to it")
    p_create.add_argument("title", help="document title")
    p_create.add_argument("--file", "-f", default="-",
                          metavar="FILE", help="content file (default: stdin)")
    p_create.add_argument("--owner", metavar="OPEN_ID",
                          help="transfer ownership to this open_id (default: config default_owner_open_id)")

    # larkhelm doc append <url> [--file <path|-]
    p_append = doc_sub.add_parser("append", help="append content to an existing doc")
    p_append.add_argument("url", help="Feishu doc/wiki URL")
    p_append.add_argument("--file", "-f", default="-",
                          metavar="FILE", help="content file (default: stdin)")

    # larkhelm doc write <url> [--file <path|-]
    p_write = doc_sub.add_parser("write", help="overwrite an existing doc with new content")
    p_write.add_argument("url", help="Feishu doc/wiki URL")
    p_write.add_argument("--file", "-f", default="-",
                         metavar="FILE", help="content file (default: stdin)")

    # `larkhelm memory` — export / import persistent state
    memory_parser = sub.add_parser(
        "memory",
        help="export or import persistent state (backup / restore)",
    )
    memory_sub = memory_parser.add_subparsers(dest="memory_command")

    p_mem_export = memory_sub.add_parser("export", help="export state to a .zip archive")
    p_mem_export.add_argument(
        "output", nargs="?", metavar="OUTPUT",
        help="output .zip file (default: larkhelm-export-{timestamp}.zip in current dir)",
    )
    p_mem_export.add_argument(
        "--chat-ids", nargs="+", metavar="CHAT_ID",
        help="export only these chat IDs",
    )
    p_mem_export.add_argument(
        "--data-dir", metavar="DIR",
        help="data directory (default: auto-detect)",
    )
    p_mem_export.add_argument(
        "--include-debug-log", action="store_true",
        help="include larkhelm.log in the archive (can be large)",
    )

    p_mem_import = memory_sub.add_parser("import", help="restore state from a .zip archive")
    p_mem_import.add_argument("archive", metavar="ARCHIVE", help="path to .zip archive")
    p_mem_import.add_argument(
        "--replace", action="store_true",
        help="overwrite all files unconditionally (default: merge)",
    )
    p_mem_import.add_argument(
        "--dry-run", action="store_true",
        help="show what would be written without making changes",
    )
    p_mem_import.add_argument(
        "--data-dir", metavar="DIR",
        help="data directory (default: auto-detect)",
    )

    p_mem_status = memory_sub.add_parser("status", help="show summary of persistent state sizes")
    p_mem_status.add_argument(
        "--data-dir", metavar="DIR",
        help="data directory (default: auto-detect)",
    )

    # Keep the version sub-command for backward compatibility (prefer the --version flag)
    sub.add_parser("version", help="print version (prefer --version)")

    # `larkhelm mcp-server` — MCP stdio server spawned by Claude Code per session
    mcp_parser = sub.add_parser(
        "mcp-server",
        help="run as an MCP stdio server (spawned automatically by Claude Code)",
    )
    mcp_parser.add_argument("--config", metavar="PATH", help="path to config file")
    mcp_parser.add_argument("--data-dir", metavar="DIR", help="path to data directory")

    # `larkhelm voice probe` — install-time capability check for local STT
    voice_parser = sub.add_parser(
        "voice",
        help="voice (STT) related subcommands; run 'larkhelm voice probe' once after install",
    )
    voice_sub = voice_parser.add_subparsers(dest="voice_command")
    p_probe = voice_sub.add_parser(
        "probe",
        help="probe system capability for local STT (faster-whisper) + write verdict to config.json",
    )
    p_probe.add_argument("--no-benchmark", action="store_true",
                         help="skip the real-inference benchmark (CPU flags + RAM only)")
    p_probe.add_argument("--no-write", action="store_true",
                         help="don't write results to config.json; print report only")
    p_probe.add_argument("--config", metavar="PATH",
                         help="path to config.json (default: auto-detect)")

    args = parser.parse_args()

    if args.command == "version":
        print(f"larkhelm {__version__}")
    elif args.command == "start":
        from larkhelm.bridge import main
        main(config_path=args.config, data_dir=args.data_dir)
    elif args.command == "memory":
        _cmd_memory(args, memory_parser)
    elif args.command == "doc":
        if not args.doc_command:
            doc_parser.print_help()
            sys.exit(0)
        _cmd_doc(args)
    elif args.command == "mcp-server":
        from larkhelm.mcp_server import run as mcp_run
        mcp_run(config_path=args.config, data_dir=args.data_dir)
    elif args.command == "voice":
        if args.voice_command == "probe":
            from larkhelm.voice.system_probe import cli_main as _probe_cli
            argv: list[str] = []
            if args.no_benchmark:
                argv.append("--no-benchmark")
            if args.no_write:
                argv.append("--no-write")
            if args.config:
                argv += ["--config", args.config]
            sys.exit(_probe_cli(argv))
        else:
            voice_parser.print_help()
            sys.exit(0)
    else:
        # No subcommand given: show help and require the user to make an explicit choice
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    cli()
