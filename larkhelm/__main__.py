#!/usr/bin/env python3
"""larkhelm CLI entry point"""
import argparse
import os
import sys
from pathlib import Path


def _cmd_memory(args, memory_parser):
    """Handler for `larkhelm memory` subcommands."""
    from larkhelm.memory_io import export_memory, import_memory, _resolve_data_dir, _MEMORY_HOME

    if args.memory_command == "status":
        import json

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

    elif args.memory_command == "audit-summary":
        # Phase D / Phase 2 (REQ-37) — operator-facing aggregation over the
        # retriever audit JSONL trail. Runtime-independent: no live bridge
        # required, scans rotated files under DATA_DIR/<audit_path>.
        import contextlib
        import io
        import json
        from datetime import timedelta
        # 1) ``LARKHELM_TEST_MODE`` short-circuits backend health-probe /
        #    model probe / MemoryGC daemon startup inside ``_init_runtime``.
        # 2) ``_init_ai_sem`` is called UNCONDITIONALLY and emits a one-line
        #    ``[Runner] MAX_AI_PROCS=...`` info log via ``larkhelm.log.info``,
        #    which goes to stdout (preserved for systemd journal). Redirect
        #    stdout for the duration of ``_init_runtime`` so the JSON
        #    contract (AC-07) is not polluted when this CLI is piped into
        #    ``python -c "json.load(sys.stdin)"``.
        os.environ.setdefault("LARKHELM_TEST_MODE", "1")
        import larkhelm.config as _cfg
        _init_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(_init_buf):
                _cfg._init_runtime(data_dir=getattr(args, "data_dir", None) or None)
        except SystemExit:
            # Missing APP_ID in test envs — degrade and continue with module
            # defaults (DATA_DIR may be unset; iter falls back to tempdir).
            pass
        print("audit subcommand removed (retriever infrastructure removed)", file=sys.stderr)
        sys.exit(1)

    elif args.memory_command == "unstale":
        print("unstale subcommand removed (lifecycle infrastructure removed)", file=sys.stderr)
        sys.exit(1)

    else:
        memory_parser.print_help()
        sys.exit(0)


# ── audit-summary helpers ────────────────────────────────────────────────


def _parse_audit_summary_duration(spec: str | None):
    """Parse "30m" / "2h" / "1d" / "PT15M"; default 1 hour."""
    from datetime import timedelta
    if not spec:
        return timedelta(hours=1)
    s = str(spec).strip().lower()
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
    try:
        if s and s[-1] in units:
            return timedelta(seconds=int(s[:-1] or "0") * units[s[-1]])
        return timedelta(seconds=int(s))
    except Exception:
        return timedelta(hours=1)


def _compute_llm_router_summary(records: list) -> dict | None:
    """Aggregate Stage C (LLM-router) telemetry from Phase 3 audit records.

    Returns ``None`` when **no** record in the window carries any
    ``llm_router_*`` field — meaning Stage C gate didn't fire in this
    window, so the operator sees a clean Phase-2-shaped summary instead
    of a section full of zeroes.

    When at least one record has the fields, returns a dict with:

      * ``gate_fired_count``    — records that reached the Stage C gate
        (the dispatcher set ``llm_router_diag`` on them)
      * ``invoked_count``       — successful invokes ONLY (LLM call
        completed AND response parsed AND a non-empty selection was
        used). Excludes invoke-then-failed records, which are counted
        in ``skipped_breakdown`` instead — see disjointness note below.
      * ``cache_hit_count``     — subset that reused a cached verdict
      * ``cache_hit_rate``      — hits / (hits + invokes); 0.0 when both 0
      * ``avg_selected_n``      — avg LLM-selected slice count among
        successful invokes
      * ``skipped_breakdown``   — Counter dict of non-empty
        ``llm_router_skipped`` reasons (rate_limit, no_cheap_caller,
        caller_exception, parse_failed, empty_response,
        underlying_failure, ...)

    Disjointness invariant
    ----------------------
    The three buckets ``invoked_count`` + ``cache_hit_count`` +
    ``sum(skipped_breakdown.values())`` must equal ``gate_fired_count``.

    The producer (``memory_llm_router.py``) sets ``diag.invoked = True``
    *before* calling the LLM as a debug trail, then sets a
    ``skipped_reason`` on post-call failures (``parse_failed`` /
    ``empty_response`` / ``caller_exception``) without flipping
    ``invoked`` back. This aggregator restores disjointness by
    treating ``invoked=True AND skipped_reason!=""`` as **skipped only**.
    Without that filter ``avg_selected_n`` got poisoned by failed
    invokes (always selected_n=0) — review MF-01 round-2.

    All counts are operator-facing — they answer "is Stage C alive,
    is it cheap, is the LLM picking anything useful?".
    """
    # A record was considered by Stage C iff it carries any llm_router_* key.
    # ``llm_router_invoked`` is the canonical sentinel; the dispatcher
    # writes it on every Stage C path (invoke / cache-hit / skip).
    gate_fired = [r for r in records if "llm_router_invoked" in r]
    if not gate_fired:
        return None

    def _skipped(r: dict) -> str:
        return str(r.get("llm_router_skipped") or "")

    # Disjoint buckets (review MF-01 round-2): a record with
    # ``invoked=True`` AND ``skipped_reason`` non-empty is the
    # "tried LLM, post-call failure" case — count it as skipped,
    # NOT as a successful invoke.
    successful_invokes = [
        r for r in gate_fired
        if r.get("llm_router_invoked") and not _skipped(r)
    ]
    cache_hits = [r for r in gate_fired if r.get("llm_router_cache_hit")]
    skipped_counter: dict[str, int] = {}
    for r in gate_fired:
        reason = _skipped(r)
        if reason:
            skipped_counter[reason] = skipped_counter.get(reason, 0) + 1

    def _safe_int(v: object, default: int = 0) -> int:
        """Tolerate malformed ``selected_n`` (string / None / non-numeric)
        without crashing the whole audit-summary CLI — review SF-02
        round-2."""
        try:
            return int(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    selected_total = sum(
        _safe_int(r.get("llm_router_selected_n")) for r in successful_invokes
    )
    avg_selected = (
        selected_total / len(successful_invokes)
        if successful_invokes else 0.0
    )

    n_inv = len(successful_invokes)
    n_hit = len(cache_hits)
    cache_denom = n_inv + n_hit
    cache_rate = (n_hit / cache_denom) if cache_denom else 0.0

    return {
        "gate_fired_count": len(gate_fired),
        "invoked_count":    n_inv,
        "cache_hit_count":  n_hit,
        "cache_hit_rate":   cache_rate,
        "avg_selected_n":   avg_selected,
        "skipped_breakdown": skipped_counter,
    }


def _compute_audit_summary(records: list, delta) -> dict:
    """Aggregate a list of audit records into the JSON contract dict (design.md §3.7)."""
    from datetime import datetime, timezone
    until = datetime.now(timezone.utc).astimezone()
    since = until - delta
    if not records:
        return {
            "schema_version": "2",
            "since": since.isoformat(timespec="seconds"),
            "until": until.isoformat(timespec="seconds"),
            "total_records": 0,
            "mode_distribution": {},
            "avg_elapsed_ms": 0.0,
            "p95_elapsed_ms": 0.0,
            "fail_open_rate": 0.0,
            "avg_selected_chars": 0,
            "avg_selected_slice_count": 0.0,
            "by_agent_type": {},
        }
    elapsed_ms = [int(r.get("elapsed_ms", 0)) for r in records]
    elapsed_ms_sorted = sorted(elapsed_ms)
    p95_idx = max(0, int(round(0.95 * (len(elapsed_ms_sorted) - 1))))
    fail_open_count = sum(1 for r in records if r.get("fail_open"))
    mode_distribution: dict[str, int] = {}
    by_agent: dict[str, list] = {}
    for r in records:
        mode = str(r.get("mode", "?"))
        mode_distribution[mode] = mode_distribution.get(mode, 0) + 1
        by_agent.setdefault(str(r.get("agent_type", "?")), []).append(r)
    by_agent_summary: dict[str, dict] = {}
    for agent, rs in by_agent.items():
        e = sorted(int(x.get("elapsed_ms", 0)) for x in rs)
        idx = max(0, int(round(0.95 * (len(e) - 1))))
        f_open = sum(1 for x in rs if x.get("fail_open"))
        by_agent_summary[agent] = {
            "count": len(rs),
            "p95_elapsed_ms": float(e[idx]),
            "fail_open_rate": f_open / len(rs),
        }
    total = len(records)
    avg_selected_chars = sum(int(r.get("selected_token_chars", 0)) for r in records) / total
    avg_slice_count = sum(len(r.get("selected_slice_ids", []) or []) for r in records) / total
    out = {
        "schema_version": "2",
        "since": since.isoformat(timespec="seconds"),
        "until": until.isoformat(timespec="seconds"),
        "total_records": total,
        "mode_distribution": mode_distribution,
        "avg_elapsed_ms": sum(elapsed_ms) / total,
        "p95_elapsed_ms": float(elapsed_ms_sorted[p95_idx]),
        "fail_open_rate": fail_open_count / total,
        "avg_selected_chars": int(avg_selected_chars),
        "avg_selected_slice_count": avg_slice_count,
        "by_agent_type": by_agent_summary,
    }
    # Phase 3 — Stage C LLM-router telemetry. Only emitted when at least
    # one record in the window has llm_router_* fields, so the summary
    # stays Phase-2-shaped for installations that haven't rolled out
    # Stage C yet (avoids confusing zero-noise in the JSON contract).
    llm_router_summary = _compute_llm_router_summary(records)
    if llm_router_summary is not None:
        out["llm_router"] = llm_router_summary
    return out


def _render_audit_summary_text(d: dict) -> str:
    """Render the JSON summary dict as a human-readable text block."""
    lines = [
        f"window     : {d.get('since')} → {d.get('until')}",
        f"records    : {d.get('total_records', 0)}",
        f"elapsed    : avg={d.get('avg_elapsed_ms', 0):.1f}ms  p95={d.get('p95_elapsed_ms', 0):.0f}ms",
        f"fail-open  : {d.get('fail_open_rate', 0):.2%}",
        f"avg slice  : {d.get('avg_selected_slice_count', 0):.1f} per query  ({d.get('avg_selected_chars', 0)} chars)",
    ]
    md = d.get("mode_distribution") or {}
    if md:
        lines.append("modes      : " + ", ".join(f"{k}={v}" for k, v in sorted(md.items())))
    bya = d.get("by_agent_type") or {}
    if bya:
        lines.append("per agent  :")
        for agent, st in sorted(bya.items()):
            lines.append(
                f"  {agent:<8} n={st['count']:<4} p95={st['p95_elapsed_ms']:.0f}ms"
                f" fail-open={st['fail_open_rate']:.2%}"
            )
    # Phase 3 — Stage C LLM-router section. Only rendered when the
    # ``llm_router`` key is present (i.e. at least one record in the
    # window passed the Stage C gate). Layout mirrors the existing
    # "per agent" block so operators reading the text output get
    # consistent spacing.
    llm_r = d.get("llm_router")
    if llm_r:
        lines.append("llm router :")
        gate    = int(llm_r.get("gate_fired_count", 0))
        invoked = int(llm_r.get("invoked_count", 0))
        hits    = int(llm_r.get("cache_hit_count", 0))
        rate    = float(llm_r.get("cache_hit_rate", 0.0))
        avg_n   = float(llm_r.get("avg_selected_n", 0.0))
        lines.append(
            f"  gate-fired={gate} invoked={invoked} cache-hit={hits}"
            f" cache-rate={rate:.2%} avg-selected={avg_n:.1f}"
        )
        breakdown = llm_r.get("skipped_breakdown") or {}
        if breakdown:
            parts = ", ".join(f"{k}={v}" for k, v in sorted(breakdown.items()))
            lines.append(f"  skipped: {parts}")
    return "\n".join(lines)


def _cmd_doc(args):
    """Handler for `larkhelm doc` subcommands."""
    import larkhelm.config as _cfg
    import lark_oapi as lark
    import larkhelm.lark_client as _lc
    from larkhelm.lark_client import FeishuDocClient, parse_doc_url

    _cfg._init_runtime()
    _lc.client = lark.Client.builder().app_id(_cfg.APP_ID).app_secret(_cfg.APP_SECRET).build()
    doc_client = FeishuDocClient()

    content = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")

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

    # Phase D / Phase 2 (REQ-37) — aggregate audit JSONL records.
    p_mem_audit = memory_sub.add_parser(
        "audit-summary",
        help="aggregate Phase D retriever audit records (mode/p95/fail-open/agent breakdown)",
    )
    p_mem_audit.add_argument(
        "--since", metavar="DURATION", default="1h",
        help="time window (e.g. 30m / 2h / 1d). Default: 1h",
    )
    p_mem_audit.add_argument(
        "--chat-id", metavar="CHAT_ID",
        help="filter to one chat only",
    )
    p_mem_audit.add_argument(
        "--mode", choices=["keyword", "embedding", "hybrid"],
        help="filter to records with this actual mode",
    )
    p_mem_audit.add_argument(
        "--json", action="store_true",
        help="emit a single JSON object instead of human-readable text",
    )
    p_mem_audit.add_argument(
        "--data-dir", metavar="DIR",
        help="data directory (default: auto-detect)",
    )

    # Phase D / Phase 2 (REQ-47) — re-activate a slice that was demoted to stale.
    p_mem_unstale = memory_sub.add_parser(
        "unstale",
        help="remove a slice_id from every .meta.json sidecar (re-activate after demotion)",
    )
    p_mem_unstale.add_argument(
        "--slice-id", metavar="ID", required=True,
        help="the 12-char slice id (md5 hex prefix). See `/memory diagnose` output.",
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

    # `larkhelm user-login` — OAuth authorize as a user so subsequent doc-create
    # calls happen under user_access_token (no transfer_owner notifications).
    # See larkhelm/oauth_user.py for the full flow.
    user_login_parser = sub.add_parser(
        "user-login",
        help="OAuth-authorize this user so doc creates skip the transfer-ownership notification",
    )
    user_login_parser.add_argument("--config", metavar="PATH",
                                   help="path to config file (default: auto-detect)")
    user_login_parser.add_argument("--data-dir", metavar="DIR",
                                   help="path to data directory (default: auto-detect)")

    user_logout_parser = sub.add_parser(
        "user-logout",
        help="clear the saved user_access_token (reverts to tenant + transfer behavior)",
    )
    user_logout_parser.add_argument("--config", metavar="PATH")
    user_logout_parser.add_argument("--data-dir", metavar="DIR")

    user_status_parser = sub.add_parser(
        "user-status",
        help="show user_token status (open_id / scope / expiry)",
    )
    user_status_parser.add_argument("--config", metavar="PATH")
    user_status_parser.add_argument("--data-dir", metavar="DIR")

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
    elif args.command in ("user-login", "user-logout", "user-status"):
        import larkhelm.config as _cfg
        _cfg._init_runtime(config_path=args.config, data_dir=args.data_dir)
        from larkhelm import oauth_user as _ou
        if args.command == "user-login":
            sys.exit(_ou.cli_login())
        elif args.command == "user-logout":
            sys.exit(_ou.cli_logout())
        else:
            sys.exit(_ou.cli_status())
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
