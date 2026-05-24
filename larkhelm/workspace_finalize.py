"""Workspace finalisation hook — shared by /plan and /dev.

After a multi-step task (``/plan`` or ``/dev``) finishes successfully, the
workspace under ``<cwd>/.crew_workspace`` is left in a state that needs two
small pieces of post-processing before the user can move on:

  1. **Flip ``workspace_meta.json`` to ``completed=true`` when the run's
     ``review.md`` ends with ``APPROVED``.** Without this, a follow-up
     ``/dev`` or ``/plan`` with the same ``task_hash`` (within the 24h stale
     TTL) silently reuses ``design.md`` / ``tasks.json`` as if the previous
     run was still in progress, causing cross-task contamination.

  2. **Emit a Feishu card listing the files the run touched, plus a copy-
     paste-able ``git add`` / ``git commit`` hint.** Plan / dev artefacts
     (new tests, scripts, doc changes) routinely stay ``untracked`` after
     a run because the user has to compose the commit by hand from the
     workspace ``file_changes.json``. Surfacing the list and a ready-made
     command removes a step where it's easy to drop a file.

History
-------
This module was extracted from ``cmd_plan.py`` (commit 83d9312) so that
``/dev`` could use the exact same hook. The previous /dev meta-flip block
in ``crew/_commands.py:903-905`` only handled (1) and lacked (2); both
flows now share this implementation.

Fail-soft
---------
Every external interaction (file read, git invoke, card send) is wrapped
so a failure here never affects the caller's reported outcome — the
caller's ``finally`` block must not see exceptions from this module.
"""
from __future__ import annotations

from pathlib import Path

from larkhelm.log import _debug_log


# ── Tunable constants ────────────────────────────────────────────────────

# git status timeout. Was 8s when this lived inside /plan; bumped to 30s
# because large mono-repos with thousands of untracked files (e.g. node_modules
# left over by an aborted install) can legitimately take >8s. We're already
# in the post-run finally path so a slow git status delays nothing user-visible.
_GIT_STATUS_TIMEOUT_SEC = 30

# Per-bucket display cap on the summary card. The card is a hint, not a
# manifest — long lists hurt scanability in the Feishu chat.
_DISPLAY_LIMIT = 12

# Cap on how many paths the suggested ``git add ...`` line embeds. Past this
# the line wraps awkwardly in the Feishu code block and is rarely the right
# command anyway (the user will refine by hand).
_COMMIT_TARGETS_LIMIT = 20

# B2: Drift threshold for the finalize auto-commit path. When the user
# enables ``dev_auto_commit=true`` and a /dev or /plan run lands on
# review APPROVED, we want to commit — but ONLY if the working tree
# state matches the run's declared ``file_changes.json``. If the run
# touched ``drift_count`` files that were not declared, the safer
# behaviour is to NOT commit and surface the drift in the summary card
# so the user can decide. ``3`` chosen per reviewer guidance — a couple
# of incidental file syncs are routine; more than that smells like the
# run quietly modified files outside its declared scope.
_DRIFT_THRESHOLD = 3


# ── Public API ───────────────────────────────────────────────────────────

def finalize_workspace(chat_id: str, title: str, *, kind: str = "plan") -> None:
    """Post-run workspace cleanup: meta flip on APPROVED + Feishu summary card.

    Parameters
    ----------
    chat_id:
        Feishu chat that originated the run. Used to (a) resolve the cwd
        via ``chat_state._get_cwd`` and (b) target the summary card.
    title:
        Display label embedded in the card body and used as the suggested
        ``git commit -m <title>`` text. Should be a short human-readable
        description (plan title or first 80 chars of /dev requirement).
    kind:
        ``"plan"`` or ``"dev"`` — only affects the card title prefix
        (``📦 Plan 收尾`` vs ``📦 Dev 收尾``) so the user can tell at a
        glance which flow produced the card.
    """
    from larkhelm.chat_state import _get_cwd
    from larkhelm.lark_client import send_card

    cwd = _get_cwd(chat_id)
    if not cwd:
        return
    ws = Path(cwd) / ".crew_workspace"
    if not ws.is_dir():
        return

    # 1. Inspect ``review.md`` — flip meta on APPROVED.
    review_file = ws / "review.md"
    review_ok = False
    try:
        if review_file.exists():
            # Last *non-blank* line so trailing whitespace / newline doesn't
            # break the check; ``.endswith("APPROVED")`` (the historical /dev
            # approach) also requires last-line APPROVED so the contract is
            # equivalent.
            tail = next((ln for ln in reversed(review_file.read_text(encoding="utf-8").splitlines())
                         if ln.strip()), "")
            review_ok = tail.strip().endswith("APPROVED")
    except Exception as _e:
        _debug_log(f"[WorkspaceFinalize] review.md read failed: {_e}")

    # ``review_meta_pre`` captured here so the B2 auto-commit can use
    # ``meta.chat_id`` / ``meta.plan_id`` when writing the final extended
    # schema below. We flip ``completed=true`` first (B3 invariant: the
    # boolean must be set even if the auto-commit doesn't run, so future
    # stale-checks honour the user's last successful run).
    review_meta_pre: dict = {}
    if review_ok:
        try:
            from larkhelm.crew._commands import _read_workspace_meta, _write_workspace_meta
            review_meta_pre = _read_workspace_meta(ws) or {}
            task_hash = review_meta_pre.get("task_hash", "")
            if task_hash and not review_meta_pre.get("completed"):
                _write_workspace_meta(
                    ws,
                    task_hash=task_hash,
                    completed=True,
                    commit_sha=review_meta_pre.get("commit_sha", "") or "",
                    finalized_at=review_meta_pre.get("finalized_at", 0.0) or 0.0,
                    chat_id=review_meta_pre.get("chat_id", "") or "",
                    plan_id=review_meta_pre.get("plan_id", "") or "",
                )
                _debug_log(
                    f"[WorkspaceFinalize] workspace_meta flipped to completed=true "
                    f"(task_hash={task_hash[:8]}, kind={kind}, title={title!r})"
                )
        except Exception as _e:
            _debug_log(f"[WorkspaceFinalize] meta flip failed: {_e}")

    # 2. Build files-to-commit list + git-add hint card.
    files = _collect_plan_artifacts(ws, cwd)
    # ``files`` is always a dict with the three bucket keys present, so the
    # truthiness check on the dict itself is meaningless. What we actually
    # want to know is "did anything land in any bucket". An empty run skips
    # the card.
    if not any(files.values()):
        return

    # B2: opportunistic auto-commit on review APPROVED + dev_auto_commit=true.
    # ``_maybe_auto_commit_finale`` is fail-soft; it returns a dict whose
    # ``commit_sha`` field is "" on any of: feature disabled, drift exceeds
    # threshold, git error, no review APPROVED, empty change set. The
    # remaining card-rendering path is unchanged for the disabled / drift
    # / failure cases, so /dev's existing UX is preserved.
    commit_info: dict = {"commit_sha": "", "drift_count": 0,
                         "drift_paths": [], "skipped_reason": ""}
    if review_ok:
        try:
            commit_info = _maybe_auto_commit_finale(cwd, files, title)
        except Exception as _e:
            _debug_log(f"[WorkspaceFinalize] auto-commit raised: {_e}")
        # B3: if we got a real sha, persist it + finalized_at into the
        # extended workspace_meta schema so future stale-detection /
        # ``/plan`` linking can disambiguate completed-and-committed
        # runs from completed-but-not-committed ones.
        sha = commit_info.get("commit_sha", "") if isinstance(commit_info, dict) else ""
        if sha:
            try:
                import time as _time
                from larkhelm.crew._commands import (
                    _read_workspace_meta, _write_workspace_meta,
                )
                _meta = _read_workspace_meta(ws) or {}
                if _meta.get("task_hash"):
                    _write_workspace_meta(
                        ws,
                        task_hash=_meta["task_hash"],
                        completed=True,
                        commit_sha=sha,
                        finalized_at=_time.time(),
                        chat_id=_meta.get("chat_id", "") or "",
                        plan_id=_meta.get("plan_id", "") or "",
                    )
            except Exception as _e:
                _debug_log(f"[WorkspaceFinalize] meta sha-write failed: {_e}")

    # Gather "本次摘要" metrics (U15): review verdict + test pass rate (from
    # changes.md) + diff line counts (from git) + Feishu doc URLs (from
    # changes.md). All sources are read-only / fail-soft; if a signal is
    # absent the relevant line is just omitted from the summary block.
    metrics = _collect_run_metrics(ws, cwd, files)
    try:
        body, color = _format_workspace_summary(
            files, review_ok, title, metrics, commit_info=commit_info,
        )
        kind_label = "Dev" if kind == "dev" else "Plan"
        send_card(chat_id, f"📦 {kind_label} 收尾 · 改动文件", body, color=color)
    except Exception as _e:
        _debug_log(f"[WorkspaceFinalize] summary card failed: {_e}")


# ── Internal helpers ─────────────────────────────────────────────────────


def _compute_drift(declared: "list[str]", actual_dirty: "list[str]") -> dict:
    """Compute drift between the run's declared file_changes.json and the
    actual git-dirty working tree.

    Pure function — no I/O. Returns a dict with these keys:

      * ``in_both``      — declared ∩ actual, ordered like ``declared``
      * ``drift``        — actual − declared (the run touched files it
                           didn't declare; this is what trips the
                           ``_DRIFT_THRESHOLD`` gate)
      * ``missing``      — declared − actual (declared but no change; not
                           necessarily a problem — could be a no-op edit
                           that hit existing content)
      * ``all_to_add``   — ``in_both ∪ drift``, the white-list that the
                           safe auto-commit path stages

    De-duped within each bucket; order-preserving for stable display.
    """
    declared_set = set(declared)
    actual_set = set(actual_dirty)
    in_both: list[str] = [p for p in declared if p in actual_set]
    drift: list[str] = [p for p in actual_dirty if p not in declared_set]
    missing: list[str] = [p for p in declared if p not in actual_set]
    # Stable union for the commit white-list.
    all_to_add: list[str] = []
    seen: set[str] = set()
    for p in in_both + drift:
        if p not in seen:
            seen.add(p)
            all_to_add.append(p)
    return {
        "in_both":    in_both,
        "drift":      drift,
        "missing":    missing,
        "all_to_add": all_to_add,
    }


def _maybe_auto_commit_finale(cwd: str, files: dict, title: str) -> dict:
    """B2: opportunistic auto-commit on review APPROVED.

    Behaviour matrix:

      ============================  ====================================
      Condition                     Outcome
      ============================  ====================================
      ``dev_auto_commit`` is false  No commit; returns empty sha.
      Empty change set              No commit; returns empty sha.
      ``len(drift) >= 3``           No commit; returns drift list for the
                                    summary card so the user decides.
      Otherwise                     ``git add -- <in_both ∪ drift>`` then
                                    commit with a body that includes the
                                    drift section; returns sha.
      ============================  ====================================

    Never raises — every external interaction is wrapped so the rest of
    ``finalize_workspace`` keeps rendering the summary card even if git
    blows up.
    """
    import larkhelm.config as _cfg
    if not _cfg.config.get("dev_auto_commit", False):
        return {"commit_sha": "", "drift_count": 0, "drift_paths": [],
                "skipped_reason": "dev_auto_commit=false"}

    declared = list(files.get("from_file_changes") or [])
    actual_dirty = list(files.get("tracked_modified") or []) + \
                   list(files.get("untracked") or [])
    if not actual_dirty:
        return {"commit_sha": "", "drift_count": 0, "drift_paths": [],
                "skipped_reason": "no dirty changes"}

    drift_info = _compute_drift(declared, actual_dirty)
    drift_count = len(drift_info["drift"])
    if drift_count >= _DRIFT_THRESHOLD:
        return {
            "commit_sha":      "",
            "drift_count":     drift_count,
            "drift_paths":     list(drift_info["drift"]),
            "skipped_reason":  f"drift_count {drift_count} ≥ {_DRIFT_THRESHOLD}",
        }

    add_targets = drift_info["all_to_add"]
    if not add_targets:
        return {"commit_sha": "", "drift_count": 0, "drift_paths": [],
                "skipped_reason": "no targets after drift filter"}

    # Build a commit message that embeds the drift summary so future
    # archeology can see what slipped past the declared scope without
    # tripping the gate. Subject line is short and tagged so it groups
    # well in ``git log --oneline``; body lists drift paths (if any).
    title_short = (title or "").splitlines()[0][:60] or "auto-finalize"
    msg_lines = [f"[finalize] {title_short}", ""]
    if drift_info["in_both"]:
        msg_lines.append(
            f"Declared ({len(drift_info['in_both'])} file(s)):"
        )
        for p in drift_info["in_both"][:20]:
            msg_lines.append(f"  - {p}")
        if len(drift_info["in_both"]) > 20:
            msg_lines.append(f"  - …+{len(drift_info['in_both']) - 20} more")
        msg_lines.append("")
    if drift_info["drift"]:
        msg_lines.append(
            f"Drift, auto-included ({len(drift_info['drift'])} file(s) "
            f"below {_DRIFT_THRESHOLD}-threshold):"
        )
        for p in drift_info["drift"]:
            msg_lines.append(f"  - {p}")
        msg_lines.append("")
    msg_lines.append("Auto-committed by larkhelm workspace_finalize (B2).")
    commit_msg = "\n".join(msg_lines)

    try:
        from larkhelm.crew._state import _git_auto_commit
        sha = _git_auto_commit(
            cwd, "finalize",
            add_targets=add_targets,
            commit_message=commit_msg,
        )
    except Exception as _e:
        _debug_log(f"[WorkspaceFinalize] _git_auto_commit raised: {_e}")
        sha = ""

    return {
        "commit_sha":      sha,
        "drift_count":     drift_count,
        "drift_paths":     list(drift_info["drift"]),
        "skipped_reason":  "" if sha else "git error or disabled",
    }


def _collect_plan_artifacts(ws: Path, cwd: str) -> dict:
    """Return a dict ``{tracked_modified: [...], untracked: [...], from_file_changes: [...]}``.

    Three sources, merged:

      * ``workspace/file_changes.json`` — the design-time list of files the
        run *intended* to modify. Always included so the user sees the
        run's own intent.
      * ``git status --porcelain`` filtered to repo-relative paths — actual
        working-tree state (catches off-plan edits the run didn't predict).
      * Files appearing in both are de-duplicated downstream by
        ``_format_workspace_summary``; the intent list is considered
        authoritative for ordering.
    """
    import json as _json
    result = {
        "from_file_changes": [],   # design-time intent
        "tracked_modified":  [],   # M in git status
        "untracked":         [],   # ?? in git status
    }
    # (a) file_changes.json — design-time intent.
    fc_path = ws / "file_changes.json"
    if fc_path.exists():
        try:
            data = _json.loads(fc_path.read_text(encoding="utf-8"))
            for entry in data.get("files", []) or []:
                p = entry.get("path") if isinstance(entry, dict) else None
                if p:
                    result["from_file_changes"].append(p)
        except Exception as _e:
            _debug_log(f"[WorkspaceFinalize] file_changes.json parse failed: {_e}")
    # (b) git status --porcelain.
    #
    # Two non-obvious parse hazards:
    #
    #   1. Renamed (``R`` status) / copied (``C``) entries appear as
    #      ``R  old.py -> new.py`` on a single line. Naively taking
    #      ``line[3:]`` would feed the literal ``"old.py -> new.py"``
    #      into the git-add hint, which the shell then can't resolve.
    #      Take the path AFTER ``->`` when the arrow is present.
    #
    #   2. Non-ASCII filenames are octal-escaped + double-quoted by git
    #      by default: ``?? "\344\270\255\346\226\207.py"`` for ``中文.py``.
    #      Setting ``core.quotePath=false`` makes git emit the literal
    #      UTF-8 path instead, so a copy-paste ``git add 中文.py`` works.
    try:
        import subprocess
        proc = subprocess.run(
            ["git", "-c", "core.quotePath=false", "-C", cwd,
             "status", "--porcelain"],
            capture_output=True, text=True, timeout=_GIT_STATUS_TIMEOUT_SEC,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                # Porcelain v1 format: ``XY path`` (XY = 2-char status code)
                if len(line) < 4:
                    continue
                status, rest = line[:2], line[3:].lstrip()
                if " -> " in rest:
                    # Rename / copy — take destination path (the file the
                    # user will want to ``git add``).
                    rest = rest.split(" -> ", 1)[1]
                if "?" in status:
                    result["untracked"].append(rest)
                else:
                    result["tracked_modified"].append(rest)
    except Exception as _e:
        _debug_log(f"[WorkspaceFinalize] git status failed: {_e}")
    return result


def _collect_run_metrics(ws: "Path", cwd: str, files: dict) -> dict:
    """Read post-run signals into a metrics dict for the "📊 本次摘要" block.

    Returns a dict with these keys (each may be ``None`` / ``[]`` when the
    underlying signal isn't present — the summary renderer skips empty
    fields rather than show "N/A"):

      * ``tests``       — first ``\\d+ passed`` pattern in ``changes.md``,
                          rendered as ``"N passed"`` or with failures
                          (``"N passed, M failed"``)
      * ``diff_stats``  — output of ``git diff --shortstat HEAD`` against
                          the current cwd, e.g. ``"3 files changed,
                          120 insertions(+), 5 deletions(-)"``
      * ``docx_urls``   — unique ``https://*.feishu.cn/docx/<token>`` URLs
                          appearing in ``changes.md`` (deduped, order-preserving)
      * ``file_count``  — total distinct paths across the three buckets in
                          ``files``, for the "改动 N 个文件" tally

    All file IO + subprocess calls are wrapped — never raises so the caller
    can always render *something*.
    """
    import re
    import subprocess

    metrics = {"tests": None, "diff_stats": None, "docx_urls": [], "file_count": 0}

    # Count distinct paths across the three file buckets (de-duped).
    seen: set[str] = set()
    for bucket in ("from_file_changes", "tracked_modified", "untracked"):
        for p in files.get(bucket, []):
            seen.add(p)
    metrics["file_count"] = len(seen)

    # Parse ``changes.md`` for test count + Feishu doc URLs.
    changes_path = ws / "changes.md"
    if changes_path.exists():
        try:
            text = changes_path.read_text(encoding="utf-8")
            # Common pytest output shapes:
            #   "N passed"  /  "N passed, M failed"  /  "N passed in 1.23s"
            # Prefer the LAST occurrence — multiple pytest runs append upward.
            test_matches = list(re.finditer(
                r"(\d+)\s+passed(?:,\s+(\d+)\s+failed)?", text))
            if test_matches:
                m = test_matches[-1]
                passed = m.group(1)
                failed = m.group(2)
                metrics["tests"] = (f"{passed} passed, {failed} failed"
                                    if failed and int(failed) > 0
                                    else f"{passed} passed")
            # Feishu doc URLs. We accept any subdomain (``feishu.cn`` /
            # ``my.feishu.cn`` / ``open.feishu.cn``) since plan output
            # quoting style varies. Strip trailing punctuation so a URL
            # at the end of a sentence (e.g. ``…文档：https://…feishu.cn/docx/X.``)
            # doesn't keep the period.
            url_matches = re.findall(
                r"https?://[a-zA-Z0-9.-]*feishu\.cn/(?:docx|wiki)/[A-Za-z0-9_-]+",
                text)
            seen_urls: set[str] = set()
            for u in url_matches:
                if u not in seen_urls:
                    seen_urls.add(u)
                    metrics["docx_urls"].append(u)
        except Exception as _e:
            _debug_log(f"[WorkspaceFinalize] changes.md parse failed: {_e}")

    # ``git diff --shortstat HEAD`` for total +/- and file count.
    # Note: this captures *both* staged and unstaged working-tree changes
    # since the last commit, matching the user's mental model of "what
    # this run produced". Untracked files don't show up in this stat —
    # the file_count above does cover them.
    try:
        proc = subprocess.run(
            ["git", "-C", cwd, "diff", "--shortstat", "HEAD"],
            capture_output=True, text=True, timeout=_GIT_STATUS_TIMEOUT_SEC,
        )
        if proc.returncode == 0:
            stat_line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
            if stat_line:
                metrics["diff_stats"] = stat_line
    except Exception as _e:
        _debug_log(f"[WorkspaceFinalize] git diff --shortstat failed: {_e}")

    return metrics


def _format_workspace_summary(files: dict, review_ok: bool, title: str,
                              metrics: "dict | None" = None,
                              *,
                              commit_info: "dict | None" = None) -> tuple[str, str]:
    """Render the workspace-summary card body + colour.

    Body sections (omitted when empty):
      * 📊 本次摘要 (U15) — review verdict + test pass rate + diff stats
        + doc-URL count, when ``metrics`` is provided and non-empty
      * 🔖 自动提交 (B2) — commit sha + drift status, when ``commit_info``
        is provided
      * intent — files the run declared in file_changes.json
      * modified / untracked — current working-tree state
      * git-add hint — ready-to-paste shell line covering both

    Colour:
      * green if review APPROVED, blue otherwise (no review or REJECTED).
    """
    lines: list[str] = []
    if review_ok:
        lines.append("**Review**: ✅ APPROVED — `workspace_meta.completed=true` 已刷")
    else:
        lines.append("**Review**: ⚠️ 未 APPROVED — `workspace_meta` 保留 `completed=false`")

    # B2: 🔖 自动提交 — surface commit_sha or drift warning right at the
    # top so the user notices before reading the file lists below.
    if commit_info:
        sha = commit_info.get("commit_sha", "")
        drift = commit_info.get("drift_count", 0)
        drift_paths = commit_info.get("drift_paths", []) or []
        reason = commit_info.get("skipped_reason", "")
        if sha:
            extra = f"（漂移 {drift} 个已并入）" if drift else ""
            lines.append(f"\n**🔖 自动提交**: `{sha}`{extra}")
        elif drift_paths:
            preview = ", ".join(f"`{p}`" for p in drift_paths[:5])
            extra = f"，余 {len(drift_paths) - 5} 个" if len(drift_paths) > 5 else ""
            lines.append(
                f"\n**🔖 自动提交**: ⚠️ 已跳过——漂移 {drift} 个文件"
                f"（≥ {_DRIFT_THRESHOLD}-threshold）：{preview}{extra}"
            )
            lines.append("> 请人工审视下方文件清单后手动执行 `git add` / `git commit`。")
        elif reason and reason != "dev_auto_commit=false":
            lines.append(f"\n**🔖 自动提交**: ⚠️ 已跳过（{reason}）")

    # 📊 本次摘要 (U15). Render only the rows we actually have data for,
    # so the section doesn't become a sea of "N/A".
    if metrics:
        summary_rows: list[str] = []
        if metrics.get("tests"):
            summary_rows.append(f"- 测试: `{metrics['tests']}`")
        if metrics.get("diff_stats"):
            summary_rows.append(f"- Diff: `{metrics['diff_stats']}`")
        if metrics.get("file_count"):
            summary_rows.append(f"- 改动文件: {metrics['file_count']} 个")
        urls = metrics.get("docx_urls") or []
        if urls:
            # Show the first 3 URLs inline; tally the rest. >3 飞书 docs in
            # one run is unusual but possible for big plans.
            preview = "\n".join(f"  - {u}" for u in urls[:3])
            extra = f"\n  - _… 余 {len(urls) - 3} 个略_" if len(urls) > 3 else ""
            summary_rows.append(f"- 飞书文档产出: {len(urls)} 个\n{preview}{extra}")
        if summary_rows:
            lines.append("\n**📊 本次摘要**")
            lines.extend(summary_rows)

    def _list_block(header: str, items: list[str], limit: int = _DISPLAY_LIMIT) -> None:
        if not items:
            return
        lines.append(f"\n**{header}**（{len(items)} 个）")
        for p in items[:limit]:
            lines.append(f"- `{p}`")
        if len(items) > limit:
            lines.append(f"- _… 余 {len(items) - limit} 个略_")

    _list_block("📋 file_changes.json 声明", files["from_file_changes"])
    _list_block("✏️ 工作树 modified", files["tracked_modified"])
    _list_block("📥 工作树 untracked", files["untracked"])

    # Build a git-add / git-commit hint. Combine intent + modified + untracked,
    # quote any path containing spaces / special chars.
    import shlex
    add_targets = []
    seen = set()
    for source in (files["from_file_changes"],
                   files["tracked_modified"],
                   files["untracked"]):
        for p in source:
            if p not in seen:
                seen.add(p)
                add_targets.append(p)
    if add_targets:
        quoted = " ".join(shlex.quote(p) for p in add_targets[:_COMMIT_TARGETS_LIMIT])
        more = ("" if len(add_targets) <= _COMMIT_TARGETS_LIMIT
                else f"  # +{len(add_targets) - _COMMIT_TARGETS_LIMIT} more — adjust as needed")
        # Quote title too — run titles can contain ``"`` / ``$(...)`` /
        # backticks that would break or shell-inject the bare ``-m "..."``
        # form. ``shlex.quote`` returns single-quoted form for strings
        # with no special chars (e.g. ``"MyPlan"`` → ``'MyPlan'``), keeping
        # the line paste-able for everyday titles.
        lines.append(
            "\n**一键提交命令**\n```bash\n"
            f"git add {quoted}{more}\n"
            f"git commit -m {shlex.quote(title)}\n"
            "```"
        )

    color = "green" if review_ok else "blue"
    return "\n".join(lines), color
