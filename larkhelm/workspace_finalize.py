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

    if review_ok:
        try:
            from larkhelm.crew._commands import _read_workspace_meta, _write_workspace_meta
            meta = _read_workspace_meta(ws)
            task_hash = meta.get("task_hash", "") if isinstance(meta, dict) else ""
            if task_hash and not meta.get("completed"):
                _write_workspace_meta(ws, task_hash=task_hash, completed=True)
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
    try:
        body, color = _format_workspace_summary(files, review_ok, title)
        kind_label = "Dev" if kind == "dev" else "Plan"
        send_card(chat_id, f"📦 {kind_label} 收尾 · 改动文件", body, color=color)
    except Exception as _e:
        _debug_log(f"[WorkspaceFinalize] summary card failed: {_e}")


# ── Internal helpers ─────────────────────────────────────────────────────

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


def _format_workspace_summary(files: dict, review_ok: bool, title: str) -> tuple[str, str]:
    """Render the workspace-summary card body + colour.

    Body sections (omitted when empty):
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
