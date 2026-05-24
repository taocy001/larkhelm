"""Workspace artifact reconciliation (B4).

After ``implementer`` / ``fixer`` finishes editing files, the architect's
``file_changes.json`` is sometimes incomplete — agents routinely modify
files outside the declared list (test files, doc syncs, neighbouring
modules pulled in by a refactor). This module reconciles
``file_changes.json`` with the actual ``git status --porcelain`` output,
adding any drifted file as an entry tagged ``"auto_added": true``.

Reviewer (independent expert) raised this in the B2 review: "the
architect writes file_changes.json completely by prompt self-discipline.
implementer may modify clean-list-outside files." We chose B2 to handle
this with a drift threshold (≥ 3 = skip auto-commit). B4 closes the
remaining gap — it makes the data structure honest so future tooling /
B3 stale-detection / human review sees the full picture.

Schema additions (v2)
---------------------

``file_changes.json``:

    {
      "schema_version": "2",
      "files": [
        { "path": ..., "action": ..., "desc": ..., "auto_added": true? },
        ...
      ]
    }

``tasks.json``:

    {
      "schema_version": "2",
      "required_packages": [...],
      "logic_analysis": [[path, desc, {"anchors": [...]}], ...],
      "task_list": [...]
    }

``schema_version`` is OPTIONAL — readers must default to ``"1"`` when
missing (legacy 2-field metas from pre-B1 runs).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from larkhelm.log import _debug_log


_GIT_STATUS_TIMEOUT_SEC = 30

# Schema version marker. ``"2"`` covers all post-B1 artifacts (anchors +
# auto_added fields). Legacy artifacts are version ``"1"``.
SCHEMA_VERSION = "2"


def _read_porcelain(cwd: str) -> "list[str]":
    """Return repo-relative paths from ``git status --porcelain``.

    Mirrors the parsing logic in ``workspace_finalize._collect_plan_artifacts``
    (rename targets, untracked, modified). Returns an empty list on any
    git failure — caller treats no-data as "no reconcile possible".
    """
    paths: list[str] = []
    try:
        proc = subprocess.run(
            ["git", "-c", "core.quotePath=false", "-C", cwd,
             "status", "--porcelain"],
            capture_output=True, text=True, timeout=_GIT_STATUS_TIMEOUT_SEC,
        )
        if proc.returncode != 0:
            return paths
        for line in proc.stdout.splitlines():
            if len(line) < 4:
                continue
            rest = line[3:].lstrip()
            if " -> " in rest:
                rest = rest.split(" -> ", 1)[1]
            paths.append(rest)
    except Exception as e:
        _debug_log(f"[WorkspaceReconcile] git status failed: {e}")
    return paths


def reconcile_file_changes(cwd: str) -> dict:
    """Reconcile ``<cwd>/.crew_workspace/file_changes.json`` with the
    actual git working tree.

    Returns a dict ``{"added": [...], "schema_version": ..., "noop": bool}``:

      * ``added``           — paths newly appended with ``auto_added=true``
      * ``schema_version``  — the file's schema_version (post-reconcile)
      * ``noop``            — True when nothing changed on disk

    Never raises. If the file is missing / unparseable / the workspace
    doesn't exist, returns a noop result.
    """
    ws = Path(cwd) / ".crew_workspace"
    fc_path = ws / "file_changes.json"
    if not fc_path.exists():
        return {"added": [], "schema_version": "0", "noop": True}

    try:
        raw = json.loads(fc_path.read_text(encoding="utf-8"))
    except Exception as e:
        _debug_log(f"[WorkspaceReconcile] file_changes.json parse failed: {e}")
        return {"added": [], "schema_version": "0", "noop": True}

    if not isinstance(raw, dict):
        return {"added": [], "schema_version": "0", "noop": True}

    files = raw.get("files")
    if not isinstance(files, list):
        files = []
        raw["files"] = files

    declared_paths: set[str] = set()
    for entry in files:
        if isinstance(entry, dict):
            p = entry.get("path")
            if p:
                declared_paths.add(p)

    actual_dirty = _read_porcelain(cwd)
    if not actual_dirty:
        return {"added": [], "schema_version": raw.get("schema_version", "1"),
                "noop": True}

    added: list[str] = []
    for p in actual_dirty:
        if p in declared_paths:
            continue
        declared_paths.add(p)
        files.append({
            "path":       p,
            "action":     "modify",   # safest default; reconcile can't know intent
            "desc":       "auto-reconciled drift after implementer/fixer",
            "auto_added": True,
        })
        added.append(p)

    if not added:
        return {"added": [], "schema_version": raw.get("schema_version", "1"),
                "noop": True}

    # Stamp schema_version on first write that touches the structure so
    # future readers can distinguish v1 (no auto_added) from v2.
    raw.setdefault("schema_version", SCHEMA_VERSION)

    try:
        fc_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    except Exception as e:
        _debug_log(f"[WorkspaceReconcile] file_changes.json write failed: {e}")
        return {"added": [], "schema_version": raw.get("schema_version", "1"),
                "noop": True}

    _debug_log(
        f"[WorkspaceReconcile] appended {len(added)} drift file(s) to "
        f"file_changes.json (auto_added=true)"
    )
    return {"added": added, "schema_version": raw["schema_version"], "noop": False}


def stamp_schema_version_on_tasks(cwd: str) -> bool:
    """Ensure ``tasks.json`` carries ``schema_version: "2"`` when it has
    the post-B1 anchor schema. Returns True if a write happened.

    Heuristic: if ``logic_analysis[i]`` has length ≥ 3 and the third
    slot is a dict, the architect emitted anchor metadata — stamp v2.
    Pure-2-tuple entries leave the file at v1 (no stamp).
    """
    ws = Path(cwd) / ".crew_workspace"
    tasks_path = ws / "tasks.json"
    if not tasks_path.exists():
        return False
    try:
        raw = json.loads(tasks_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    if raw.get("schema_version") == SCHEMA_VERSION:
        return False
    la = raw.get("logic_analysis") or []
    has_anchor = any(
        isinstance(e, (list, tuple)) and len(e) >= 3 and isinstance(e[2], dict)
        for e in la
    )
    if not has_anchor:
        return False
    raw["schema_version"] = SCHEMA_VERSION
    try:
        tasks_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
        return True
    except Exception as e:
        _debug_log(f"[WorkspaceReconcile] tasks.json schema stamp write failed: {e}")
        return False
