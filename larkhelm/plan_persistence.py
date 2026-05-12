"""Plan state persistence — survives bridge restart (U17).

Problem
-------
A ``/plan`` that's in flight when the bridge dies (cgroup OOM-kill,
SIGTERM during deploy, kernel panic, anything) loses its in-memory
``MultiPlanState``. The Feishu progress card the user sees becomes a
permanent "⏳ 思考中" ghost — the bridge no longer knows that plan
ever existed, so no error card is ever sent, no cancel signal reaches
any subprocess (they're dead anyway), and the user is left guessing
"is it still running? did it succeed silently? do I send it again?".

Scope of this module (deliberately narrow)
------------------------------------------
* **Persist a state snapshot** to ``DATA_DIR/_active_plans/<plan_id>.json``
  on every step status change + once at plan start. Only the fields the
  startup notifier needs are serialised — no ``threading.Event`` / ``Lock``,
  no ``cancel_ev``. The on-disk file is informational, not resumable.

* **Notify on restart**, NOT resume. ``resume_interrupted_plans`` is
  called once during bridge boot (in ``bridge.main`` alongside the
  existing ``resume_interrupted_crews``); it scans pending state files
  and pushes one card per affected chat:

      ⚠️ Plan 被中断 (bridge 重启) · MyPlan
      运行到第 3 / 5 步 ([dev] 实现 OAuth flow) 时中断。
      产出已保存在 .crew_workspace/。可重新发送同样需求复用
      或运行 /plan 全新开始。
      [按钮: 清除提示]

  The card button just removes the state file — no execution restart.
  Restarting execution would require rebuilding crew sub-task state
  that isn't checkpointed at step granularity, well beyond U17's value.

Why not in ``cmd_plan.py``
--------------------------
Keeps the persistence concern out of the hot path — ``_run_plan``'s
business logic stays focused on flow control, while this module owns
the on-disk schema and the "card on restart" UX. The two communicate
through three calls (``save_plan_state`` / ``delete_plan_state`` /
``mark_plan_phase``); the boundary is small enough that the persistence
behaviour can be swapped (e.g. to SQLite later) without touching
``cmd_plan``.

Fail-soft
---------
Every disk operation is wrapped — a plan run must never fail because
the persistence file couldn't be written. The startup notifier wraps
its whole scan loop so one corrupted state file can't prevent the
bridge from starting.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from larkhelm.log import _debug_log

if TYPE_CHECKING:
    # Import only for type hints to avoid a runtime circular import via
    # cmd_plan → plan_persistence → cmd_plan.
    from larkhelm.cmd_plan import MultiPlanState


# On-disk schema version. Bump when the JSON shape changes so the startup
# scanner can recognise + ignore (instead of crashing on) old files left
# over from a prior bridge build.
_STATE_SCHEMA_VERSION = 1

# Per-plan state file lives under ``DATA_DIR/_active_plans/<plan_id>.json``.
# Directory chosen for symmetry with the existing per-chat persistence
# folder ``DATA_DIR/_chat_state/`` — operators looking for "what state
# does the bridge keep on disk" find both in adjacent subdirs.
_DIRNAME = "_active_plans"

# Phases that mean "the plan thread already finished by the time the
# state file was last written"; the file only survives because the
# finally-block delete didn't get to run (e.g. SIGKILL between
# ``save_plan_state(phase=done)`` and the finally clause's
# ``delete_plan_state``). On next startup, surfacing these as
# "interrupted" would be a false alarm — the user already saw the
# proper completion / failure / cancellation card. Skip + delete.
# Audit ref: round-2 review #15.
_TERMINAL_PHASES = frozenset({"done", "failed", "cancelled"})

# Audit ref: round-2 review #11 — crash-loop flooding. When the bridge
# OOM-restarts repeatedly, the same plan would otherwise spawn one
# "⚠️ Plan 被中断" card per restart, drowning the user's chat. The
# notifier writes back ``notify_count`` + ``last_notified_at`` after
# each successful send and uses these thresholds to throttle:
#
#   * ``_FLOOD_THROTTLE_SEC`` — minimum gap between successive
#     notifications for the same plan. 30 min covers a typical
#     OOM-restart loop window without delaying meaningful re-notify
#     when a different problem causes a restart hours later.
#   * ``_MAX_NOTIFY_COUNT`` — after this many notifications without
#     user action (i.e. the "🗑️ 清除提示" button), the file is
#     auto-deleted on next scan. The user has either acknowledged
#     and moved on, or doesn't care to triage.
_FLOOD_THROTTLE_SEC = 30 * 60
_MAX_NOTIFY_COUNT = 3

# Serialize on-disk writes per plan_id. Without this, two near-simultaneous
# ``save_plan_state`` calls (e.g. step transition + phase change) could
# write half a record each. ``threading.Lock`` is enough — the persistence
# path isn't async-contended; concurrent calls only happen within a single
# plan thread + occasional heartbeat-style writes.
_write_locks: "dict[str, threading.Lock]" = {}
_write_locks_meta = threading.Lock()


def _state_dir() -> Path:
    """Return ``DATA_DIR/_active_plans``, creating it if needed.

    Read ``_cfg.DATA_DIR`` lazily — this module imports early enough
    that ``_init_runtime`` may not have run yet at import time on some
    paths (e.g. test fixtures).
    """
    import larkhelm.config as _cfg
    p = Path(_cfg.DATA_DIR) / _DIRNAME
    p.mkdir(parents=True, exist_ok=True)
    return p


def _state_path(plan_id: str) -> Path:
    # ``plan_id`` is generated by ``uuid.uuid4().hex[:12]`` so it's already
    # safe for use as a filename — alphanum only. No sanitisation needed.
    return _state_dir() / f"{plan_id}.json"


def _get_write_lock(plan_id: str) -> threading.Lock:
    with _write_locks_meta:
        lk = _write_locks.get(plan_id)
        if lk is None:
            lk = threading.Lock()
            _write_locks[plan_id] = lk
        return lk


# ── Public API: lifecycle hooks called by ``cmd_plan._run_plan`` ─────────

def save_plan_state(state: "MultiPlanState") -> None:
    """Persist the current plan state to disk. Idempotent + fail-soft.

    Called once at plan start and again on every step status / phase
    transition so the on-disk record matches what the user is seeing.
    Cost is one short JSON serialize + atomic file write (``os.replace``
    after writing a temp file), measured at < 1 ms even on slow disks
    — fine to call frequently.
    """
    try:
        payload = _serialise(state)
        path = _state_path(state.plan_id)
        tmp = path.with_suffix(".json.tmp")
        with _get_write_lock(state.plan_id):
            tmp.write_text(json.dumps(payload, ensure_ascii=False),
                           encoding="utf-8")
            os.replace(tmp, path)
    except Exception as e:
        _debug_log(f"[PlanPersist] save_plan_state({state.plan_id[:8]}) failed: {e}")


def delete_plan_state(plan_id: str) -> None:
    """Remove the on-disk state for a finished / cancelled plan.

    Called from the ``finally`` block of ``_run_plan`` regardless of
    success / failure / cancel — once the plan thread has fully exited,
    its state is no longer "interrupted by bridge death". Idempotent.
    """
    try:
        _state_path(plan_id).unlink(missing_ok=True)
        with _write_locks_meta:
            _write_locks.pop(plan_id, None)
    except Exception as e:
        _debug_log(f"[PlanPersist] delete_plan_state({plan_id[:8]}) failed: {e}")


def list_pending_plan_states() -> list[dict]:
    """Return all on-disk plan state records (the "interrupted" set).

    Reading is read-only — no side effects, safe to call multiple times.
    Each entry is the dict shape produced by ``_serialise`` (see schema
    doc on that function). Corrupted / wrong-schema files are silently
    skipped so the startup notifier never crashes on bad data.
    """
    try:
        d = _state_dir()
    except Exception as e:
        _debug_log(f"[PlanPersist] list state dir failed: {e}")
        return []
    out: list[dict] = []
    for f in d.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            if data.get("schema_version") != _STATE_SCHEMA_VERSION:
                _debug_log(f"[PlanPersist] skipping {f.name}: schema mismatch")
                continue
            out.append(data)
        except Exception as e:
            _debug_log(f"[PlanPersist] skipping unreadable {f.name}: {e}")
    return out


# ── Internals — serialisation ────────────────────────────────────────────

def _serialise(state: "MultiPlanState") -> dict:
    """Snapshot the bits of ``MultiPlanState`` the startup notifier needs.

    Skips non-pickleable members (``threading.Event`` / ``Lock``) and
    state-bag fields that have no meaning across a bridge restart
    (``_confirm_ev`` / ``_confirm_result``). The kept set is intentionally
    minimal — large state files just slow the startup scan, and the
    notifier only needs:

      * which plan + which chat (so the card goes to the right place)
      * how far it got (which step number / what its desc was)
      * what the current phase is when it died (so we can say "running"
        vs "waiting for confirm" vs "failed mid-step")

    Schema (locked at version 1):

      {
        "schema_version":     1,
        "plan_id":            "abc123def456",
        "chat_id":            "oc_xxxx",
        "title":              "<plan title>",
        "phase":              "running" | "waiting" | "done" | ...,
        "current_idx":        <int>,
        "start_time":         <epoch float>,
        "saved_at":           <epoch float>,
        "notify_count":       <int>,         # round-2 follow-up #11
        "last_notified_at":   <epoch float>, # round-2 follow-up #11
        "steps":              [{idx, type, desc, status, error, retry_count}],
      }

    ``notify_count`` / ``last_notified_at`` start at 0 and are
    updated by the startup notifier after each successful notify
    (see ``_persist_notify_state``). They are forward-compatible
    additions — older records without these keys default via
    ``.get(..., 0)`` so the schema version is **not** bumped.
    """
    return {
        "schema_version":   _STATE_SCHEMA_VERSION,
        "plan_id":          state.plan_id,
        "chat_id":          state.chat_id,
        "title":            state.title,
        "phase":            state.phase,
        "current_idx":      state.current_idx,
        "start_time":       state.start_time,
        "saved_at":         time.time(),
        "notify_count":     0,
        "last_notified_at": 0.0,
        "steps": [
            {
                "idx":         s.idx,
                "type":        s.type,
                "desc":        s.desc,
                "status":      s.status,
                "error":       (s.error or "")[:200],
                "retry_count": s.retry_count,
                # Round-2 audit #7: persist per-step timing so the
                # interrupted-plan card can show "卡在该步 X 分钟" — a
                # 3-minute hang and a 3-hour hang look identical without
                # this. ``None`` (step never started / never finished) is
                # preserved via ``or None`` since dataclass default float
                # would round-trip through json as 0.0 and lose the
                # "never started" distinction.
                "start_time":  s.start_time,
                "end_time":    s.end_time,
            }
            for s in state.steps
        ],
    }


# ── Bridge-startup notifier ──────────────────────────────────────────────

def resume_interrupted_plans() -> int:
    """Called from ``bridge.main`` on startup. Sends one notification card
    per interrupted plan + does NOT auto-resume execution.

    Returns the count of cards sent (mostly useful for the startup log
    + tests). Each on-disk state file is left in place so the user's
    card-button "清除提示" callback can find + delete it; a periodic
    GC of orphaned files isn't necessary because the user is exactly
    the right person to triage them.

    Wraps the whole scan loop in a broad except so one bad file (or
    a Feishu API outage during the scan) never blocks bridge startup.
    """
    try:
        records = list_pending_plan_states()
    except Exception as e:
        _debug_log(f"[PlanPersist] resume scan failed: {e}")
        return 0
    if not records:
        return 0

    sent = 0
    for rec in records:
        try:
            if _notify_chat_of_interrupted_plan(rec):
                sent += 1
        except Exception as e:
            _debug_log(
                f"[PlanPersist] notify failed for plan="
                f"{rec.get('plan_id', '?')[:8]}: {e}"
            )
    if sent:
        _debug_log(f"[PlanPersist] notified {sent} interrupted plan(s) on startup")
    return sent


def _format_step_duration(step: dict) -> str:
    """Build a "(已运行 X 分钟 / 已耗时 Y 分钟)" suffix from a step's
    start_time / end_time. Empty string when timing data is missing or
    out of range — caller appends conditionally.

    Cases (in priority order):
      * step is not a dict (None / list / str) → ``""``  (defensive — see below)
      * No start_time → ``""``                          (step never started)
      * start_time but no end_time → "已运行 N 分钟"     (was active at crash)
      * Both present → "已耗时 N 分钟"                  (finished before crash)

    Sub-minute durations show as "N 秒" to avoid the common-but-useless
    "已运行 0 分钟". Very long durations use "N 小时" past 90 minutes.

    Defensive type check: ``rec.get("steps")`` from a corrupted on-disk
    record can yield anything (None, list-of-strings if someone schema-
    bumped wrong, etc). The outer ``_notify_chat_of_interrupted_plan``
    catches exceptions globally, but failing here would silently kill
    the whole notification card. Returning "" lets the caller fall
    back to the bare ``step_line`` without duration suffix, which is
    still useful information for the user. Round-2 audit #6 follow-up.
    """
    if not isinstance(step, dict):
        return ""
    start = step.get("start_time")
    end   = step.get("end_time")
    if not start:
        return ""
    try:
        start_f = float(start)
    except (TypeError, ValueError):
        return ""
    if end:
        try:
            elapsed = float(end) - start_f
        except (TypeError, ValueError):
            return ""
        verb = "已耗时"
    else:
        elapsed = time.time() - start_f
        verb = "已运行"
    if elapsed < 0:
        return ""
    if elapsed < 60:
        return f"({verb} {int(elapsed)} 秒)"
    if elapsed < 90 * 60:
        return f"({verb} {int(elapsed / 60)} 分钟)"
    return f"({verb} {int(elapsed / 3600)} 小时)"


def _format_workspace_artefacts(chat_id: str) -> str:
    """Render a "**产出**: ..." line listing files in ``.crew_workspace/``
    and their sizes (or "未生成" if absent).

    Without this, "产出已保存在 .crew_workspace/" was opaque — the user
    had to drop to a terminal and ``ls -la`` just to know whether the
    interrupted plan got past the PM stage. Now they see at a glance
    whether there's meaningful partial work to resume from.

    Fail-soft: any IO error returns the original opaque hint so the
    card never breaks.
    """
    fallback = ("**产出**: 已保存在 `.crew_workspace/`（design.md / tasks.json"
                " / changes.md / review.md 视进度而定）。")
    try:
        from larkhelm.chat_state import _get_cwd
        cwd = _get_cwd(chat_id)
        if not cwd:
            return fallback
        ws = Path(cwd) / ".crew_workspace"
        if not ws.is_dir():
            return fallback
        # Plan-relevant files in roughly the order a /plan run produces them.
        # workspace_meta.json is internal bookkeeping — skip from the card.
        # Plan-relevant files in roughly the order a /plan run produces them.
        # workspace_meta.json is internal bookkeeping — skip from the card.
        # ``qa_report.md`` is produced by the [test] step (cmd_plan.py:545+)
        # and was missing from the original list — audit follow-up #20.
        watched = ["prd.md", "design.md", "tasks.json", "file_changes.json",
                   "changes.md", "qa_report.md", "review.md"]
        rows: list[str] = []
        any_present = False
        for name in watched:
            p = ws / name
            if p.exists():
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                if size > 0:
                    rows.append(f"  - {name} ({_humanise_bytes(size)})")
                    any_present = True
                else:
                    rows.append(f"  - {name} (空)")
        if not any_present:
            # "PM 阶段" is internal /plan jargon (the first crew sub-agent role);
            # most users don't know what PM is in this context. Audit follow-up
            # #19: prefer plain "尚未生成任何文档" phrasing instead.
            return ("**产出**: `.crew_workspace/` 内文件均为空或不存在 — "
                    "plan 中断时还未生成任何文档。")
        return "**产出**:\n" + "\n".join(rows)
    except Exception as e:
        _debug_log(f"[PlanPersist] artefact listing failed: {e}")
        return fallback


def _humanise_bytes(n: int) -> str:
    """``5421`` → ``"5.3 KB"``, ``523`` → ``"523 B"``. Plan artefacts top
    out around 50KB (review.md / changes.md for a multi-step plan), so
    we only need B / KB precision."""
    if n < 1024:
        return f"{n} B"
    return f"{n / 1024:.1f} KB"


def _notify_chat_of_interrupted_plan(rec: dict) -> bool:
    """Build + send the "Plan 被中断" card for a single interrupted plan.

    Returns True iff a card was actually sent. ``False`` indicates one
    of:

      * malformed record (missing chat_id / plan_id) → can't send
      * **terminal-phase record** (#15) — the plan finished cleanly but
        its file outlived the finally-block delete (likely a SIGKILL
        during ``finalize_workspace``). The user already saw the proper
        completion card; surfacing it as "interrupted" would be a false
        alarm. Silently delete the stale file.
      * **throttle-skipped** (#11) — either we've already notified the
        max number of times (``_MAX_NOTIFY_COUNT``) or we're inside the
        ``_FLOOD_THROTTLE_SEC`` cooldown after the most recent send.
        Bridge crash-loops would otherwise spam the chat with one card
        per restart. After ``_MAX_NOTIFY_COUNT`` notifications the file
        is auto-deleted (user clearly isn't going to triage).
    """
    from larkhelm.lark_client import send_card

    chat_id = rec.get("chat_id", "")
    plan_id = rec.get("plan_id", "")
    title   = rec.get("title", "(无标题)")
    if not chat_id or not plan_id:
        return False

    steps = rec.get("steps", []) or []
    cur_idx = int(rec.get("current_idx", 0))
    total = len(steps)
    phase = rec.get("phase", "running")

    # #15 — terminal-phase record. Plan thread already ran the
    # phase-write line; the only reason this file still exists is the
    # finally-block delete didn't get a chance (SIGKILL mid-finalize).
    # The user already saw the real "✅/❌ Plan 完成" card — surfacing
    # this as "interrupted" would confuse, not inform. Drop silently.
    if phase in _TERMINAL_PHASES:
        _debug_log(
            f"[PlanPersist] dropping terminal-phase record "
            f"plan={plan_id[:8]} phase={phase} (no notify, file removed)"
        )
        delete_plan_state(plan_id)
        return False

    # #11 — throttle. Read previous notify state from the record (which
    # ``_serialise`` initialised to 0 + we overwrite via
    # ``_persist_notify_state`` below after every successful send).
    notify_count     = int(rec.get("notify_count", 0))
    last_notified_at = float(rec.get("last_notified_at", 0.0))
    now              = time.time()

    if notify_count >= _MAX_NOTIFY_COUNT:
        # Auto-GC: user has had 3 chances to triage; if they haven't
        # pressed the button by now, the file is just clutter. Remove
        # it so subsequent restarts don't keep scanning + skipping it.
        _debug_log(
            f"[PlanPersist] auto-removing plan={plan_id[:8]} after "
            f"{notify_count} notifications (no user action)"
        )
        delete_plan_state(plan_id)
        return False

    cooldown = now - last_notified_at
    if last_notified_at > 0 and cooldown < _FLOOD_THROTTLE_SEC:
        _debug_log(
            f"[PlanPersist] throttling plan={plan_id[:8]} — "
            f"last notified {int(cooldown)}s ago (< "
            f"{_FLOOD_THROTTLE_SEC}s window); file retained for next restart"
        )
        return False

    # Locate the step that was running when the bridge died — use
    # ``current_idx`` first, fall back to the first non-done step. This
    # mirrors how the progress card itself shows the "active step".
    active_step = None
    if 0 <= cur_idx < total:
        active_step = steps[cur_idx]
    else:
        for s in steps:
            if s.get("status") not in ("done", "skipped"):
                active_step = s
                break

    if active_step is not None:
        step_line = (
            f"运行到第 {active_step.get('idx', cur_idx) + 1} / {total} 步 "
            f"(**[{active_step.get('type', '?')}]** "
            f"{(active_step.get('desc') or '').strip()[:80]}) 时中断。"
        )
        # Audit #7: render step duration if start_time is present. A
        # 3-min hang vs a 3-hour hang look identical without this — the
        # user can't tell if the step was actually doing work when the
        # bridge died or had been wedged for ages.
        step_dur = _format_step_duration(active_step)
        if step_dur:
            step_line += f" {step_dur}"
    else:
        step_line = f"共 {total} 个阶段，中断时阶段未知。"

    age_min = max(0, int((time.time() - float(rec.get("saved_at", 0))) / 60))
    age_str = f"{age_min} 分钟前" if age_min < 60 else f"{age_min // 60} 小时前"

    # N5: list the .crew_workspace artefacts that survived the crash with
    # their sizes — lets the user decide "is there enough partial work
    # to resume from, or do I just /plan again". Without this, "产出已
    # 保存在 .crew_workspace/" is opaque — they have to open a terminal
    # and `ls -la` to find out.
    artefacts_line = _format_workspace_artefacts(chat_id)

    body = (
        f"**{title}**\n\n"
        f"{step_line}\n"
        f"上次心跳: {age_str} (phase=`{phase}`)\n\n"
        f"{artefacts_line}\n\n"
        "可重发同样需求复用前次产出，或运行 `/plan` 全新开始。"
    )
    buttons = [("🗑️ 清除提示", f"plan_persist_clear:{plan_id}")]
    send_card(chat_id, f"⚠️ Plan 被中断 (bridge 重启)", body,
              color="orange", buttons=buttons)
    # #11 — record this notification so the next bridge restart can
    # throttle / GC accordingly. Persist failure here doesn't fail
    # the notify (the card was already sent); next scan just won't
    # know about it and may double-send, which is the previous bug
    # we're trying to fix — log loudly.
    try:
        _persist_notify_state(plan_id,
                              notify_count=notify_count + 1,
                              last_notified_at=now)
    except Exception as e:
        _debug_log(
            f"[PlanPersist] _persist_notify_state failed for "
            f"plan={plan_id[:8]}: {e} — next restart will not throttle"
        )
    return True


def _persist_notify_state(plan_id: str, *, notify_count: int,
                          last_notified_at: float) -> None:
    """Update only the throttle fields on an existing state record,
    leaving the rest of the snapshot untouched.

    Read-modify-write inside the per-plan write lock so a concurrent
    ``save_plan_state`` from a re-running plan thread (rare, but
    possible if the user's chat had crash → restart → user re-runs
    the same plan_id manually) doesn't get clobbered. If the record
    has vanished between read and write — e.g. the user pressed
    "🗑️ 清除提示" mid-call — we drop the update silently rather than
    re-create the file.
    """
    path = _state_path(plan_id)
    lock = _get_write_lock(plan_id)
    with lock:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return   # user cleared it while we were sending; respect that
        if not isinstance(data, dict):
            return
        data["notify_count"]     = int(notify_count)
        data["last_notified_at"] = float(last_notified_at)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, path)


def clear_plan_state_button(plan_id: str) -> bool:
    """Card-button callback: user pressed "🗑️ 清除提示" on the
    interrupted-plan notification. Just deletes the on-disk state.
    Returns True iff a file actually existed (for the button confirmation
    text). Safe to call multiple times.
    """
    path = _state_path(plan_id)
    existed = path.exists()
    delete_plan_state(plan_id)
    return existed
