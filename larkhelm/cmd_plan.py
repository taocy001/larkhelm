"""
larkhelm · /plan multi-phase dev orchestrator

Executes a sequence of [dev], [review], [fix], [test] steps sequentially,
with a human-confirmation breakpoint between each step.

Format:
    /plan
    可选标题行
    [dev] 实现用户登录
    [dev] 实现商品目录
    [review] 检查数据安全
    [fix] 修复遗留问题
    [test] 回归测试

Or load from a Feishu doc:
    /plan https://feishu.cn/docx/xxx
"""
from __future__ import annotations

import dataclasses
import re
import threading
import time
import uuid
from enum import Enum
from pathlib import Path

from functools import lru_cache

from larkhelm.log import _debug_log, log_entry
from larkhelm.card_builder import _fmt_elapsed, _make_card
from larkhelm.plan_retry import PlanRetryEngine


# NIT-2 (P3 review): ``PlanRetryEngine`` is stateless — just a strategy
# string. Building a fresh instance on every step failure is wasteful even
# though the cost is negligible. Cache by strategy so the hot retry-failure
# path on a long plan reuses the same engine. Three valid strategies + a
# slot for any unknown passthrough (which defaults to ``"off"`` internally)
# fits in 4 entries.
@lru_cache(maxsize=4)
def _get_plan_retry_engine(strategy: str) -> PlanRetryEngine:
    return PlanRetryEngine(strategy)


def _resolve_plan_retry_strategy() -> str:
    """Look up the operator-configured plan_retry_strategy lazily.

    Lazy because ``cmd_plan`` imports happen before ``_init_runtime``
    on the cold path of some bootstrap tests, so reading the module
    global at module-import time would freeze the value at "off".
    """
    try:
        import larkhelm.config as _cfg
        return str(getattr(_cfg, "PLAN_RETRY_STRATEGY", "off") or "off").lower()
    except Exception:
        return "off"


def decide_retry_action(
    strategy: str,
    step_retry_count: int,
    auto_retried: int,
    max_retries: int,
) -> tuple[str, str]:
    """Pure routing helper for the /plan failure branch (P3 REQ-04).

    Returns ``(action, reason)`` where:

    * ``action`` is ``"auto_retry"`` (silently retry the same step) or
      ``"user_prompt"`` (render the retry/continue/cancel card and wait
      for the operator).
    * ``reason`` is one of ``"below_threshold"``, ``"retries_exhausted"``,
      ``"manual_required"``, ``"disabled"`` — purely for logging.

    Strategy semantics:

    * ``"off"``    — auto-retry only while ``auto_retried < max_retries``
      (the P2 behaviour; ``PlanRetryEngine`` is bypassed).
    * ``"now"``    — consult :class:`PlanRetryEngine` with
      ``step_retry_count`` so retries persist across user-initiated
      re-runs that already incremented the counter.
    * ``"manual"`` — always defer to the user card.
    * unknown     — collapse to ``"off"`` semantics for safety.
    """
    s = (strategy or "off").lower()
    if s == "manual":
        return ("user_prompt", "manual_required")
    if s == "now":
        engine = _get_plan_retry_engine("now")
        decision = engine.evaluate({
            "retry_count": step_retry_count,
            "max_retries": max_retries,
        })
        if decision.should_retry:
            return ("auto_retry", decision.reason)
        return ("user_prompt", decision.reason)
    if auto_retried < max_retries:
        return ("auto_retry", "below_threshold")
    return ("user_prompt", "retries_exhausted")


# ── Constants ────────────────────────────────────────────────────

_STATUS_ICON = {
    "pending":  "⏸",
    "running":  "⚙️",
    "done":     "✅",
    "failed":   "❌",
    "skipped":  "⏭",
}

_TYPE_LABEL = {
    "dev":    "Dev",
    "review": "Review",
    "fix":    "Fix",
    "test":   "Test",
}


# ── Data structures ──────────────────────────────────────────────

@dataclasses.dataclass
class PlanStep:
    idx:         int
    type:        str           # "dev" | "review" | "fix" | "test"
    desc:        str
    status:      str           = "pending"
    error:       str           = ""
    start_time:  float | None  = None
    end_time:    float | None  = None
    retry_count: int           = 0


class PlanPhase(Enum):
    """State-machine phases for a multi-step /plan run.

    The earlier ``phase: str`` field was a pure string token compared in
    ~25 sites across this module; a typo (``"runing"``) would silently
    skip the branch and leak through to the persistence file. P3-7
    cleanup converts the contract to an Enum so the typo is a
    ``ValueError`` at the boundary instead of a silent miss.

    Values are preserved verbatim from the pre-migration strings so
    ``state.phase.value`` round-trips byte-identically through the
    persistence JSON — existing on-disk plan_state files keep working.
    """
    PLANNING   = "planning"
    CONFIRMING = "confirming"
    RUNNING    = "running"
    WAITING    = "waiting"
    DONE       = "done"
    CANCELLED  = "cancelled"
    FAILED     = "failed"


def _coerce_phase(value) -> "PlanPhase":
    """Boundary helper: accept ``PlanPhase`` or its string value.

    Used by:
      * test fixtures that still pass ``phase="running"`` for ergonomics
      * the persistence loader, where the on-disk JSON stores the raw value
      * any third-party tooling that pokes at ``state.phase`` from outside
        this module (none known today, but the surface is public-ish)

    Unknown strings raise ``ValueError`` so a typo is caught loudly.
    """
    if isinstance(value, PlanPhase):
        return value
    return PlanPhase(value)


@dataclasses.dataclass
class MultiPlanState:
    plan_id:         str
    chat_id:         str
    title:           str
    steps:           list[PlanStep]
    card_mid:        str | None        = None
    trigger_msg_id:  str | None        = None
    cancel_ev:       threading.Event   = dataclasses.field(default_factory=threading.Event)
    lock:            threading.Lock    = dataclasses.field(default_factory=threading.Lock)
    phase:           PlanPhase         = PlanPhase.RUNNING
    current_idx:     int               = 0
    start_time:      float             = dataclasses.field(default_factory=time.time)
    _confirm_ev:     threading.Event   = dataclasses.field(default_factory=threading.Event)
    _confirm_result: str               = "continue"  # "continue" | "skip" | "cancel" | "retry"
    last_step_failed: bool             = False        # True when entering waiting after a failure
    max_retries:     int               = 1            # auto-retries before notifying user
    no_confirm:      bool              = False        # skip between-step confirmations

    def __post_init__(self) -> None:
        # Accept either ``PlanPhase`` or the raw string value at construction
        # so existing test fixtures (``phase="running"``) and persistence
        # round-tripping (raw string from JSON) keep working with no caller
        # changes. Unknown strings raise ``ValueError`` here — caught loudly
        # at the boundary rather than silently miscompared later.
        if not isinstance(self.phase, PlanPhase):
            self.phase = _coerce_phase(self.phase)


# plan_id → MultiPlanState
_active_plans:      dict[str, MultiPlanState] = {}
_active_plans_lock: threading.Lock            = threading.Lock()

# Test hook: poll interval used by _wait_for_confirm_or_cancel. Tests override
# this to a small value so chat-cancel propagation is observed quickly.
_WAIT_POLL_INTERVAL: float = 1.0


# ── Parser ───────────────────────────────────────────────────────

# Keyword table for the natural-language fallback in _parse_plan.
# Structure: {type: (chinese_keywords, english_tokens)}.
# Iteration order is also the conflict-resolution priority: dev > fix > review > test.
_STEP_KEYWORDS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "dev":    (("开发", "实现", "编写"), ()),
    "fix":    (("修复", "修改"),         ("fix",)),
    "review": (("检视", "审查"),         ("review",)),
    "test":   (("测试", "验证", "回归"), ("test",)),
}

_STEP_TYPE_PRIORITY: tuple[str, ...] = ("dev", "fix", "review", "test")

_EXPLICIT_TYPE_RE = re.compile(r'\[(dev|review|fix|test)\]\s*(.*)', re.IGNORECASE)

_LLM_CLASSIFY_PROMPT: str = """\
你是一个 plan step 分类器。下面每行是用户写的一个步骤描述。
请把每行归类为 dev / review / fix / test 之一，并按 `[类型] 描述` 的格式逐行输出，
**不要输出任何其他内容**（不要序号、不要解释、不要空行）。
未知类别一律归 dev。

输入：
{lines}
"""


def _strip_prefix_kw(line: str, kw: str, ignore_case: bool) -> str:
    """If ``line`` starts with ``kw`` (optionally case-insensitively),
    strip it and any following whitespace; otherwise return ``line`` unchanged.
    Falls back to the original line if stripping leaves an empty string.
    """
    head = line[: len(kw)]
    matches = head.lower() == kw.lower() if ignore_case else head == kw
    if not matches:
        return line
    rest = line[len(kw):].lstrip()
    return rest if rest else line


def _map_step_type(line: str) -> tuple[str, str] | None:
    """Match keyword table; return ``(type, stripped_desc)`` or ``None``.

    - Chinese keywords: substring (``in``) match.
    - English tokens: ``\\b<token>\\b`` regex with ``re.IGNORECASE``.
    - On multi-type hit, picks by ``_STEP_TYPE_PRIORITY`` and ``_debug_log`` the conflict.
    - ``desc`` strips the triggering keyword only when it is the line prefix.

    English token behaviour in mixed CJK context
    --------------------------------------------
    Python 3's ``re`` defaults to Unicode ``\\w`` — meaning CJK characters
    count as "word" chars. That makes ``\\b<token>\\b`` reject English
    tokens embedded in Chinese: ``re一下view`` and ``走fix流程`` both
    return ``None`` because there's no word boundary between ``走`` /
    ``流`` and ``fix``.

    This is **intentional**, not a bug: it avoids false-positive
    matches like ``previewer`` triggering ``review``, and matches the
    user's intuition that ``fix`` written inside a Chinese sentence
    isn't really an English keyword invocation. The token branch is
    designed for whitespace-separated English input (``Review API`` /
    ``FIX login bug``); pure-Chinese input goes through the substring
    branch above.

    The downside is that a user typing ``开fix掉那个 bug`` gets a None
    here (no Chinese keyword present either) and ends up in the LLM
    fallback — slower but still correct.
    """
    stripped = line.strip()
    if not stripped:
        return None

    hits: list[tuple[str, str]] = []   # (type, desc_after_strip)
    for typ in _STEP_TYPE_PRIORITY:
        zh_kws, en_kws = _STEP_KEYWORDS[typ]
        zh_hit_kw: str | None = None
        for kw in zh_kws:
            if kw in stripped:
                zh_hit_kw = kw
                break
        if zh_hit_kw is not None:
            hits.append((typ, _strip_prefix_kw(stripped, zh_hit_kw, ignore_case=False)))
            continue
        en_hit_kw: str | None = None
        for tok in en_kws:
            if re.search(rf'\b{re.escape(tok)}\b', stripped, re.IGNORECASE):
                en_hit_kw = tok
                break
        if en_hit_kw is not None:
            hits.append((typ, _strip_prefix_kw(stripped, en_hit_kw, ignore_case=True)))

    if not hits:
        return None
    if len(hits) > 1:
        _debug_log(
            f"[Plan] keyword conflict on line: {line!r} → "
            f"types={[h[0] for h in hits]}, picked={hits[0][0]}"
        )
    return hits[0][0], hits[0][1]


def _llm_classify_steps(lines: list[str], chat_id: str | None = None) -> list[tuple[str, str]]:
    """Batch LLM classification for ambiguous /plan lines.

    Single ``_spawn_claude_proc`` call; prompt enforces ``[type] desc`` per line.
    Output is parsed with the same regex used for explicit syntax. Output rows
    that fail to parse are silently dropped and ``_debug_log``-ed. Any exception
    (subprocess failure, timeout, cancellation) propagates up — ``_parse_plan``
    catches it and drops the whole ambiguous batch (fail-soft per PRD §3 P1).
    """
    if not lines:
        return []

    from larkhelm.ai_runner import _spawn_claude_proc
    from larkhelm.chat_state import _get_cwd
    from larkhelm.perm import grant_yolo, revoke_yolo

    cwd = _get_cwd(chat_id) if chat_id else None
    ns  = f"{chat_id or 'plan'}__classifier_{uuid.uuid4().hex[:8]}"
    payload = "\n".join(l.strip() for l in lines)
    prompt  = _LLM_CLASSIFY_PROMPT.format(lines=payload)

    grant_yolo(ns)
    try:
        output = _spawn_claude_proc(
            chat_id=ns, message=prompt, sid=None, cwd=cwd,
            cancel_ev=None, on_text=None, allow_retry=False,
            session_namespace=ns,
        )
    finally:
        revoke_yolo(ns)

    results: list[tuple[str, str]] = []
    for raw in (output or "").splitlines():
        out_line = raw.strip()
        if not out_line:
            continue
        m = _EXPLICIT_TYPE_RE.match(out_line)
        if not m:
            _debug_log(f"[Plan] classifier output unparseable: {out_line!r}")
            continue
        results.append((m.group(1).lower(), m.group(2).strip()))
    return results


def _parse_plan(text: str) -> tuple[str, list[PlanStep]]:
    """Return (title, steps) from /plan body text.

    Three-level fallback (PRD §1.2): explicit ``[type]`` syntax → keyword table
    (``_map_step_type``) → batched LLM classification (``_llm_classify_steps``).
    The LLM call only happens when at least one line is ambiguous. Failures in
    the LLM step are fail-soft: the ambiguous batch is dropped silently and the
    /plan still launches with whatever explicit/keyword steps were parsed.
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    title = "多阶段开发计划"

    pass1_slots: list[PlanStep | None] = []
    ambiguous_lines: list[tuple[int, str]] = []   # (slot_idx, original_line)
    candidate_title_slot: int | None = None       # reserved slot for the title

    for i, line in enumerate(lines):
        m = _EXPLICIT_TYPE_RE.match(line)
        if m:
            pass1_slots.append(PlanStep(idx=0, type=m.group(1).lower(),
                                        desc=m.group(2).strip()))
            continue
        kw = _map_step_type(line)
        if kw is not None:
            pass1_slots.append(PlanStep(idx=0, type=kw[0], desc=kw[1]))
            continue
        # Tentative title: only the very first line, only if it would otherwise
        # be ambiguous, and only if no step has been parsed yet. Whether it
        # ultimately *becomes* the title is decided after Pass 1.
        if i == 0 and not pass1_slots:
            candidate_title_slot = len(pass1_slots)
            pass1_slots.append(None)
            continue
        slot_idx = len(pass1_slots)
        pass1_slots.append(None)
        ambiguous_lines.append((slot_idx, line))

    # Title resolution: keep the candidate as title only when at least one
    # explicit/keyword step exists; otherwise demote it to ambiguous so it
    # gets a fair LLM classification (PRD AC-03).
    if candidate_title_slot is not None:
        has_concrete_step = any(s is not None for s in pass1_slots)
        if has_concrete_step:
            title = lines[candidate_title_slot]
        else:
            ambiguous_lines.insert(
                0, (candidate_title_slot, lines[candidate_title_slot])
            )

    if ambiguous_lines:
        ambiguous_payload = [orig for _, orig in ambiguous_lines]
        try:
            classified = _llm_classify_steps(ambiguous_payload)
        except Exception as e:
            _debug_log(
                f"[Plan] LLM fallback failed: {e}; "
                f"dropping {len(ambiguous_lines)} ambiguous line(s)"
            )
            classified = []
        for (slot_idx, _orig), parsed in zip(ambiguous_lines, classified):
            t, d = parsed
            if t not in _STEP_TYPE_PRIORITY:
                _debug_log(f"[Plan] classifier returned unknown type {t!r}; dropping line")
                continue
            pass1_slots[slot_idx] = PlanStep(idx=0, type=t, desc=d.strip())

    steps: list[PlanStep] = []
    for slot in pass1_slots:
        if slot is None:
            continue
        slot.idx = len(steps)
        steps.append(slot)
    return title, steps


# ── Card ─────────────────────────────────────────────────────────

def _build_plan_card(state: MultiPlanState) -> str:
    elapsed = _fmt_elapsed(time.time() - state.start_time)
    n_done  = sum(1 for s in state.steps if s.status in ("done", "skipped"))
    n_total = len(state.steps)

    if state.phase == PlanPhase.PLANNING:
        title, color = "🧠 Plan · 生成计划中…", "grey"
    elif state.phase == PlanPhase.CONFIRMING:
        title, color = f"📋 Plan 已生成 · 共 {n_total} 步，确认后开始执行", "blue"
    elif state.phase == PlanPhase.RUNNING:
        title, color = f"⚙️ Plan · {n_done}/{n_total} ({elapsed})", "blue"
    elif state.phase == PlanPhase.WAITING:
        if state.last_step_failed:
            title, color = f"⚠️ Plan · 步骤失败，请处理  {n_done}/{n_total} ({elapsed})", "orange"
        else:
            title, color = f"⏸ Plan · 等待确认  {n_done}/{n_total} ({elapsed})", "yellow"
    elif state.phase == PlanPhase.DONE:
        title, color = f"✅ Plan · 完成  {n_total} 阶段  ({elapsed})", "green"
    elif state.phase == PlanPhase.CANCELLED:
        title, color = f"🛑 Plan · 已取消 ({elapsed})", "orange"
    else:
        title, color = f"❌ Plan · 失败 ({elapsed})", "red"

    lines = [f"**{state.title}**\n"]
    for s in state.steps:
        icon  = _STATUS_ICON.get(s.status, "?")
        label = _TYPE_LABEL.get(s.type, s.type.upper())
        t_str = ""
        if s.start_time and s.end_time:
            t_str = f" · {_fmt_elapsed(s.end_time - s.start_time)}"
        elif s.start_time and s.status == "running":
            t_str = f" · {_fmt_elapsed(time.time() - s.start_time)}…"
        marker = " ◀" if s.status == "running" else ""
        retry_str = f" 🔄{s.retry_count}" if s.retry_count > 0 and s.status == "running" else ""
        lines.append(f"{icon} **[{label}]** {s.desc}{t_str}{retry_str}{marker}")
        if s.status == "failed" and s.error:
            lines.append(f"   ⚠️ {s.error[:120]}")

    body = "\n".join(lines)

    if state.phase == PlanPhase.PLANNING:
        return _make_card(title, f"**需求：** {state.title}\n\n生成多阶段执行计划中，请稍候…",
                          color=color,
                          buttons=[("🛑 取消", f"plan_cancel:{state.plan_id}")])

    if state.phase == PlanPhase.CONFIRMING:
        return _make_card(title, body, color=color,
                          buttons=[
                              ("▶ 开始执行", f"plan_continue:{state.plan_id}"),
                              ("🛑 取消", f"plan_cancel:{state.plan_id}"),
                          ])

    if state.phase == PlanPhase.RUNNING:
        return _make_card(title, body, color=color,
                          buttons=[("🛑 取消", f"plan_cancel:{state.plan_id}")])

    if state.phase == PlanPhase.WAITING:
        # Show failed step error if applicable
        if state.last_step_failed:
            failed_steps = [s for s in state.steps if s.status == "failed"]
            if failed_steps:
                fs = failed_steps[-1]
                err_detail = fs.error[:200] if fs.error else "执行失败（无详细信息）"
                body += f"\n\n---\n⚠️ **[{_TYPE_LABEL.get(fs.type, fs.type.upper())}] {fs.desc[:60]}** 失败：\n{err_detail}"
        # Show next step info
        idx = state.current_idx
        if idx < n_total and not state.last_step_failed:
            nxt    = state.steps[idx]
            nlabel = _TYPE_LABEL.get(nxt.type, nxt.type.upper())
            body  += f"\n\n---\n**下一步：** [{nlabel}] {nxt.desc}"
        buttons = []
        if state.last_step_failed:
            buttons.append(("🔄 重试本步", f"plan_retry:{state.plan_id}"))
        buttons.append(("▶ 继续", f"plan_continue:{state.plan_id}"))
        if not state.last_step_failed:
            buttons.append(("⏭ 跳过下一步", f"plan_skip:{state.plan_id}"))
        buttons.append(("🛑 取消", f"plan_cancel:{state.plan_id}"))
        return _make_card(title, body, color=color, buttons=buttons)

    return _make_card(title, body, color=color)


def _update_plan_card(state: MultiPlanState) -> None:
    if not state.card_mid:
        return
    from larkhelm.lark_client import _patch_card_raw, _send_card_raw
    card = _build_plan_card(state)
    try:
        ok = _patch_card_raw(state.card_mid, card)
    except Exception as e:
        _debug_log(f"[Plan] card patch error: {e}")
        ok = False
    if not ok:
        new_mid = _send_card_raw(state.chat_id, card)
        if new_mid:
            with state.lock:
                state.card_mid = new_mid
            _debug_log(f"[Plan] replaced stale card → {new_mid}")


# ── Smart planner ────────────────────────────────────────────────

_PLANNER_SYSTEM = """\
你是一个多阶段开发计划制定专家。根据用户需求，输出一份多步骤执行计划。

**输出格式（严格遵守，不要输出任何其他内容）：**
第一行：计划标题（15字以内）
后续每行：一个步骤，格式为 `[类型] 步骤描述`

步骤类型：
- [dev]    主要开发步骤（实现功能）
- [review] 代码检视（检查安全、逻辑、规范）
- [fix]    修复检视发现的问题
- [test]   运行测试或回归验证

**规则：**
- 步骤描述要具体，工程师能直接理解并开始工作
- [review] 后面通常跟 [fix]
- 只在必要时插入 [test]，避免冗余
- 不要输出序号、解释或额外说明，只输出步骤列表
- **严格遵守用户指定的范围**：如果用户说"phase5到phase10"，只输出那几个 phase 的步骤，不要擅自补充范围之外的内容
"""

def _auto_plan(requirement: str, chat_id: str,
               cancel_ev: threading.Event,
               doc_context: str = "") -> tuple[str, list[PlanStep]]:
    """Use Claude to generate a structured plan from a natural-language requirement."""
    import larkhelm.config as _cfg
    from larkhelm.ai_runner import _spawn_claude_proc
    from larkhelm.chat_state import _get_cwd
    from larkhelm.perm import grant_yolo, revoke_yolo
    # Path already imported at module top — no need to re-import.

    cwd = _get_cwd(chat_id)
    ns  = f"{chat_id}__planner_{uuid.uuid4().hex[:8]}"

    # Background context: injected Feishu doc content takes priority; fall back to
    # local workspace files hint when no doc was provided.
    if doc_context:
        ctx_hint = f"\n\n## 背景文档（仅作参考，严格按上述用户需求的范围执行，不要规划需求范围之外的步骤）\n\n{doc_context[:8000]}"
    else:
        ws = Path(cwd) / ".crew_workspace"
        ctx_hint = ""
        if (ws / "prd.md").exists() or (ws / "design.md").exists():
            ctx_hint = (
                "\n\n项目工作区已有文件，请先读取 .crew_workspace/prd.md"
                "（若存在）和 .crew_workspace/design.md（若存在）了解背景后再制定计划。"
            )

    # Inject memory context so the planner is aware of global/project preferences.
    # Phase C wrap-up (REQ-22 symmetry): use v2 with ``requirement`` as
    # the gating query so lazy global / project conditional can decide
    # whether each layer actually adds value for this plan — same change
    # as crew/_commands.py:_augment_requirement_with_context.
    _mem_ctx = ""
    try:
        from larkhelm.memory import get_memory_context_v2
        _mem_ctx, _ = get_memory_context_v2(chat_id, cwd=cwd, query=requirement)
    except Exception as e:
        _debug_log(f"[Plan] memory load failed: {e}")
    _mem_prefix = f"\n\n[Background Context from Memory]\n{_mem_ctx}" if _mem_ctx else ""

    # Put user requirement BEFORE doc context so Claude knows the scope
    # constraint before reading the full document.
    prompt = f"{_PLANNER_SYSTEM}{_mem_prefix}\n\n用户需求：{requirement}{ctx_hint}"

    grant_yolo(ns)
    try:
        output = _spawn_claude_proc(
            chat_id=ns, message=prompt, sid=None, cwd=cwd,
            cancel_ev=cancel_ev, on_text=None, allow_retry=False,
            session_namespace=ns,
        )
    finally:
        revoke_yolo(ns)

    return _parse_plan(output.strip())


# ── Confirmation signal (from card button callback) ───────────────

def signal_plan(plan_id: str, action: str) -> bool:
    """Called by card button: action = 'continue' | 'skip' | 'cancel' | 'retry'."""
    with _active_plans_lock:
        state = _active_plans.get(plan_id)
    if not state:
        return False
    with state.lock:
        state._confirm_result = action
        if action == "cancel":
            state.cancel_ev.set()
        state._confirm_ev.set()
    return True


# ── Step executors ───────────────────────────────────────────────

def _run_dev_step(state: MultiPlanState, step: PlanStep, crew_id: str) -> bool:
    """Run a full dev pipeline. _active_crew is managed by _run_dev_crew_inner."""
    from larkhelm.crew._commands import _run_dev_crew_inner
    try:
        _run_dev_crew_inner(
            chat_id=state.chat_id,
            requirement=step.desc,
            user_msg_id=None,   # each step sends its own card into the chat
            no_confirm=True,    # plan handles human-in-the-loop between steps
            crew_id=crew_id,
            force_replan=True,  # each [dev] step re-runs PM/Architect for its specific requirement
            suppress_done_signal=True,  # plan signals done itself when all steps finish
            suppress_finalize=True,     # plan emits its own finalize card after all steps
        )
        return not state.cancel_ev.is_set()
    except Exception as e:
        step.error = str(e)[:200]
        _debug_log(f"[Plan] dev step {step.idx} error: {e}")
        return False


def _run_single_agent_step(state: MultiPlanState, step: PlanStep) -> bool:
    """Run a single-agent mini-crew (review / fix / test)."""
    import larkhelm.config as _cfg
    from larkhelm.crew_types import CrewState, AgentState, CrewPlan, AgentSpec
    from larkhelm.crew._runner import _run_crew
    from larkhelm.chat_state import _get_cwd
    from larkhelm.lark_client import _send_card_raw, _pin_task_card

    crew_id = uuid.uuid4().hex[:12]

    if step.type == "review":
        spec = AgentSpec(
            id="reviewer", role="代码审查员", model="claude",
            system=(
                "你是一个严格的代码审查员。\n\n"
                "**必须逐条检查以下 8 项，每项给出 ✅ 或 ❌ 及说明：**\n"
                "1. 安全：无 SQL 注入/命令注入/XSS，无硬编码密钥\n"
                "2. 错误处理：异常是否被捕获并合理处理\n"
                "3. 边界条件：空值、零值、极大值、并发访问\n"
                "4. 代码规范：命名一致、无重复代码、函数职责单一\n"
                "5. 性能：无明显 N+1 查询、无不必要循环\n"
                "6. 测试覆盖：核心逻辑和边界条件是否有测试\n"
                "7. 文档：公共接口和复杂逻辑是否有必要注释\n"
                "8. 完整性：参考 .crew_workspace/changes.md 确认无漏改/多改\n\n"
                "将检查结果输出到 .crew_workspace/review.md，不要自行修改代码。\n\n"
                "⚠️ 输出的最后一行必须且只能是 APPROVED 或 REJECTED"
            ),
            prompt=f"{step.desc or '审查所有本次改动的代码，按 8 项标准检查'}\n\n请先读取 .crew_workspace/changes.md 了解改动范围，完成后输出到 .crew_workspace/review.md。\n\n**重要**：请直接输出结果，不要等待用户确认。",
            depends_on=[], timeout=_cfg.RESPONSE_TIMEOUT * 8,
            output_file="review.md",
        )
    elif step.type == "fix":
        spec = AgentSpec(
            id="fixer", role="工程师（修复）", model="claude",
            system=(
                "你是一个资深工程师，专注于修复问题。\n"
                "读取 .crew_workspace/qa_report.md 和 .crew_workspace/review.md（若存在），"
                "修复发现的所有问题，将修复摘要追加到 .crew_workspace/changes.md。\n"
                "只修复明确列出的问题，不要顺手重构其他代码。"
            ),
            prompt=step.desc or "修复 qa_report.md 和 review.md 中列出的所有问题，更新 changes.md。\n\n**重要**：请直接输出结果，不要等待用户确认。",
            depends_on=[], timeout=_cfg.RESPONSE_TIMEOUT * 8,
            output_file="changes.md",
        )
    elif step.type == "test":
        spec = AgentSpec(
            id="qa", role="测试工程师", model="gemini",
            system=(
                "你是一个测试工程师。先确保测试环境就绪（安装依赖、配置环境），再运行所有测试。\n"
                "发现代码 bug 时记录到 .crew_workspace/qa_report.md，不要自行修复代码。\n\n"
                "⚠️ 输出的最后一行必须且只能是 TESTS_PASSED 或 TESTS_FAILED"
            ),
            prompt=step.desc or "确保环境就绪，运行所有测试，将 bug 记录到 qa_report.md。\n\n**重要**：请直接输出结果，不要等待用户确认。",
            depends_on=[], timeout=_cfg.RESPONSE_TIMEOUT * 4,
            output_file="qa_report.md",
        )
    else:
        return True

    label    = _TYPE_LABEL.get(step.type, step.type.upper())
    n_total  = len(state.steps)
    init_card = _make_card(
        f"⚙️ {label} · {step.desc[:40]}",
        f"**任务：** {step.desc}\n\n阶段 {step.idx + 1}/{n_total}",
        color="blue",
        buttons=[("🛑 取消", f"plan_cancel:{state.plan_id}")],
    )
    card_mid = _send_card_raw(state.chat_id, init_card)
    if card_mid:
        _pin_task_card(state.chat_id, card_mid)

    crew_state = CrewState(
        crew_id=crew_id, chat_id=state.chat_id,
        plan=CrewPlan(title=step.desc, agents=[spec]),
        agents={spec.id: AgentState(spec=spec)},
        card_mid=card_mid,
        cancel_ev=state.cancel_ev,
        phase="planned", kind="crew",
    )

    try:
        _run_crew(crew_state, _cfg.RESPONSE_TIMEOUT * 8)
        ag = crew_state.agents.get(spec.id)
        return (not state.cancel_ev.is_set()
                and ag is not None
                and ag.status.value == "done")
    except Exception as e:
        step.error = str(e)[:200]
        _debug_log(f"[Plan] {step.type} step {step.idx} error: {e}")
        return False


# ── Confirmation wait ────────────────────────────────────────────

def _wait_for_confirm_or_cancel(state: MultiPlanState, timeout: float = 86400.0) -> bool:
    """Block until either a card button signals state._confirm_ev or a cancel
    is observed. threading.Event lacks native multi-event wait, so we poll the
    confirm event with a short timeout and re-check the cancel sources.

    Cancel sources observed:
      - state.cancel_ev: set by signal_plan('cancel') and propagated here
      - chat-level cancel event (_get_cancel_event): set by /cancel command;
        when this fires we mark the plan as cancelled by setting state.cancel_ev
        and state._confirm_result so the rest of _run_plan exits cleanly.

    Returns True when state._confirm_ev fired (button click or propagated cancel),
    False on timeout.
    """
    from larkhelm.concurrency import _get_cancel_event
    chat_cancel = _get_cancel_event(state.chat_id)

    deadline = time.time() + timeout
    poll = _WAIT_POLL_INTERVAL
    while True:
        remaining = deadline - time.time()
        if remaining <= 0:
            return False
        if state._confirm_ev.wait(timeout=min(poll, remaining)):
            return True
        # Card-button cancel already propagated to state.cancel_ev — nothing
        # extra to do, the wait above will have fired via signal_plan.
        # /cancel command only sets the chat-level event; bridge it here.
        if chat_cancel.is_set():
            with state.lock:
                state._confirm_result = "cancel"
                state.cancel_ev.set()
            state._confirm_ev.set()
            return True
        if state.cancel_ev.is_set():
            return True


def _wait_confirm(state: MultiPlanState) -> str:
    """Enter waiting phase, show confirm card, block until user acts.
    Returns 'continue' | 'skip' | 'cancel' | 'retry'.

    Wakes up on either:
      - User clicks a card button (signal_plan → _confirm_ev)
      - User sends /cancel command (chat-level cancel event)
    """
    with state.lock:
        state.phase = PlanPhase.WAITING
        state._confirm_ev.clear()
        state._confirm_result = "cancel"   # default on timeout / cancel
    _update_plan_card(state)
    _wait_for_confirm_or_cancel(state)
    with state.lock:
        return state._confirm_result


# ── Main executor ────────────────────────────────────────────────

def _run_plan(state: MultiPlanState) -> None:
    from larkhelm.crew._commands import _register_crew_thread, _unregister_crew_thread
    from larkhelm.crew._state import _active_crew, _active_crew_lock
    from larkhelm.lark_client import send_card
    from larkhelm.token_stats import evict_crew_agent_tokens
    from larkhelm.plan_persistence import save_plan_state, delete_plan_state

    # U17: persist plan state at the start so a bridge crash mid-step is
    # discoverable on next startup. The state file is updated on every
    # step transition + deleted in the ``finally`` block when the plan
    # exits cleanly. See ``plan_persistence`` module docstring.
    save_plan_state(state)

    # Heartbeat: keep the plan card's elapsed time ticking
    _hb_stop = threading.Event()
    def _heartbeat():
        while not _hb_stop.is_set():
            _update_plan_card(state)
            _hb_stop.wait(timeout=15)
    threading.Thread(target=_heartbeat, daemon=True,
                     name=f"plan-hb-{state.plan_id[:6]}").start()

    def _release_slot():
        with _active_crew_lock:
            if _active_crew.get(state.chat_id, "").startswith("plan:"):
                _active_crew.pop(state.chat_id, None)

    def _hold_slot():
        with _active_crew_lock:
            _active_crew[state.chat_id] = f"plan:{state.plan_id}"

    try:
        idx = 0
        auto_retried = 0  # auto-retry counter for the current step
        while idx < len(state.steps):
            if state.cancel_ev.is_set():
                break

            step = state.steps[idx]
            if step.status == "skipped":
                idx += 1
                auto_retried = 0
                continue

            with state.lock:
                state.current_idx = idx
                state.phase       = PlanPhase.RUNNING
                step.status       = "running"
                step.start_time   = time.time()
                step.end_time     = None
            _update_plan_card(state)

            if step.type == "dev":
                crew_id = uuid.uuid4().hex[:12]
                _register_crew_thread(crew_id, threading.current_thread())
                # Clear any plan-marker from the previous wait so _run_dev_crew_inner
                # can claim _active_crew[chat_id].
                _release_slot()
                try:
                    ok = _run_dev_step(state, step, crew_id)
                finally:
                    _unregister_crew_thread(crew_id)
                    try:
                        evict_crew_agent_tokens(f"{state.chat_id}__crew_{crew_id}")
                    except Exception as e:
                        _debug_log(f"[Plan] token eviction failed: {e}")
            else:
                _hold_slot()
                try:
                    ok = _run_single_agent_step(state, step)
                finally:
                    _release_slot()

            step.end_time = time.time()

            if state.cancel_ev.is_set():
                step.status = "failed"
                step.error  = "已取消"
                break

            step.status = "done" if ok else "failed"
            _update_plan_card(state)
            save_plan_state(state)   # U17: persist after each step transition

            is_last = (idx == len(state.steps) - 1)

            if not ok:
                label = _TYPE_LABEL.get(step.type, step.type.upper())

                # P3 REQ-04: route through decide_retry_action so the
                # ``plan_retry_strategy`` config actually affects behaviour.
                # ``off`` keeps the P2 auto_retried-counter path; ``now``
                # consults PlanRetryEngine against ``step.retry_count``;
                # ``manual`` skips silent auto-retry entirely.
                _retry_strategy = _resolve_plan_retry_strategy()
                _action, _reason = decide_retry_action(
                    _retry_strategy,
                    step.retry_count,
                    auto_retried,
                    state.max_retries,
                )
                _debug_log(
                    f"[PlanRetry] step {idx} failed; strategy={_retry_strategy} "
                    f"action={_action} reason={_reason}"
                )

                if _action == "auto_retry":
                    auto_retried     += 1
                    step.retry_count += 1
                    step.status       = "pending"
                    step.error        = ""
                    step.start_time   = None
                    step.end_time     = None
                    _debug_log(f"[Plan] step {idx} auto-retry {auto_retried}/{state.max_retries}")
                    _update_plan_card(state)
                    continue  # retry same idx silently

                # All auto-retries exhausted → proactive alert + wait for user decision
                err_msg = step.error[:120] if step.error else "执行失败（无详细信息）"
                retry_note = f"（已自动重试 {auto_retried} 次）" if auto_retried else ""
                send_card(
                    state.chat_id,
                    f"⚠️ Plan · [{label}] 步骤失败{retry_note}",
                    f"**{state.title}**\n\n"
                    f"**失败步骤：** [{label}] {step.desc[:60]}\n"
                    f"**错误：** {err_msg}\n\n"
                    "请在任务卡片上选择「🔄 重试本步」、「▶ 继续」或「🛑 取消」。",
                    color="orange",
                )
                with state.lock:
                    state.last_step_failed = True
                    state.current_idx = idx
                _hold_slot()
                action = _wait_confirm(state)
                with state.lock:
                    state.last_step_failed = False
                _release_slot()

                if action == "cancel":
                    break
                elif action == "retry":
                    # User-triggered retry: reset auto-retry counter for a fresh start
                    auto_retried     = 0
                    step.retry_count += 1
                    step.status       = "pending"
                    step.error        = ""
                    step.start_time   = None
                    step.end_time     = None
                    _update_plan_card(state)
                    continue  # retry same idx
                else:  # "continue" — accept failure, advance
                    auto_retried = 0
                    idx += 1
                    continue

            # Step succeeded; advance to next step (with optional confirmation)
            auto_retried = 0
            if is_last or state.no_confirm:
                idx += 1
                continue

            _hold_slot()
            with state.lock:
                state.last_step_failed = False
                state.current_idx = idx + 1
            action = _wait_confirm(state)

            if action == "cancel":
                break
            if action == "skip" and idx + 1 < len(state.steps):
                nxt = state.steps[idx + 1]
                nxt.status     = "skipped"
                nxt.start_time = nxt.end_time = time.time()
                _update_plan_card(state)
            # _active_crew still held; cleared at start of next step branch

            idx += 1

        # ── Final state ───────────────────────────────────────────
        with state.lock:
            if state.cancel_ev.is_set():
                state.phase = PlanPhase.CANCELLED
            elif any(s.status == "failed" for s in state.steps):
                state.phase = PlanPhase.FAILED
            else:
                state.phase = PlanPhase.DONE

        _release_slot()
        _update_plan_card(state)
        save_plan_state(state)   # U17: persist final phase (done/failed/cancelled)
                                 # before finally-block delete, so a crash in
                                 # the finally itself still leaves a recoverable
                                 # record of the terminal state.

        if state.phase == PlanPhase.DONE:
            n       = len(state.steps)
            elapsed = _fmt_elapsed(time.time() - state.start_time)
            send_card(state.chat_id, "✅ Plan 全部完成",
                      f"**{state.title}**\n\n共 {n} 个阶段 · 耗时 {elapsed}",
                      color="green")
            # P1: workspace post-plan finalisation — flip
            # ``workspace_meta.json`` to ``completed=true`` when the included
            # [review] step output ``APPROVED``, and surface a git-add /
            # git-commit hint listing the files the plan touched. Shared
            # implementation with /dev (``workspace_finalize`` module).
            # Fail-soft: only logs on error.
            try:
                from larkhelm.workspace_finalize import finalize_workspace
                finalize_workspace(state.chat_id, state.title, kind="plan")
            except Exception as _fe:
                _debug_log(f"[Plan] workspace finalisation failed: {_fe}")

    finally:
        _hb_stop.set()
        # U17: plan exited cleanly (success / fail / cancel / exception) →
        # delete the persisted state. From here on, the on-disk record
        # would only mislead the next bridge boot's "interrupted plans"
        # scanner. delete is idempotent + fail-soft so it's safe in finally.
        try:
            delete_plan_state(state.plan_id)
        except Exception as _pe:
            _debug_log(f"[Plan] delete_plan_state failed: {_pe}")
        with _active_plans_lock:
            _active_plans.pop(state.plan_id, None)
        from larkhelm.crew._state import _signal_crew_done
        with _active_crew_lock:
            if _active_crew.get(state.chat_id, "").startswith("plan:"):
                _active_crew.pop(state.chat_id, None)
            _signal_crew_done(state.chat_id)
        # Capture this /plan completion into session memory (debounced).
        # ``record_milestone`` itself swallows failures, but we wrap once
        # more so a memory module import error never propagates up here.
        try:
            from larkhelm.memory import record_milestone
            record_milestone(state.chat_id, "plan",
                             summary=f"{state.title} ({len(state.steps)} steps, phase={state.phase})")
        except Exception as _e:
            _debug_log(f"[Plan] milestone record failed: {_e}")


# ── Entry point ──────────────────────────────────────────────────

def cmd_plan(chat_id: str, args_str: str, user_msg_id: str = None) -> None:
    """/plan command entry point.

    Two modes:
    - Manual:  input contains [dev]/[review]/[fix]/[test] markers → parse and run directly
    - Smart:   plain natural-language input → Claude generates the step list, user confirms
    """
    from larkhelm.lark_client import send_card, _reply_card_raw, _send_card_raw, _pin_task_card
    from larkhelm.crew._state import _active_crew, _active_crew_lock
    from larkhelm.concurrency import _reset_cancel

    # Clear any stale chat-level cancel signal from a previous task so the new
    # plan doesn't get falsely woken up as cancelled during _wait_for_confirm_or_cancel.
    # Mirrors the reset done by the normal-query dispatch path in handlers/_message.py.
    _reset_cancel(chat_id)

    text = args_str.strip()

    # Parse --retry=N flag (default: 1 auto-retry before notifying user)
    _retry_re  = re.compile(r'--retry(?:=| +)(\d+)')
    _retry_m   = _retry_re.search(text)
    max_retries = int(_retry_m.group(1)) if _retry_m else 1
    if _retry_m:
        text = _retry_re.sub("", text).strip()

    # Parse --no-confirm flag: skip between-step human confirmations
    no_confirm = "--no-confirm" in text
    if no_confirm:
        text = text.replace("--no-confirm", "").strip()

    if not text:
        send_card(chat_id, "⚠️ 用法",
                  "**手动编排**\n"
                  "```\n/plan\n可选标题\n[dev] 第一阶段\n[review] 安全审查\n"
                  "[fix] 修复问题\n[test] 回归测试\n```\n\n"
                  "**智能规划**（自然语言描述需求，自动生成计划）\n"
                  "`/plan 实现 Phase 5~10，每个阶段之间做代码检视和修复`\n\n"
                  "也支持从飞书文档读取：`/plan https://feishu.cn/docx/xxx`\n\n"
                  "**选项：**\n"
                  "- `--retry=N` 步骤失败时自动重试 N 次（默认 1）\n"
                  "- `--no-confirm` 步骤成功后跳过确认，自动连续执行\n\n"
                  "也支持自然语言：`开发登录` / `审查安全` 等",
                  color="orange")
        return

    # Feishu doc URL handling
    # ① URL only (no other text) → load doc content as the plan body (manual or smart)
    # ② URL(s) mixed with natural language → read docs as background context for smart planner
    _feishu_url_re = re.compile(r'https://[a-zA-Z0-9-]+\.feishu\.cn/[a-zA-Z0-9/_\-?.=#%&+]+')
    _doc_context   = ""
    _urls          = _feishu_url_re.findall(text)
    if _urls:
        from larkhelm.lark_client import FeishuDocClient, parse_doc_url
        _doc_client  = FeishuDocClient()
        _doc_parts: list[str] = []
        for _url in _urls:
            _ref = parse_doc_url(_url)
            if _ref is None:
                continue
            try:
                _res = _doc_client.read(_ref)
                _doc_parts.append(f"[文档：《{_res.title or _url}》]\n{_res.content}\n[/文档]")
            except Exception as _e:
                _debug_log(f"[Plan] read doc {_url} error: {_e}")

        _text_no_urls = _feishu_url_re.sub("", text).strip()
        if not _text_no_urls:
            # URL-only input: use doc content as plan body
            if not _doc_parts:
                send_card(chat_id, "⚠️ 无法读取飞书文档", "所有文档读取均失败，请检查权限。", color="orange")
                return
            text = "\n\n".join(_doc_parts)
        else:
            # URL + requirement: doc(s) become background context for smart planner
            _doc_context = "\n\n".join(_doc_parts)
            text = _text_no_urls

    with _active_crew_lock:
        if chat_id in _active_crew:
            send_card(chat_id, "⚠️ 任务冲突",
                      "当前有任务正在运行，请等待完成或发送 `/cancel` 后再试。",
                      color="orange")
            return

    has_markers = bool(re.search(r'\[(dev|review|fix|test)\]', text, re.IGNORECASE))

    plan_id = uuid.uuid4().hex[:12]

    if not has_markers:
        # ── Smart plan mode: generate steps via Claude ────────────
        requirement = text
        state = MultiPlanState(
            plan_id=plan_id, chat_id=chat_id,
            title=requirement[:40],   # placeholder until planner generates real title
            steps=[],
            phase=PlanPhase.PLANNING,
            trigger_msg_id=user_msg_id,
            max_retries=max_retries,
            no_confirm=no_confirm,
        )
        # Show "generating" card immediately
        planning_card = _build_plan_card(state)
        if user_msg_id:
            card_mid = _reply_card_raw(user_msg_id, planning_card, in_thread=False)
        else:
            card_mid = _send_card_raw(chat_id, planning_card)
        if card_mid:
            _pin_task_card(chat_id, card_mid)
        state.card_mid = card_mid

        with _active_plans_lock:
            _active_plans[plan_id] = state

        log_entry(chat_id, "user", f"/plan (smart) {requirement[:80]}", model="plan")

        # Generate plan (blocking; runs in the plan-handler thread)
        try:
            title, steps = _auto_plan(requirement, chat_id, state.cancel_ev,
                                      doc_context=_doc_context)
        except Exception as e:
            _debug_log(f"[Plan] auto_plan error: {e}")
            with state.lock:
                state.phase = PlanPhase.FAILED
            _update_plan_card(state)
            with _active_plans_lock:
                _active_plans.pop(plan_id, None)
            return

        if state.cancel_ev.is_set() or not steps:
            with state.lock:
                state.phase = PlanPhase.CANCELLED if state.cancel_ev.is_set() else "failed"
            _update_plan_card(state)
            with _active_plans_lock:
                _active_plans.pop(plan_id, None)
            return

        # Update state with generated plan, enter confirming phase
        with state.lock:
            state.title = title
            state.steps = steps
            state.phase = PlanPhase.CONFIRMING
            state._confirm_ev.clear()
            state._confirm_result = "cancel"
        _update_plan_card(state)

        # Wait for user to click "开始执行" or "取消".
        # Watches chat-level /cancel as well, so a typed /cancel can abort the plan
        # before any step runs (without this, the wait would block forever).
        _wait_for_confirm_or_cancel(state)
        with state.lock:
            action = state._confirm_result

        if action == "cancel" or state.cancel_ev.is_set():
            with state.lock:
                state.phase = PlanPhase.CANCELLED
            _update_plan_card(state)
            with _active_plans_lock:
                _active_plans.pop(plan_id, None)
            return

        # Confirmed → run
        _run_plan(state)
        return

    # ── Manual mode: parse [xxx] markers and run directly ─────────
    title, steps = _parse_plan(text)
    if not steps:
        send_card(chat_id, "⚠️ 未找到有效步骤",
                  "请在每行开头用 `[dev]`、`[review]`、`[fix]` 或 `[test]` 标注步骤类型。",
                  color="orange")
        return

    state = MultiPlanState(
        plan_id=plan_id, chat_id=chat_id, title=title, steps=steps,
        trigger_msg_id=user_msg_id,
        max_retries=max_retries,
        no_confirm=no_confirm,
    )

    init_card = _build_plan_card(state)
    if user_msg_id:
        card_mid = _reply_card_raw(user_msg_id, init_card, in_thread=False)
    else:
        card_mid = _send_card_raw(chat_id, init_card)
    if card_mid:
        _pin_task_card(chat_id, card_mid)
    state.card_mid = card_mid

    with _active_plans_lock:
        _active_plans[plan_id] = state

    log_entry(chat_id, "user", f"/plan {title} ({len(steps)} 步)", model="plan")
    _run_plan(state)
