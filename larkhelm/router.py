"""larkhelm · backend routing — resolve_backend()

Routing rules (priority high → low):
  0. locked_backend in chat state → raise LockedBackendUnavailableError if unhealthy
  1. has_images → get_by_tag(["vision"])
  2. has_doc_urls → get_by_tag(["tools"])  (also flips require_tools=True downstream)
  3. enable_cheap_routing + short message + no images/docs → get_by_tag(["cheap", "fast"])
     **NEW**: cheap routing is suppressed when ``_likely_needs_tools(message)``
     so a short query like "看下 commands.py" / "grep foo" doesn't get sent to
     a text-only API backend (e.g. DeepSeek) that can't actually read files.
  4. user preference backend_id (chat_state) → registry.get(backend_id)
  5. config default_backend → orchestrator chain → first healthy enabled

Fallback: RuntimeError if no healthy backend found.

Capability unification (Approach C, 2026-05-16): both the chat path
(this module) and the crew path (``crew/_backend_resolver`` →
``BACKEND_REGISTRY.rank_for_task``) now consult the same
``require_tools`` signal. Previously, only crew enforced it via
``TaskProfile.require_tools``; chat-path Rule 3 cheap routing happily
selected DeepSeek for short queries that needed file I/O. The shared
``_likely_needs_tools(message)`` heuristic closes that gap.
"""
from __future__ import annotations

import re

import larkhelm.config as _cfg
from larkhelm.backend_registry import BACKEND_REGISTRY, BackendSpec
from larkhelm.chat_state import _get_chat_state
from larkhelm.log import _debug_log

_SHORT_MSG_THRESHOLD = 100


# ── Capability signal: does this message need actual tools? ──────────────
# Detects:
#   - file extensions  (.py / .md / .json / etc — same set used by
#     memory_context._PATH_RE for "code-flavoured" detection)
#   - filesystem paths (/etc/foo, ../bar)
#   - shell-ish commands the LLM will want to actually run
#   - markdown code fences (user is asking about literal code)
# All matches are case-insensitive. Bigger nets = more conservative
# routing (more queries go to tool-capable backends); errs on the side
# of correctness because the alternative is DeepSeek giving a confident
# wrong answer about a file it never read.
_TOOL_NEED_RE = re.compile(
    # File extensions: ``.py`` ``.md`` etc. Aligns with memory_context
    # ``_PATH_RE`` so chat-side capability detection and memory
    # injection use the same code-flavour signal.
    r"\.(?:py|md|json|toml|sh|ts|tsx|js|yml|yaml|go|rs|java|cpp|c|h|sql)\b"
    # Filesystem paths: ``/etc/foo`` ``~/.config/...``.
    # Lookbehind excludes ``[A-Za-z0-9:/]`` — round-1 review #4 fix v2.
    # The bug was URL paths like ``https://example.com/foo`` matching
    # because the SECOND ``/`` in ``://`` had the FIRST ``/`` before
    # it (not ``:``); ``:``-only exclusion didn't help. Adding ``/``
    # to the exclusion ensures the second ``/`` of ``//`` is also
    # disqualified. Net effect: a path triggers iff preceded by
    # whitespace / start-of-string / a non-URL boundary character.
    # Feishu doc URLs are still handled via has_doc_urls / Rule 2.
    r"|(?<![A-Za-z0-9:/])/[\w\-/.]+"
    # Shell commands. **Deliberately conservative** — only verbs that
    # almost never appear in casual prose. Excluded: ``python``,
    # ``node``, ``bash``, ``npm`` (all appear in language-discussion
    # context like "推荐 Python 学习路径"). Excluded: ``rm`` ``cp``
    # ``mv`` ``mkdir`` ``curl`` ``wget`` (also too ambiguous).
    # Included: tools whose name is essentially never spoken outside
    # a CLI invocation context.
    r"|(?<![A-Za-z0-9])"
        r"(?:grep|cat|ls|find|head|tail|sed|awk|git|pytest|"
        r"systemctl|journalctl|docker|kubectl)\b"
    # Markdown code fence → user is asking about literal code.
    r"|```",
    re.IGNORECASE,
)


def _likely_needs_tools(message: str) -> bool:
    """Return True iff ``message`` contains a strong signal that the
    response will require file I/O or shell execution.

    Tuned for false-positive over false-negative: a chat about "Python
    development" without specific files / commands doesn't match (no
    file extension or command verb), but "看一下 commands.py" / "grep
    -n foo" / a fenced code block does.

    Empty / blank messages → False (no tools needed for empty input).
    """
    if not message or not message.strip():
        return False
    return bool(_TOOL_NEED_RE.search(message))


class LockedBackendUnavailableError(RuntimeError):
    """Raised when the chat's locked_backend is set but currently unhealthy.

    Callers must catch this before the broad Exception handler to show a
    user-facing error card rather than silently falling back to other backends.
    """
    def __init__(self, backend_id: str, last_error: str = ""):
        self.backend_id = backend_id
        detail = f"：{last_error}" if last_error else ""
        super().__init__(f"锁定的后端 {backend_id} 当前不可用{detail}")


def resolve_backend(
    chat_id: str,
    message: str,
    has_images: bool = False,
    has_doc_urls: bool = False,
    force_backend_id: str | None = None,
) -> BackendSpec:
    """Route the query to the best available BackendSpec."""
    try:
        enable_cheap = getattr(_cfg, "config", {}).get("enable_cheap_routing", False)
    except Exception:
        enable_cheap = False

    # Capability signal: does this query implicitly need file I/O / shell
    # exec / doc parsing? If yes, downstream rules MUST NOT pick a
    # text-only API backend (DeepSeek etc). has_doc_urls is a hard signal
    # already (Rule 2 picks tools-tagged); _likely_needs_tools picks up
    # the "短消息看代码" pattern Rule 3 used to send to DeepSeek.
    require_tools = has_doc_urls or _likely_needs_tools(message)

    # Rule 0.5: per-message backend force (e.g. /c, /g, /k — override orchestrator routing)
    if force_backend_id:
        spec = BACKEND_REGISTRY.get(force_backend_id)
        if spec is None:
            # Provider fallback: legacy model names "claude"/"gemini"/"kimi" → provider lookup
            _provider_map: dict[str, tuple[str, ...]] = {
                "claude": ("claude_cli", "anthropic_api"),
                "gemini": ("gemini_cli", "google_api"),
                "kimi":   ("kimi_cli",),
            }
            _providers = _provider_map.get(force_backend_id, ())
            spec = next(
                (s for s in BACKEND_REGISTRY.all_enabled()
                 if s.healthy and s.provider in _providers),
                None
            )
        if spec and spec.enabled and spec.healthy:
            _debug_log(f"[router] {chat_id}: force → {spec.id}")
            return spec
        _debug_log(f"[router] {chat_id}: force {force_backend_id!r} unavailable, falling through")

    # Rule 0: locked_backend in chat state → fast-fail if unhealthy, else return spec
    locked_state = _get_chat_state(chat_id)
    locked_id = locked_state.get("locked_backend")
    if locked_id:
        spec = BACKEND_REGISTRY.get(locked_id)
        if spec is None or not spec.enabled:
            # Backend removed from config or explicitly disabled — treat as unavailable
            _id = locked_id if spec is None else spec.id
            _err = "" if spec is None else "已禁用"
            raise LockedBackendUnavailableError(_id, _err)
        if not spec.healthy:
            raise LockedBackendUnavailableError(spec.id, spec.last_error or "")
        _debug_log(f"[router] {chat_id}: locked_backend → {spec.id}")
        return spec

    # Rule 1: image → vision-capable backend (prefer orchestrator so delegation works)
    if has_images:
        spec = BACKEND_REGISTRY.get_by_tag(["vision"], prefer_role="orchestrator")
        if spec:
            _debug_log(f"[router] {chat_id}: image → {spec.id}")
            return spec
        _debug_log(f"[router] {chat_id}: no vision backend available, falling through")

    # Rule 2: doc URLs → tools-capable backend (prefer orchestrator so delegation works)
    if has_doc_urls:
        spec = BACKEND_REGISTRY.get_by_tag(["tools"], prefer_role="orchestrator")
        if spec:
            _debug_log(f"[router] {chat_id}: doc_url → {spec.id}")
            return spec
        _debug_log(f"[router] {chat_id}: no tools backend available, falling through")

    # Rule 3: cheap routing for short messages.
    #
    # Suppressed when ``require_tools`` — e.g. user said "看一下
    # commands.py" (short, but needs Read tool). Without this guard,
    # cheap routing happily picked DeepSeek API which has no tool-use
    # capability and produced confident wrong answers about files it
    # never read. (Approach C unification: same require_tools logic as
    # crew/_backend_resolver.)
    if (
        enable_cheap
        and not has_images
        and not has_doc_urls
        and not require_tools
        and len(message) < _SHORT_MSG_THRESHOLD
    ):
        spec = BACKEND_REGISTRY.get_by_tag(["cheap", "fast"])
        if spec:
            _debug_log(f"[router] {chat_id}: short+cheap → {spec.id}")
            return spec

    # Rule 4: user preference (backend_id or model set via /model command)
    # Note: legacy configs use "model" field (values: claude/gemini/kimi) which
    # happen to match the auto-migrated backend IDs. New configs should use backend_id.
    preferred_id = locked_state.get("backend_id") or locked_state.get("model")
    if preferred_id:
        spec = BACKEND_REGISTRY.get(preferred_id)
        if spec and spec.healthy and spec.enabled:
            # Even a user-pinned backend must be filtered out if the
            # query needs tools and the backend doesn't have them —
            # otherwise the user's /model deepseek setting silently
            # routes a "grep -n foo" message to a text-only API.
            if require_tools and "tools" not in (spec.tags or []):
                _debug_log(
                    f"[router] {chat_id}: user_pref {spec.id} lacks 'tools'"
                    f" tag but query needs tools — falling through"
                )
            else:
                _debug_log(f"[router] {chat_id}: user_pref → {spec.id}")
                return spec

    # Rule 5: config default_backend → orchestrator chain → first healthy enabled
    default_bid = getattr(_cfg, "config", {}).get("default_backend", "")
    if default_bid:
        spec = BACKEND_REGISTRY.get(default_bid)
        if spec and spec.healthy and spec.enabled:
            if require_tools and "tools" not in (spec.tags or []):
                _debug_log(
                    f"[router] {chat_id}: default_backend {spec.id} lacks"
                    f" 'tools' tag but query needs tools — falling through"
                )
            else:
                _debug_log(f"[router] {chat_id}: default_backend → {spec.id}")
                return spec

    orch_chain = BACKEND_REGISTRY.get_orchestrator_chain()
    if orch_chain:
        # Orchestrators are large general-purpose backends; prefer the
        # first one that has tools when the query needs them. Falls
        # back to chain[0] if none have tools (no harm: chain[0] still
        # gets attempted, but in practice all orchestrators carry the
        # "tools" tag in default config).
        if require_tools:
            for cand in orch_chain:
                if "tools" in (cand.tags or []):
                    _debug_log(f"[router] {chat_id}: orchestrator(tools) → {cand.id}")
                    return cand
        _debug_log(f"[router] {chat_id}: orchestrator → {orch_chain[0].id}")
        return orch_chain[0]

    raise RuntimeError("No healthy backend available — all backends are down or disabled")
