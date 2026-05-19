"""larkhelm · KimiRunner — Kimi CLI subprocess runner."""
import json
import re
import threading
import time
from pathlib import Path

import larkhelm.config as _cfg
from larkhelm.log import _debug_log
from larkhelm.chat_state import _save_sid
from larkhelm.runner_base import BaseProcessRunner, _truncate_tool_result

_KIMI_TOOL_MAP = {"Shell": "Bash", "FetchURL": "WebFetch", "SearchWeb": "WebSearch"}

# kimi 1.43 emits the session ID **only** on stderr in the form
#   "\nTo resume this session: kimi -r <uuid>\n"
# No ``session_id`` field appears anywhere in stream-json stdout, so this
# regex is the sole source of truth for session continuity. Pre-fix:
# ``parse_stdout_event`` looked for ``ev.get("session_id")`` and never
# matched → every kimi turn opened a fresh session, no memory carry-over.
#
# Round-1 review (Kimi 2026-05-19) P2: character class permissive enough
# to accept base64-url style sids (alnum + ``-_=.+/``) if a future kimi
# version switches encoding. Greedy ``\S+`` would over-capture into the
# next stderr token, so stick with an explicit positive class.
_KIMI_STDERR_SESSION_RE = re.compile(
    r"To resume this session:\s*kimi\s+-r\s+([A-Za-z0-9_\-=.+/]+)"
)


def _estimate_tokens_cjk_aware(text: str) -> int:
    """Mixed-CJK token estimator for the ``cleanup_extra`` fallback path.

    Background — round-3 review R3-4
    --------------------------------
    The Kimi CLI never emits a usage envelope on stdout (see ``cleanup_extra``
    below), so token counts are estimated from raw text length. The original
    formula ``len(text) // 4`` is the rough Anthropic / OpenAI BPE
    tokens-per-char heuristic for English text — but it under-counts Chinese
    by ~4×. In BPE / SentencePiece tokenizers each CJK ideograph typically
    occupies one token (occasionally split into 2 across rare-char
    boundaries), so a 500-character zh response was being recorded as ~125
    tokens when the real cost is closer to 500. For a bilingual user this
    silently halved-to-quartered the "monthly kimi tokens" display.

    Heuristic
    ---------
    Split chars into two buckets and apply separate ratios:

    * **CJK** (CJK Unified Ideographs + Hiragana/Katakana ranges): 1 token
      per char. Slight over-count for the few rare ideographs that get split,
      but right order-of-magnitude for everyday text.
    * **Non-CJK** (Latin / digits / punctuation / spaces): retain ``// 4``
      from the original heuristic — backward-compatible for pure-English
      prompts, so the existing TestKimiCleanupExtraEstimate fixtures still
      pass without numeric drift.

    Empty / whitespace-only input returns ``0``. Caller is responsible for
    the ``max(0, …)`` clamp + the ``estimated=True`` flag.
    """
    if not text:
        return 0
    cjk = 0
    for c in text:
        # CJK Unified Ideographs: U+4E00–U+9FFF
        # Hiragana: U+3040–U+309F; Katakana: U+30A0–U+30FF
        # (CJK Ext-A U+3400–U+4DBF excluded — rare in chat traffic,
        # and the BPE split rate there is high enough that the 1:1
        # heuristic would over-count. Acceptable miss in the cleanup
        # fallback path; precise counts come from a real usage envelope.)
        if "一" <= c <= "鿿" or "぀" <= c <= "ヿ":
            cjk += 1
    non_cjk = len(text) - cjk
    return cjk + non_cjk // 4


def _build_kimi_stream_input(text: str, image_paths: list[str]) -> str:
    """Build multimodal stdin for Kimi --input-format stream-json."""
    import base64
    content: list[dict] = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            ext = Path(path).suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{data}"},
            })
        except Exception as e:
            _debug_log(f"[Kimi Image] failed to read {path}: {e}")
    content.append({"type": "text", "text": text})
    return json.dumps({"role": "user", "content": content})


class KimiRunner(BaseProcessRunner):
    _KIMI_TOOL_MAP = _KIMI_TOOL_MAP

    def __init__(
        self,
        chat_id: str,
        message: str,
        sid: str | None,
        cwd: str,
        *,
        cancel_ev=None,
        on_text=None,
        on_tool=None,
        on_tool_result=None,
        on_soft_timeout=None,
        on_start=None,
        allow_retry: bool = False,
        images: list | None = None,
        session_namespace: str | None = None,
        command: str | None = None,
        model: str | None = None,
        extra_args: list | None = None,
        session_key: str | None = None,
    ) -> None:
        super().__init__(
            "kimi", chat_id, message, sid, cwd,
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            on_start=on_start, allow_retry=allow_retry, images=images,
            session_namespace=session_namespace, command=command,
        )
        self._model = model
        self._extra_args = list(extra_args) if extra_args else []
        self._session_key = session_key or "kimi"
        # Round-1 review (Kimi 2026-05-19) P2 — session-id race lock.
        # ``_on_stderr_line`` runs on the stderr-drain thread while
        # ``parse_stdout_event`` runs on the stdout-iterator thread.
        # kimi 1.43 only emits sid on stderr today, but a future version
        # that re-introduces ``session_id`` to stdout would race against
        # the stderr regex. Single lock guards the (read _new_sid →
        # write _new_sid → _save_sid) critical section in both call
        # sites so the "first-match-wins" contract holds across threads.
        self._sid_lock = threading.Lock()
        self._ctor_kwargs = dict(
            cancel_ev=cancel_ev, on_text=on_text, on_tool=on_tool,
            on_tool_result=on_tool_result, on_soft_timeout=on_soft_timeout,
            on_start=on_start, allow_retry=allow_retry, images=images,
            session_namespace=session_namespace, command=command,
            model=model, extra_args=extra_args, session_key=session_key,
        )

    def build_args(self) -> list[str]:
        cmd = self.command or _cfg.KIMI_CMD
        args = [
            cmd, "--print", "--output-format", "stream-json",
            "--input-format", "stream-json",
            "--verbose",
            "--work-dir", self.cwd,
        ]
        if self._model:
            args += ["--model", self._model]
        if self.sid:
            args += ["--session", self.sid]
        if _cfg.SKIP_PERMISSIONS:
            args += ["--yolo"]
        args += self._extra_args
        _debug_log(
            f"[kimi] starting cwd={self.cwd} sid={self.sid} "
            f"skip_perm={_cfg.SKIP_PERMISSIONS} images={len(self.images)} "
            f"ns={self._ns} cmd={cmd} model={self._model or '(default)'}"
        )
        return args

    def build_stdin(self) -> str | None:
        if self.images:
            return _build_kimi_stream_input(self.message, self.images)
        return json.dumps({"role": "user", "content": self.message})

    def _on_stderr_line(self, line: str) -> None:
        """Parse the "To resume this session: kimi -r <uuid>" hint that
        kimi 1.43.x emits on stderr after each turn.

        kimi-cli stream-json stdout has no ``session_id`` envelope at all
        (verified against 1.43.x on 2026-05-19), so this stderr scrape is
        the only way to persist a session for `kimi -r SID` continuity.
        Idempotent — first match wins per turn, guarded by ``_sid_lock``
        so a future stdout-side ``session_id`` event can't race this one.
        """
        m = _KIMI_STDERR_SESSION_RE.search(line)
        if not m:
            return
        sid = m.group(1)
        with self._sid_lock:
            if self._new_sid:
                return
            self._new_sid = sid
        try:
            _save_sid(self._ns, sid, self._session_key)
        except Exception as e:
            _debug_log(f"[Kimi] _save_sid failed: {e}")

    def _extract_kimi_content_text(self, content) -> str:
        """Collect the visible-text bytes from a kimi 1.43 content payload.

        kimi 1.43 schema: ``content`` is either a string (rare / legacy)
        or a list of typed parts ``[{"type": "think"|"text", ...}]``. We
        only forward ``text`` parts to the user — ``think`` is the
        model's private reasoning trace and shouldn't be streamed.
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                c.get("text", "")
                for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            )
        return ""

    def parse_stdout_event(self, ev: dict) -> bool:
        # Some hypothetical future kimi version may re-add ``session_id``
        # to stdout — keep the line, but the live source of truth is
        # ``_on_stderr_line`` (see _KIMI_STDERR_SESSION_RE). Use the
        # same ``_sid_lock`` as the stderr path so a stdout-event and a
        # stderr-line arriving on different threads cannot both win.
        cand_sid = ev.get("session_id") or ev.get("session")
        if cand_sid:
            with self._sid_lock:
                first_write = not self._new_sid
                if first_write:
                    self._new_sid = cand_sid
            if first_write:
                try:
                    _save_sid(self._ns, cand_sid, self._session_key)
                except Exception as e:
                    _debug_log(f"[Kimi] _save_sid failed: {e}")

        role = ev.get("role", "")
        etype = ev.get("type", "")

        if role == "assistant":
            content = ev.get("content", "")
            tool_calls = ev.get("tool_calls") or []

            chunk = self._extract_kimi_content_text(content)
            if chunk:
                self._result_text += chunk
                if self.on_text:
                    self.on_text(self._result_text, status="typing")

            for tc in tool_calls:
                tc_id = tc.get("id", "")
                func = tc.get("function", {})
                kimi_name = func.get("name", "?")
                name = self._KIMI_TOOL_MAP.get(kimi_name, kimi_name)
                try:
                    inp = json.loads(func.get("arguments", "{}"))
                except Exception:
                    inp = {}
                self._tool_start_times[tc_id] = time.monotonic()
                if self.on_tool:
                    self.on_tool(name, self._summarize_tool_input(name, inp), tool_id=tc_id)

        elif role == "tool":
            tc_id = ev.get("tool_call_id", "")
            content = ev.get("content", "")
            is_error = bool(ev.get("is_error", False))
            # kimi 1.43 sends tool results as ``content: [{type:"text",
            # text:"..."}, ...]``. Pre-fix did ``str(content)`` on the
            # list which produced an ugly Python repr blob.
            #
            # Round-1 review (Kimi 2026-05-19) P0: previous fix used
            # ``_extract(...) or str(content)`` which still produced
            # ugly repr when the list contained only empty-text parts
            # (a *legal* "command produced no output" tool result).
            # ``or`` couldn't distinguish "extraction failed" from
            # "extraction succeeded with empty string". Replace with
            # type-dispatch: list → trust extraction (incl. ""),
            # string → passthrough, other → repr.
            if isinstance(content, list):
                pretty = self._extract_kimi_content_text(content)
            elif isinstance(content, str):
                pretty = content
            else:
                pretty = str(content)
            start_t = self._tool_start_times.get(tc_id, time.monotonic())
            elapsed = time.monotonic() - start_t
            if self.on_tool_result:
                self.on_tool_result(tc_id, _truncate_tool_result(pretty, is_error), is_error, elapsed)

        elif etype == "result" or role == "result":
            # Defensive: kimi 1.43 does NOT emit this envelope, but if a
            # future version starts to, honour it and skip cleanup_extra
            # estimation entirely.
            usage = ev.get("usage", {})
            cost = ev.get("total_cost_usd", 0.0)
            if usage:
                self._record_tokens("kimi", {
                    "input_tokens":  usage.get("input_tokens", usage.get("prompt_tokens", 0)),
                    "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
                    "cache_read":    0,
                    "cache_create":  0,
                }, cost)
            return True

        return False

    def cleanup_extra(self) -> None:
        # Kimi CLI 1.43.x emits Message / ToolResult / Notification /
        # PlanDisplay envelopes via ``--output-format stream-json`` but
        # NEVER surfaces a usage / result envelope on stdout — token
        # accounting therefore got 0 contributions for every kimi query
        # (independent audit confirmed: 91 ``model=kimi`` user records
        # → 0 ``role=token`` kimi records on the live JSONL). Without
        # at least an estimate the user's "本月 kimi tokens" stays at
        # 0 forever, masking real usage entirely.
        #
        # Best-effort fallback: estimate via ``_estimate_tokens_cjk_aware``
        # which keeps the original ``len // 4`` for ASCII / Latin chars
        # but counts each CJK ideograph + kana char as ~1 token. The
        # naive ``len(text) // 4`` formula under-counted Chinese by ~4×
        # (BPE / SentencePiece typically gives 1 token per CJK char),
        # so bilingual users saw their "本月 kimi tokens" silently
        # halved-to-quartered. Round-3 review R3-4 fix. Marked
        # ``estimated=True`` in the JSONL record so downstream tooling
        # can distinguish from precise SDK-reported counts. cache_read
        # and cache_create stay 0 — there's no schema to read them from
        # on the CLI path.
        # Short-circuit if a (hypothetical) future kimi CLI version
        # starts emitting usage and ``parse_stdout_event`` already
        # recorded — don't double-count. Mirrors the base
        # ``record_partial_tokens_if_needed`` guard.
        if self._tokens_recorded:
            return
        try:
            text = getattr(self, "_result_text", "") or ""
            prompt = getattr(self, "message", "") or ""
            if not text and not prompt:
                return
            self._record_tokens("kimi", {
                "input_tokens":  max(0, _estimate_tokens_cjk_aware(prompt)),
                "output_tokens": max(0, _estimate_tokens_cjk_aware(text)),
                "cache_read":    0,
                "cache_create":  0,
                "estimated":     True,
            }, 0.0)
        except Exception as e:
            _debug_log(f"[kimi] cleanup_extra estimate failed: {e}")
