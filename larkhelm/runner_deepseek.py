"""larkhelm · DeepSeekRunner — HTTP adapter for DeepSeek's OpenAI-compatible chat API.

Architectural note
------------------
DeepSeek is HTTP-only (no official CLI). Per the runner-architecture analysis
(see ``.crew_workspace/runner_architecture_analysis.md`` §4.4), HTTP backends
should NOT subclass :class:`BaseProcessRunner` — its template method is built
around ``subprocess.Popen``, stderr draining, kill-9, and stream-json line
parsing, none of which apply here. Instead this module provides a structurally
parallel ``DeepSeekRunner`` class with the same callback contract
(``on_text`` / ``on_tool`` / ``on_tool_result`` / ``on_soft_timeout`` /
``cancel_ev`` + ``run() -> str``) so callers in ``backend_cli`` / ``ai_runner``
can treat it like any other runner.

What is shared with ``BaseProcessRunner``:

* The global ``_acquire_ai_sem`` / ``_inc_active`` / ``_dec_active`` semaphore,
  so HTTP requests count against the ``MAX_AI_PROCS`` budget.
* The ``QueryCancelledError`` exception type, so ``_do_query`` exception
  handling is uniform.
* ``_record_tokens`` semantics (delegated via ``record_token_usage`` directly,
  with the same crew-namespace double-record rule).

What is NOT shared (because it doesn't apply to HTTP):

* ``Popen`` / stdin / stderr drain / kill -9.
* ``--input-format stream-json`` / ``--resume sid`` / ``_clone()`` retry.
  DeepSeek is stateless; "session" = JSON-encoded message history persisted
  in the standard ``.sid`` file via ``_load_sid`` / ``_save_sid``.

Tool/function calling is intentionally not implemented in this revision; the
streaming text path is the v1 surface area. Tool-use can be layered on later
without changing the public class contract.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Callable

import larkhelm.config as _cfg
from larkhelm.log import _debug_log, safe_log
from larkhelm.chat_state import _load_sid, _save_sid
from larkhelm.runner_base import (
    QueryCancelledError,
    _acquire_ai_sem,
    _inc_active,
    _dec_active,
    get_ai_sem,
)


_DEFAULT_BASE_URL = "https://api.deepseek.com"
_DEFAULT_MODEL = "deepseek-chat"
_HISTORY_TURN_CAP = 40   # last N {user, assistant} pairs kept; mirrors api_session._MAX_HIST
_REQUEST_BACKOFF = (1.0, 3.0, 6.0)   # exponential backoff on 429/503
_HTTP_CONNECT_TIMEOUT = 30.0
_HTTP_READ_TIMEOUT = None   # rely on watch thread / cancel_ev for stalled streams

# SSE delta keys the parser knows how to handle. Anything outside this set
# triggers a one-shot debug log per chat (see ``_consume_sse``) so future
# protocol additions surface immediately instead of being silently dropped.
# References: design.md §8 first risk row mentions ``reasoning_content`` /
# protocol drift; current set covers OpenAI-compat + DeepSeek extensions.
_KNOWN_DELTA_KEYS: frozenset[str] = frozenset({
    "role", "content", "reasoning_content",
    "tool_calls", "function_call",
    "refusal",
})

# Lazy-initialized shared HTTP session for TLS/connection reuse. DeepSeek
# /chat/completions endpoints get hit in tight bursts during /dev /crew
# /plan pipelines (each agent step → 1 HTTP call); reusing the session
# saves 50–150ms TLS handshake per call. requests.Session is thread-safe
# at the adapter level for concurrent gets/posts, which matches our
# threading model (each runner instance lives on its own thread, but they
# all share this pool).
_session = None  # type: ignore[var-annotated]
_session_lock = threading.Lock()


def _get_session():
    """Return the shared :class:`requests.Session`, creating it on first call.

    Importing ``requests`` lazily here keeps the module importable in
    test environments that haven't installed it yet (``run()`` raises a
    clear RuntimeError on the first actual request instead of an import-time
    ModuleNotFoundError).
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                import requests
                _session = requests.Session()
    return _session


def _load_history(chat_id: str) -> list[dict]:
    """Load DeepSeek conversation history from the standard .sid file.

    Returns an empty list if the file is missing or corrupt — DeepSeek is
    stateless, so a missing/broken history is equivalent to a new session.
    """
    raw = _load_sid(chat_id, "deepseek")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [m for m in data if isinstance(m, dict) and "role" in m]
    except (json.JSONDecodeError, ValueError) as e:
        _debug_log(f"[DeepSeek] history parse failed, starting fresh: {e}")
    return []


def _save_history(chat_id: str, history: list[dict]) -> None:
    """Persist conversation history; trims to the last _HISTORY_TURN_CAP exchanges."""
    if len(history) > _HISTORY_TURN_CAP * 2:
        history = history[-_HISTORY_TURN_CAP * 2:]
    try:
        _save_sid(chat_id, json.dumps(history, ensure_ascii=False), "deepseek")
    except Exception as e:
        _debug_log(f"[DeepSeek] history save failed: {e}")


def _clear_history(chat_id: str) -> None:
    """Delete persisted history for this chat (mirrors _clear_sid contract)."""
    from larkhelm.chat_state import _clear_sid
    _clear_sid(chat_id, "deepseek")


class DeepSeekRunner:
    """Mirror of :class:`BaseProcessRunner` for DeepSeek's HTTP chat API.

    The constructor signature matches the subprocess runners by intent so that
    ``backend_cli`` / ``ai_runner`` shims remain near-identical across backends,
    even though several arguments (``cwd``, ``images``, ``allow_retry``,
    ``session_namespace``, ``command``) are no-ops for HTTP and accepted only
    for source-level uniformity.
    """

    def __init__(
        self,
        chat_id: str,
        message: str,
        sid: str | None,                # accepted for parity; ignored (history file is canonical)
        cwd: str,
        *,
        cancel_ev: threading.Event | None = None,
        on_text: Callable | None = None,
        on_tool: Callable | None = None,
        on_tool_result: Callable | None = None,
        on_soft_timeout: Callable | None = None,
        on_start: Callable | None = None,
        allow_retry: bool = False,      # accepted for parity; HTTP retry is internal (429/503)
        images: list | None = None,     # accepted for parity; not yet supported
        session_namespace: str | None = None,
        command: str | None = None,     # accepted for parity; HTTP has no command
        use_session: bool = True,
        record_under: str | None = None,
        model: str | None = None,
        extra_args: list | None = None, # accepted for parity; ignored
        session_key: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self.chat_id = chat_id
        self.message = message
        self.cwd = cwd
        self.cancel_ev = cancel_ev
        self.on_text = on_text
        self.on_tool = on_tool
        self.on_tool_result = on_tool_result
        self.on_soft_timeout = on_soft_timeout
        self.on_start = on_start
        self.use_session = use_session
        self.record_under = record_under
        self.session_namespace = session_namespace
        self._session_key = session_key or "deepseek"
        self._model = model or getattr(_cfg, "DEEPSEEK_MODEL", _DEFAULT_MODEL) or _DEFAULT_MODEL
        self._api_key = api_key or getattr(_cfg, "DEEPSEEK_API_KEY", "") or ""
        self._base_url = (base_url or getattr(_cfg, "DEEPSEEK_BASE_URL", _DEFAULT_BASE_URL) or _DEFAULT_BASE_URL).rstrip("/")
        self._system_prompt = system_prompt or ""

        self._result_text: str = ""
        # Chain-of-thought stream from ``deepseek-reasoner`` arrives on
        # ``delta.reasoning_content``. Kept in a separate buffer (not folded
        # into ``_result_text``) so it doesn't pollute the assistant turn
        # written back to history, but IS surfaced to ``on_text(...,
        # status="thinking")`` so live cards can render it (matches Kimi's
        # "thinking" block treatment).
        self._reasoning_text: str = ""
        # Per-chat dedup for unknown ``delta`` keys: only debug-log once
        # per (key, chat_id) so a noisy upstream protocol change can't
        # spam DEBUG_LOG every chunk for an entire run.
        self._seen_unknown_delta_keys: set[str] = set()
        self._completed = threading.Event()
        self._cancelled_flag = threading.Event()
        self._soft_timeout_flag = threading.Event()
        # Idle-timeout tracking — see ``BaseProcessRunner.__init__`` for
        # rationale. ``_consume_sse`` calls ``_touch_activity`` on every
        # SSE line so a slow but steadily-streaming DeepSeek response
        # never trips HARD_TIMEOUT, only a fully-stalled one does.
        self._last_activity_ts: float = time.time()
        self._activity_lock = threading.Lock()

    @property
    def _ns(self) -> str:
        return self.session_namespace if self.session_namespace is not None else self.chat_id

    # ------------------------------------------------------------------ helpers

    def _record_tokens(self, usage: dict, cost: float = 0.0) -> None:
        """Mirror BaseProcessRunner._record_tokens semantics for crew double-recording."""
        if self.record_under:
            record_id = self.record_under
        elif "__crew_" in self.chat_id:
            record_id = self.chat_id.split("__crew_")[0]
        else:
            record_id = self.chat_id
        full_usage = {**usage, "cost_usd": cost}
        try:
            from larkhelm.token_stats import record_token_usage
            record_token_usage(record_id, "deepseek", full_usage)
        except Exception as e:
            _debug_log(f"[DeepSeek] token_stats update failed: {e}")
        if "__crew_" in self.chat_id:
            try:
                from larkhelm.token_stats import record_crew_agent_tokens
                record_crew_agent_tokens(self.chat_id, "deepseek", full_usage)
            except Exception as e:
                _debug_log(f"[DeepSeek] token_stats update failed: {e}")

    def _touch_activity(self) -> None:
        """Liveness signal — called from ``_consume_sse`` per SSE line."""
        with self._activity_lock:
            self._last_activity_ts = time.time()

    def _watch(self) -> None:
        """Idle-clock watcher (mirrors ``BaseProcessRunner._watch``).

        Idle is measured from the last SSE line received, so a long but
        actively-streaming DeepSeek response (reasoner CoT on a hard
        problem, large output) never trips the hard timeout — only a
        request that has truly stopped sending bytes does.
        """
        soft_fired = False
        while True:
            if self._completed.is_set():
                return
            if self.cancel_ev and self.cancel_ev.is_set():
                _debug_log("[DeepSeek] user cancelled")
                self._cancelled_flag.set()
                return
            with self._activity_lock:
                idle = time.time() - self._last_activity_ts
            if idle >= _cfg.HARD_TIMEOUT:
                _debug_log(
                    f"[DeepSeek] hard idle timeout ({idle:.0f}s ≥ "
                    f"{_cfg.HARD_TIMEOUT}s without output), forcing cancel"
                )
                self._cancelled_flag.set()
                return
            if not soft_fired and idle >= _cfg.RESPONSE_TIMEOUT:
                soft_fired = True
                _debug_log(
                    f"[DeepSeek] soft idle timeout ({idle:.0f}s ≥ "
                    f"{_cfg.RESPONSE_TIMEOUT}s), releasing lock but keeping request running"
                )
                self._soft_timeout_flag.set()
                if self.on_soft_timeout:
                    try:
                        self.on_soft_timeout()
                    except Exception as e:
                        _debug_log(f"[DeepSeek] on_soft_timeout callback failed: {e}")
            time.sleep(0.3)

    def _build_messages(self) -> list[dict]:
        """Compose the ``messages`` array from system prompt + history + new user turn."""
        messages: list[dict] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        if self.use_session:
            messages.extend(_load_history(self.chat_id))
        messages.append({"role": "user", "content": self.message})
        return messages

    # ------------------------------------------------------------------ run

    def run(self) -> str:
        if not self._api_key:
            raise RuntimeError(
                "DeepSeek API key missing. Set deepseek_api_key in config.json "
                "or DEEPSEEK_API_KEY env var."
            )
        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise RuntimeError(f"DeepSeek runner requires the 'requests' package: {e}")

        _debug_log(
            f"[DeepSeek] starting chat_id={self.chat_id} model={self._model} "
            f"ns={self._ns} use_session={self.use_session}"
        )

        sem_acquired = _acquire_ai_sem(self.cancel_ev)
        sem_held = True
        active_inc = False
        try:
            _inc_active()
            active_inc = True

            if self.on_start:
                try:
                    self.on_start()
                except Exception as e:
                    _debug_log(f"[DeepSeek] on_start callback failed: {e}")

            if self.on_text:
                try:
                    self.on_text("", status="init")
                except Exception as e:
                    _debug_log(f"[DeepSeek] on_text init callback failed: {e}")

            threading.Thread(target=self._watch, daemon=True, name=f"deepseek-watch").start()

            messages = self._build_messages()
            self._stream_chat(messages)

            if self._cancelled_flag.is_set():
                raise QueryCancelledError("query cancelled")

            # Persist updated history only if we got something coherent and sessions are on
            if self.use_session and self._result_text.strip():
                new_history = _load_history(self.chat_id)
                new_history.append({"role": "user", "content": self.message})
                new_history.append({"role": "assistant", "content": self._result_text})
                _save_history(self.chat_id, new_history)
        finally:
            # Single point of _completed.set(): covers both the success path
            # AND any raise from _build_messages / _stream_chat / cancel-check
            # / history save. Earlier draft had an extra inner try/finally
            # around _stream_chat that set _completed up to ~300ms earlier
            # (one watcher poll); reviewer flagged the redundancy correctly
            # since the watcher is daemon-only and exits within one tick.
            self._completed.set()
            if sem_held:
                # Release the *exact* sem instance we acquired (P0 fix —
                # see runner_base.get_ai_sem docstring).
                sem_acquired.release()
            if active_inc:
                _dec_active()

        return self._result_text.strip()

    # ------------------------------------------------------------------ HTTP plumbing

    def _stream_chat(self, messages: list[dict]) -> None:
        """Stream a chat completion, applying exponential backoff on 429/503."""
        import requests

        url = f"{self._base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        body = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        # Use the shared Session so TLS handshake / TCP connection are reused
        # across rapid back-to-back requests (e.g. /dev pipeline agent steps).
        session = _get_session()

        last_err: Exception | None = None
        for attempt, delay in enumerate([0.0] + list(_REQUEST_BACKOFF)):
            if self._cancelled_flag.is_set():
                return
            if delay:
                # Wait in small slices so cancel_ev / hard timeout can interrupt
                slept = 0.0
                while slept < delay:
                    if self._cancelled_flag.is_set():
                        return
                    time.sleep(min(0.3, delay - slept))
                    slept += 0.3

            try:
                resp = session.post(
                    url, headers=headers, json=body, stream=True,
                    timeout=(_HTTP_CONNECT_TIMEOUT, _HTTP_READ_TIMEOUT),
                )
            except requests.RequestException as e:
                last_err = e
                _debug_log(f"[DeepSeek] connect attempt {attempt + 1} failed: {e}")
                continue

            if resp.status_code in (429, 503):
                # Drain to free the connection; backoff and retry
                snippet = ""
                try:
                    snippet = resp.text[:200]
                except Exception as e:
                    safe_log(f"[DeepSeek] resp.text drain failed: {e}")
                last_err = RuntimeError(
                    f"HTTP {resp.status_code} from DeepSeek (attempt {attempt + 1}): {snippet}"
                )
                _debug_log(f"[DeepSeek] {last_err}")
                continue

            if resp.status_code >= 400:
                snippet = ""
                try:
                    snippet = resp.text[:500]
                except Exception as e:
                    safe_log(f"[DeepSeek] error body read failed: {e}")
                raise RuntimeError(
                    f"DeepSeek HTTP {resp.status_code}: {snippet or '(no body)'}"
                )

            # 2xx: consume the SSE stream
            self._consume_sse(resp)
            return

        raise RuntimeError(
            f"DeepSeek request failed after {len(_REQUEST_BACKOFF) + 1} attempts: {last_err}"
        )

    def _consume_sse(self, resp) -> None:
        """Iterate the SSE response, dispatching text deltas and recording usage.

        We check both ``_cancelled_flag`` (set by the watch thread on a 0.3s
        poll for cancel/hard-timeout) and ``cancel_ev`` directly — the latter
        gives tighter responsiveness for cancels that arrive mid-stream.
        """
        usage_seen: dict | None = None
        for raw_line in resp.iter_lines(decode_unicode=True):
            # Liveness signal regardless of whether the line is data /
            # comment / keep-alive — same reasoning as runner_base: the
            # watcher only needs to know bytes are still arriving.
            self._touch_activity()
            if self._cancelled_flag.is_set() or (self.cancel_ev and self.cancel_ev.is_set()):
                self._cancelled_flag.set()
                try:
                    resp.close()
                except Exception as e:
                    safe_log(f"[DeepSeek] resp.close on cancel failed: {e}")
                return
            if not raw_line:
                continue
            if raw_line.startswith(":"):
                # SSE comment / keep-alive
                continue
            if not raw_line.startswith("data:"):
                continue
            payload = raw_line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                ev = json.loads(payload)
            except json.JSONDecodeError:
                _debug_log(f"[DeepSeek] non-JSON SSE payload: {payload[:120]}")
                continue

            choices = ev.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}

                # ── content (final answer text) ──────────────────────────
                chunk = delta.get("content") or ""
                if chunk:
                    self._result_text += chunk
                    if self.on_text:
                        try:
                            self.on_text(self._result_text, status="typing")
                        except Exception as e:
                            _debug_log(f"[DeepSeek] on_text callback failed: {e}")

                # ── reasoning_content (deepseek-reasoner chain-of-thought) ──
                # Live-stream the thinking buffer to the same on_text callback
                # but with status="thinking" so card renderers can show it
                # distinctly (mirrors Kimi's thinking-block treatment). The
                # buffer is NOT folded into _result_text — only content goes
                # into history, so the next round's prompt isn't polluted
                # with chain-of-thought.
                rchunk = delta.get("reasoning_content") or ""
                if rchunk:
                    self._reasoning_text += rchunk
                    if self.on_text:
                        try:
                            self.on_text(self._reasoning_text, status="thinking")
                        except Exception as e:
                            _debug_log(f"[DeepSeek] on_text(thinking) callback failed: {e}")

                # ── unknown delta keys → drift monitor ──────────────────
                # First time we see a key outside _KNOWN_DELTA_KEYS for this
                # runner instance, log once and remember. Catches new fields
                # like a future delta.image / delta.audio_transcript before
                # users notice missing output.
                for k in delta.keys():
                    if k in _KNOWN_DELTA_KEYS or k in self._seen_unknown_delta_keys:
                        continue
                    self._seen_unknown_delta_keys.add(k)
                    _debug_log(
                        f"[DeepSeek] unknown delta key {k!r} chat={self.chat_id[:12]} "
                        f"(silently dropped; consider extending _KNOWN_DELTA_KEYS)"
                    )

            # DeepSeek surfaces usage on the final chunk (after [DONE] in some
            # variants, on the last data event in others). Capture whenever present.
            if ev.get("usage"):
                usage_seen = ev["usage"]

        if usage_seen:
            self._record_tokens({
                "input_tokens":  usage_seen.get("prompt_tokens", 0),
                "output_tokens": usage_seen.get("completion_tokens", 0),
                "cache_read":    usage_seen.get("prompt_cache_hit_tokens", 0),
                "cache_create":  usage_seen.get("prompt_cache_miss_tokens", 0),
            }, cost=0.0)
