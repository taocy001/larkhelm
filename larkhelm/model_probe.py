"""larkhelm · model probe — verify model availability at startup.

Each CLI backend is probed with a minimal request; results update BackendRegistry
asynchronously so the service starts immediately without blocking.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from larkhelm.log import _debug_log

PROBE_TIMEOUT = 12  # seconds per probe
_MAX_WORKERS   = 4  # concurrent probes

# Sentinel returned by CLI probes when the subprocess times out — meaning
# we could neither confirm health (False) nor confirm reachability (True).
# ``BackendRegistry.set_probe_result`` accepts ``ok=None`` and does NOT
# mutate ``healthy`` — only updates last_probed_at. Real-traffic
# ``record_call_failure`` becomes the authoritative health signal for
# this backend until the next probe tick succeeds or fails definitively.
#
# Pre-fix (the bug this constant exists to prevent): timeout returned
# ``True`` with rationale "slow start = model exists" — a Gemini API
# outage causing the CLI to hang indefinitely was silently relabeled as
# "healthy".
PROBE_INDETERMINATE_TIMEOUT = (None, "subprocess timeout")


# ── Per-provider probe functions ─────────────────────────────────────────────

def _probe_gemini(spec) -> tuple[bool | None, str]:
    cmd = [spec.command or "gemini"]
    if spec.model:
        cmd += ["-m", spec.model]
    cmd += ["-y", "-p", ".", "--output-format", "stream-json"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        for line in r.stdout.splitlines():
            try:
                ev = json.loads(line)
                if ev.get("type") in ("init", "result", "message"):
                    return True, ""
            except Exception:
                pass
        if r.returncode == 0:
            return True, ""
        err = r.stderr[:200].strip()
        if "ModelNotFound" in err or "not found" in err.lower():
            return False, f"model not found: {spec.model or '(default)'}"
        return False, err
    except subprocess.TimeoutExpired:
        # Indeterminate — see PROBE_INDETERMINATE_TIMEOUT comment for why
        # this is None and not True.
        return PROBE_INDETERMINATE_TIMEOUT
    except Exception as e:
        return False, str(e)[:200]


def _probe_claude(spec) -> tuple[bool | None, str]:
    cmd = [spec.command or "claude", "--print", "--verbose", "--output-format", "stream-json"]
    if spec.model:
        cmd += ["--model", spec.model]
    cmd += ["-p", "."]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=PROBE_TIMEOUT)
        for line in r.stdout.splitlines():
            try:
                ev = json.loads(line)
                if ev.get("type") in ("system", "assistant", "result"):
                    return True, ""
            except Exception:
                pass
        if r.returncode == 0:
            return True, ""
        err = r.stderr[:200].strip()
        if "model" in err.lower() and ("not found" in err.lower() or "invalid" in err.lower()):
            return False, f"model not available: {spec.model or '(default)'}"
        return False, err
    except subprocess.TimeoutExpired:
        return PROBE_INDETERMINATE_TIMEOUT
    except Exception as e:
        return False, str(e)[:200]


def _probe_kimi(spec) -> tuple[bool | None, str]:
    cmd = [spec.command or "kimi",
           "--print", "--output-format", "stream-json", "--input-format", "stream-json"]
    if spec.model:
        cmd += ["--model", spec.model]
    stdin_data = json.dumps({"type": "user", "message": "."}) + "\n"
    try:
        r = subprocess.run(cmd, input=stdin_data, capture_output=True,
                           text=True, timeout=PROBE_TIMEOUT)

        # Step 1 — happy path: any valid stream-json event means the call
        # succeeded end-to-end. Don't scan for auth markers here, because
        # a successful response could legitimately contain the substring
        # "401" inside its content (e.g. the model echoing back something
        # about HTTP status codes). Trust the structured event.
        for line in r.stdout.splitlines():
            try:
                ev = json.loads(line)
                if ev.get("type") in ("system", "assistant", "result"):
                    return True, ""
            except Exception:
                pass

        # Step 2 — no structured event observed. Now it's safe to scan for
        # auth/quota markers because the only output is error text. Kimi CLI
        # is known to print "Error code: 401" / "Invalid Authentication" to
        # stdout AND exit 0, so we can't rely on rc alone — but we CAN rely
        # on "no stream-json event observed".
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        from larkhelm.health_signals import classify_error, AUTH, QUOTA, MODEL_NOT_FOUND
        cat = classify_error(combined)
        if cat in (AUTH, QUOTA, MODEL_NOT_FOUND):
            snippet = combined.strip().splitlines()[0][:200] if combined.strip() else cat.lower()
            return False, f"{cat.lower()}: {snippet}"

        # Step 3 — no event, no recognizable error, exited cleanly.
        # Same bug class as the TimeoutExpired path (round-1 review #2):
        # "rc=0 with no stream-json output" doesn't prove the upstream
        # API works, only that the CLI exited without an error. A
        # kimi-cli that hangs internally then quits cleanly would land
        # here and be silently marked healthy under the pre-fix policy.
        # Return INDETERMINATE so real-traffic record_call_failure is
        # the authoritative signal.
        if r.returncode == 0:
            return None, "rc=0 but no stream-json event observed"
        err = r.stderr[:200].strip() or r.stdout[:200].strip() or f"rc={r.returncode}"
        return False, err
    except subprocess.TimeoutExpired:
        return PROBE_INDETERMINATE_TIMEOUT
    except Exception as e:
        return False, str(e)[:200]


def _probe_api(spec) -> tuple[bool, str]:
    """API backends: just check api_key is present (no network call).

    Used as the fallback when ``BACKEND_PROBE_API_REAL_CALL`` is False (e.g.
    on metered networks or when probe traffic is unwanted).
    """
    if spec.api_key and "${" not in spec.api_key:
        return True, ""
    return False, "api_key not configured"


def _with_thread_deadline(fn, timeout_sec: float):
    """Run ``fn`` on a worker thread and abandon it after ``timeout_sec``.

    Used to enforce a hard wall-clock deadline on SDK calls whose underlying
    HTTP client doesn't honor a per-request timeout (notably google-genai's
    grpc path on some versions). The thread itself can't be killed — Python
    has no thread-cancel primitive — but we stop *waiting* on the result and
    let the runtime collect it eventually. Probe is a low-frequency event,
    so a leaked worker for one tick is acceptable.

    Uses explicit ``pool.shutdown(wait=False, cancel_futures=True)`` rather
    than a ``with`` block: ``ThreadPoolExecutor.__exit__`` calls
    ``shutdown(wait=True)`` which BLOCKS until the worker finishes, defeating
    the whole purpose of the deadline. The not-yet-started cancellation flag
    is harmless when the only future has already started.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="probe-deadline")
    try:
        fut = pool.submit(fn)
        try:
            return fut.result(timeout=timeout_sec)
        except FuturesTimeout:
            # Round-1 review #3: align API SDK probe timeout policy with CLI.
            # Pre-fix returned ``(False, ...)`` here, immediately flipping
            # healthy=False — asymmetric vs the CLI policy which returns
            # ``(None, ...)`` as INDETERMINATE. Real-traffic
            # record_call_failure (3-strike threshold) is the authoritative
            # signal in both cases now.
            return None, f"probe timeout (>{int(timeout_sec)}s)"
    finally:
        # Critical: do NOT wait. A hung SDK call would block us indefinitely.
        # The worker thread continues running in the background until the SDK
        # call returns or the bridge restarts; one leaked thread per timed-out
        # probe is acceptable given probe cadence (~once per spec per 24h idle).
        pool.shutdown(wait=False, cancel_futures=True)


def _classify_api_exception(exc: Exception) -> tuple[bool | None, str]:
    """Map an API SDK exception to a probe result.

    Symmetric with the CLI policy (round-1 review #3): exceptions that
    indicate the network couldn't deliver a verdict (timeouts, connection
    refused, DNS) return ``None`` — INDETERMINATE. Definitive errors
    (auth, quota, model-not-found, schema mismatch) return ``False`` so
    the registry flips healthy=False immediately and stops sending
    traffic at a backend with a known config error.

    Heuristic on exception class name + message text. Tolerates absent
    SDK imports — we never need to ``isinstance(exc, anthropic.APITimeoutError)``,
    just inspect the str.
    """
    name = type(exc).__name__.lower()
    msg = str(exc)[:300].lower()
    # Timeout family.
    timeout_markers = ("timeout", "timedout", "timed out", "deadline")
    if any(m in name for m in timeout_markers) or any(m in msg for m in timeout_markers):
        return None, f"probe timeout: {type(exc).__name__}"
    # Network unreachable / TLS handshake family — also indeterminate
    # (the backend isn't proven unhealthy; we just can't get there).
    netfail_markers = (
        "connection refused", "connectionrefused", "connecterror",
        "name or service not known", "nameresolution",
        "remote end closed", "ssl",
    )
    if any(m in name for m in netfail_markers) or any(m in msg for m in netfail_markers):
        return None, f"probe network: {type(exc).__name__}"
    # Everything else (auth, quota, schema, etc.) is definitive.
    return False, str(exc)[:200]


def _probe_anthropic_real(spec) -> tuple[bool | None, str]:
    """Real probe: 1-token messages.create call. Catches AUTH/QUOTA/MODEL_NOT_FOUND.

    Passes ``timeout=PROBE_TIMEOUT`` to the SDK client so a hung TLS handshake
    or stuck server doesn't block the probe thread for the SDK default (600s).
    """
    if not spec.api_key or "${" in spec.api_key:
        return False, "api_key not configured"
    try:
        import anthropic
    except ImportError:
        return False, "anthropic SDK not installed"
    try:
        kwargs: dict = {"api_key": spec.api_key, "timeout": float(PROBE_TIMEOUT)}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        client = anthropic.Anthropic(**kwargs)
        client.messages.create(
            model=spec.model or "claude-sonnet-4-6",
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
        return True, ""
    except Exception as e:
        return _classify_api_exception(e)


def _probe_google_real(spec) -> tuple[bool | None, str]:
    """google-genai's per-call timeout knobs vary by SDK version, so we wrap
    the SDK invocation with our own thread-deadline guard. Belt-and-suspenders:
    set ``http_options.timeout`` if the SDK accepts it; in any case bail at
    PROBE_TIMEOUT via the wrapper.
    """
    if not spec.api_key or "${" in spec.api_key:
        return False, "api_key not configured"
    try:
        from google import genai
    except ImportError:
        return False, "google-genai SDK not installed"

    def _do_call():
        try:
            client_kwargs: dict = {"api_key": spec.api_key}
            # Best-effort: newer SDK versions accept http_options. Tolerate older.
            try:
                from google.genai import types as _gtypes
                client_kwargs["http_options"] = _gtypes.HttpOptions(timeout=PROBE_TIMEOUT * 1000)
            except Exception:
                pass
            client = genai.Client(**client_kwargs)
            client.models.generate_content(
                model=spec.model or "gemini-2.5-flash",
                contents=".",
                config={"max_output_tokens": 1},
            )
            return True, ""
        except Exception as e:
            return _classify_api_exception(e)

    return _with_thread_deadline(_do_call, PROBE_TIMEOUT)


def _probe_openai_compat_real(spec) -> tuple[bool | None, str]:
    """Real probe for openai_compat_api provider — uses the openai SDK with an
    explicit per-request timeout."""
    if not spec.api_key or "${" in spec.api_key:
        return False, "api_key not configured"
    try:
        import openai
    except ImportError:
        return False, "openai SDK not installed"
    try:
        kwargs: dict = {"api_key": spec.api_key, "timeout": float(PROBE_TIMEOUT)}
        if spec.base_url:
            kwargs["base_url"] = spec.base_url
        client = openai.OpenAI(**kwargs)
        client.chat.completions.create(
            model=spec.model or "gpt-4o-mini",
            max_tokens=1,
            messages=[{"role": "user", "content": "."}],
        )
        return True, ""
    except Exception as e:
        return _classify_api_exception(e)


def _probe_deepseek_real(spec) -> tuple[bool | None, str]:
    """Real probe for deepseek_api provider — uses ``requests`` directly to mirror
    ``runner_deepseek.DeepSeekRunner._stream_chat`` (which avoids the openai SDK
    dependency). Hits ``/chat/completions`` with max_tokens=1.
    """
    if not spec.api_key or "${" in spec.api_key:
        return False, "api_key not configured"
    try:
        import requests
    except ImportError:
        return False, "requests not installed"
    base_url = spec.base_url or "https://api.deepseek.com"
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {spec.api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": spec.model or "deepseek-chat",
        "messages": [{"role": "user", "content": "."}],
        "max_tokens": 1,
        "stream": False,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=PROBE_TIMEOUT)
    except requests.RequestException as e:
        # Round-1 review #3: network-level failure (timeout, conn refused,
        # DNS, TLS) is INDETERMINATE — backend health unknown. Auth/quota/
        # model errors arrive via r.status_code below and stay definitive.
        return _classify_api_exception(e)
    if r.status_code == 200:
        return True, ""
    # Map common HTTP errors to classify_error-friendly text so the registry
    # can flip healthy=False instantly on AUTH/QUOTA without waiting for the
    # TRANSIENT threshold.
    body_text = (r.text or "")[:200]
    if r.status_code in (401, 403):
        return False, f"auth: HTTP {r.status_code} {body_text}"
    if r.status_code == 429:
        return False, f"quota: HTTP 429 {body_text}"
    if r.status_code == 404:
        return False, f"model not found: HTTP 404 {body_text}"
    return False, f"HTTP {r.status_code}: {body_text}"


# ── Dispatcher ────────────────────────────────────────────────────────────────

def probe_spec(spec) -> tuple[bool | None, str]:
    """Run availability probe for a single BackendSpec. Returns (ok, error_msg).

    Return-type semantics:
      * (True, "")             — probe confirmed reachable.
      * (False, "<reason>")    — probe failed clearly (auth / quota / not-found / etc).
      * (None, "<reason>")     — INDETERMINATE (CLI subprocess timeout).
        ``set_probe_result`` interprets None as "do not mutate healthy".
        Real-traffic ``record_call_failure`` becomes the authoritative
        health signal until the next definitive probe result.

    For API backends, dispatch based on ``_cfg.BACKEND_PROBE_API_REAL_CALL``:
    when True (default), make a real 1-token call to validate auth + model;
    when False, fall back to the cheap ``_probe_api`` key-presence check.
    """
    try:
        if spec.provider == "gemini_cli":
            return _probe_gemini(spec)
        elif spec.provider == "claude_cli":
            return _probe_claude(spec)
        elif spec.provider == "kimi_cli":
            return _probe_kimi(spec)
        # API backends — prefer real probe, fall back to key-presence check
        try:
            from larkhelm import config as _cfg
            real = bool(getattr(_cfg, "BACKEND_PROBE_API_REAL_CALL", True))
        except Exception:
            real = True
        if not real:
            return _probe_api(spec)
        if spec.provider == "anthropic_api":
            return _probe_anthropic_real(spec)
        if spec.provider == "google_api":
            return _probe_google_real(spec)
        if spec.provider == "openai_compat_api":
            return _probe_openai_compat_real(spec)
        if spec.provider == "deepseek_api":
            return _probe_deepseek_real(spec)
        return _probe_api(spec)
    except Exception as e:
        return False, str(e)[:200]


# ── Async runner ──────────────────────────────────────────────────────────────

def run_probes_async(specs: list, registry) -> None:
    """Probe all specs concurrently in a background daemon thread.

    Updates registry.set_probe_result() as each probe completes.
    Service remains fully available during probing (specs start healthy=True
    from health_check; probes may flip unhealthy ones or confirm healthy ones).
    """
    # Respect enabled=False — disabled backends shouldn't burn probe traffic
    # (matches the behavior of recover_check / health-tick loop).
    specs = [s for s in (specs or []) if getattr(s, "enabled", True)]
    if not specs or "PYTEST_CURRENT_TEST" in os.environ:
        return

    def _worker():
        _debug_log(f"[ModelProbe] probing {len(specs)} backends")
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS,
                                thread_name_prefix="model-probe") as pool:
            future_to_spec = {pool.submit(probe_spec, s): s for s in specs}
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    ok, err = future.result()
                except Exception as e:
                    ok, err = False, str(e)[:200]
                registry.set_probe_result(spec.id, ok, err)
                # Three-state icon (round-1 review #1): ✓ True / ✗ False / ? None.
                if ok is True:
                    icon = "✓"
                elif ok is False:
                    icon = "✗"
                else:
                    icon = "?"
                suffix = f": {err}" if err else ""
                _debug_log(f"[ModelProbe] {icon} {spec.id}"
                           f" ({spec.model or 'default'}){suffix}")
        _debug_log("[ModelProbe] all probes completed")

    t = threading.Thread(target=_worker, daemon=True, name="model-probe")
    t.start()
