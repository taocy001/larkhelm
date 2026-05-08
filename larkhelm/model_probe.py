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

PROBE_TIMEOUT = 12  # seconds per probe; timeout = model exists but slow → treat as available
_MAX_WORKERS   = 4  # concurrent probes


# ── Per-provider probe functions ─────────────────────────────────────────────

def _probe_gemini(spec) -> tuple[bool, str]:
    cmd = [spec.command or "gemini"]
    if spec.model:
        cmd += ["-m", spec.model]
    cmd += ["--skip-trust", "-p", ".", "--output-format", "stream-json"]
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
        return True, ""  # slow start = model exists
    except Exception as e:
        return False, str(e)[:200]


def _probe_claude(spec) -> tuple[bool, str]:
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
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


def _probe_kimi(spec) -> tuple[bool, str]:
    cmd = [spec.command or "kimi",
           "--print", "--output-format", "stream-json", "--input-format", "stream-json"]
    if spec.model:
        cmd += ["--model", spec.model]
    stdin_data = json.dumps({"type": "user", "message": "."}) + "\n"
    try:
        r = subprocess.run(cmd, input=stdin_data, capture_output=True,
                           text=True, timeout=PROBE_TIMEOUT)
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
        return False, err
    except subprocess.TimeoutExpired:
        return True, ""
    except Exception as e:
        return False, str(e)[:200]


def _probe_api(spec) -> tuple[bool, str]:
    """API backends: just check api_key is present (no network call at startup)."""
    if spec.api_key and "${" not in spec.api_key:
        return True, ""
    return False, "api_key not configured"


# ── Dispatcher ────────────────────────────────────────────────────────────────

def probe_spec(spec) -> tuple[bool, str]:
    """Run availability probe for a single BackendSpec. Returns (ok, error_msg)."""
    try:
        if spec.provider == "gemini_cli":
            return _probe_gemini(spec)
        elif spec.provider == "claude_cli":
            return _probe_claude(spec)
        elif spec.provider == "kimi_cli":
            return _probe_kimi(spec)
        else:
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
                icon = "✓" if ok else "✗"
                suffix = f": {err}" if err else ""
                _debug_log(f"[ModelProbe] {icon} {spec.id}"
                           f" ({spec.model or 'default'}){suffix}")
        _debug_log("[ModelProbe] all probes completed")

    t = threading.Thread(target=_worker, daemon=True, name="model-probe")
    t.start()
