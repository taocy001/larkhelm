"""larkhelm · BackendSpec dataclass and BackendRegistry singleton"""
from __future__ import annotations

import dataclasses
import os
import re
import shutil
import threading

from larkhelm.log import _debug_log


@dataclasses.dataclass
class BackendSpec:
    id: str
    provider: str           # "claude_cli" | "gemini_cli" | "kimi_cli"
                            # | "anthropic_api" | "google_api" | "openai_compat_api"
    display_name: str
    role: str               # "orchestrator" | "worker" | "cheap"
    tags: list[str]         # e.g. ["vision", "tools", "cheap", "fast"]
    command: str = ""       # CLI backends: executable path
    model: str = ""         # API backends: model name
    api_key: str = ""       # API backends: resolved key (no ${} placeholders)
    base_url: str = ""      # API backends: custom endpoint
    healthy: bool = True
    enabled: bool = True
    last_error: str | None = None  # last health_check failure reason


def _resolve_env_vars(raw: str) -> str:
    """Expand ${ENV_VAR} placeholders using os.environ."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r'\$\{([^}]+)\}', _replace, raw)


# Normalize PRD hyphen-style provider names to internal underscore names
_PROVIDER_ALIASES: dict[str, str] = {
    "claude-cli":   "claude_cli",
    "gemini-cli":   "gemini_cli",
    "kimi-cli":     "kimi_cli",
    "claude-api":   "anthropic_api",
    "gemini-api":   "google_api",
    "openai-api":   "openai_compat_api",
}


class BackendRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, BackendSpec] = {}
        self._lock = threading.RLock()

    def load(self, specs: list[dict]) -> None:
        with self._lock:
            self._specs.clear()
            for s in specs:
                tags = s.get("tags", [])
                if isinstance(tags, str):
                    tags = [t.strip() for t in tags.split(",") if t.strip()]
                raw_provider = s.get("provider", "")
                provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
                # Also pull api_key/base_url from legacy "extra" dict if not top-level
                extra = s.get("extra", {})

                raw_api_key = s.get("api_key", "") or extra.get("api_key", "")
                resolved_api_key = _resolve_env_vars(raw_api_key)
                enabled = s.get("enabled", True)
                # Unresolved placeholder → disable backend without logging (avoids leaking var names)
                if "${" in resolved_api_key:
                    # Unresolved placeholder — backend never usable
                    resolved_api_key = ""
                    enabled = False
                elif raw_api_key and not resolved_api_key:
                    # Key was specified but env var resolved to empty string
                    resolved_api_key = ""
                    enabled = False

                # Auto-infer role when not explicitly set
                raw_role = s.get("role", "")
                if raw_role:
                    role = raw_role
                elif provider.endswith("_cli") and "tools" in list(tags):
                    role = "orchestrator"
                else:
                    role = "worker"

                spec = BackendSpec(
                    id=s["id"],
                    provider=provider,
                    display_name=s.get("display_name", s["id"]),
                    role=role,
                    tags=list(tags),
                    command=s.get("command", "") or extra.get("command", ""),
                    model=s.get("model", ""),
                    api_key=resolved_api_key,
                    base_url=s.get("base_url", "") or extra.get("base_url", ""),
                    healthy=True,
                    enabled=enabled,
                )
                self._specs[spec.id] = spec

    def health_check(self) -> None:
        with self._lock:
            for spec in self._specs.values():
                if not spec.enabled:
                    continue
                spec.healthy = True   # reset before re-checking
                spec.last_error = None
                try:
                    if spec.provider.endswith("_cli"):
                        if not spec.command or not shutil.which(spec.command):
                            spec.healthy = False
                            spec.last_error = f"command not found: {spec.command!r}"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "anthropic_api":
                        try:
                            import anthropic  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "anthropic SDK not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "google_api":
                        try:
                            import google.genai  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "google-genai SDK not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                    elif spec.provider == "openai_compat_api":
                        try:
                            import openai  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            spec.last_error = "openai SDK not installed"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                            continue
                        if not spec.api_key or "${" in spec.api_key:
                            spec.healthy = False
                            spec.last_error = "api_key missing or unresolved"
                            _debug_log(f"[BackendRegistry] {spec.id}: {spec.last_error}")
                except Exception as e:
                    spec.healthy = False
                    spec.last_error = str(e)
                    _debug_log(f"[BackendRegistry] {spec.id}: health_check error: {e}")

    def get(self, id: str) -> BackendSpec | None:
        with self._lock:
            return self._specs.get(id)

    def get_by_tag(self, tags: list[str], prefer_role: str = "") -> BackendSpec | None:
        """Return first BackendSpec matching ALL tags, healthy+enabled.

        If prefer_role is set, a matching spec with that role is returned first;
        falls back to any matching spec if no preferred-role match exists.
        """
        with self._lock:
            candidates = [
                s for s in self._specs.values()
                if s.enabled and s.healthy and all(t in s.tags for t in tags)
            ]
        if not candidates:
            return None
        if prefer_role:
            preferred = [s for s in candidates if s.role == prefer_role]
            if preferred:
                return preferred[0]
        return candidates[0]

    def get_orchestrator(self) -> BackendSpec | None:
        """Return first role='orchestrator' healthy+enabled BackendSpec."""
        with self._lock:
            for spec in self._specs.values():
                if spec.role == "orchestrator" and spec.healthy and spec.enabled:
                    return spec
        return None

    def get_orchestrator_chain(self) -> list[BackendSpec]:
        """Return ordered list of healthy+enabled backends for failover.

        Priority:
          1. role=orchestrator + 'tools' in tags + CLI provider
          2. role=orchestrator + 'tools' in tags + API provider
          3. role=orchestrator (any)
          4. any healthy+enabled (fallback)
        Deduplication via seen-set; all groups concatenated.
        """
        with self._lock:
            candidates = [s for s in self._specs.values() if s.healthy and s.enabled]

        seen: set[str] = set()
        result: list[BackendSpec] = []

        groups = [
            [s for s in candidates
             if s.role == "orchestrator" and "tools" in s.tags and s.provider.endswith("_cli")],
            [s for s in candidates
             if s.role == "orchestrator" and "tools" in s.tags and not s.provider.endswith("_cli")],
            [s for s in candidates if s.role == "orchestrator"],
            candidates,
        ]
        for group in groups:
            for spec in group:
                if spec.id not in seen:
                    seen.add(spec.id)
                    result.append(spec)
        return result

    def recover_check(self) -> None:
        """Re-probe only healthy=False backends; set healthy=True on success.

        Does NOT reset healthy=True backends (avoids unnecessary overhead).
        Thread-safe. Exceptions per-spec are caught and logged.
        """
        with self._lock:
            to_check = [s for s in self._specs.values() if not s.healthy and s.enabled]

        for spec in to_check:
            try:
                recovered = False
                if spec.provider.endswith("_cli"):
                    if spec.command and shutil.which(spec.command):
                        recovered = True
                elif spec.provider == "anthropic_api":
                    try:
                        import anthropic  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass
                elif spec.provider == "google_api":
                    try:
                        import google.genai  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass
                elif spec.provider == "openai_compat_api":
                    try:
                        import openai  # noqa: F401
                        if spec.api_key and "${" not in spec.api_key:
                            recovered = True
                    except ImportError:
                        pass

                if recovered:
                    with self._lock:
                        if not spec.healthy:  # re-check: could have failed again since snapshot
                            spec.healthy = True
                            spec.last_error = None
                    _debug_log(f"[BackendRegistry] recovered: {spec.id}")
                else:
                    _debug_log(f"[BackendRegistry] still unhealthy: {spec.id}")
            except Exception as e:
                _debug_log(f"[BackendRegistry] recover_check error for {spec.id}: {e}")

    def all_enabled(self) -> list[BackendSpec]:
        with self._lock:
            return [s for s in self._specs.values() if s.enabled]


# Singleton — populated by config._init_runtime()
BACKEND_REGISTRY: BackendRegistry = BackendRegistry()
