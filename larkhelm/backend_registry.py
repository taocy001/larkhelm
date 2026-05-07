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


def _resolve_env_vars(raw: str) -> str:
    """Expand ${ENV_VAR} placeholders using os.environ."""
    def _replace(m: re.Match) -> str:
        return os.environ.get(m.group(1), m.group(0))
    return re.sub(r'\$\{([^}]+)\}', _replace, raw)


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
                spec = BackendSpec(
                    id=s["id"],
                    provider=s["provider"],
                    display_name=s.get("display_name", s["id"]),
                    role=s.get("role", "worker"),
                    tags=list(tags),
                    command=s.get("command", ""),
                    model=s.get("model", ""),
                    api_key=_resolve_env_vars(s.get("api_key", "")),
                    base_url=s.get("base_url", ""),
                    healthy=True,
                    enabled=s.get("enabled", True),
                )
                self._specs[spec.id] = spec

    def health_check(self) -> None:
        with self._lock:
            for spec in self._specs.values():
                if not spec.enabled:
                    continue
                spec.healthy = True  # reset before re-checking; stays True if no failure found
                try:
                    if spec.provider.endswith("_cli"):
                        if not spec.command or not shutil.which(spec.command):
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: command not found: {spec.command!r}")
                    elif spec.provider == "anthropic_api":
                        try:
                            import anthropic  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: anthropic SDK not installed")
                            continue
                        if not spec.api_key:
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: api_key empty")
                    elif spec.provider == "google_api":
                        try:
                            import google.genai  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: google-genai SDK not installed")
                            continue
                        if not spec.api_key:
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: api_key empty")
                    elif spec.provider == "openai_compat_api":
                        try:
                            import openai  # noqa: F401
                        except ImportError:
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: openai SDK not installed")
                            continue
                        if not spec.api_key:
                            spec.healthy = False
                            _debug_log(f"[BackendRegistry] {spec.id}: api_key empty")
                except Exception as e:
                    spec.healthy = False
                    _debug_log(f"[BackendRegistry] {spec.id}: health_check error: {e}")

    def get(self, id: str) -> BackendSpec | None:
        with self._lock:
            return self._specs.get(id)

    def get_by_tag(self, tags: list[str]) -> BackendSpec | None:
        """Return first BackendSpec that contains ALL specified tags and is healthy+enabled."""
        with self._lock:
            for spec in self._specs.values():
                if spec.enabled and spec.healthy and all(t in spec.tags for t in tags):
                    return spec
        return None

    def get_orchestrator(self) -> BackendSpec | None:
        """Return first role='orchestrator' healthy+enabled BackendSpec."""
        with self._lock:
            for spec in self._specs.values():
                if spec.role == "orchestrator" and spec.healthy and spec.enabled:
                    return spec
        return None

    def all_enabled(self) -> list[BackendSpec]:
        with self._lock:
            return [s for s in self._specs.values() if s.enabled]


# Singleton — populated by config._init_runtime()
BACKEND_REGISTRY: BackendRegistry = BackendRegistry()
