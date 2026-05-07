"""larkhelm · orchestration — delegation protocol

Provides:
  build_orchestrator_system_prompt(registry)  — generate specialist listing for system prompt
  _detect_delegation(buffer)                  — detect DELEGATE block in streaming buffer
"""
from __future__ import annotations

import re

from larkhelm.backend_registry import BackendRegistry


def build_orchestrator_system_prompt(registry: BackendRegistry) -> str:
    """Generate system prompt listing available specialists.

    Returns empty string if no specialist (non-orchestrator, healthy, enabled) backends found.
    Only called when delegation is potentially useful (i.e., at least one worker/specialist exists).
    """
    specialists = [
        s for s in registry.all_enabled()
        if s.healthy and s.role != "orchestrator"
    ]
    if not specialists:
        return ""

    lines = [
        "You are an orchestrator with access to specialist backends.",
        "To delegate a sub-task, respond with EXACTLY this format (no other text before it):",
        "",
        "DELEGATE <backend_id>",
        "<sub_query>",
        "END_DELEGATE",
        "",
        "Available specialists:",
    ]
    for s in specialists:
        tags_str = ", ".join(s.tags) if s.tags else "general"
        lines.append(f"  - {s.id} ({s.display_name}): tags=[{tags_str}]")
    lines.append("")
    lines.append("Only delegate when the task clearly benefits from a specialist. Otherwise answer directly.")

    return "\n".join(lines)


def _detect_delegation(buffer: str) -> tuple[str, str] | None:
    """Detect DELEGATE protocol in streaming buffer.

    Pattern: DELEGATE <backend_id>\\n<sub_query>\\nEND_DELEGATE

    Args:
        buffer: accumulated streaming text (should be called when len >= 60)
    Returns:
        (backend_id, sub_query) if DELEGATE block found and complete
        None if no delegation or block incomplete
    """
    m = re.search(r'DELEGATE\s+(\S+)\s*\n(.*?)\nEND_DELEGATE', buffer, re.DOTALL)
    if m:
        backend_id = m.group(1).strip()
        sub_query = m.group(2).strip()
        return (backend_id, sub_query)
    return None
