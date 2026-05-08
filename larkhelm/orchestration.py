"""larkhelm · orchestration — delegation protocol

Provides:
  build_orchestrator_system_prompt(registry)  — generate specialist listing for system prompt
  _detect_delegation(buffer)                  — detect DELEGATE block in streaming buffer
"""
from __future__ import annotations

import re

from larkhelm.backend_registry import BackendRegistry

# Only alphanumeric, hyphens, and underscores are valid backend IDs (max 64 chars).
# Rejects whitespace, path separators, injection characters, etc.
_BACKEND_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')

# Max allowed sub_query length to prevent context explosion via malformed delegation
_MAX_SUB_QUERY_LEN = 8000

_FALLBACK_SYSTEM = (
    "Answer the user's question directly and completely. "
    "Do NOT use the DELEGATE format — all specialists are unavailable."
)


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
        "You are a routing orchestrator. Decide whether to answer directly or delegate to a specialist.",
        "",
        "ANSWER DIRECTLY (do NOT delegate) when the task is:",
        "- Reading, displaying, or summarizing a file or document the user mentioned",
        "- Simple factual questions, status checks, short explanations",
        "- Casual conversation, greetings, quick lookups",
        "- Any task you can answer confidently in one short response",
        "",
        "DELEGATE when the task genuinely requires specialist capability:",
        "- Non-trivial code writing, debugging, or architecture work → specialist with tools tag",
        "- Hard reasoning, long multi-step analysis, or research → pro-tier specialist",
        "- Math / logic puzzles requiring extended step-by-step thinking → thinking specialist",
        "- Tasks where a specialist would clearly do better than you",
        "",
        "To delegate, respond with EXACTLY this format (nothing else in your response):",
        "",
        "DELEGATE <backend_id>",
        "<self-contained sub_query with all context the specialist needs>",
        "END_DELEGATE",
        "",
        "Available specialists:",
    ]
    for s in specialists:
        tags_str = ", ".join(s.tags) if s.tags else "general"
        cap_str = f" — {s.capabilities}" if s.capabilities else ""
        lines.append(f"  - {s.id} ({s.display_name}): tags=[{tags_str}]{cap_str}")
    lines.append("")
    lines.append("Default to answering directly unless a specialist is clearly the better choice.")

    return "\n".join(lines)


def _detect_delegation(buffer: str) -> tuple[str, str] | None:
    """Detect DELEGATE protocol in streaming buffer.

    Pattern: DELEGATE <backend_id>\\n<sub_query>\\nEND_DELEGATE

    Args:
        buffer: accumulated streaming text (should be called when len >= 60)
    Returns:
        (backend_id, sub_query) if DELEGATE block found, complete, and backend_id valid
        None if no delegation, block incomplete, or backend_id fails validation
    """
    m = re.search(r'DELEGATE\s+(\S+)\s*\n(.*?)\nEND_DELEGATE', buffer, re.DOTALL)
    if m:
        backend_id = m.group(1).strip()
        if not _BACKEND_ID_RE.match(backend_id):
            return None  # reject injected or malformed backend_id
        sub_query = m.group(2).strip()[:_MAX_SUB_QUERY_LEN]
        return (backend_id, sub_query)
    return None
