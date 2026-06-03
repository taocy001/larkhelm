"""larkhelm · shared query constants (REQ-23 circular-dependency breaker).

This module is a zero-import leaf: it must NOT import from _query.py,
_query_session.py, or any larkhelm business module. Both _query.py and
_query_session.py import from here, breaking the former circular dependency
(_query_session importing constants from _query which itself imported from
_query_session for QuerySession).
"""

CARD_PUSH_INTERVAL: float = 5.0    # seconds between heartbeat card pushes
CURSOR_INTERVAL: float = 0.3       # seconds between cursor animation frames
STALL_THRESHOLD: float = 30.0      # seconds of no output before stall warning
CURSOR_FRAMES: list = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
TOOL_HISTORY_CAP: int = 20         # max tool-use history entries to keep
