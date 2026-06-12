"""larkhelm · shared query constants.

This module is a zero-import leaf: it must NOT import from _query.py
or any larkhelm business module, so any query component can import it
without dependency cycles.
"""

CARD_PUSH_INTERVAL: float = 5.0    # seconds between heartbeat card pushes
CURSOR_INTERVAL: float = 0.3       # seconds between cursor animation frames
STALL_THRESHOLD: float = 30.0      # seconds of no output before stall warning
CURSOR_FRAMES: list = ["⠋", "⠙", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
TOOL_HISTORY_CAP: int = 20         # max tool-use history entries to keep
