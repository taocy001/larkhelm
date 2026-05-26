"""AC-01: _cmd_reset for deepseek must call clear_session_counters."""
from __future__ import annotations

from unittest.mock import patch, call


def test_reset_deepseek_clears_counters(
    init_test_config, fake_card_sender, monkeypatch,
):
    """AC-01: /reset deepseek must call clear_session_counters(chat_id) to align
    with the claude and kimi branches that already do so (prevents DeepSeek
    session counters from carrying stale values across reset boundaries).
    """
    import larkhelm.commands as _cmds

    chat = "test_chat_reset_deepseek"

    with patch.object(
        _cmds, "clear_session_counters", wraps=_cmds.clear_session_counters
    ) as mock_clear:
        _cmds._cmd_reset(chat, which="deepseek", msg_id=None)

    mock_clear.assert_called_once_with(chat), (
        f"clear_session_counters was not called exactly once for /reset deepseek; "
        f"calls={mock_clear.call_args_list}"
    )
