"""AC-01/AC-05: _cmd_reset must call session_guard.clear_session_counters for each backend."""
from __future__ import annotations

from unittest.mock import patch


def test_reset_deepseek_clears_counters(
    init_test_config, fake_card_sender, monkeypatch,
):
    """AC-01: /reset deepseek must call session_guard.clear_session_counters(chat_id, 'deepseek')
    (Week-4 REQ-03: deepseek reset now targets the universal guard, not the Claude-scoped one).
    """
    import larkhelm.session_guard as _sg

    chat = "test_chat_reset_deepseek"

    with patch.object(_sg, "clear_session_counters") as mock_clear:
        import larkhelm.commands as _cmds
        _cmds._cmd_reset(chat, which="deepseek", msg_id=None)

    mock_clear.assert_called_once_with(chat, "deepseek"), (
        f"session_guard.clear_session_counters not called with (chat, 'deepseek'); "
        f"calls={mock_clear.call_args_list}"
    )


def test_reset_gemini_clears_counters(
    init_test_config, fake_card_sender, monkeypatch,
):
    """AC-05: /reset gemini must call session_guard.clear_session_counters(chat_id, 'gemini')."""
    import larkhelm.session_guard as _sg

    chat = "test_chat_reset_gemini"

    with patch.object(_sg, "clear_session_counters") as mock_clear:
        import larkhelm.commands as _cmds
        _cmds._cmd_reset(chat, which="gemini", msg_id=None)

    mock_clear.assert_called_once_with(chat, "gemini"), (
        f"session_guard.clear_session_counters not called with (chat, 'gemini'); "
        f"calls={mock_clear.call_args_list}"
    )


def test_reset_kimi_clears_counters(
    init_test_config, fake_card_sender, monkeypatch,
):
    """AC-05: /reset kimi must call session_guard.clear_session_counters(chat_id, 'kimi')."""
    import larkhelm.session_guard as _sg

    chat = "test_chat_reset_kimi"

    with patch.object(_sg, "clear_session_counters") as mock_clear:
        import larkhelm.commands as _cmds
        _cmds._cmd_reset(chat, which="kimi", msg_id=None)

    mock_clear.assert_called_once_with(chat, "kimi"), (
        f"session_guard.clear_session_counters not called with (chat, 'kimi'); "
        f"calls={mock_clear.call_args_list}"
    )


def test_cmd_stats_shows_multi_backend_names(
    init_test_config, fake_card_sender, monkeypatch,
):
    """AC-04: _cmd_stats must include gemini backend name when thresholds >0."""
    import larkhelm.session_guard as _sg
    import larkhelm.commands as _cmds

    chat = "test_chat_stats_ac04"
    sent_bodies = []

    # Patch the module-level send_card_reply binding in commands.py (_cmd_stats uses send_card_reply)
    monkeypatch.setattr(_cmds, "send_card", lambda cid, title, body, *a, **kw: sent_bodies.append(body or ""), raising=False)
    monkeypatch.setattr(_cmds, "send_card_reply", lambda cid, mid, title, body, *a, **kw: sent_bodies.append(body or ""), raising=False)

    # Mock session_guard.get_session_counters to return thresholds >0 for gemini
    def mock_get(cid, backend="claude"):
        if backend == "gemini":
            return {"turns": 3, "cache_read": 0, "threshold_turns": 30, "threshold_cache_read": 0}
        return {"turns": 0, "cache_read": 0, "threshold_turns": 0, "threshold_cache_read": 0}
    monkeypatch.setattr(_sg, "get_session_counters", mock_get)

    _cmds._cmd_stats(chat, msg_id=None, args="")

    combined = " ".join(sent_bodies)
    assert "gemini" in combined, f"Expected 'gemini' in stats output; got: {combined[:500]}"
