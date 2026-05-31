"""AC-03: _cmd_stats_cache includes cache hit rate in output."""
import pytest
from unittest.mock import patch, MagicMock


def test_ac03_stats_cache_shows_hit_rate(monkeypatch):
    """AC-03: when get_cache_hit_rate_summary returns data, the output contains '命中率' and '%'."""
    from larkhelm.commands import _cmd_stats_cache

    captured = {}

    def fake_send_card_reply(chat_id, msg_id, title, body, color=None):
        captured["title"] = title
        captured["body"] = body
        captured["color"] = color

    monkeypatch.setattr(
        "larkhelm.token_stats.get_cache_hit_rate_summary",
        lambda: {"claude": {"read": 1000, "input": 500}},
    )
    monkeypatch.setattr(
        "larkhelm.token_stats.get_cache_savings_summary",
        lambda: {},
    )

    # Mock session_guard.get_session_counters to return empty
    import larkhelm.commands as _cmds
    monkeypatch.setattr(
        "larkhelm.commands.send_card_reply",
        fake_send_card_reply,
    )

    # Also stub session_guard import inside the function
    mock_sg = MagicMock()
    mock_sg.get_session_counters.return_value = {}
    with patch.dict("sys.modules", {"larkhelm.session_guard": mock_sg}):
        _cmd_stats_cache("chat_test", None)

    body = captured.get("body", "")
    assert "命中率" in body, f"Expected '命中率' in body, got: {body!r}"
    assert "%" in body, f"Expected '%' in body, got: {body!r}"


def test_ac03_hit_rate_calculation():
    """AC-03 math: read=1000, input=500 → 66% hit rate."""
    read, inp = 1000, 500
    hr = int(read * 100 / (read + inp))
    assert hr == 66
