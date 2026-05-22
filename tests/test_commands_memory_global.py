"""MEM-C1 tests — AC-08.

Verifies that _cmd_memory and _cmd_btw accept and forward sender_open_id
to the global memory functions.
"""
from __future__ import annotations

import inspect
from unittest.mock import patch, MagicMock


def test_cmd_memory_accepts_sender_open_id():
    """_cmd_memory has a sender_open_id keyword parameter."""
    from larkhelm.commands import _cmd_memory
    sig = inspect.signature(_cmd_memory)
    assert "sender_open_id" in sig.parameters


def test_cmd_btw_accepts_sender_open_id():
    """_cmd_btw has a sender_open_id keyword parameter."""
    from larkhelm.commands import _cmd_btw
    sig = inspect.signature(_cmd_btw)
    assert "sender_open_id" in sig.parameters


def test_cmd_memory_show_passes_sender_open_id(tmp_path, monkeypatch):
    """_cmd_memory (default show path) passes sender_open_id to load_global_memory."""
    import larkhelm.memory as mem
    monkeypatch.setattr(mem, "MEMORY_HOME_DIR", tmp_path)
    monkeypatch.setattr(mem, "_ensure_dir", lambda: None)

    captured = {}

    def fake_load_global(chat_id, *, sender_open_id=None):
        captured["sender_open_id"] = sender_open_id
        return ""

    with patch("larkhelm.memory.load_global_memory", side_effect=fake_load_global), \
         patch("larkhelm.memory.load_project_memory", return_value=""), \
         patch("larkhelm.memory.load_memory", return_value=""), \
         patch("larkhelm.memory._global_memory_file", return_value=None), \
         patch("larkhelm.memory._project_memory_file", return_value=tmp_path / "p.md"), \
         patch("larkhelm.memory._session_memory_file", return_value=tmp_path / "s.md"), \
         patch("larkhelm.memory._ensure_dir"), \
         patch("larkhelm.commands._get_cwd", return_value=str(tmp_path)), \
         patch("larkhelm.commands.send_card_reply"), \
         patch("larkhelm.chat_state._get_turn_count", return_value=5):

        from larkhelm.commands import _cmd_memory
        _cmd_memory("chat1", "", msg_id="mid1", sender_open_id="user_X")

    assert captured.get("sender_open_id") == "user_X"


def test_cmd_memory_set_global_passes_sender_open_id(tmp_path, monkeypatch):
    """_cmd_memory 'set global' passes sender_open_id to save_global_memory."""
    import larkhelm.memory as mem
    monkeypatch.setattr(mem, "MEMORY_HOME_DIR", tmp_path)
    monkeypatch.setattr(mem, "_ensure_dir", lambda: None)

    captured = {}

    def fake_save_global(text, *, chat_id=None, sender_open_id=None, **kw):
        captured["sender_open_id"] = sender_open_id

    def fake_global_file(chat_id, *, sender_open_id=None):
        captured["file_open_id"] = sender_open_id
        return tmp_path / f"global_{sender_open_id}.md"

    with patch("larkhelm.memory.save_global_memory", side_effect=fake_save_global), \
         patch("larkhelm.memory._global_memory_file", side_effect=fake_global_file), \
         patch("larkhelm.memory._ensure_dir"), \
         patch("larkhelm.memory.GLOBAL_MAX_CHARS", 500), \
         patch("larkhelm.commands._get_cwd", return_value=str(tmp_path)), \
         patch("larkhelm.commands.send_card_reply"):

        from larkhelm.commands import _cmd_memory
        _cmd_memory("chat1", "set global hello", msg_id="mid1", sender_open_id="user_Y")

    assert captured.get("sender_open_id") == "user_Y"
    assert captured.get("file_open_id") == "user_Y"
