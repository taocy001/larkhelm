"""AC-06 / AC-07: /memory set project_guide command tests."""
import pytest
from unittest.mock import patch


@pytest.fixture()
def _cfg_patch(monkeypatch):
    """Set up a mutable _cfg.config dict and reset module globals."""
    import larkhelm.config as _cfg
    cfg_dict = {
        "project_guide_enabled": False,
        "project_guide_path": "",
        "project_guide_auto_discover": False,
    }
    monkeypatch.setattr(_cfg, "config", cfg_dict, raising=False)
    monkeypatch.setattr(_cfg, "PROJECT_GUIDE_ENABLED", False, raising=False)
    monkeypatch.setattr(_cfg, "PROJECT_GUIDE_PATH", "", raising=False)
    monkeypatch.setattr(_cfg, "PROJECT_GUIDE_AUTO_DISCOVER", False, raising=False)
    return _cfg


def test_ac06_set_auto(monkeypatch, _cfg_patch):
    """AC-06: 'auto' sets enabled=True, auto_discover=True, path=''."""
    import larkhelm.config as _cfg
    import larkhelm.commands as _cmds

    captured = {}

    def fake_send(chat_id, msg_id, title, body, color=None):
        captured.update({"title": title, "body": body, "color": color})

    monkeypatch.setattr("larkhelm.commands.send_card_reply", fake_send)

    from larkhelm.commands import _cmd_memory_set_project_guide
    _cmd_memory_set_project_guide("chat_test", "auto", None)

    assert _cfg.config.get("project_guide_auto_discover") is True, "auto_discover should be True"
    assert _cfg.config.get("project_guide_enabled") is True, "enabled should be True"
    assert _cfg.config.get("project_guide_path") == "", "path should be empty"
    assert captured.get("color") == "green", f"Expected green card, got {captured.get('color')!r}"


def test_ac07_set_off(monkeypatch, _cfg_patch):
    """AC-07: 'off' sets enabled=False, auto_discover=False, path=''."""
    import larkhelm.config as _cfg
    import larkhelm.commands as _cmds

    # Pre-set to something truthy so we can verify the reset
    _cfg.config["project_guide_enabled"] = True
    _cfg.config["project_guide_auto_discover"] = True
    _cfg.config["project_guide_path"] = "/some/path"

    captured = {}

    def fake_send(chat_id, msg_id, title, body, color=None):
        captured.update({"title": title, "body": body, "color": color})

    monkeypatch.setattr("larkhelm.commands.send_card_reply", fake_send)

    from larkhelm.commands import _cmd_memory_set_project_guide
    _cmd_memory_set_project_guide("chat_test", "off", None)

    assert _cfg.config.get("project_guide_enabled") is False, "enabled should be False"
    assert _cfg.config.get("project_guide_auto_discover") is False, "auto_discover should be False"
    assert _cfg.config.get("project_guide_path") == "", "path should be empty"
    assert captured.get("color") == "green"


def test_set_path_nonexistent(monkeypatch, _cfg_patch, tmp_path):
    """path subcommand with non-existent path sends orange error card."""
    import larkhelm.commands as _cmds

    captured = {}

    def fake_send(chat_id, msg_id, title, body, color=None):
        captured.update({"color": color})

    monkeypatch.setattr("larkhelm.commands.send_card_reply", fake_send)

    from larkhelm.commands import _cmd_memory_set_project_guide
    _cmd_memory_set_project_guide("chat_test", "path /does/not/exist/guide.md", None)

    assert captured.get("color") == "orange"


def test_set_path_valid(monkeypatch, _cfg_patch, tmp_path):
    """path subcommand with existing file updates config."""
    import larkhelm.config as _cfg

    guide = tmp_path / "GUIDE.md"
    guide.write_text("# Guide", encoding="utf-8")

    captured = {}

    def fake_send(chat_id, msg_id, title, body, color=None):
        captured.update({"color": color, "body": body})

    monkeypatch.setattr("larkhelm.commands.send_card_reply", fake_send)

    # DATA_DIR must not overlap with tmp_path for the security check
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path / "data", raising=False)

    from larkhelm.commands import _cmd_memory_set_project_guide
    _cmd_memory_set_project_guide("chat_test", f"path {guide}", None)

    assert captured.get("color") == "green"
    assert _cfg.config.get("project_guide_enabled") is True
    assert str(guide.resolve()) in _cfg.config.get("project_guide_path", "")


def test_usage_card_on_unknown_subcommand(monkeypatch, _cfg_patch):
    """Unknown subcommand sends orange usage card."""
    captured = {}

    def fake_send(chat_id, msg_id, title, body, color=None):
        captured.update({"color": color})

    monkeypatch.setattr("larkhelm.commands.send_card_reply", fake_send)

    from larkhelm.commands import _cmd_memory_set_project_guide
    _cmd_memory_set_project_guide("chat_test", "unknown_cmd", None)

    assert captured.get("color") == "orange"
