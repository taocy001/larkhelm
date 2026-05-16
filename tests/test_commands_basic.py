"""P1-4 PR1: tests for /reset /status /cd /pwd /ls /run."""
from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

import larkhelm.config as _cfg  # noqa: E402

if not getattr(_cfg, "_runtime", None):
    import json as _json
    import pathlib as _pl
    _tmp = _pl.Path("/tmp") / "larkhelm-test-commands"
    _tmp.mkdir(parents=True, exist_ok=True)
    _cfg_path = _tmp / "config.json"
    _cfg_path.write_text(_json.dumps({
        "APP_ID": "X", "APP_SECRET": "Y",
        "response_timeout": 30, "hard_timeout": 120,
    }))
    _cfg._init_runtime(str(_cfg_path), str(_tmp))


@pytest.fixture(autouse=True)
def _silence_cards(monkeypatch):
    recorded: list[dict] = []

    def fake_send_card(chat_id, title, body, color="blue", note="", buttons=None, normalize=True):
        recorded.append({"kind": "send", "title": title, "body": body, "color": color})
        return "fake_mid"

    def fake_send_card_reply(chat_id, msg_id, title, body, color="blue", note="",
                              buttons=None, normalize=True):
        recorded.append({"kind": "reply", "title": title, "body": body, "color": color})
        return "fake_mid"

    def fake_reply_card(chat_id, mid, title, body, color="blue", note=""):
        recorded.append({"kind": "patch_reply", "title": title, "body": body, "color": color})

    import larkhelm.commands as _cm
    monkeypatch.setattr(_cm, "send_card", fake_send_card, raising=False)
    monkeypatch.setattr(_cm, "send_card_reply", fake_send_card_reply, raising=False)
    monkeypatch.setattr(_cm, "reply_card", fake_reply_card, raising=False)
    return recorded


# ── /pwd ──────────────────────────────────────────────────────────────


def test_cmd_pwd_emits_card(_silence_cards, monkeypatch):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: "/tmp/x")
    _cm._cmd_pwd("c1", msg_id="m1")
    assert any("当前目录" in r.get("title", "") for r in _silence_cards)


# ── /cd ───────────────────────────────────────────────────────────────


def test_cmd_cd_bad_dir(_silence_cards, monkeypatch):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: "/tmp")
    _cm._cmd_cd("c1", "/this/does/not/exist", msg_id="m1")
    assert any(r.get("color") == "red" for r in _silence_cards)


def test_cmd_cd_valid_dir(_silence_cards, monkeypatch, tmp_path):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: "/tmp")
    monkeypatch.setattr(_cm, "_check_cwd_root", lambda p: True)
    monkeypatch.setattr(_cm, "_set_chat_field", lambda chat_id, k, v: None)
    _cm._cmd_cd("c1", str(tmp_path), msg_id="m1")
    assert any(r.get("color") == "green" for r in _silence_cards)


# ── /ls ───────────────────────────────────────────────────────────────


def test_cmd_ls_lists_files(_silence_cards, monkeypatch, tmp_path):
    (tmp_path / "a.txt").write_text("hi")
    (tmp_path / "sub").mkdir()
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: str(tmp_path))
    monkeypatch.setattr(_cm, "_check_cwd_root", lambda p: True)
    _cm._cmd_ls("c1", "", msg_id="m1")
    body = next(r["body"] for r in _silence_cards if "文件列表" in r.get("title", ""))
    assert "a.txt" in body
    assert "sub" in body


def test_cmd_ls_outside_root_rejected(_silence_cards, monkeypatch, tmp_path):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: str(tmp_path))
    monkeypatch.setattr(_cm, "_check_cwd_root", lambda p: False)
    _cm._cmd_ls("c1", "", msg_id="m1")
    assert any(r.get("color") == "red" for r in _silence_cards)


# ── /run ──────────────────────────────────────────────────────────────


def test_cmd_run_executes_echo(_silence_cards, monkeypatch):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: "/tmp")
    monkeypatch.setattr(_cm, "_run_shell",
                        lambda chat_id, cmd: ("hello\n", "", 0))
    monkeypatch.setattr(_cm, "log_entry", lambda *a, **kw: None)
    _cm._cmd_run("c1", "echo hello", msg_id="m1")
    # final card via reply_card
    body = next(r["body"] for r in _silence_cards
                if r["kind"] == "patch_reply" and "Shell" in r.get("title", ""))
    assert "hello" in body
    assert "退出码: `0`" in body


def test_cmd_run_failure(_silence_cards, monkeypatch):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: "/tmp")
    monkeypatch.setattr(_cm, "_run_shell",
                        lambda chat_id, cmd: ("", "boom", 1))
    monkeypatch.setattr(_cm, "log_entry", lambda *a, **kw: None)
    _cm._cmd_run("c1", "false", msg_id="m1")
    body = next(r["body"] for r in _silence_cards
                if r["kind"] == "patch_reply" and "Shell" in r.get("title", ""))
    assert "boom" in body
    assert "退出码: `1`" in body


# ── /reset ────────────────────────────────────────────────────────────


def test_cmd_reset_without_arg_clears_all(_silence_cards, monkeypatch):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_clear_sid", lambda *a, **kw: None, raising=False)
    monkeypatch.setattr(_cm, "log_entry", lambda *a, **kw: None)
    # Don't fail if it tries to clear caches we don't have
    try:
        _cm._cmd_reset("c1", which=None, msg_id="m1")
    except Exception as e:
        pytest.fail(f"_cmd_reset raised: {e}")
    assert any(r.get("kind") in ("reply", "patch_reply", "send") for r in _silence_cards)


# ── /status ───────────────────────────────────────────────────────────


def test_cmd_status_emits_card(_silence_cards, monkeypatch):
    from larkhelm import commands as _cm
    monkeypatch.setattr(_cm, "_get_cwd", lambda chat_id: "/tmp/work")
    monkeypatch.setattr(_cm, "log_entry", lambda *a, **kw: None)
    try:
        _cm._cmd_status("c1", msg_id="m1")
    except Exception as e:
        pytest.fail(f"_cmd_status raised: {e}")
    assert any(r.get("kind") in ("reply", "patch_reply", "send") for r in _silence_cards)
