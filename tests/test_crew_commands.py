"""Tests for ``crew/_commands`` argument parsing and terminal-failure plumbing."""
from __future__ import annotations

import threading


def test_cmd_dev_parses_no_confirm(
    init_test_config, fake_card_sender, monkeypatch,
):
    """``--no-confirm`` should be stripped from args and propagated to
    _run_dev_crew."""
    captured = {}
    def _fake_run_dev_crew(chat_id, requirement, user_msg_id, **kwargs):
        captured["chat_id"]      = chat_id
        captured["requirement"]  = requirement
        captured["no_confirm"]   = kwargs.get("no_confirm")
        captured["force_replan"] = kwargs.get("force_replan")
    import larkhelm.crew._commands as _c
    monkeypatch.setattr(_c, "_run_dev_crew", _fake_run_dev_crew)
    monkeypatch.setattr(_c, "_expand_doc_requirement", lambda r: r)
    _c.cmd_dev("test_chat", "--no-confirm  实现登录", user_msg_id=None)
    assert captured["no_confirm"] is True
    assert captured["force_replan"] is True
    assert captured["requirement"] == "实现登录"


def test_cmd_dev_default_no_confirm_false(
    init_test_config, fake_card_sender, monkeypatch,
):
    captured = {}
    def _fake_run_dev_crew(chat_id, requirement, user_msg_id, **kwargs):
        captured["no_confirm"] = kwargs.get("no_confirm")
    import larkhelm.crew._commands as _c
    monkeypatch.setattr(_c, "_run_dev_crew", _fake_run_dev_crew)
    monkeypatch.setattr(_c, "_expand_doc_requirement", lambda r: r)
    _c.cmd_dev("test_chat", "build login", user_msg_id=None)
    assert captured["no_confirm"] is False


def test_cmd_dev_empty_arg_sends_usage(
    init_test_config, fake_card_sender,
):
    import larkhelm.crew._commands as _c
    fake_card_sender.clear()
    _c.cmd_dev("test_chat", "", user_msg_id=None)
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends
    assert "用法" in sends[0]["title"] or "/dev" in sends[0]["body"]


def test_cmd_crew_status_no_active(
    init_test_config, fake_card_sender,
):
    """``/crew status`` when no task is running → returns the status card."""
    import larkhelm.crew._commands as _c
    fake_card_sender.clear()
    _c._cmd_crew_status("test_chat")
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends


def test_terminal_failure_wrapper_catches_exception(
    init_test_config, fake_card_sender, fake_backend_registry, monkeypatch,
):
    """Outer wrapper catches an unexpected exception in the dev-crew impl."""
    import larkhelm.crew._commands as _c
    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")
    monkeypatch.setattr(_c, "_run_dev_crew_inner_impl", _boom)
    fake_card_sender.clear()
    try:
        _c._run_dev_crew_inner(
            "test_chat", "build x", None, no_confirm=True, crew_id="abc",
        )
    except RuntimeError:
        pass
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert any(c["color"] == "red" for c in sends), (
        f"expected terminal failure card; got {sends}"
    )


def test_generic_terminal_failure_wrapper_catches(
    init_test_config, fake_card_sender, fake_backend_registry, monkeypatch,
):
    import larkhelm.crew._commands as _c
    def _boom(*args, **kwargs):
        raise RuntimeError("planner crashed")
    monkeypatch.setattr(_c, "_run_generic_crew_inner_impl", _boom)
    fake_card_sender.clear()
    try:
        _c._run_generic_crew_inner(
            "test_chat", "do something", 3, 60, None, "abc",
        )
    except RuntimeError:
        pass
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert any(c["color"] == "red" for c in sends)
