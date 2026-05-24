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


def test_dev_clears_stale_cancel_event_from_prev_crew(
    init_test_config, fake_card_sender, fake_backend_registry, monkeypatch,
):
    """Regression: a previous crew's ``state.cancel_ev.set()`` (e.g. from a
    breakpoint auto-cancel) leaks into the per-chat ``_cancel_events`` map.
    The next /dev must clear it before ``_run_crew`` runs, otherwise the new
    crew immediately raises QueryCancelledError on the first wave check and
    looks like it was instantly cancelled from the user's perspective.
    """
    from larkhelm.concurrency import _get_cancel_event
    import larkhelm.crew._commands as _c

    chat = "test_chat_stale_cancel"
    # Simulate the leftover state from a previous crew that auto-cancelled.
    prev_ev = _get_cancel_event(chat)
    prev_ev.set()
    assert prev_ev.is_set()

    captured: dict = {}
    def _fake_run_crew(state, total_timeout):
        captured["cancel_set_at_run_crew"] = state.cancel_ev.is_set()

    # Stub deeper crew machinery so the test stays at the cmd-level.
    monkeypatch.setattr("larkhelm.crew._runner._run_crew", _fake_run_crew)
    monkeypatch.setattr(_c, "_augment_requirement_with_context",
                        lambda req, *_a, **_kw: req)
    # ``_pin_task_card`` is imported inside the function — patch at its
    # canonical module so the late ``from ... import`` picks up the stub.
    monkeypatch.setattr("larkhelm.lark_client._pin_task_card",
                        lambda *a, **kw: None)
    monkeypatch.setattr("larkhelm.lark_client._reply_card_raw",
                        lambda *a, **kw: "mid_fake")
    monkeypatch.setattr("larkhelm.lark_client._send_card_raw",
                        lambda *a, **kw: "mid_fake")

    # Drive the dev entry impl directly to bypass thread plumbing.
    _c._run_dev_crew_inner_impl(
        chat, "regression req", user_msg_id=None,
        no_confirm=True, crew_id="cid_stale", force_replan=True,
    )

    # The newly-started crew must NOT inherit the stale cancel signal.
    assert captured.get("cancel_set_at_run_crew") is False, (
        "stale cancel_ev from previous crew leaked into new /dev — "
        "regression of breakpoint-timeout cross-contamination bug"
    )
    # And the per-chat event itself must be cleared (defensive cross-check).
    assert _get_cancel_event(chat).is_set() is False


def test_generic_crew_clears_stale_cancel_event(
    init_test_config, fake_card_sender, fake_backend_registry, monkeypatch,
):
    """Same regression for /crew (generic) entry — they share the bug class."""
    from larkhelm.concurrency import _get_cancel_event
    import larkhelm.crew._commands as _c

    chat = "test_chat_stale_cancel_generic"
    _get_cancel_event(chat).set()

    captured: dict = {}
    def _fake_run_crew(state, total_timeout):
        captured["cancel_set_at_run_crew"] = state.cancel_ev.is_set()
    monkeypatch.setattr("larkhelm.crew._runner._run_crew", _fake_run_crew)
    monkeypatch.setattr("larkhelm.lark_client._pin_task_card",
                        lambda *a, **kw: None)
    monkeypatch.setattr("larkhelm.lark_client._reply_card_raw",
                        lambda *a, **kw: "mid_fake")
    monkeypatch.setattr("larkhelm.lark_client._send_card_raw",
                        lambda *a, **kw: "mid_fake")
    monkeypatch.setattr("larkhelm.lark_client._patch_card_raw",
                        lambda *a, **kw: None)
    # Stub Manager planning to a trivial 1-agent plan so we reach _run_crew.
    from larkhelm.crew_types import CrewPlan, AgentSpec
    monkeypatch.setattr(_c, "_crew_plan", lambda *a, **kw: CrewPlan(
        title="x", agents=[AgentSpec(
            id="a", role="r", model="claude", system="", prompt="p",
            depends_on=[], timeout=60,
        )], synthesis_prompt="",
    ))

    _c._run_generic_crew_inner_impl(
        chat, "regression req", max_agents=1, total_timeout=60,
        user_msg_id=None, crew_id="cid_stale_g",
    )

    assert captured.get("cancel_set_at_run_crew") is False
    assert _get_cancel_event(chat).is_set() is False


# ── C4 #10 — generic-crew fallback uses task_profile ───────────────────


def test_generic_crew_fallback_single_agent_uses_task_profile(
    init_test_config, fake_card_sender, fake_backend_registry, monkeypatch,
):
    """C4 #10 (sister of C3 #8): when Manager planning fails AND the cancel
    event isn't set, the fallback CrewPlan must route through
    ``task_profile`` instead of hardcoding ``model="claude"``. Pins both
    the migration AND guards against any future regression that pins a
    specific backend at construction time.
    """
    from larkhelm.concurrency import _get_cancel_event
    import larkhelm.crew._commands as _c

    chat = "test_chat_fallback_profile"
    # Make sure the cancel-clear behaviour from the other test doesn't
    # interfere with this one's fixture state.
    _get_cancel_event(chat).clear()

    # Manager planning returns None → fallback path activates.
    monkeypatch.setattr(_c, "_crew_plan", lambda *a, **kw: None)

    captured: dict = {}

    def _capture_run_crew(state, total_timeout):
        # CrewState carries the resolved plan; snapshot the only-agent's spec.
        captured["agents"] = list(state.plan.agents)

    monkeypatch.setattr("larkhelm.crew._runner._run_crew", _capture_run_crew)
    monkeypatch.setattr("larkhelm.lark_client._pin_task_card",
                        lambda *a, **kw: None)
    monkeypatch.setattr("larkhelm.lark_client._reply_card_raw",
                        lambda *a, **kw: "mid_fake")
    monkeypatch.setattr("larkhelm.lark_client._send_card_raw",
                        lambda *a, **kw: "mid_fake")
    monkeypatch.setattr("larkhelm.lark_client._patch_card_raw",
                        lambda *a, **kw: None)

    _c._run_generic_crew_inner_impl(
        chat, "ambiguous request", max_agents=1, total_timeout=60,
        user_msg_id=None, crew_id="cid_fallback",
    )

    specs = captured.get("agents") or []
    assert len(specs) == 1, f"fallback should be a single-agent CrewPlan; got {specs}"
    spec = specs[0]
    assert spec.id == "agent_1"
    # The actual migration: ``model`` must be empty so resolve_backend
    # walks the task_profile path; ``task_profile`` must be set.
    assert spec.model == "", (
        f"fallback agent still hardcodes model={spec.model!r}; "
        "C3 #8 / C4 #10 migration requires model=''"
    )
    assert spec.task_profile, "fallback agent must declare a task_profile"
    # Profile choice itself is a contract: ``chat`` because the fallback is
    # a generic helper with no tool / vision requirement; heavier profiles
    # would silently demand tool-capable backends and surprise on rank.
    assert spec.task_profile == "chat", (
        f"fallback profile should be 'chat'; got {spec.task_profile!r}"
    )


# ── C4 #11 — /crew status surfaces plan ownership ─────────────────────


def test_cmd_crew_status_reports_plan_owner_when_plan_holds_slot(
    init_test_config, fake_card_sender, monkeypatch,
):
    """C4 #11 (sister of C3 #9): when ``/plan`` writes ``_active_crew``
    but ``_active_crew_states`` is empty (plan never populates the state
    dict), ``/crew status`` must NOT lie "no task running" — it should
    decode the owner token via ``describe_active_owner`` so the user
    knows a plan is serialising the chat.
    """
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _active_crew_states,
    )
    import larkhelm.crew._commands as _c

    chat = "test_chat_plan_owner"
    plan_id = "abcdef123456"
    fake_card_sender.clear()

    with _active_crew_lock:
        _active_crew[chat] = f"plan:{plan_id}"
    try:
        # Critical: ``_active_crew_states`` must stay empty — this is
        # exactly the production shape that exposes the bug. Assert
        # inside ``try`` so a fixture-leak assert failure still pops the
        # ``_active_crew`` entry we just wrote.
        with _active_crew_lock:
            assert chat not in _active_crew_states
        _c._cmd_crew_status(chat)
    finally:
        with _active_crew_lock:
            _active_crew.pop(chat, None)

    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends, "expected at least one card emitted by /crew status"
    # The pre-C4 bug emitted the "no task running" body. The fix routes
    # through describe_active_owner → body must mention /plan + id prefix.
    card = sends[0]
    # Both "无" and "没有" are negative-result wording from the pre-C4
    # "no crew task running" body; either appearing means the fix
    # regressed. Use ``and`` so the assertion fails if EITHER leaks.
    assert "无正在运行" not in card["body"] and "没有正在运行" not in card["body"], (
        f"crew_status still says 'no task': {card!r}"
    )
    assert "/plan" in card["body"], (
        f"crew_status body must name the active plan owner; got {card!r}"
    )
    assert plan_id[:8] in card["body"], (
        f"crew_status body must include plan id prefix; got {card!r}"
    )


def test_cmd_crew_status_unknown_owner_falls_back_to_describe(
    init_test_config, fake_card_sender, monkeypatch,
):
    """Defensive: a non-plan owner token (raw hex crew_id) without any
    matching CrewState should still surface a useful card rather than
    crashing or silently lying. This covers the recovery path where a
    crashed crew left ``_active_crew`` set but ``_active_crew_states``
    cleared.
    """
    from larkhelm.crew._state import (
        _active_crew, _active_crew_lock, _active_crew_states,
    )
    import larkhelm.crew._commands as _c

    chat = "test_chat_orphan_token"
    fake_card_sender.clear()

    with _active_crew_lock:
        _active_crew[chat] = "deadbeef0000"  # hex crew_id, no state
    try:
        with _active_crew_lock:
            assert chat not in _active_crew_states
        _c._cmd_crew_status(chat)
    finally:
        with _active_crew_lock:
            _active_crew.pop(chat, None)

    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends
    body = sends[0]["body"]
    # ``describe_active_owner`` renders raw hex as "/crew 或 /dev 任务".
    assert "/crew" in body and "/dev" in body, (
        f"crew_status didn't surface owner descriptor; got {body!r}"
    )


def test_cmd_crew_status_no_active_no_history_unchanged(
    init_test_config, fake_card_sender,
):
    """Regression-pin: pre-C4 behaviour for the clean "no task" path must
    keep emitting the same 'no task running' card (we didn't accidentally
    widen the new ``elif owner_token`` to swallow it).
    """
    import larkhelm.crew._commands as _c

    fake_card_sender.clear()
    _c._cmd_crew_status("test_chat_truly_idle_xyz")
    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends
    # Either the "no recent task" card OR the "recent task" card is OK
    # (depends on log state); the key requirement is no "/plan" leakage.
    assert "/plan" not in sends[0]["body"]
