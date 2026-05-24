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


# ── C5 #13 — Manager-default model → task_profile fallback ────────────


def test_agents_from_manager_plan_no_model_uses_engineer_profile(
    init_test_config,
):
    """Pre-C5 the comprehension default ``a.get("model", "claude")``
    pinned every Manager-emitted field-less agent to Claude. C5 changes
    the default to ``model=""`` + ``task_profile="engineer"`` so the
    dispatcher's ``rank_for_task`` picks the highest-ranked healthy
    tool-capable backend. This pins the new fallback contract.
    """
    from larkhelm.crew._commands import _agents_from_manager_plan
    out = _agents_from_manager_plan(
        [{"id": "a", "role": "researcher", "prompt": "do x"}],  # no model field
        max_agents=5,
    )
    assert len(out) == 1
    spec = out[0]
    assert spec.model == "", (
        f"Manager-default model must be empty so resolve_backend walks "
        f"the task_profile path; got {spec.model!r}"
    )
    assert spec.task_profile == "engineer", (
        f"Manager-default fallback should pick 'engineer' (Manager-planned "
        f"agents typically do non-trivial work); got {spec.task_profile!r}"
    )


def test_agents_from_manager_plan_explicit_model_wins(init_test_config):
    """Legacy compat: when Manager explicitly emits ``model``, honour it
    and DO NOT silently override with a task_profile — Manager has
    already picked a backend on purpose (e.g. vision-required task).
    """
    from larkhelm.crew._commands import _agents_from_manager_plan
    out = _agents_from_manager_plan(
        [{"id": "a", "role": "r", "prompt": "p", "model": "kimi"}],
        max_agents=5,
    )
    spec = out[0]
    assert spec.model == "kimi"
    # Explicit model means no profile fallback — profile-only resolution
    # would override Manager's deliberate choice.
    assert spec.task_profile == "", (
        f"explicit model must NOT auto-acquire a task_profile; "
        f"got task_profile={spec.task_profile!r}"
    )


def test_agents_from_manager_plan_explicit_profile_respected(init_test_config):
    """If Manager LLM emits ``task_profile`` explicitly (forward-compat
    with a future prompt revision that teaches it the field), honour
    that even when no model is given. Default fallback must only fire
    when *both* fields are absent.
    """
    from larkhelm.crew._commands import _agents_from_manager_plan
    out = _agents_from_manager_plan(
        [{"id": "a", "role": "r", "prompt": "p", "task_profile": "reviewer"}],
        max_agents=5,
    )
    spec = out[0]
    assert spec.model == ""
    assert spec.task_profile == "reviewer"


def test_agents_from_manager_plan_respects_max_agents_cap(init_test_config):
    """Refactor-pin: the previous inline comprehension capped at
    ``raw_agents[:max_agents]``. The extracted helper must preserve that
    so a misbehaving Manager LLM can't explode the agent count.
    """
    from larkhelm.crew._commands import _agents_from_manager_plan
    raw = [{"id": f"a{i}", "role": "r", "prompt": "p"} for i in range(10)]
    out = _agents_from_manager_plan(raw, max_agents=3)
    assert len(out) == 3
    assert [s.id for s in out] == ["a0", "a1", "a2"]


# ── C5 #14 — queue card label per owner type ─────────────────────────


def _patch_active_crew(monkeypatch, owner_token: str):
    """Helper: simulate a chat slot held by ``owner_token`` for the
    duration of the test, without touching the real ``_active_crew``
    dict (avoids cross-test fixture leakage).
    """
    import larkhelm.crew._state as _state
    monkeypatch.setattr(_state, "is_crew_running", lambda chat_id: True)
    monkeypatch.setattr(_state, "current_owner", lambda chat_id: owner_token)
    # subscribe_crew_done must return an already-set event so the
    # background ``_after`` thread doesn't sit forever.
    fake_ev = threading.Event()
    fake_ev.set()
    monkeypatch.setattr(_state, "subscribe_crew_done", lambda chat_id: fake_ev)


def _captured_queue_card_title(fake_card_sender) -> tuple[str, str]:
    """Pull the queue card title + body from the fake sender's record.

    The queue card can emit through ``_reply_card_raw`` (when
    ``user_msg_id`` is present) OR ``_send_card_raw`` (when not). The
    fixture records the former as ``kind="reply_raw"`` and the latter
    as ``kind="send_raw"``. Both stash the card payload under ``"card"``;
    when the underlying helper is the patched JSON path the payload is
    a JSON *string*, otherwise a dict.
    """
    import json as _json
    sends = [c for c in fake_card_sender
             if c["kind"] in ("send_raw", "reply_raw", "patch")]
    assert sends, f"no queue card emitted; recorded={fake_card_sender}"
    card_payload = sends[0]["card"]
    card = _json.loads(card_payload) if isinstance(card_payload, str) else card_payload
    title = card.get("header", {}).get("title", {}).get("content", "")
    body_parts: list[str] = []
    for elem in card.get("body", {}).get("elements", []) or []:
        if isinstance(elem, dict):
            content = elem.get("content")
            if isinstance(content, str):
                body_parts.append(content)
    body = "\n".join(body_parts)
    return title, body


def _run_query_until_queue_card(
    monkeypatch, fake_card_sender, owner_token: str,
    chat_id: str, user_msg_id: str,
):
    """Drive ``_do_query`` just far enough to emit the queue card, then
    short-circuit. ``_update_pending_card_mid`` is the first call AFTER
    the queue card lands in ``fake_card_sender`` — monkey-patching it
    to raise lets the inner ``except Exception`` in the queue branch
    fire ``_pop_pending`` + ``return`` cleanly, so we never enter the
    backend-routing code path.

    ``_query.py`` does ``from lark_client import _reply_card_raw, ...``
    at module load, so the conftest fixture's patch on
    ``lark_client._reply_card_raw`` only catches calls in the test
    where ``_query`` was first imported. We re-bind on the ``_query``
    module directly so the recording works regardless of test order.
    """
    from larkhelm.handlers import _query as _q
    from larkhelm.concurrency import _pop_pending
    # Defensive: a previous test (queued plan/crew owner) may have left
    # ``_pending_msg[chat_id]`` populated if its cleanup didn't run; pop
    # so ``_set_pending`` here returns ``None`` (the "no prior pending"
    # path) and we exercise the ``_reply_card_raw`` branch consistently
    # regardless of test order.
    _pop_pending(chat_id)
    _patch_active_crew(monkeypatch, owner_token)

    # Re-route _query.py's frozen-at-import bindings to fake_card_sender
    # so the queue card is recorded regardless of test ordering.
    import json as _json

    def _q_reply(message_id, card_json, in_thread=True):
        fake_card_sender.append({
            "kind": "reply_raw", "message_id": message_id,
            "card": card_json,
            "mid": f"fake_reply_raw_mid_{len(fake_card_sender)}",
        })
        return f"fake_reply_raw_mid_{len(fake_card_sender)}"

    def _q_send(chat_id_arg, card, _fallback_text=None):
        fake_card_sender.append({
            "kind": "send_raw", "chat_id": chat_id_arg,
            "card": card,
            "mid": f"fake_send_raw_mid_{len(fake_card_sender)}",
        })
        return f"fake_send_raw_mid_{len(fake_card_sender)}"

    def _q_patch(mid, card):
        fake_card_sender.append({"kind": "patch", "mid": mid, "card": card})
        return True

    monkeypatch.setattr(_q, "_reply_card_raw", _q_reply)
    monkeypatch.setattr(_q, "_send_card_raw", _q_send)
    monkeypatch.setattr(_q, "_patch_card_raw", _q_patch)

    sentinel = RuntimeError("halt after queue card")

    def _explode(*a, **kw):
        raise sentinel
    monkeypatch.setattr(
        "larkhelm.handlers._query._update_pending_card_mid", _explode,
    )

    fake_card_sender.clear()
    _q._do_query(chat_id, "please answer", "claude", user_msg_id)
    # Defensive cleanup so the next test starts from a known state
    # even if this test's _pop_pending in the except branch didn't fire.
    _pop_pending(chat_id)


def test_query_queue_card_label_for_plan_owner(
    init_test_config, fake_card_sender, monkeypatch,
):
    """C5 #14: when a ``/plan`` holds the slot, the queue card title +
    body must name the plan (not "Crew"). Drives the legacy
    ``_query._do_query`` queue branch via the
    ``is_crew_running``-True path and asserts the title says
    "Plan 运行中".
    """
    _run_query_until_queue_card(
        monkeypatch, fake_card_sender,
        owner_token="plan:abcdef123456",
        chat_id="oc_q_plan_owner", user_msg_id="msg1",
    )
    title, body = _captured_queue_card_title(fake_card_sender)
    assert "Plan" in title, f"queue card title must say Plan, got {title!r}"
    assert "Crew" not in title, f"queue card title still says Crew, got {title!r}"
    assert "/plan" in body and "abcdef12" in body, (
        f"queue body must name the plan owner; got {body!r}"
    )


def test_query_queue_card_label_for_crew_owner(
    init_test_config, fake_card_sender, monkeypatch,
):
    """Regression-pin: when an actual crew/dev holds the slot, the
    queue card must still say "Crew 运行中" (we didn't break the legacy
    happy path by introducing the owner-aware branching).
    """
    _run_query_until_queue_card(
        monkeypatch, fake_card_sender,
        owner_token="deadbeef0000",   # raw hex == crew/dev
        chat_id="oc_q_crew_owner", user_msg_id="msg2",
    )
    title, body = _captured_queue_card_title(fake_card_sender)
    assert "Crew" in title, f"queue card title must say Crew, got {title!r}"
    assert "Plan" not in title
    assert "Crew" in body


def _drive_session_queue(monkeypatch, owner_token: str, chat_id: str):
    """Build a QuerySession and call ``_maybe_queue_behind_crew`` directly.

    Returns the recording list (list of card dicts). Same module-binding
    workaround as :func:`_run_query_until_queue_card` — re-routes
    ``_query_session.py``'s frozen-at-import lark_client bindings to
    in-test fakes so the recording survives test order.
    """
    from larkhelm.handlers import _query_session as _qs
    from larkhelm.concurrency import _pop_pending
    _pop_pending(chat_id)
    _patch_active_crew(monkeypatch, owner_token)

    recorded: list[dict] = []

    def _reply(message_id, card_json, in_thread=True):
        recorded.append({"kind": "reply_raw", "message_id": message_id,
                         "card": card_json,
                         "mid": f"fmid_{len(recorded)}"})
        return f"fmid_{len(recorded)}"

    def _send(chat_id_arg, card, _fallback_text=None):
        recorded.append({"kind": "send_raw", "chat_id": chat_id_arg,
                         "card": card, "mid": f"fmid_{len(recorded)}"})
        return f"fmid_{len(recorded)}"

    def _patch(mid, card):
        recorded.append({"kind": "patch", "mid": mid, "card": card})
        return True

    monkeypatch.setattr(_qs, "_reply_card_raw", _reply, raising=False)
    monkeypatch.setattr(_qs, "_send_card_raw", _send, raising=False)
    monkeypatch.setattr(_qs, "_patch_card_raw", _patch, raising=False)
    # ``_update_pending_card_mid`` only fires AFTER card emit on the
    # non-existing-mid path; let it succeed (or no-op) so the queue
    # branch returns ``True`` normally.
    monkeypatch.setattr(_qs, "_update_pending_card_mid",
                        lambda *a, **k: None, raising=False)
    # Background ``_after`` thread spawns ``_re_dispatch_query`` once
    # the fake done-event fires — gag it so we don't spawn a live
    # Claude process under pytest. Daemon thread, but cleaner output.
    monkeypatch.setattr(_qs, "_re_dispatch_query",
                        lambda *a, **k: None, raising=False)

    sess = _qs.QuerySession(
        chat_id=chat_id, message="please answer", model="claude",
        user_msg_id="msg_q",
    )
    queued = sess._maybe_queue_behind_crew()
    _pop_pending(chat_id)
    return queued, recorded


def test_session_queue_card_label_for_plan_owner(
    init_test_config, monkeypatch,
):
    """C5 #14: v2 ``QuerySession._maybe_queue_behind_crew`` must mirror
    the legacy ``_query._do_query`` queue-card owner-aware labelling.
    Both sites are intentionally kept in sync — fix one, fix the other.
    """
    queued, recorded = _drive_session_queue(
        monkeypatch, "plan:abcdef123456", "oc_qs_plan_owner",
    )
    assert queued is True
    title, body = _captured_queue_card_title(recorded)
    assert "Plan" in title and "Crew" not in title, (
        f"v2 queue card title must say Plan, got {title!r}"
    )
    assert "/plan" in body and "abcdef12" in body


def test_session_queue_card_label_for_crew_owner(
    init_test_config, monkeypatch,
):
    """Regression-pin for the v2 path's crew/dev happy case."""
    queued, recorded = _drive_session_queue(
        monkeypatch, "deadbeef0000", "oc_qs_crew_owner",
    )
    assert queued is True
    title, body = _captured_queue_card_title(recorded)
    assert "Crew" in title and "Plan" not in title


# ── C5 #15 — _run_pipeline_inner conflict card decodes owner ──────────


def test_run_pipeline_inner_conflict_card_decodes_plan_owner(
    init_test_config, fake_card_sender, monkeypatch,
):
    """C5 #15 (sister of C4 #11/#12 + C5 #14): the third ``_active_crew``
    conflict-card site (Phase 5 PipelineAgent path) was the last one
    still hard-coding "⚠️ Crew 已在运行". When a ``/plan`` holds the
    slot, the user must see the plan in the card body, not a wrong
    "crew" label.
    """
    from larkhelm.crew._state import _active_crew, _active_crew_lock
    from larkhelm.crew import _commands as _c
    from larkhelm.crew_types import CrewPlan, AgentSpec

    chat = "oc_pipeline_conflict_plan"
    fake_card_sender.clear()
    # Pre-occupy the slot as a plan owner so _run_pipeline_inner's
    # ``if chat_id in _active_crew`` branch fires immediately.
    with _active_crew_lock:
        _active_crew[chat] = "plan:abcdef123456"
    try:
        # A minimal-but-valid CrewPlan is enough — the conflict branch
        # short-circuits before any agent runs.
        plan = CrewPlan(
            title="t",
            agents=[AgentSpec(id="a", role="r",
                              model="", task_profile="engineer",
                              system="", prompt="p",
                              depends_on=[], timeout=60)],
            synthesis_prompt="",
        )
        _c._run_pipeline_inner(
            chat_id=chat, plan=plan, requirement="any",
            user_msg_id=None, crew_id="cid_pipe_conf",
            sender_open_id="",
        )
    finally:
        with _active_crew_lock:
            _active_crew.pop(chat, None)

    sends = [c for c in fake_card_sender if c["kind"] == "send_card"]
    assert sends, f"conflict card not emitted; recorded={fake_card_sender}"
    body = sends[0]["body"]
    title = sends[0]["title"]
    # Old hardcoded title "Crew 已在运行" must NOT leak when a plan owns
    # the slot — pre-C5 #15 it would have.
    assert "Crew 已在运行" not in title, (
        f"pipeline conflict card still hardcodes Crew title; got {title!r}"
    )
    assert "/plan" in body and "abcdef12" in body, (
        f"pipeline conflict card must surface plan owner; got body={body!r}"
    )


# ── owner_kind unit pin ───────────────────────────────────────────────


def test_owner_kind_classifies_known_tokens():
    """Pin C5's ``owner_kind`` contract so future additions of new owner
    classes (e.g. remote agents) can't silently regress callers that
    branch on the return value (e.g. queue card label).
    """
    from larkhelm.crew._state import owner_kind
    assert owner_kind("") == "", "empty owner must classify as ''"
    assert owner_kind("plan:abc123def") == "plan"
    assert owner_kind("deadbeef0000") == "crew", (
        "raw hex must classify as 'crew' (sibling of describe_active_owner)"
    )
    assert owner_kind("plan:") == "plan", (
        "even an empty plan_id still classifies as plan kind"
    )
