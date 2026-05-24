"""P2 AC-02: tests for ``larkhelm.bridge`` six-function decomposition.

The bridge's ``main()`` is now a thin composition over six helpers; this
test drives each helper with a fake config dataclass + fake lark.Client
so unit coverage stays achievable without spinning up a real bridge.
"""
from __future__ import annotations

import os
import signal
import sys
import types
from pathlib import Path

import pytest

os.environ.setdefault("LARKHELM_TEST_MODE", "1")

from larkhelm import bridge as _br  # noqa: E402


# ── Shared fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def fake_cfg(tmp_path: Path):
    """Build the smallest config-like object the bridge helpers need."""
    return types.SimpleNamespace(
        APP_ID="test_app_id",
        APP_SECRET="test_app_secret",
        DATA_DIR=tmp_path,
        CLAUDE_CMD="claude",
        GEMINI_CMD="gemini",
        DEFAULT_MODEL="claude",
        DEFAULT_CWD=str(tmp_path / "work"),
        RESPONSE_TIMEOUT=30,
        ALLOWED_CHATS=set(),
        SKIP_PERMISSIONS=True,  # avoids _start_perm_server during tests
        HEALTH_ENDPOINT_PORT=0,  # disables health server
        HEALTH_BIND_ADDR="127.0.0.1",
        MEMORY_LIMIT_MB=512,
    )


# ── _install_pid_lock ────────────────────────────────────────────────────


def test_pid_lock_acquires_then_blocks(tmp_path: Path):
    # First acquire must succeed.
    assert _br._install_pid_lock(tmp_path) is True
    # A second attempt from the same process — flock(EX|NB) on a fresh fd
    # of the same path fails because the previous fd still holds the lock.
    assert _br._install_pid_lock(tmp_path) is False


def test_pid_lock_back_compat_alias_acquire(tmp_path: Path):
    """The legacy ``_acquire_pid_lock`` name still routes to the new helper."""
    assert _br._acquire_pid_lock(tmp_path) is True


# ── _install_signal_handlers ─────────────────────────────────────────────


def test_install_signal_handlers_registers_sigterm(monkeypatch):
    recorded = []

    def _fake_signal(signum, handler):
        recorded.append((signum, handler))

    # Reset the idempotency guard so a previous test's run doesn't shadow.
    monkeypatch.setattr(_br, "_signal_handlers_installed", False)
    monkeypatch.setattr(signal, "signal", _fake_signal)
    _br._install_signal_handlers()
    assert any(s == signal.SIGTERM for s, _ in recorded), recorded


def test_install_signal_handlers_idempotent(monkeypatch):
    calls: list = []
    monkeypatch.setattr(signal, "signal", lambda s, h: calls.append(s))
    monkeypatch.setattr(_br, "_signal_handlers_installed", False)
    _br._install_signal_handlers()
    _br._install_signal_handlers()  # second call must be no-op
    assert len(calls) == 1


# ── _initialise_clients ──────────────────────────────────────────────────


def test_initialise_clients_builds_lark_client(monkeypatch, fake_cfg):
    import lark_oapi as lark
    import larkhelm.lark_client as _lc

    built_specs = {}

    class _FakeBuilder:
        def app_id(self, v):
            built_specs["app_id"] = v
            return self

        def app_secret(self, v):
            built_specs["app_secret"] = v
            return self

        def build(self):
            return types.SimpleNamespace(_fake=True)

    monkeypatch.setattr(lark.Client, "builder", staticmethod(_FakeBuilder))
    monkeypatch.setattr(_lc, "_fetch_bot_open_id", lambda: None)

    client = _br._initialise_clients(fake_cfg)
    assert client is _lc.client
    assert getattr(client, "_fake", False) is True
    assert built_specs["app_id"] == fake_cfg.APP_ID
    assert built_specs["app_secret"] == fake_cfg.APP_SECRET


# ── _register_handlers ───────────────────────────────────────────────────


def test_register_handlers_returns_dispatcher(monkeypatch):
    import lark_oapi as lark

    class _FakeBuilder:
        def __init__(self, *args, **kwargs):
            self.routes = []

        def register_p2_im_message_receive_v1(self, fn):
            self.routes.append(("msg", fn))
            return self

        def register_p2_card_action_trigger(self, fn):
            self.routes.append(("card", fn))
            return self

        def register_p2_im_message_reaction_created_v1(self, fn):
            self.routes.append(("reaction", fn))
            return self

        def build(self):
            return ("dispatcher", self.routes)

    monkeypatch.setattr(
        lark.EventDispatcherHandler, "builder", staticmethod(_FakeBuilder),
    )
    out = _br._register_handlers(client=None)
    assert isinstance(out, tuple) and out[0] == "dispatcher"
    kinds = {k for k, _ in out[1]}
    assert kinds == {"msg", "card", "reaction"}


# ── _start_background_threads ────────────────────────────────────────────


def test_background_threads_skips_perm_when_skip_permissions(monkeypatch, fake_cfg):
    """When ``SKIP_PERMISSIONS=True``, _start_perm_server must not be called."""
    perm_called = []

    monkeypatch.setattr(_br, "_start_perm_server", lambda: perm_called.append(True))
    monkeypatch.setattr(_br, "_start_cron_scheduler", lambda: None)
    monkeypatch.setattr(_br, "_start_gc_thread", lambda: None)
    monkeypatch.setattr(_br, "_start_memory_boot_warmup", lambda: None)
    import larkhelm.memory_watchdog as _mw
    monkeypatch.setattr(_mw, "start_memory_watchdog", lambda _x: None)
    import larkhelm.crew as _crew
    monkeypatch.setattr(_crew, "resume_interrupted_crews", lambda: None)
    import larkhelm.plan_persistence as _pp
    monkeypatch.setattr(_pp, "resume_interrupted_plans", lambda: None, raising=False)
    # Health server: monkey-patch so the test doesn't bind a port.
    import larkhelm.health_server as _hs
    monkeypatch.setattr(_hs, "start_health_server", lambda *a, **kw: True)

    _br._start_background_threads(fake_cfg)
    assert perm_called == []


# ── _post_init_notify ────────────────────────────────────────────────────


def test_post_init_notify_handles_missing_restart_marker(tmp_path, fake_cfg):
    # When the marker file is absent, the helper should print the banner
    # without raising and without sending any cards.
    _br._post_init_notify(fake_cfg)


def test_post_init_notify_sends_restart_card(monkeypatch, tmp_path, fake_cfg):
    import json
    import larkhelm.lark_client as _lc

    sent = []
    monkeypatch.setattr(
        _lc, "send_card",
        lambda chat, title, body, color="blue", **kw: sent.append((chat, title, body)),
    )
    marker = fake_cfg.DATA_DIR / "_restart_notify.json"
    marker.write_text(json.dumps({"chat_id": "oc_x", "ts": 9e18}))
    _br._post_init_notify(fake_cfg)
    # File deleted, send_card invoked once with our chat_id.
    assert not marker.exists()
    assert sent and sent[0][0] == "oc_x"
    # Bare payload (no prev/new/subject) keeps the minimal body.
    assert "服务已成功重启" in sent[0][2]


def test_post_init_notify_renders_prev_new_subject(monkeypatch, tmp_path, fake_cfg):
    """When _do_upgrade writes prev_head / new_head / commit_subject, the
    "升级完成" card should embed them so the operator can see *what* shipped."""
    import json
    import larkhelm.lark_client as _lc

    sent = []
    monkeypatch.setattr(
        _lc, "send_card",
        lambda chat, title, body, color="blue", **kw: sent.append((chat, title, body)),
    )
    marker = fake_cfg.DATA_DIR / "_restart_notify.json"
    marker.write_text(json.dumps({
        "chat_id": "oc_y",
        "ts": 9e18,
        "prev_head": "cd28e0b",
        "new_head": "f998f87",
        "commit_subject": "fix(upgrade): re-resolve SOURCE_DIR on entry",
    }))
    _br._post_init_notify(fake_cfg)
    assert not marker.exists()
    assert sent and sent[0][0] == "oc_y"
    body = sent[0][2]
    assert "cd28e0b" in body and "f998f87" in body, "diff arrow must show both hashes"
    assert "fix(upgrade)" in body, "subject must appear"


def test_post_init_notify_truncates_long_subject(monkeypatch, tmp_path, fake_cfg):
    """Pathological 200-char commit subjects must not blow up the card."""
    import json
    import larkhelm.lark_client as _lc

    sent = []
    monkeypatch.setattr(
        _lc, "send_card",
        lambda chat, title, body, color="blue", **kw: sent.append(body),
    )
    long_subj = "x" * 200
    marker = fake_cfg.DATA_DIR / "_restart_notify.json"
    marker.write_text(json.dumps({
        "chat_id": "oc_z", "ts": 9e18,
        "new_head": "abcdef0", "commit_subject": long_subj,
    }))
    _br._post_init_notify(fake_cfg)
    assert sent
    body = sent[0]
    assert "…" in body, "truncation ellipsis missing"
    # No line longer than ~125 chars (120 + small overhead)
    assert all(len(line) <= 125 for line in body.splitlines()), \
        f"some body line exceeded budget: {[len(l) for l in body.splitlines()]}"


# ── _handle_sigterm ──────────────────────────────────────────────────────


def test_sigterm_handler_does_not_exit(monkeypatch):
    """AC-05: _handle_sigterm must not call sys.exit and must only set shutting_down."""
    import larkhelm.concurrency as _conc

    exit_called: list = []
    monkeypatch.setattr(sys, "exit", lambda code=0: exit_called.append(code))
    # Reset the flag so we can assert it was set by the handler.
    monkeypatch.setattr(_conc, "_shutting_down", False)

    _br._handle_sigterm(signal.SIGTERM, None)

    assert exit_called == [], "sys.exit must not be called from the signal handler"
    assert _conc.is_shutting_down(), "set_shutting_down() must be called"


# ── main() composition order ─────────────────────────────────────────────


def test_main_invokes_helpers_in_order(monkeypatch, tmp_path, fake_cfg):
    """Verify main() calls the six helpers in the documented order.

    We monkeypatch every helper to a name-recording stub and assert on
    the sequence. ``ws.Client.start`` is replaced with a no-op so the
    test doesn't try to open a real websocket.
    """
    order: list[str] = []
    import larkhelm.config as _cfg
    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path, raising=False)
    # Make config look like our fake_cfg for the helpers' attribute reads.
    for attr in ("APP_ID", "APP_SECRET", "CLAUDE_CMD", "GEMINI_CMD",
                 "DEFAULT_MODEL", "DEFAULT_CWD", "RESPONSE_TIMEOUT",
                 "ALLOWED_CHATS"):
        monkeypatch.setattr(_cfg, attr, getattr(fake_cfg, attr), raising=False)

    # ``bridge.py`` imports these names directly (``from ... import _init_runtime``)
    # so we must patch the BRIDGE's local binding, not the source module.
    monkeypatch.setattr(_br, "_init_runtime",
                        lambda *a, **kw: order.append("init_runtime"))
    monkeypatch.setattr(_br, "_install_pid_lock",
                        lambda d: order.append("pid_lock") or True)
    monkeypatch.setattr(_br, "rotate_jsonl_if_needed",
                        lambda: order.append("rotate"))
    monkeypatch.setattr(_br, "_load_global_state",
                        lambda: order.append("load_global"))
    monkeypatch.setattr(_br, "_initialise_clients",
                        lambda c: order.append("clients") or types.SimpleNamespace())
    monkeypatch.setattr(_br, "_register_handlers",
                        lambda c: order.append("handlers") or "evh")
    monkeypatch.setattr(_br, "_install_signal_handlers",
                        lambda: order.append("signal"))
    monkeypatch.setattr(_br, "_start_background_threads",
                        lambda c: order.append("threads"))
    monkeypatch.setattr(_br, "_post_init_notify",
                        lambda c: order.append("notify"))
    # _make_interruptible_select returns a truthy sentinel so main() takes
    # the direct ws_client.start() path (not the daemon-thread fallback).
    monkeypatch.setattr(_br, "_make_interruptible_select",
                        lambda: object())
    monkeypatch.setattr(_br, "_run_shutdown_sequence",
                        lambda: order.append("shutdown_seq") or {})
    # Reset the executed guard so re-running the test doesn't skip the sequence.
    monkeypatch.setattr(_br, "_shutdown_sequence_executed", False)

    class _FakeWS:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            order.append("ws_start")

    import lark_oapi.ws as _ws
    monkeypatch.setattr(_ws, "Client", _FakeWS)

    _br.main()
    assert order == [
        "init_runtime", "pid_lock", "rotate", "load_global",
        "clients", "handlers",
        "signal", "threads", "notify",
        "ws_start", "shutdown_seq",
    ]


def test_main_cleanup_sequence_on_shutdown(monkeypatch, tmp_path, fake_cfg):
    """AC-06: _run_shutdown_sequence is called after ws_client.start() returns."""
    import larkhelm.config as _cfg

    sequence: list[str] = []

    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path, raising=False)
    for attr in ("APP_ID", "APP_SECRET", "CLAUDE_CMD", "GEMINI_CMD",
                 "DEFAULT_MODEL", "DEFAULT_CWD", "RESPONSE_TIMEOUT",
                 "ALLOWED_CHATS"):
        monkeypatch.setattr(_cfg, attr, getattr(fake_cfg, attr), raising=False)

    monkeypatch.setattr(_br, "_init_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(_br, "_install_pid_lock", lambda d: True)
    monkeypatch.setattr(_br, "rotate_jsonl_if_needed", lambda: None)
    monkeypatch.setattr(_br, "_load_global_state", lambda: None)
    monkeypatch.setattr(_br, "_initialise_clients",
                        lambda c: types.SimpleNamespace())
    monkeypatch.setattr(_br, "_register_handlers", lambda c: "evh")
    monkeypatch.setattr(_br, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(_br, "_start_background_threads", lambda c: None)
    monkeypatch.setattr(_br, "_post_init_notify", lambda c: None)
    monkeypatch.setattr(_br, "_make_interruptible_select", lambda: object())
    monkeypatch.setattr(_br, "_run_shutdown_sequence",
                        lambda: sequence.append("shutdown_seq") or {})
    monkeypatch.setattr(_br, "_shutdown_sequence_executed", False)

    class _FakeWS:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            sequence.append("ws_start")

    import lark_oapi.ws as _ws
    monkeypatch.setattr(_ws, "Client", _FakeWS)

    _br.main()

    assert "ws_start" in sequence, "ws_client.start() must be called"
    assert "shutdown_seq" in sequence, "_run_shutdown_sequence must be called"
    ws_idx = sequence.index("ws_start")
    seq_idx = sequence.index("shutdown_seq")
    assert seq_idx > ws_idx, "_run_shutdown_sequence must be called after ws_client.start()"


def test_main_uses_daemon_thread_fallback_when_select_unavailable(monkeypatch, tmp_path, fake_cfg):
    """When _make_interruptible_select() returns None, main() exits via the is_shutting_down() poll."""
    import larkhelm.config as _cfg
    import larkhelm.concurrency as _conc

    sequence: list[str] = []

    monkeypatch.setattr(_cfg, "DATA_DIR", tmp_path, raising=False)
    for attr in ("APP_ID", "APP_SECRET", "CLAUDE_CMD", "GEMINI_CMD",
                 "DEFAULT_MODEL", "DEFAULT_CWD", "RESPONSE_TIMEOUT",
                 "ALLOWED_CHATS"):
        monkeypatch.setattr(_cfg, attr, getattr(fake_cfg, attr), raising=False)

    monkeypatch.setattr(_br, "_init_runtime", lambda *a, **kw: None)
    monkeypatch.setattr(_br, "_install_pid_lock", lambda d: True)
    monkeypatch.setattr(_br, "rotate_jsonl_if_needed", lambda: None)
    monkeypatch.setattr(_br, "_load_global_state", lambda: None)
    monkeypatch.setattr(_br, "_initialise_clients", lambda c: types.SimpleNamespace())
    monkeypatch.setattr(_br, "_register_handlers", lambda c: "evh")
    monkeypatch.setattr(_br, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(_br, "_start_background_threads", lambda c: None)
    monkeypatch.setattr(_br, "_post_init_notify", lambda c: None)
    # Return None to trigger the daemon-thread fallback path
    monkeypatch.setattr(_br, "_make_interruptible_select", lambda: None)
    monkeypatch.setattr(_br, "_run_shutdown_sequence",
                        lambda: sequence.append("shutdown_seq") or {})
    monkeypatch.setattr(_br, "_shutdown_sequence_executed", False)
    # Pre-set shutting_down so the polling loop exits immediately without sleeping
    monkeypatch.setattr(_conc, "_shutting_down", True)

    class _FakeWS:
        def __init__(self, *a, **kw):
            pass

        def start(self):
            sequence.append("ws_start")  # called from daemon thread, may race

    import lark_oapi.ws as _ws
    monkeypatch.setattr(_ws, "Client", _FakeWS)

    _br.main()

    # Shutdown sequence must be called in the fallback path too
    assert "shutdown_seq" in sequence, "_run_shutdown_sequence must be called after fallback loop"


def test_run_shutdown_sequence_is_idempotent(monkeypatch):
    """_run_shutdown_sequence returns {} on second call without re-running steps."""
    import time as _time

    monkeypatch.setattr(_br, "_shutdown_sequence_executed", False)
    # Speed up wait_for_idle which is bound into the bridge module at import time
    monkeypatch.setattr(_br, "wait_for_idle", lambda timeout=120.0: True)
    monkeypatch.setattr(_time, "sleep", lambda s: None)

    result1 = _br._run_shutdown_sequence()
    assert isinstance(result1, dict) and "final_status" in result1, \
        "First call must return a non-empty result dict with final_status"

    # Idempotency guard must prevent re-execution
    result2 = _br._run_shutdown_sequence()
    assert result2 == {}, "Second call must return empty dict (idempotency guard)"
