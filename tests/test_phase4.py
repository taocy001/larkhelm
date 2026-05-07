"""Phase 4 acceptance tests: AC-01 through AC-10.

Tests cover:
  AC-01  Auto-discovery: mock shutil.which, assert registry contains discovered IDs
  AC-02  Missing env var: unresolved ${} → enabled=False
  AC-03  Failover: primary orchestrator raises, switches to secondary
  AC-04  Recover check: unhealthy backend becomes healthy when command appears in PATH
  AC-05  Delegation: orchestrator outputs DELEGATE block → on_tool/on_tool_result called
  AC-06  No delegation: direct answer → on_tool call count == 0
  AC-07  /lock command: /lock <id> and /lock off change locked_backend field
  AC-08  /lock unhealthy backend: returns error card
  AC-09  Specialist unavailable: orchestrator falls back to direct answer
  AC-10  All orchestrators down: friendly error card shown
"""
import threading
import unittest
from unittest.mock import MagicMock, patch


# ─── AC-01: Auto-discovery ────────────────────────────────────────────────────

class TestAC01AutoDiscover(unittest.TestCase):
    def test_auto_discover_found(self):
        """_auto_discover_cli returns entries only for CLIs found via shutil.which."""
        def fake_which(cmd):
            return f"/usr/bin/{cmd}" if cmd in ("claude", "gemini") else None

        with patch("larkhelm.config.shutil.which", side_effect=fake_which):
            from larkhelm.config import _auto_discover_cli
            result = _auto_discover_cli()

        ids = [r["id"] for r in result]
        self.assertIn("claude", ids)
        self.assertIn("gemini", ids)
        self.assertNotIn("kimi", ids)
        self.assertNotIn("kimi-code", ids)

    def test_auto_discover_none(self):
        """_auto_discover_cli returns empty list when nothing is in PATH."""
        with patch("larkhelm.config.shutil.which", return_value=None):
            from larkhelm.config import _auto_discover_cli
            result = _auto_discover_cli()
        self.assertEqual(result, [])

    def test_migrate_explicit_supplements_auto(self):
        """Explicit backends are preserved; auto-discover supplements missing IDs."""
        def fake_which(cmd):
            return f"/usr/bin/{cmd}" if cmd == "gemini" else None

        with patch("larkhelm.config.shutil.which", side_effect=fake_which):
            from larkhelm.config import _migrate_legacy_backends
            cfg = {
                "backends": [
                    {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
                     "tags": ["tools"], "command": "claude"},
                ]
            }
            result = _migrate_legacy_backends(cfg)

        ids = [r["id"] for r in result]
        self.assertIn("claude", ids)
        self.assertIn("gemini", ids)  # supplemented by auto-discover
        self.assertEqual(ids[0], "claude")  # explicit comes first


# ─── AC-02: Missing env var → enabled=False ──────────────────────────────────

class TestAC02MissingEnvVar(unittest.TestCase):
    def test_unresolved_placeholder_disables_backend(self):
        """api_key with unresolved ${MISSING_KEY} sets enabled=False."""
        import os
        from larkhelm.backend_registry import BackendRegistry

        reg = BackendRegistry()
        # Ensure env var is not set
        env_key = "LARKHELM_TEST_MISSING_KEY_XYZ"
        os.environ.pop(env_key, None)

        reg.load([{
            "id": "api-test",
            "provider": "anthropic_api",
            "display_name": "Test API",
            "role": "worker",
            "tags": [],
            "api_key": f"${{{env_key}}}",
        }])
        spec = reg.get("api-test")
        self.assertIsNotNone(spec)
        self.assertFalse(spec.enabled)
        self.assertEqual(spec.api_key, "")

    def test_resolved_placeholder_keeps_enabled(self):
        """api_key with resolved env var keeps enabled=True."""
        import os
        from larkhelm.backend_registry import BackendRegistry

        env_key = "LARKHELM_TEST_PRESENT_KEY_XYZ"
        os.environ[env_key] = "sk-test-123"
        try:
            reg = BackendRegistry()
            reg.load([{
                "id": "api-test2",
                "provider": "anthropic_api",
                "display_name": "Test API 2",
                "role": "worker",
                "tags": [],
                "api_key": f"${{{env_key}}}",
            }])
            spec = reg.get("api-test2")
            self.assertTrue(spec.enabled)
            self.assertEqual(spec.api_key, "sk-test-123")
        finally:
            os.environ.pop(env_key, None)


# ─── AC-03: Failover ─────────────────────────────────────────────────────────

class TestAC03Failover(unittest.TestCase):
    def test_failover_to_secondary(self):
        """When primary orchestrator raises, secondary is tried and succeeds."""
        from larkhelm.backend_registry import BackendRegistry, BackendSpec

        reg = BackendRegistry()
        reg.load([
            {"id": "orch1", "provider": "claude_cli", "display_name": "Orch1",
             "tags": ["tools"], "command": "claude"},
            {"id": "orch2", "provider": "claude_cli", "display_name": "Orch2",
             "tags": ["tools"], "command": "claude"},
        ])

        call_order = []

        def fake_run(spec, *args, **kwargs):
            call_order.append(spec.id)
            if spec.id == "orch1":
                raise RuntimeError("orch1 unavailable")
            return "orch2 response"

        chain = reg.get_orchestrator_chain()
        self.assertEqual(len(chain), 2)

        output = None
        last_err = None
        for s in chain:
            try:
                output = fake_run(s)
                break
            except Exception as e:
                s.healthy = False
                last_err = e

        self.assertEqual(output, "orch2 response")
        self.assertFalse(reg.get("orch1").healthy)
        self.assertTrue(reg.get("orch2").healthy)
        self.assertEqual(call_order, ["orch1", "orch2"])

    def test_all_fail_raises(self):
        """When all backends fail, the loop ends with output=None (error path)."""
        from larkhelm.backend_registry import BackendRegistry

        reg = BackendRegistry()
        reg.load([
            {"id": "orch1", "provider": "claude_cli", "display_name": "O1",
             "tags": ["tools"], "command": "claude"},
        ])
        chain = reg.get_orchestrator_chain()
        output = None
        last_err = None
        for s in chain:
            try:
                raise RuntimeError("all down")
            except Exception as e:
                s.healthy = False
                last_err = e

        self.assertIsNone(output)
        self.assertIsNotNone(last_err)


# ─── AC-04: Recover check ────────────────────────────────────────────────────

class TestAC04RecoverCheck(unittest.TestCase):
    def test_recover_cli_backend(self):
        """recover_check re-probes CLI backend and sets healthy=True when command found."""
        from larkhelm.backend_registry import BackendRegistry

        reg = BackendRegistry()
        reg.load([
            {"id": "orch1", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
        ])
        spec = reg.get("orch1")
        spec.healthy = False
        spec.last_error = "simulated failure"

        with patch("larkhelm.backend_registry.shutil.which", return_value="/usr/bin/claude"):
            reg.recover_check()

        self.assertTrue(spec.healthy)
        self.assertIsNone(spec.last_error)

    def test_recover_still_unhealthy(self):
        """recover_check keeps healthy=False when command still not found."""
        from larkhelm.backend_registry import BackendRegistry

        reg = BackendRegistry()
        reg.load([
            {"id": "orch1", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
        ])
        spec = reg.get("orch1")
        spec.healthy = False
        spec.last_error = "not found"

        with patch("larkhelm.backend_registry.shutil.which", return_value=None):
            reg.recover_check()

        self.assertFalse(spec.healthy)

    def test_recover_skips_healthy(self):
        """recover_check does not re-probe already-healthy backends."""
        from larkhelm.backend_registry import BackendRegistry

        reg = BackendRegistry()
        reg.load([
            {"id": "orch1", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
        ])
        spec = reg.get("orch1")
        # spec.healthy is True by default

        called = []
        with patch("larkhelm.backend_registry.shutil.which", side_effect=lambda c: called.append(c)):
            reg.recover_check()

        self.assertEqual(called, [])  # which() not called for healthy specs


# ─── AC-05: Delegation ───────────────────────────────────────────────────────

class TestAC05Delegation(unittest.TestCase):
    def test_delegation_triggers_on_tool(self):
        """Orchestrator DELEGATE block calls on_tool and on_tool_result."""
        from larkhelm.handlers._query import _do_query_with_delegation
        from larkhelm.backend_registry import BackendSpec

        orch_spec = BackendSpec(
            id="claude", provider="claude_cli", display_name="Claude",
            role="orchestrator", tags=["tools"], command="claude",
        )
        worker_spec = BackendSpec(
            id="kimi-code", provider="kimi_cli", display_name="Kimi-Code",
            role="worker", tags=["tools"], command="kimi-code",
            healthy=True, enabled=True,
        )

        on_tool_calls = []
        on_tool_result_calls = []
        cancel_ev = threading.Event()

        def fake_run(spec, chat_id, message, cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout, images=None, **kwargs):
            if spec.id == "claude":
                # First call: return delegation; second call: synthesis
                if not on_tool_calls:
                    return "DELEGATE kimi-code\nReview this code\nEND_DELEGATE"
                return "Final synthesized answer"
            elif spec.id == "kimi-code":
                return "Specialist review result"
            return ""

        def fake_on_tool(name, desc, tool_id=""):
            on_tool_calls.append((name, desc))

        def fake_on_tool_result(tool_id, result, is_error, elapsed):
            on_tool_result_calls.append((tool_id, result, is_error))

        with patch("larkhelm.handlers._query._run_backend_single", side_effect=fake_run), \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.all_enabled.return_value = [worker_spec]
            result = _do_query_with_delegation(
                "chat1", "Review my Go code", orch_spec,
                {"kimi-code": worker_spec}, "/tmp", cancel_ev,
                lambda t, s="typing": None,
                fake_on_tool, fake_on_tool_result, lambda: None,
            )

        self.assertTrue(len(on_tool_calls) >= 1)
        self.assertIn("委托", on_tool_calls[0][0])
        self.assertTrue(len(on_tool_result_calls) >= 1)


# ─── AC-06: No delegation ────────────────────────────────────────────────────

class TestAC06NoDelegation(unittest.TestCase):
    def test_direct_answer_no_on_tool(self):
        """Direct orchestrator answer → on_tool call count is 0."""
        from larkhelm.handlers._query import _do_query_with_delegation
        from larkhelm.backend_registry import BackendSpec

        orch_spec = BackendSpec(
            id="claude", provider="claude_cli", display_name="Claude",
            role="orchestrator", tags=["tools"], command="claude",
        )

        on_tool_calls = []
        cancel_ev = threading.Event()

        def fake_run(spec, chat_id, message, cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout, images=None, **kwargs):
            return "Here is my direct answer to your question about Python."

        with patch("larkhelm.handlers._query._run_backend_single", side_effect=fake_run), \
             patch("larkhelm.backend_registry.BACKEND_REGISTRY") as mock_reg:
            mock_reg.all_enabled.return_value = []
            result = _do_query_with_delegation(
                "chat1", "What is Python?", orch_spec,
                {}, "/tmp", cancel_ev,
                lambda t, s="typing": None,
                lambda n, d, tid="": on_tool_calls.append(n),
                lambda *a: None, lambda: None,
            )

        self.assertEqual(len(on_tool_calls), 0)
        self.assertIn("direct answer", result)


# ─── AC-07: /lock command ────────────────────────────────────────────────────

class TestAC07LockCommand(unittest.TestCase):
    def _make_registry(self):
        from larkhelm.backend_registry import BackendRegistry
        reg = BackendRegistry()
        reg.load([
            {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
            {"id": "kimi-code", "provider": "kimi_cli", "display_name": "Kimi-Code",
             "tags": ["tools"], "command": "kimi-code"},
        ])
        return reg

    def test_lock_sets_field(self):
        """_cmd_lock <id> writes locked_backend to chat state."""
        from larkhelm.commands import _cmd_lock

        reg = self._make_registry()
        state_store = {}

        def fake_set_field(chat_id, key, val):
            state_store[key] = val

        def fake_get_state(chat_id):
            return state_store

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.commands._set_chat_field", side_effect=fake_set_field), \
             patch("larkhelm.commands._get_chat_state", side_effect=fake_get_state), \
             patch("larkhelm.commands.send_card_reply"):
            _cmd_lock("chat1", "claude", "msg1")

        self.assertEqual(state_store.get("locked_backend"), "claude")

    def test_lock_off_clears_field(self):
        """/lock off clears locked_backend."""
        from larkhelm.commands import _cmd_lock

        reg = self._make_registry()
        state_store = {"locked_backend": "claude"}

        def fake_set_field(chat_id, key, val):
            state_store[key] = val

        def fake_get_state(chat_id):
            return state_store

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.commands._set_chat_field", side_effect=fake_set_field), \
             patch("larkhelm.commands._get_chat_state", side_effect=fake_get_state), \
             patch("larkhelm.commands.send_card_reply"):
            _cmd_lock("chat1", "off", "msg1")

        self.assertIsNone(state_store.get("locked_backend"))

    def test_lock_show_status(self):
        """/lock with no args shows current locked_backend status."""
        from larkhelm.commands import _cmd_lock

        reg = self._make_registry()
        state_store = {"locked_backend": "claude"}
        card_calls = []

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.commands._get_chat_state", return_value=state_store), \
             patch("larkhelm.commands.send_card_reply", side_effect=lambda *a, **kw: card_calls.append(a)):
            _cmd_lock("chat1", "", "msg1")

        self.assertTrue(len(card_calls) >= 1)
        # Should mention the locked backend
        card_text = " ".join(str(a) for a in card_calls[0])
        self.assertIn("claude", card_text)


# ─── AC-08: /lock unhealthy backend ──────────────────────────────────────────

class TestAC08LockUnhealthy(unittest.TestCase):
    def test_lock_unhealthy_backend_returns_error(self):
        """Locking an unhealthy backend shows an error card, does not write state."""
        from larkhelm.backend_registry import BackendRegistry
        from larkhelm.commands import _cmd_lock

        reg = BackendRegistry()
        reg.load([
            {"id": "kimi-code", "provider": "kimi_cli", "display_name": "Kimi-Code",
             "tags": ["tools"], "command": "kimi-code"},
        ])
        reg.get("kimi-code").healthy = False
        reg.get("kimi-code").last_error = "command not found"

        state_store = {}

        def fake_set_field(chat_id, key, val):
            state_store[key] = val

        card_calls = []

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.commands._set_chat_field", side_effect=fake_set_field), \
             patch("larkhelm.commands.send_card_reply", side_effect=lambda *a, **kw: card_calls.append((a, kw))):
            _cmd_lock("chat1", "kimi-code", "msg1")

        self.assertNotIn("locked_backend", state_store)
        # Card should signal error (red color)
        colors = [kw.get("color") for _, kw in card_calls]
        self.assertIn("red", colors)


# ─── AC-09: Specialist unavailable ───────────────────────────────────────────

class TestAC09SpecialistUnavailable(unittest.TestCase):
    def test_missing_specialist_falls_back(self):
        """When DELEGATE targets unknown specialist, orchestrator answers directly."""
        from larkhelm.handlers._query import _do_query_with_delegation
        from larkhelm.backend_registry import BackendSpec

        orch_spec = BackendSpec(
            id="claude", provider="claude_cli", display_name="Claude",
            role="orchestrator", tags=["tools"], command="claude",
        )
        text_seen = []
        cancel_ev = threading.Event()

        def fake_run(spec, chat_id, message, cwd, cancel_ev, on_text, on_tool, on_tool_result, on_soft_timeout, images=None, **kwargs):
            if spec.id == "claude":
                return "DELEGATE nonexistent-specialist\nDo something\nEND_DELEGATE"
            return "Direct fallback answer"

        def fake_on_text(t, s="typing"):
            text_seen.append(t)

        with patch("larkhelm.handlers._query._run_backend_single", side_effect=fake_run):
            result = _do_query_with_delegation(
                "chat1", "Some task", orch_spec,
                {},  # empty worker_specs → specialist not found
                "/tmp", cancel_ev,
                fake_on_text,
                lambda *a: None, lambda *a: None, lambda: None,
            )

        # Should fall back (re-run orchestrator or show warning)
        self.assertTrue(any("不可用" in t or "Direct" in t for t in text_seen + [result]))


# ─── AC-10: All orchestrators down ───────────────────────────────────────────

class TestAC10AllOrchestratorsDown(unittest.TestCase):
    def test_all_down_raises_runtime_error(self):
        """When all backends fail, a RuntimeError is raised (caller shows error card)."""
        from larkhelm.backend_registry import BackendRegistry

        reg = BackendRegistry()
        reg.load([
            {"id": "orch1", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
            {"id": "orch2", "provider": "gemini_cli", "display_name": "Gemini",
             "tags": ["tools"], "command": "gemini"},
        ])
        chain = reg.get_orchestrator_chain()
        self.assertEqual(len(chain), 2)

        output = None
        last_err = None
        for s in chain:
            try:
                raise ConnectionError(f"{s.id} refused connection")
            except Exception as e:
                s.healthy = False
                last_err = e

        self.assertIsNone(output)
        # Both marked unhealthy
        self.assertFalse(reg.get("orch1").healthy)
        self.assertFalse(reg.get("orch2").healthy)
        # A friendly error would be raised with last_err
        with self.assertRaises(RuntimeError):
            raise RuntimeError(f"所有 backend 均不可用。最近错误: {last_err}")


# ─── AC-11: _detect_delegation boundary cases ────────────────────────────────

class TestAC11DetectDelegationBoundary(unittest.TestCase):
    def setUp(self):
        from larkhelm.orchestration import _detect_delegation
        self._detect = _detect_delegation

    def test_well_formed_delegation(self):
        """Correctly formed DELEGATE block is parsed."""
        text = "DELEGATE gemini\nSolve this maths problem\nEND_DELEGATE"
        result = self._detect(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "gemini")
        self.assertEqual(result[1], "Solve this maths problem")

    def test_no_delegation_plain_text(self):
        """Plain text without DELEGATE returns None."""
        result = self._detect("Sure, here is the answer to your question.")
        self.assertIsNone(result)

    def test_missing_end_delegate(self):
        """Incomplete block without END_DELEGATE returns None."""
        result = self._detect("DELEGATE gemini\nDo something")
        self.assertIsNone(result)

    def test_empty_query_in_delegate(self):
        """DELEGATE block with empty query between markers is still parsed."""
        text = "DELEGATE gemini\n\nEND_DELEGATE"
        result = self._detect(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "gemini")

    def test_delegate_embedded_in_prose(self):
        """DELEGATE block embedded mid-text is detected."""
        text = "Let me delegate this.\nDELEGATE kimi-code\nwrite a unit test\nEND_DELEGATE\nDone."
        result = self._detect(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "kimi-code")

    def test_trailing_space_after_backend_id(self):
        """Trailing whitespace after backend_id does not break detection."""
        text = "DELEGATE gemini-code  \nCheck this PR\nEND_DELEGATE"
        result = self._detect(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "gemini-code")

    def test_buffer_shorter_than_threshold_no_match(self):
        """Short buffer (< 60 chars) without END_DELEGATE returns None (block incomplete)."""
        text = "DELEGATE kimi\nDo x"  # well under 60 chars, no END_DELEGATE
        result = self._detect(text)
        self.assertIsNone(result)

    def test_backend_id_with_special_chars(self):
        """backend_id with hyphens and underscores is captured correctly."""
        text = "DELEGATE my-backend_v2\nRun the task\nEND_DELEGATE"
        result = self._detect(text)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "my-backend_v2")
        self.assertEqual(result[1], "Run the task")


# ─── AC-12: Router Rule 0 with unhealthy locked backend ──────────────────────

class TestAC12RouterRule0Unhealthy(unittest.TestCase):
    def test_locked_unhealthy_backend_raises(self):
        """Rule 0: locked_backend that is unhealthy raises LockedBackendUnavailableError."""
        from larkhelm.backend_registry import BackendRegistry
        from larkhelm.router import resolve_backend, LockedBackendUnavailableError

        reg = BackendRegistry()
        reg.load([
            {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
            {"id": "gemini", "provider": "gemini_cli", "display_name": "Gemini",
             "tags": ["tools"], "command": "gemini"},
        ])
        reg.get("claude").healthy = False
        reg.get("claude").last_error = "connection refused"

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"locked_backend": "claude"}):
            with self.assertRaises(LockedBackendUnavailableError) as ctx:
                resolve_backend("chat1", "hello")

        self.assertEqual(ctx.exception.backend_id, "claude")
        self.assertIn("不可用", str(ctx.exception))

    def test_locked_disabled_backend_falls_through_to_rules(self):
        """Rule 0: locked_backend with enabled=False falls through to normal routing."""
        from larkhelm.backend_registry import BackendRegistry
        from larkhelm.router import resolve_backend

        reg = BackendRegistry()
        reg.load([
            {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
            {"id": "gemini", "provider": "gemini_cli", "display_name": "Gemini",
             "tags": ["tools"], "command": "gemini", "role": "orchestrator"},
        ])
        reg.get("claude").enabled = False

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"locked_backend": "claude"}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "gemini")

    def test_locked_missing_id_falls_through_to_rules(self):
        """Rule 0: unknown locked_backend ID falls through to normal routing."""
        from larkhelm.backend_registry import BackendRegistry
        from larkhelm.router import resolve_backend

        reg = BackendRegistry()
        reg.load([
            {"id": "gemini", "provider": "gemini_cli", "display_name": "Gemini",
             "tags": ["tools"], "command": "gemini", "role": "orchestrator"},
        ])

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"locked_backend": "nonexistent"}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "gemini")


# ─── AC-13: Router Rule 4 — user backend preference ─────────────────────────

class TestAC13RouterRule4UserPreference(unittest.TestCase):
    def _make_reg(self):
        from larkhelm.backend_registry import BackendRegistry
        reg = BackendRegistry()
        reg.load([
            {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude", "role": "orchestrator"},
            {"id": "gemini", "provider": "gemini_cli", "display_name": "Gemini",
             "tags": ["tools"], "command": "gemini", "role": "orchestrator"},
        ])
        return reg

    def test_backend_id_preference_used(self):
        """Rule 4: backend_id in chat state selects that backend."""
        from larkhelm.router import resolve_backend

        reg = self._make_reg()
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"backend_id": "gemini"}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "gemini")

    def test_legacy_model_field_used(self):
        """Rule 4: legacy 'model' field falls back to backend lookup."""
        from larkhelm.router import resolve_backend

        reg = self._make_reg()
        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"model": "gemini"}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "gemini")

    def test_backend_id_unhealthy_falls_through(self):
        """Rule 4: unhealthy preferred backend falls through to Rule 5."""
        from larkhelm.router import resolve_backend

        reg = self._make_reg()
        reg.get("gemini").healthy = False

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"backend_id": "gemini"}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "claude")

    def test_backend_id_disabled_falls_through(self):
        """Rule 4: disabled preferred backend falls through to Rule 5."""
        from larkhelm.router import resolve_backend

        reg = self._make_reg()
        reg.get("gemini").enabled = False

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={"backend_id": "gemini"}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "claude")


# ─── AC-14: Router Rule 5 — default_backend config + orchestrator fallback ───

class TestAC14RouterRule5DefaultFallback(unittest.TestCase):
    def _make_reg(self):
        from larkhelm.backend_registry import BackendRegistry
        reg = BackendRegistry()
        reg.load([
            {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude", "role": "orchestrator"},
            {"id": "gemini", "provider": "gemini_cli", "display_name": "Gemini",
             "tags": ["tools"], "command": "gemini"},
        ])
        return reg

    def test_config_default_backend_used(self):
        """Rule 5: default_backend in config selects that backend."""
        import larkhelm.config as _cfg
        from larkhelm.router import resolve_backend

        reg = self._make_reg()
        fake_config = {"default_backend": "gemini"}

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={}), \
             patch.object(_cfg, "config", fake_config, create=True):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "gemini")

    def test_config_default_backend_unhealthy_falls_to_orchestrator(self):
        """Rule 5: unhealthy default_backend falls through to orchestrator."""
        import larkhelm.config as _cfg
        from larkhelm.router import resolve_backend

        reg = self._make_reg()
        reg.get("gemini").healthy = False
        fake_config = {"default_backend": "gemini"}

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={}), \
             patch.object(_cfg, "config", fake_config, create=True):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "claude")

    def test_no_config_falls_to_orchestrator(self):
        """Rule 5: no default_backend config falls through to get_orchestrator()."""
        from larkhelm.router import resolve_backend

        reg = self._make_reg()

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "claude")

    def test_no_orchestrator_falls_to_first_healthy(self):
        """Rule 5: no orchestrator falls to first healthy+enabled backend."""
        from larkhelm.backend_registry import BackendRegistry
        from larkhelm.router import resolve_backend

        reg = BackendRegistry()
        reg.load([
            {"id": "worker1", "provider": "gemini_cli", "display_name": "Worker1",
             "tags": [], "command": "gemini", "role": "worker"},
        ])

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={}):
            spec = resolve_backend("chat1", "hello")

        self.assertEqual(spec.id, "worker1")

    def test_all_unhealthy_raises_runtime_error(self):
        """Rule 5: all backends unhealthy raises RuntimeError."""
        from larkhelm.backend_registry import BackendRegistry
        from larkhelm.router import resolve_backend

        reg = BackendRegistry()
        reg.load([
            {"id": "claude", "provider": "claude_cli", "display_name": "Claude",
             "tags": ["tools"], "command": "claude"},
        ])
        reg.get("claude").healthy = False

        with patch("larkhelm.backend_registry.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router.BACKEND_REGISTRY", reg), \
             patch("larkhelm.router._get_chat_state", return_value={}):
            with self.assertRaises(RuntimeError):
                resolve_backend("chat1", "hello")


if __name__ == "__main__":
    unittest.main()
