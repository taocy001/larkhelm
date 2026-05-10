"""AC-11: orchestration._detect_agent_protocol parses AGENT/BACKEND/MODE/TASK blocks."""
import unittest

from larkhelm.orchestration import _detect_agent_protocol


class TestDetectAgentProtocol(unittest.TestCase):

    def test_well_formed_block(self):
        buf = (
            "AGENT dev\n"
            "BACKEND claude\n"
            "MODE pipeline\n"
            "TASK 实现登录模块，包括前后端\n"
            "END_AGENT"
        )
        result = _detect_agent_protocol(buf)
        self.assertIsNotNone(result)
        self.assertEqual(result.agent_type, "dev")
        self.assertEqual(result.backend_id, "claude")
        self.assertEqual(result.mode, "pipeline")
        self.assertIn("实现登录模块", result.task)

    def test_multiline_task(self):
        buf = (
            "AGENT crew\n"
            "BACKEND gemini-pro\n"
            "MODE research\n"
            "TASK line1\nline2\nline3\n"
            "END_AGENT"
        )
        result = _detect_agent_protocol(buf)
        self.assertIsNotNone(result)
        self.assertIn("line1", result.task)
        self.assertIn("line3", result.task)

    def test_returns_none_when_no_match(self):
        self.assertIsNone(_detect_agent_protocol(""))
        self.assertIsNone(_detect_agent_protocol("hello world"))

    def test_returns_none_when_block_incomplete(self):
        buf = "AGENT dev\nBACKEND claude\nMODE pipeline\nTASK hi"
        self.assertIsNone(_detect_agent_protocol(buf))

    def test_rejects_invalid_agent_type(self):
        buf = (
            "AGENT bad/type!\n"
            "BACKEND claude\n"
            "MODE pipeline\n"
            "TASK do thing\n"
            "END_AGENT"
        )
        self.assertIsNone(_detect_agent_protocol(buf))

    def test_rejects_invalid_backend_id(self):
        buf = (
            "AGENT dev\n"
            "BACKEND ../etc/passwd\n"
            "MODE pipeline\n"
            "TASK do thing\n"
            "END_AGENT"
        )
        self.assertIsNone(_detect_agent_protocol(buf))

    def test_rejects_empty_task(self):
        buf = (
            "AGENT dev\n"
            "BACKEND claude\n"
            "MODE pipeline\n"
            "TASK \n"
            "END_AGENT"
        )
        self.assertIsNone(_detect_agent_protocol(buf))

    def test_does_not_collide_with_delegate_block(self):
        # DELEGATE blocks must not be parsed as AGENT blocks.
        from larkhelm.orchestration import _detect_delegation
        buf = "DELEGATE claude\nhello\nEND_DELEGATE"
        self.assertIsNone(_detect_agent_protocol(buf))
        self.assertIsNotNone(_detect_delegation(buf))


if __name__ == "__main__":
    unittest.main()
