"""Tests for _lark_api_call_with_retry (AC-03, AC-04)."""
import unittest
from unittest.mock import MagicMock, patch, call


def _make_resp(code, success=None):
    resp = MagicMock()
    resp.code = code
    resp.success.return_value = (success if success is not None else code == 200)
    resp.data = MagicMock()
    resp.data.message_id = "msg_test_123"
    resp.msg = "ok" if code == 200 else "rate limited"
    return resp


class TestLarkApiRetry(unittest.TestCase):
    def test_retry_429_then_success(self):
        from larkhelm.lark_client import _lark_api_call_with_retry
        resp_429 = _make_resp(429, success=False)
        resp_200 = _make_resp(200, success=True)
        fn = MagicMock(side_effect=[resp_429, resp_200])

        with patch("time.sleep"), \
             patch("larkhelm.metrics.inc_lark_api_retry") as mock_metric:
            result = _lark_api_call_with_retry(fn, "arg1", method_name="send_card")

        self.assertEqual(fn.call_count, 2)
        self.assertEqual(result.code, 200)
        mock_metric.assert_called_once_with("send_card", "success_after_retry")

    def test_retry_exhausted(self):
        from larkhelm.lark_client import _lark_api_call_with_retry
        resp_429 = _make_resp(429, success=False)
        fn = MagicMock(return_value=resp_429)

        with patch("time.sleep"), \
             patch("larkhelm.metrics.inc_lark_api_retry") as mock_metric:
            result = _lark_api_call_with_retry(fn, "arg1", max_retries=3,
                                               method_name="send_card")

        # 1 initial + 3 retries = 4 total calls
        self.assertEqual(fn.call_count, 4)
        self.assertEqual(result.code, 429)
        mock_metric.assert_called_once_with("send_card", "exhausted")

    def test_no_retry_on_other_error(self):
        from larkhelm.lark_client import _lark_api_call_with_retry
        resp_400 = _make_resp(400, success=False)
        fn = MagicMock(return_value=resp_400)

        with patch("time.sleep") as mock_sleep, \
             patch("larkhelm.metrics.inc_lark_api_retry") as mock_metric:
            result = _lark_api_call_with_retry(fn, "arg1", method_name="send_card")

        self.assertEqual(fn.call_count, 1)
        mock_sleep.assert_not_called()
        mock_metric.assert_not_called()
        self.assertEqual(result.code, 400)

    def test_no_metric_on_first_success(self):
        from larkhelm.lark_client import _lark_api_call_with_retry
        resp_200 = _make_resp(200, success=True)
        fn = MagicMock(return_value=resp_200)

        with patch("larkhelm.metrics.inc_lark_api_retry") as mock_metric:
            result = _lark_api_call_with_retry(fn, "arg1", method_name="send_card")

        self.assertEqual(fn.call_count, 1)
        mock_metric.assert_not_called()


class TestSendCardRetry(unittest.TestCase):
    def test_send_card_retry(self):
        """_send_card_raw should not raise when 429->200 occurs."""
        import larkhelm.lark_client as lc
        from larkhelm.lark_client import _send_card_raw

        resp_429 = _make_resp(429, success=False)
        resp_200 = _make_resp(200, success=True)

        mock_client = MagicMock()
        mock_client.im.v1.message.create.side_effect = [resp_429, resp_200]
        orig_client = lc.client
        lc.client = mock_client

        try:
            with patch("time.sleep"), \
                 patch("larkhelm.metrics.inc_lark_api_retry"):
                result = _send_card_raw("chat_abc", '{"type":"card"}')
            self.assertEqual(result, "msg_test_123")
        finally:
            lc.client = orig_client


if __name__ == "__main__":
    unittest.main()
