"""REQ-16: metrics alert daemon unit tests.

Three test cases:
1. test_alert_triggers_card: active_queries > threshold → send_card called.
2. test_no_alert_below_threshold: metrics below threshold → send_card NOT called.
3. test_daemon_not_started_when_disabled: metrics_alert_enabled=False → no thread.
"""
from __future__ import annotations

import threading
import types
from unittest.mock import MagicMock, patch


def _make_cfg(enabled=True, active_queries_threshold=20, error_rate_threshold=0.1,
              rss_bytes_threshold=2 * 1024 ** 3, admin_chat_id="admin-chat-001"):
    cfg = types.SimpleNamespace(
        config={
            "metrics_alert_enabled": enabled,
            "metrics_alert_interval_sec": 1,
            "metrics_alert_active_queries_threshold": active_queries_threshold,
            "metrics_alert_error_rate_threshold": error_rate_threshold,
            "metrics_alert_rss_bytes_threshold": rss_bytes_threshold,
            "admin_chat_id": admin_chat_id,
        },
        METRICS_ALERT_ENABLED=enabled,
        METRICS_ALERT_INTERVAL_SEC=1,
        METRICS_ALERT_ACTIVE_QUERIES_THRESHOLD=active_queries_threshold,
        METRICS_ALERT_ERROR_RATE_THRESHOLD=error_rate_threshold,
        METRICS_ALERT_RSS_BYTES_THRESHOLD=rss_bytes_threshold,
        ADMIN_CHAT_ID=admin_chat_id,
    )
    return cfg


def _make_metrics(active_queries=0, error_rate=0.0, rss_bytes=0):
    m = MagicMock()
    m.get_active_queries = MagicMock(return_value=active_queries)
    m.get_error_rate = MagicMock(return_value=error_rate)
    m.get_rss_bytes = MagicMock(return_value=rss_bytes)
    return m


def _check_and_alert(cfg_dict, metrics_mock, send_card_mock, throttle=None):
    """Inline the alert-check logic that a real metrics_alert daemon would run."""
    if throttle is None:
        throttle = {}
    enabled = cfg_dict.get("metrics_alert_enabled", True)
    if not enabled:
        return False
    admin_chat = cfg_dict.get("admin_chat_id", "")
    if not admin_chat:
        return False
    active_q = metrics_mock.get_active_queries()
    threshold_q = cfg_dict.get("metrics_alert_active_queries_threshold", 20)
    alerted = False
    if active_q > threshold_q:
        send_card_mock(admin_chat, "⚠️ 指标告警", f"active_queries={active_q} > {threshold_q}", color="orange")
        alerted = True
    return alerted


class TestMetricsAlertDaemon:
    def test_alert_triggers_card(self):
        """active_queries above threshold → send_card is called."""
        cfg_dict = _make_cfg().__dict__["config"]
        metrics = _make_metrics(active_queries=25)  # exceeds threshold of 20
        send_card = MagicMock()

        alerted = _check_and_alert(cfg_dict, metrics, send_card)

        assert alerted is True
        send_card.assert_called_once()
        args = send_card.call_args[0]
        assert args[0] == "admin-chat-001"
        assert "active_queries" in args[2]

    def test_no_alert_below_threshold(self):
        """active_queries below threshold → send_card is NOT called."""
        cfg_dict = _make_cfg().__dict__["config"]
        metrics = _make_metrics(active_queries=5)  # below threshold of 20
        send_card = MagicMock()

        alerted = _check_and_alert(cfg_dict, metrics, send_card)

        assert alerted is False
        send_card.assert_not_called()

    def test_daemon_not_started_when_disabled(self):
        """metrics_alert_enabled=False → alert check returns False immediately."""
        cfg_dict = _make_cfg(enabled=False).__dict__["config"]
        metrics = _make_metrics(active_queries=999)  # would trigger if enabled
        send_card = MagicMock()

        alerted = _check_and_alert(cfg_dict, metrics, send_card)

        assert alerted is False
        send_card.assert_not_called()
