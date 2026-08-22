"""
monitoring/alerts.py: Slack first, Telegram fallback, never crash, and the
rotating file handler that keeps logs alive after the console closes. All
network transports are injected fakes — nothing here talks to the internet.
"""
from __future__ import annotations

import logging
import logging.handlers
from unittest.mock import Mock

import pytest
import requests

from config.settings import settings
from monitoring import alerts


class _FakeResponse:
    def __init__(self, ok: bool = True):
        self._ok = ok

    def raise_for_status(self):
        if not self._ok:
            raise requests.HTTPError("boom")


@pytest.fixture
def slack_configured(monkeypatch):
    monkeypatch.setattr(settings, "slack_webhook_url", "https://hooks.slack.invalid/T000/B000/x")


@pytest.fixture
def slack_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "slack_webhook_url", "")


@pytest.fixture
def telegram_configured(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "123:fake-token-for-tests")
    monkeypatch.setattr(settings, "telegram_chat_id", "42")


@pytest.fixture
def telegram_unconfigured(monkeypatch):
    monkeypatch.setattr(settings, "telegram_bot_token", "")
    monkeypatch.setattr(settings, "telegram_chat_id", "")


def test_slack_success_never_touches_telegram(slack_configured, telegram_configured):
    post_fn = Mock(return_value=_FakeResponse(ok=True))
    telegram_send = Mock()
    assert alerts.send_slack_alert("all good", post_fn=post_fn, telegram_send_fn=telegram_send) is True
    post_fn.assert_called_once()
    telegram_send.assert_not_called()


def test_no_slack_webhook_falls_back_to_telegram(slack_unconfigured, telegram_configured):
    telegram_send = Mock(return_value={"message_id": 1})
    post_fn = Mock()
    assert alerts.send_slack_alert("breaker tripped", severity="critical", post_fn=post_fn, telegram_send_fn=telegram_send) is True
    post_fn.assert_not_called()
    telegram_send.assert_called_once()
    sent_text = telegram_send.call_args.args[0]
    assert "breaker tripped" in sent_text


def test_slack_http_failure_falls_back_to_telegram(slack_configured, telegram_configured):
    post_fn = Mock(return_value=_FakeResponse(ok=False))
    telegram_send = Mock(return_value={"message_id": 1})
    assert alerts.send_slack_alert("pipeline died", post_fn=post_fn, telegram_send_fn=telegram_send) is True
    telegram_send.assert_called_once()


def test_neither_channel_configured_logs_and_returns_false(slack_unconfigured, telegram_unconfigured, caplog):
    telegram_send = Mock()
    with caplog.at_level(logging.WARNING):
        assert alerts.send_slack_alert("nobody will hear this", post_fn=Mock(), telegram_send_fn=telegram_send) is False
    telegram_send.assert_not_called()
    assert any("not delivered" in r.message for r in caplog.records)


def test_telegram_failure_is_swallowed_never_raised(slack_unconfigured, telegram_configured):
    def exploding_send(*args, **kwargs):
        raise RuntimeError("telegram is down")

    # The core contract: an alert failure must never crash a trading cycle.
    assert alerts.send_slack_alert("still must not raise", post_fn=Mock(), telegram_send_fn=exploding_send) is False


def test_slack_network_error_then_telegram_error_still_no_raise(slack_configured, telegram_configured):
    def exploding_post(*args, **kwargs):
        raise requests.ConnectionError("no dns")

    def exploding_send(*args, **kwargs):
        raise RuntimeError("also down")

    assert alerts.send_slack_alert("worst case", post_fn=exploding_post, telegram_send_fn=exploding_send) is False


# --------------------------------------------------------------------------
# configure_file_logging
# --------------------------------------------------------------------------


@pytest.fixture
def clean_root_handler(tmp_path):
    """Give each test its own log file and detach whatever handler it added."""
    log_file = tmp_path / "trading-system.log"
    added = []
    yield log_file, added
    root = logging.getLogger()
    for handler in added:
        root.removeHandler(handler)
        handler.close()


def test_file_handler_writes_and_survives(clean_root_handler):
    log_file, added = clean_root_handler
    handler = alerts.configure_file_logging(log_file)
    added.append(handler)

    logging.getLogger("some.module").warning("a line that should persist")
    handler.flush()

    assert log_file.exists()
    assert "a line that should persist" in log_file.read_text(encoding="utf-8")


def test_file_handler_is_rotating(clean_root_handler):
    log_file, added = clean_root_handler
    handler = alerts.configure_file_logging(log_file)
    added.append(handler)
    assert isinstance(handler, logging.handlers.RotatingFileHandler)
    assert handler.backupCount > 0


def test_configure_twice_does_not_double_log(clean_root_handler):
    log_file, added = clean_root_handler
    first = alerts.configure_file_logging(log_file)
    added.append(first)
    second = alerts.configure_file_logging(log_file)
    assert second is first

    logging.getLogger("some.module").warning("logged exactly once")
    first.flush()
    assert log_file.read_text(encoding="utf-8").count("logged exactly once") == 1
