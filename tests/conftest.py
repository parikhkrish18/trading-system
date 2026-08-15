"""
Test-suite-wide safety net: no test may reach a real outbound channel.

The alert path falls back to the Telegram approval bot when Slack is
unconfigured (monitoring/alerts.py), which is exactly the state on a dev
machine. Tests that simulate a tripped circuit breaker therefore sent real
"🚨 Circuit breaker triggered" messages to a real phone on every run —
harmless individually, corrosive in aggregate: an alert channel that cries
wolf during CI is one nobody reads when it matters.

Individual tests still inject their own fakes to assert on behavior; this
fixture only guarantees that anything they *don't* stub can't escape.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_outbound_messages(monkeypatch):
    def _blocked(*args, **kwargs):
        raise AssertionError(
            "A test tried to send a real outbound message. Inject a fake "
            "(post_fn/telegram_send_fn) or monkeypatch the sender instead."
        )

    # Block HTTP itself rather than the helpers above it: tests for the
    # Telegram transport call send_message/fetch_updates directly (injecting
    # their own fake post_fn), so stubbing those would break the very units
    # under test. Nothing legitimate in the suite performs real HTTP.
    monkeypatch.setattr("requests.post", _blocked, raising=False)
    monkeypatch.setattr("requests.get", _blocked, raising=False)
