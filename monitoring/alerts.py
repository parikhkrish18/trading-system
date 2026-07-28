"""
Alerting for: circuit breaker triggers, data pipeline failures, and any
position exceeding risk limits (Phase 8, point 2). Delivered via Slack
webhook.
"""
from __future__ import annotations

import logging

import requests

from config.settings import settings

logger = logging.getLogger(__name__)


def send_slack_alert(message: str, severity: str = "warning") -> bool:
    """
    Posts `message` to the configured Slack webhook. Returns True on success.
    Falls back to just logging if no webhook is configured, so alerting code
    elsewhere doesn't need to special-case "not configured yet".
    """
    prefix = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(severity, "")
    text = f"{prefix} {message}".strip()

    if not settings.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — alert not sent, logging only: %s", text)
        return False

    try:
        resp = requests.post(settings.slack_webhook_url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        logger.exception("Failed to send Slack alert: %s", text)
        return False


def alert_circuit_breaker(reason: str) -> None:
    send_slack_alert(f"Circuit breaker triggered: {reason}", severity="critical")


def alert_pipeline_failure(job_name: str, error: str) -> None:
    send_slack_alert(f"Pipeline job '{job_name}' failed: {error}", severity="critical")


def alert_risk_limit_exceeded(symbol: str, detail: str) -> None:
    send_slack_alert(f"Risk limit exceeded for {symbol}: {detail}", severity="warning")
