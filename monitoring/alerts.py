"""
Alerting for: circuit breaker triggers, data pipeline failures, and any
position exceeding risk limits (Phase 8, point 2).

Delivery order: Slack webhook first, then the Telegram approval bot as a
fallback when Slack is unconfigured or down (execution/telegram.py is the
transport — same bot and chat the approval gate already uses, so if trade
proposals can reach the phone, alerts can too). If neither channel is
configured the alert is logged and dropped; an alert failure must never
take a trading cycle down with it, so nothing in this module raises.

Also home to configure_file_logging(): a rotating file handler on the
root logger (logs/trading-system.log) so a closed console doesn't mean
lost history. Entry points call it once at startup.
"""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import requests

from config.settings import settings
from execution import telegram

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = _REPO_ROOT / "logs" / "trading-system.log"


def configure_file_logging(
    log_file: Path | str = DEFAULT_LOG_FILE,
    max_bytes: int = 5_000_000,
    backup_count: int = 5,
) -> logging.Handler:
    """
    Attach a RotatingFileHandler for `log_file` to the root logger.
    Idempotent: calling twice for the same file reuses the existing handler
    rather than double-logging every line.
    """
    log_file = Path(log_file)
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.handlers.RotatingFileHandler) and Path(handler.baseFilename) == log_file:
            return handler

    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(handler)
    return handler


def _post_to_slack(text: str, post_fn) -> bool:
    if not settings.slack_webhook_url:
        return False
    try:
        resp = post_fn(settings.slack_webhook_url, json={"text": text}, timeout=10)
        resp.raise_for_status()
        return True
    except requests.RequestException:
        logger.exception("Failed to send Slack alert: %s", text)
        return False


def _post_to_telegram(text: str, send_fn) -> bool:
    token, chat_id = telegram.credentials()
    if not token or not chat_id:
        return False
    try:
        send_fn(text, token=token, chat_id=chat_id)
        return True
    except Exception:
        # Broad on purpose: an alert failure must never crash a cycle.
        logger.exception("Failed to send Telegram alert: %s", text)
        return False


def send_slack_alert(
    message: str,
    severity: str = "warning",
    post_fn=requests.post,
    telegram_send_fn=telegram.send_message,
) -> bool:
    """
    Deliver `message` to Slack, falling back to Telegram; returns True if
    either channel accepted it. With neither configured (or both down) the
    alert is logged and False returned — callers never need to special-case
    "not configured yet", and nothing here ever raises.
    """
    prefix = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(severity, "")
    text = f"{prefix} {message}".strip()

    if _post_to_slack(text, post_fn):
        return True
    if _post_to_telegram(text, telegram_send_fn):
        return True

    logger.warning("Alert not delivered on any channel (Slack/Telegram unconfigured or down): %s", text)
    return False


def alert_circuit_breaker(reason: str) -> None:
    send_slack_alert(f"Circuit breaker triggered: {reason}", severity="critical")


def alert_pipeline_failure(job_name: str, error: str) -> None:
    send_slack_alert(f"Pipeline job '{job_name}' failed: {error}", severity="critical")


def alert_risk_limit_exceeded(symbol: str, detail: str) -> None:
    send_slack_alert(f"Risk limit exceeded for {symbol}: {detail}", severity="warning")
