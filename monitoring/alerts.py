"""
Alerting for: circuit breaker triggers, data pipeline failures, and any
position exceeding risk limits (Phase 8, point 2).

Delivery order: Slack webhook first, then the Telegram bot as a fallback
when Slack is unconfigured or down (execution/telegram.py is the transport
— same bot and chat the post-trade notifications already use, so if a
"cycle complete" message can reach the phone, alerts can too). If neither
channel is configured the alert is logged and dropped; an alert failure must never
take a trading cycle down with it, so nothing in this module raises.

Also home to configure_file_logging(): a rotating file handler on the
root logger (logs/trading-system.log) so a closed console doesn't mean
lost history. Entry points call it once at startup.
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from collections.abc import Callable
from pathlib import Path

import requests

from config.settings import settings
from execution import telegram

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_FILE = _REPO_ROOT / "logs" / "trading-system.log"


def configure_console_encoding() -> None:
    """
    Make stdout/stderr UTF-8 capable.

    On Windows the console defaults to cp1252, which cannot encode the emoji
    and box-drawing characters this project prints — in alerts, in MLflow's
    own progress output, in the cycle summary. The failure is a
    UnicodeEncodeError raised from a print, so a training run dies partway
    through for no reason connected to training, and the traceback points at
    logging rather than at the cause.

    PYTHONIOENCODING=utf-8 fixes it from outside, but every local developer
    would have to know that, and forgetting it looks like a broken pipeline.
    Doing it in-process means nobody has to know.

    Errors are replaced rather than raised: a character that still cannot be
    rendered should cost a mangled glyph in a log line, never a dead run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:  # already-wrapped or captured streams
            continue
        encoding = (getattr(stream, "encoding", "") or "").lower().replace("-", "")
        if encoding in ("utf8", "utf8mb4"):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass  # a stream that refuses is not worth failing a run over


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
    # Every CLI entrypoint already calls this before doing anything, which
    # makes it the one place that fixes the console for all of them.
    configure_console_encoding()

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


def alert_pipeline_progress(job_name: str, detail: str) -> None:
    """
    A job finished successfully — the "still in the loop" counterpart to
    alert_pipeline_failure. info severity, so it never carries the same
    urgency as a real failure; same fire-and-forget contract (never raises,
    logged and dropped if neither channel is configured).
    """
    send_slack_alert(f"'{job_name}' done: {detail}", severity="info")


def alert_risk_limit_exceeded(symbol: str, detail: str) -> None:
    send_slack_alert(f"Risk limit exceeded for {symbol}: {detail}", severity="warning")


def make_progress_alert(job_name: str, total: int, step_pct: int = 10) -> Callable[[int], None]:
    """
    A progress ticker for a long-running pipeline phase (fundamentals/news
    ingest, sentiment backfill, feature building) that otherwise goes
    silent for minutes with nothing to show for it. Returns a callback —
    call it once per unit of work completed (once per symbol, once per
    scored batch, once per build stage — whatever `total` counts), passing
    how many units that call represents (default 1).

    Fires an alert_pipeline_progress "X% done" update the first time
    cumulative progress crosses each new step_pct threshold, and never
    twice for the same one — a batch that jumps several thresholds at once
    (e.g. a single call advancing progress from 25% to 60%) still only
    reports each threshold crossed exactly once, in order, rather than
    skipping straight to the final one.

    total <= 0 returns a no-op callback: nothing to divide by, and there is
    no meaningful percentage of an empty job.
    """
    if total <= 0:
        return lambda done=1: None

    state = {"done": 0, "next_threshold": step_pct}

    def _tick(done: int = 1) -> None:
        state["done"] += done
        pct = (state["done"] / total) * 100
        while state["next_threshold"] <= 100 and pct >= state["next_threshold"]:
            alert_pipeline_progress(job_name, f"{state['next_threshold']}% done ({state['done']}/{total})")
            state["next_threshold"] += step_pct

    return _tick
