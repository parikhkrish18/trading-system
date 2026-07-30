"""
Between weekly screen-and-trade cycles, a held position can go stale: fresh
news can break against the direction we entered, or short-term price action
can reverse hard enough to contradict the original thesis, days before the
next scheduled screen would otherwise notice. This module checks every
currently held position against both signals and closes out any position
that's now fighting the evidence it was opened on.

Deliberately close-only, not close-and-reverse: reversing requires a fresh
conviction call (which candidate, how large), which is exactly what the
weekly screen already does properly with a trained ensemble. This module's
job is narrower and more mechanical — stop the bleeding, don't re-guess the
next trade.

Safety boundary, same as trading_loop.py: only ever calls get_broker()
without confirm_live=True.

Usage:
    python -m execution.contradiction_monitor
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging

import pandas as pd
from sqlalchemy.dialects.postgresql import JSONB

from data.ingest.db import get_engine
from data.ingest.news import ingest_news
from execution.broker import get_broker
from features.qualitative.sentiment import backfill_unscored_news
from features.quant.momentum import rolling_return
from monitoring.alerts import send_slack_alert

logger = logging.getLogger(__name__)

_FEATURE_SET_ID = "contradiction_monitor"
_MODEL_VERSION = "rule_based_v1"

# News lookback for the sentiment read — wider than the check interval so a
# position isn't judged on a single stale article between checks.
_SENTIMENT_LOOKBACK_HOURS = 24
_MIN_NEWS_COUNT = 2
_SENTIMENT_CONTRADICTION_THRESHOLD = 0.4  # mean sentiment must be this strongly opposite to trigger

_MOMENTUM_WINDOW_DAYS = 5
_MOMENTUM_CONTRADICTION_THRESHOLD = 0.04  # 4% move against the position over the window


@dataclasses.dataclass
class ContradictionResult:
    symbol: str
    side: str
    closed: bool
    reasons: list[dict]


def _recent_sentiment(engine, symbol: str) -> tuple[float | None, int]:
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=_SENTIMENT_LOOKBACK_HOURS)
    df = pd.read_sql(
        "SELECT sentiment FROM news_events WHERE symbol = %(symbol)s AND ts >= %(since)s AND sentiment IS NOT NULL",
        engine,
        params={"symbol": symbol, "since": since},
    )
    if df.empty:
        return None, 0
    return float(df["sentiment"].mean()), len(df)


def _recent_momentum(engine, symbol: str) -> float | None:
    df = pd.read_sql(
        "SELECT ts, close FROM prices WHERE symbol = %(symbol)s ORDER BY ts DESC LIMIT %(limit)s",
        engine,
        params={"symbol": symbol, "limit": _MOMENTUM_WINDOW_DAYS + 1},
    )
    if len(df) < _MOMENTUM_WINDOW_DAYS + 1:
        return None
    df = df.sort_values("ts")
    ret = rolling_return(df["close"], _MOMENTUM_WINDOW_DAYS).iloc[-1]
    return None if pd.isna(ret) else float(ret)


def _check_position(engine, symbol: str, qty: float) -> ContradictionResult:
    side = "long" if qty > 0 else "short"
    sign = 1.0 if qty > 0 else -1.0
    reasons: list[dict] = []

    sentiment, news_count = _recent_sentiment(engine, symbol)
    if sentiment is not None and news_count >= _MIN_NEWS_COUNT:
        # Contradiction: sentiment points opposite the held side, strongly enough.
        if sign * sentiment <= -_SENTIMENT_CONTRADICTION_THRESHOLD:
            reasons.append(
                {
                    "signal": "news_sentiment",
                    "value": sentiment,
                    "news_count": news_count,
                    "detail": f"mean sentiment {sentiment:.2f} over last {_SENTIMENT_LOOKBACK_HOURS}h contradicts {side} position",
                }
            )

    momentum = _recent_momentum(engine, symbol)
    if momentum is not None and sign * momentum <= -_MOMENTUM_CONTRADICTION_THRESHOLD:
        reasons.append(
            {
                "signal": "price_momentum",
                "value": momentum,
                "detail": f"{_MOMENTUM_WINDOW_DAYS}d return {momentum:.2%} contradicts {side} position",
            }
        )

    return ContradictionResult(symbol=symbol, side=side, closed=bool(reasons), reasons=reasons)


def _log_closure(result: ContradictionResult, mode: str, executed_position: float | None) -> None:
    row = {
        "ts": dt.datetime.now(tz=dt.UTC),
        "symbol": result.symbol,
        "feature_set_id": _FEATURE_SET_ID,
        "model_version": _MODEL_VERSION,
        "forecast": None,
        "regime": None,
        "target_position": 0.0,
        "executed_position": executed_position,
        "mode": mode,
        "reasoning": json.dumps(result.reasons),
    }
    pd.DataFrame([row]).to_sql("decisions", get_engine(), if_exists="append", index=False, dtype={"reasoning": JSONB})


def run_contradiction_check() -> list[ContradictionResult]:
    """
    Checks every currently held position for news/momentum contradicting the
    side it's held on, closes any that trip either threshold, logs a
    decisions row for each closure, and alerts Slack. No-ops cleanly if
    nothing is held or nothing contradicts.
    """
    broker = get_broker()  # never passes confirm_live=True — paper-only by construction
    engine = get_engine()

    if hasattr(broker, "client") and not broker.client.get_clock().is_open:
        logger.info("Market is closed — skipping this check (runs hourly during market hours).")
        return []

    positions = {s: q for s, q in broker.get_positions().items() if q != 0}
    if not positions:
        logger.info("No open positions — nothing to check.")
        return []

    symbols = list(positions.keys())
    try:
        ingest_news(symbols, since_hours=_SENTIMENT_LOOKBACK_HOURS)
        backfill_unscored_news()
    except Exception:
        logger.exception("News refresh failed — checking against whatever sentiment is already in the DB.")

    results: list[ContradictionResult] = []
    for symbol, qty in positions.items():
        result = _check_position(engine, symbol, qty)
        results.append(result)

        if not result.closed:
            continue

        detail = "; ".join(r["detail"] for r in result.reasons)
        logger.warning("Contradiction detected for %s (%s) — closing position. %s", symbol, result.side, detail)
        try:
            broker.submit_target_position(symbol, 0.0)
        except Exception:
            logger.exception("Failed to close contradicted position %s.", symbol)
            continue

        executed = broker.get_positions().get(symbol, 0.0)
        _log_closure(result, broker.mode, executed)
        send_slack_alert(
            f"Contradiction close: {symbol} ({result.side}) closed mid-week. {detail}",
            severity="warning",
        )

    return results


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    results = run_contradiction_check()
    closed = [r for r in results if r.closed]
    print(f"Checked {len(results)} position(s), closed {len(closed)}.")


if __name__ == "__main__":
    main()
