"""
Single entrypoint for the daily ingestion cron/systemd job. Wraps each
puller so one failing source doesn't stop the others, and alerts on any
failure (Phase 8, point 2: "alerts on ... data pipeline failures").

Usage:
    python -m scripts.run_daily_ingest --symbols SPY,QQQ,TQQQ,SQQQ
    python -m scripts.run_daily_ingest --universe --source yfinance
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from data.ingest.prices import ingest_prices
from data.ingest.universe import resolve_symbols
from monitoring.alerts import alert_pipeline_failure, configure_file_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Same proxy scripts/run_weekly_cycle.py ingests: not part of the tradeable
# universe, only the input to execution/trading_loop._market_regime. Without
# it in the DB the regime check silently defaults to CHOP every cycle.
_REGIME_PROXY = "SPY"


def with_regime_proxy(symbols: list[str]) -> list[str]:
    """The ingest list plus the market-regime proxy, without duplicating it."""
    if _REGIME_PROXY in symbols:
        return list(symbols)
    return [*symbols, _REGIME_PROXY]


def run_job(name: str, fn, *args, **kwargs) -> None:
    try:
        logger.info("Running job: %s", name)
        fn(*args, **kwargs)
        logger.info("Finished job: %s", name)
    except Exception as e:
        logger.exception("Job failed: %s", name)
        alert_pipeline_failure(name, str(e))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default=None)
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--source", default="alpaca", choices=["alpaca", "yfinance"])
    args = parser.parse_args()
    configure_file_logging()  # logs survive the console closing
    symbols = resolve_symbols(args.symbols, args.universe)

    today = dt.datetime.now(tz=dt.UTC).date()
    run_job(
        "price_ingest",
        ingest_prices,
        with_regime_proxy(symbols),
        today - dt.timedelta(days=7),
        today,
        args.source,
    )

    # Add fundamentals/news/macro-calendar jobs here as those vendor
    # integrations get wired up (data/ingest/fundamentals.py, news.py,
    # macro_calendar.py) — each should get its own run_job(...) call so
    # failures stay isolated per source.


if __name__ == "__main__":
    main()
