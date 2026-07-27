"""
Single entrypoint for the daily ingestion cron/systemd job. Wraps each
puller so one failing source doesn't stop the others, and alerts on any
failure (Phase 8, point 2: "alerts on ... data pipeline failures").

Usage:
    python -m scripts.run_daily_ingest --symbols SPY,QQQ,TQQQ,SQQQ
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from data.ingest.prices import ingest_prices
from monitoring.alerts import alert_pipeline_failure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    parser.add_argument("--symbols", required=True)
    parser.add_argument("--source", default="alpaca", choices=["alpaca", "yfinance"])
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    today = dt.datetime.now(tz=dt.UTC).date()
    run_job(
        "price_ingest",
        ingest_prices,
        symbols,
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
