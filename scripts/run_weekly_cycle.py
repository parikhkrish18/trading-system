"""
Single entrypoint for the full weekly pipeline: refresh the S&P 500
universe, ingest fresh prices/fundamentals/news, score sentiment, refresh
the macro calendar, rebuild features, then run one screen-and-trade cycle.

Each step is isolated via run_job (same pattern as
scripts/run_daily_ingest.py) so one failing data source doesn't take down
the week's trading cycle — e.g. if Polygon is down for news, prices and
fundamentals still refresh and the cycle still runs on what it has.

Honest cost note: fundamentals + news ingestion for the full ~500-symbol
universe is paced against Polygon's free-tier rate limit (see
data/ingest/http.py) — that's roughly two hours for those two steps alone
on this tier, not a bug, just what the vendor limit costs.

Usage:
    python -m scripts.run_weekly_cycle --feature-set-id v3
    python -m scripts.run_weekly_cycle --feature-set-id v3 --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from data.ingest.fundamentals import ingest_fundamentals
from data.ingest.macro_calendar import refresh_macro_calendar
from data.ingest.news import ingest_news
from data.ingest.prices import ingest_prices
from data.ingest.universe import load_active_universe, refresh_universe
from execution.trading_loop import run_cycle
from features.build_features import build_and_store
from features.qualitative.sentiment import backfill_unscored_news
from monitoring.alerts import alert_pipeline_failure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_REGIME_PROXY = "SPY"  # not part of the tradeable universe, only used for the market-wide regime read


def run_job(name: str, fn, *args, **kwargs):
    """Same isolation pattern as scripts/run_daily_ingest.py::run_job — logs and alerts on failure without raising."""
    try:
        logger.info("Running job: %s", name)
        result = fn(*args, **kwargs)
        logger.info("Finished job: %s", name)
        return result
    except Exception as e:
        logger.exception("Job failed: %s", name)
        alert_pipeline_failure(name, str(e))
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full weekly universe-refresh + screen + trade pipeline.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument(
        "--since-hours", type=int, default=24 * 8,
        help="News lookback window — default covers a week plus a day of slack.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Screen only, never place orders.")
    args = parser.parse_args()

    run_job("universe_refresh", refresh_universe)
    symbols = load_active_universe()
    if not symbols:
        logger.critical("Universe is empty after refresh — aborting the rest of the cycle.")
        alert_pipeline_failure("weekly_cycle", "universe table is empty, nothing to trade")
        return

    today = dt.datetime.now(tz=dt.UTC).date()
    run_job("price_ingest", ingest_prices, [*symbols, _REGIME_PROXY], today - dt.timedelta(days=7), today, "yfinance")
    run_job("fundamentals_ingest", ingest_fundamentals, symbols)
    run_job("news_ingest", ingest_news, symbols, args.since_hours)
    run_job("sentiment_backfill", backfill_unscored_news, 5000)
    run_job("macro_calendar_refresh", refresh_macro_calendar)
    run_job("build_features", build_and_store, symbols, args.feature_set_id)

    result = run_job("trading_cycle", run_cycle, args.feature_set_id, symbols, args.dry_run)
    logger.info("Weekly cycle result: %s", result)


if __name__ == "__main__":
    main()
