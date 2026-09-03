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
    python -m scripts.run_weekly_cycle --feature-set-id v4
    python -m scripts.run_weekly_cycle --feature-set-id v4 --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd

from config.settings import settings
from data.ingest.db import get_engine
from data.ingest.fundamentals import ingest_fundamentals
from data.ingest.macro_calendar import refresh_macro_calendar
from data.ingest.news import ingest_news
from data.ingest.prices import ingest_prices
from data.ingest.universe import load_active_universe, refresh_universe
from data.validators.checks import check_staleness
from execution.trading_loop import run_cycle
from features.build_features import build_and_store
from features.qualitative.sentiment import backfill_unscored_news
from monitoring.alerts import alert_pipeline_failure, configure_file_logging

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


def freshness_issues(latest_by_table: dict, max_age_days: int) -> list[str]:
    """
    Human-readable reasons the data is too old to trade on, given the newest
    row timestamp per table (None = the table has no rows at all). Empty list
    means fresh enough. Staleness logic itself is data.validators.checks.
    check_staleness, reused rather than reimplemented.
    """
    issues: list[str] = []
    for table, ts in latest_by_table.items():
        if ts is None or pd.isna(ts):
            issues.append(f"{table}: no rows at all — every ingest failed or the vendor returned nothing")
            continue
        issues += check_staleness(
            pd.DataFrame({"symbol": [table], "ts": [pd.Timestamp(ts)]}), max_age_days=max_age_days
        )
    return issues


def check_data_freshness(engine=None, max_age_days: int | None = None) -> list[str]:
    """Newest price and feature rows in the DB, run through freshness_issues."""
    engine = engine or get_engine()
    max_age_days = settings.max_data_staleness_days if max_age_days is None else max_age_days
    latest = {
        table: pd.read_sql(f"SELECT MAX(ts) AS ts FROM {table}", engine)["ts"].iloc[0]  # noqa: S608 — table names are a fixed tuple
        for table in ("prices", "features")
    }
    return freshness_issues(latest, max_age_days)


def run_guarded_trading_cycle(
    feature_set_id: str,
    symbols: list[str],
    dry_run: bool,
    run_cycle_fn=run_cycle,
    freshness_fn=check_data_freshness,
    alert_fn=alert_pipeline_failure,
):
    """
    The staleness guard between ingestion and trading. run_job deliberately
    swallows ingest failures so one dead vendor doesn't kill the week — but
    that means every job can have failed by this point. Before any broker
    call, refuse to trade unless the newest price AND feature rows are
    recent; a guard that can't even check (DB error) also refuses.
    """
    try:
        issues = freshness_fn()
    except Exception as e:  # DB unreachable etc. — refusing beats trading blind
        issues = [f"could not verify data freshness ({type(e).__name__}: {e})"]

    if issues:
        detail = "; ".join(issues)
        logger.critical("Aborting the trading cycle — data too stale to trade on: %s", detail)
        alert_fn("data_freshness_guard", f"trading cycle aborted, no orders placed: {detail}")
        return None

    return run_cycle_fn(feature_set_id, symbols, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full weekly universe-refresh + screen + trade pipeline.")
    parser.add_argument("--feature-set-id", required=True)
    parser.add_argument(
        "--since-hours", type=int, default=24 * 8,
        help="News lookback window — default covers a week plus a day of slack.",
    )
    parser.add_argument(
        "--backfill-years", type=int, default=0,
        help=(
            "Price history window. Default 0 means the normal steady-state "
            "top-up: just the last 7 days, assuming a prior backfill already "
            "gave the DB its historical depth. Set this (e.g. 5) after a "
            "fresh/truncated prices table, or every quant feature's rolling "
            "window (10/20-day momentum etc.) comes back empty and the "
            "trading cycle has nothing to train on — hit live, not "
            "hypothetical."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Screen only, never place orders.")
    args = parser.parse_args()

    configure_file_logging()  # logs survive the console closing
    run_job("universe_refresh", refresh_universe)
    symbols = load_active_universe()
    if not symbols:
        logger.critical("Universe is empty after refresh — aborting the rest of the cycle.")
        alert_pipeline_failure("weekly_cycle", "universe table is empty, nothing to trade")
        return

    today = dt.datetime.now(tz=dt.UTC).date()
    price_start = today.replace(year=today.year - args.backfill_years) if args.backfill_years else today - dt.timedelta(days=7)
    run_job("price_ingest", ingest_prices, [*symbols, _REGIME_PROXY], price_start, today, "yfinance")
    run_job("fundamentals_ingest", ingest_fundamentals, symbols)
    run_job("news_ingest", ingest_news, symbols, args.since_hours)
    run_job("sentiment_backfill", backfill_unscored_news, 5000)
    run_job("macro_calendar_refresh", refresh_macro_calendar)
    run_job("build_features", build_and_store, symbols, args.feature_set_id)

    result = run_job("trading_cycle", run_guarded_trading_cycle, args.feature_set_id, symbols, args.dry_run)
    logger.info("Weekly cycle result: %s", result)


if __name__ == "__main__":
    main()
