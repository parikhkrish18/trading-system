"""
Daily entrypoint for the dashboard's "True Report Card" panel -- computes
the system's full, cumulative, real-decisions-only track record as of
today and stores it as one row in model_report_card_history (see
data/schema/018_model_report_card.sql and monitoring/model_report_card.py
for what "cumulative" and "real" mean here and why).

Meant to run once a day, after market close -- the confirmed close is what
makes "as of today" mean something; running mid-session would grade
against an incomplete day's price bar. Deploy as its own Railway cron
service, same one-image/three-existing-services pattern as weekly-cycle-v2/
contradiction-monitor (see infra/railway/README.md), scheduled at 21:30 UTC
weekdays (4:30pm ET) -- after the 4pm close, after contradiction-monitor's
last hourly tick (its cron window ends at 20:00 UTC), and after
daily-price-ingest's 21:00 UTC run has written that day's closing prices.

Usage:
    python -m scripts.refresh_model_report_card
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from config.settings import settings
from data.ingest.db import get_engine, symbol_in_clause, upsert_dataframe
from monitoring.alerts import alert_pipeline_failure, alert_pipeline_progress, configure_file_logging
from monitoring.model_report_card import compute_report_card

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def refresh_model_report_card(as_of_date: dt.date | None = None) -> dict:
    as_of_date = as_of_date or dt.datetime.now(tz=dt.UTC).date()
    engine = get_engine()

    # The WHOLE decisions table, not filtered by date -- "cumulative" means
    # every real decision ever logged, not just today's. mode filtering
    # (paper/live only) happens inside compute_report_card.
    decisions = pd.read_sql(
        "SELECT symbol, ts, mode, forecast, target_position FROM decisions ORDER BY symbol, ts",
        engine,
    )
    if decisions.empty:
        prices = pd.DataFrame(columns=["symbol", "ts", "close"])
    else:
        symbol_list = symbol_in_clause(decisions["symbol"].unique())
        prices = pd.read_sql(
            f"SELECT symbol, ts, close FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts",  # noqa: S608 — symbols validated via symbol_in_clause
            engine,
        )

    snapshot = compute_report_card(decisions, prices, horizon_bars=settings.target_horizon_days, as_of_date=as_of_date)

    row = pd.DataFrame([{**snapshot, "as_of_date": as_of_date}])
    upsert_dataframe(row, table="model_report_card_history", conflict_cols=["as_of_date"])
    return snapshot


def main() -> None:
    configure_file_logging()  # logs survive the console closing
    try:
        logger.info("Running job: model_report_card_refresh")
        snapshot = refresh_model_report_card()
        logger.info("Finished job: model_report_card_refresh — %s", snapshot)
        alert_pipeline_progress(
            "model_report_card_refresh",
            f"{snapshot['n_trades_taken']} real trades taken, "
            f"{snapshot['n_matured']} matured, hit_rate={snapshot['hit_rate']}",
        )
    except Exception as e:
        logger.exception("Job failed: model_report_card_refresh")
        alert_pipeline_failure("model_report_card_refresh", str(e))


if __name__ == "__main__":
    main()
