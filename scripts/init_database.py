"""
First-run setup for an empty database: schema, universe, price history,
features. Everything a hosted deployment needs before its first screening
run, in the one order that works.

This is deliberately NOT scripts/run_daily_ingest.py. That job tops up the
last 7 days, which is right for a database that already has history and
useless for one that has none — the features need years behind them
(features/build_features.py::FEATURE_LOOKBACK_YEARS), so a 7-day ingest
produces a features table with nothing in it and a screener with nothing to
rank.

Deliberately skipped: fundamentals and news. On the free vendor tier those
take roughly two hours for a ~500-symbol universe (see
data/ingest/http.py's rate limiting), and the model handles their absence
natively — the columns are simply missing and LightGBM tolerates that.
Prices come from yfinance, which is free, unthrottled, and takes minutes.

Every step is idempotent, so re-running after a failure is safe and cheap:
the migration is all IF NOT EXISTS, the universe refresh upserts, prices
upsert on (symbol, ts), and features rebuild from whatever prices exist.

Usage:
    python -m scripts.init_database
    python -m scripts.init_database --years 3 --feature-set-id v4
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from config.settings import settings
from data.ingest.prices import ingest_prices
from data.ingest.universe import load_active_universe, refresh_universe
from data.schema.migrate import migrate
from features.build_features import build_and_store
from monitoring.alerts import configure_file_logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Not part of the tradeable universe — the market-wide regime read needs it
# (execution/trading_loop._market_regime). Without it in the DB the regime
# check silently defaults to CHOP, so it has to be ingested alongside.
_REGIME_PROXY = "SPY"

# A year more than the feature lookback, so the longest rolling windows have
# real data behind their first row rather than starting mid-warmup.
_DEFAULT_HISTORY_YEARS = 4


def with_regime_proxy(symbols: list[str]) -> list[str]:
    """The ingest list plus the market-regime proxy, without duplicating it."""
    if _REGIME_PROXY in symbols:
        return list(symbols)
    return [*symbols, _REGIME_PROXY]


def initialise(years: int = _DEFAULT_HISTORY_YEARS, feature_set_id: str | None = None) -> dict:
    """
    Runs the four setup steps in order and returns what each produced.

    Unlike the daily and weekly jobs, a failure here is NOT isolated and
    swallowed: each step is the precondition for the next, so continuing
    past a failure would only produce a second, more confusing error. The
    exception propagates and the caller reports it.
    """
    feature_set_id = feature_set_id or settings.feature_set_id
    results: dict = {}

    logger.info("Step 1/4: applying database schema")
    migrate()
    results["schema"] = "applied"

    logger.info("Step 2/4: scraping the S&P 500 universe")
    results["universe"] = refresh_universe()
    logger.info("Universe now holds %s active symbols", results["universe"])

    symbols = load_active_universe()
    if not symbols:
        raise RuntimeError("Universe is empty after a successful refresh — refusing to continue.")

    today = dt.datetime.now(tz=dt.UTC).date()
    start = today - dt.timedelta(days=365 * years)
    logger.info("Step 3/4: ingesting %s years of prices for %s symbols via yfinance", years, len(symbols))
    results["price_rows"] = ingest_prices(with_regime_proxy(symbols), start, today, "yfinance")
    logger.info("Ingested %s price rows", results["price_rows"])

    logger.info("Step 4/4: building features (feature_set_id=%s)", feature_set_id)
    results["feature_rows"] = build_and_store(symbols, feature_set_id)
    logger.info("Built %s feature rows", results["feature_rows"])

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an empty database for its first screening run.")
    parser.add_argument("--years", type=int, default=_DEFAULT_HISTORY_YEARS, help="Years of price history to ingest.")
    parser.add_argument("--feature-set-id", default=None, help="Defaults to FEATURE_SET_ID from the environment.")
    args = parser.parse_args()
    configure_file_logging()

    results = initialise(years=args.years, feature_set_id=args.feature_set_id)

    print("\nDatabase ready:")
    print(f"  universe symbols : {results['universe']}")
    print(f"  price rows       : {results['price_rows']}")
    print(f"  feature rows     : {results['feature_rows']}")
    print(
        "\nNext: run a trading cycle. By default (APPROVAL_MODE=auto) it trades "
        "immediately and Telegram, if configured, gets a message once orders are in."
    )


if __name__ == "__main__":
    main()
