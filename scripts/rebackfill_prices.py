"""
One-time remediation for the unadjusted-price bug fixed in
data/ingest/prices.py on 2026-09-03 (auto_adjust=False -> True; Alpaca's
adjustment=Adjustment.ALL). That fix only changes prices fetched from here
forward -- it does nothing to `prices`/`features` rows already written
under the old, unadjusted fetch. Any symbol that ever had a stock split
while it sat in the universe is carrying a corrupted rolling window around
that split date in every `prices` row and every downstream `features` row
(momentum, volatility, mean-reversion — anything derived from `close`)
until those rows are re-fetched and rebuilt with the fix in place.

What this does, for the given symbols (or --universe):
  1. Re-runs data.ingest.prices.ingest_prices over the full lookback window
     with the now-corrected (adjusted) fetch. upsert_dataframe's ON
     CONFLICT (symbol, ts) DO UPDATE means this overwrites every existing
     `prices` row in that window with the adjusted value -- no separate
     delete step needed, and no gap opens up the way a delete-only fix
     would leave (see scripts/audit_bad_prices.py's docstring for why a
     bad price row generally needs a fresh backfill, not just removal).
  2. Re-runs features.build_features.build_and_store for the same symbols,
     so every rolling feature (mom_ret_5d/20d, vol_realized_20d,
     meanrev_rsi_14, etc.) gets recomputed from the now-corrected `close`
     series rather than continuing to read whatever was already stored
     from the corrupted one.

What this does NOT do, and why it's still worth doing separately:
  - Retrain the model. models/train.py's walk-forward harness was trained
    on features built from the old, occasionally-corrupted price series —
    the *code* correcting the input data doesn't retroactively correct
    what an already-fit model learned from the bad version of it. Re-run
    models/train.py once this script has finished, for a model whose
    training data is clean going forward, before trusting new forecasts
    on any previously-affected symbol.
  - Run scripts.audit_bad_prices, a *different* class of bad price
    (non-positive/missing OHLC) than the one this fixes.

Usage:
    python -m scripts.rebackfill_prices --universe --backfill-years 5
    python -m scripts.rebackfill_prices --symbols AAPL,TSLA --backfill-years 5 --feature-set-id v4
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging

from config.settings import settings
from data.ingest.prices import ingest_prices
from data.ingest.universe import resolve_symbols
from features.build_features import build_and_store

logger = logging.getLogger(__name__)

# Deliberately a symbols-at-a-time regime proxy, matching
# scripts/run_weekly_cycle.py's own _REGIME_PROXY -- SPY has never split
# recently and wouldn't be corrupted by this bug, but re-pulling it here too
# costs nothing and keeps this script self-contained rather than assuming
# the caller remembers to include it.
_REGIME_PROXY = "SPY"


def rebackfill(symbols: list[str], backfill_years: int, feature_set_id: str, source: str = "yfinance") -> None:
    end = dt.datetime.now(tz=dt.UTC).date()
    start = end.replace(year=end.year - backfill_years)
    all_symbols = sorted({*symbols, _REGIME_PROXY})

    logger.info(
        "Re-fetching %d symbol(s) from %s to %s via %s (adjusted OHLC) -- this overwrites every "
        "existing prices row in that window.",
        len(all_symbols), start, end, source,
    )
    n = ingest_prices(all_symbols, start, end, source=source)
    logger.info("Re-wrote %d prices row(s).", n)

    logger.info("Rebuilding features for %d symbol(s) under feature_set_id=%s.", len(symbols), feature_set_id)
    n_features = build_and_store(symbols, feature_set_id, lookback_years=backfill_years)
    logger.info("Rebuilt %d feature row(s).", n_features)

    logger.info(
        "Done. Re-run models/train.py before trusting a fresh forecast on any symbol that was "
        "actually affected by a split during this window -- the model itself was fit on the old, "
        "potentially-corrupted feature values and correcting the input data doesn't retroactively "
        "correct what it already learned."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers, e.g. AAPL,TSLA")
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--backfill-years", type=int, default=5, help="How far back to re-pull (default 5).")
    parser.add_argument("--feature-set-id", default=settings.feature_set_id)
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    symbols = resolve_symbols(args.symbols, args.universe)
    rebackfill(symbols, args.backfill_years, args.feature_set_id, source=args.source)


if __name__ == "__main__":
    main()
