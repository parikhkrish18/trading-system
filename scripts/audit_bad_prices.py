"""
One-time audit for the "impossible 5-day return" class of bug reported on
the dashboard (e.g. "The stock has sold off sharply over the past 5 days
(-118.7%)" -- no real security can fall more than 100% in any window, so a
reading like that always means a bad price got into the pipeline, not a bad
week).

data/validators/checks.py::check_nonpositive_prices (wired into
data/ingest/prices.py::ingest_prices as of this fix) now stops a
zero/negative/missing OHLC value from ever being written to `prices` going
forward. This script is what finds anything that got in *before* that fix
was live, in two places:

  1. `prices` rows with a non-positive or missing open/high/low/close --
     the direct cause: rolling_return()'s plain pct_change() only produces
     something below -100% when the newer close is <= 0.
  2. `features` rows already computed from bad prices -- mom_ret_5d /
     mom_ret_20d beyond a physically-plausible band. These can persist even
     after the underlying `prices` row is fixed or removed, since nothing
     automatically recomputes `features` once its source prices change --
     they're only rebuilt the next time features.build_features runs for
     that symbol/date.

Report-only by default, matching scripts/backfill_headline_entities.py's
convention -- deleting is a separate, explicit step (--delete) since removing
a `prices` row changes what check_gaps sees for that symbol and both
`prices` and `features` may need a fresh build_features pass afterward for
the affected symbols to fully clear.

Usage:
    python -m scripts.audit_bad_prices               # report only
    python -m scripts.audit_bad_prices --delete       # also deletes the bad prices rows found
"""
from __future__ import annotations

import argparse
import logging

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine

logger = logging.getLogger(__name__)

# A 5-day or 20-day return past these is treated as "cannot be real" rather
# than "just a wild week" -- -100% is the hard physical floor (can't lose
# more than everything); +500% in 20 days is already far beyond any real
# large-cap/S&P-500-universe move and almost always means a bad price got
# blended in rather than a genuine 6x.
_MIN_PLAUSIBLE_RETURN = -1.0
_MAX_PLAUSIBLE_RETURN = 5.0


def find_bad_prices(engine) -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT symbol, ts, open, high, low, close, volume FROM prices "
            "WHERE open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 "
            "OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL "
            "ORDER BY symbol, ts"
        ),
        engine,
    )


def find_bad_return_features(engine) -> pd.DataFrame:
    return pd.read_sql(
        text(
            "SELECT symbol, ts, feature_name, value FROM features "
            "WHERE feature_name IN ('mom_ret_5d', 'mom_ret_20d') "
            "AND (value < :lo OR value > :hi) "
            "ORDER BY symbol, ts"
        ),
        engine,
        params={"lo": _MIN_PLAUSIBLE_RETURN, "hi": _MAX_PLAUSIBLE_RETURN},
    )


def audit(delete: bool = False) -> dict:
    engine = get_engine()

    bad_prices = find_bad_prices(engine)
    logger.info("%d prices row(s) with a non-positive or missing OHLC value.", len(bad_prices))
    for _, row in bad_prices.head(20).iterrows():
        logger.info("  %s %s: open=%s high=%s low=%s close=%s", row["symbol"], row["ts"], row["open"], row["high"], row["low"], row["close"])
    if len(bad_prices) > 20:
        logger.info("  ...and %d more.", len(bad_prices) - 20)

    bad_features = find_bad_return_features(engine)
    logger.info(
        "%d features row(s) with an implausible mom_ret_5d/mom_ret_20d (outside [%.0f%%, %.0f%%]).",
        len(bad_features), _MIN_PLAUSIBLE_RETURN * 100, _MAX_PLAUSIBLE_RETURN * 100,
    )
    for _, row in bad_features.head(20).iterrows():
        logger.info("  %s %s: %s = %.1f%%", row["symbol"], row["ts"], row["feature_name"], row["value"] * 100)
    if len(bad_features) > 20:
        logger.info("  ...and %d more.", len(bad_features) - 20)

    if delete and not bad_prices.empty:
        rows = bad_prices[["symbol", "ts"]].to_dict("records")
        with engine.begin() as conn:
            for row in rows:
                conn.execute(text("DELETE FROM prices WHERE symbol = :symbol AND ts = :ts"), row)
        logger.info(
            "Deleted %d bad prices row(s). Re-run features.build_features for the affected symbols "
            "to clear any features already computed from them.",
            len(rows),
        )
    elif delete:
        logger.info("Nothing to delete.")
    elif not bad_prices.empty:
        logger.info("Re-run with --delete to remove these rows (features will need a rebuild afterward).")

    return {
        "bad_prices": len(bad_prices),
        "bad_return_features": len(bad_features),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true", help="Delete the bad prices rows found (report-only otherwise).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    audit(delete=args.delete)


if __name__ == "__main__":
    main()
