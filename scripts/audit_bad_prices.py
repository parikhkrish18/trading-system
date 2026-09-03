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
was live, in three places:

  1. `prices` rows with a non-positive or missing open/high/low/close --
     the direct cause: rolling_return()'s plain pct_change() only produces
     something below -100% when the newer close is <= 0.
  2. `features` rows already computed from bad prices -- mom_ret_5d /
     mom_ret_20d beyond a physically-plausible band. These can persist even
     after the underlying `prices` row is fixed or removed, since nothing
     automatically recomputes `features` once its source prices change --
     they're only rebuilt the next time features.build_features runs for
     that symbol/date.
  3. (added 2026-09-03) Symbols with an extreme single-day close jump
     anywhere in their stored history -- the signature an unhandled stock
     split leaves in an unadjusted price series (data/ingest/prices.py used
     to fetch raw, unadjusted OHLC; fixed to auto_adjust=True /
     Adjustment.ALL). This is a *different* incident than #1/#2 above: not
     an impossible negative price, but a real, physically-possible-looking
     number (a client-reported "+252.7% in 20 days") that was still wrong.
     Report-only for a reason -- unlike #1, the fix for a flagged symbol
     isn't deleting the jump row (that just leaves a gap); it's a full
     re-fetch of that symbol's history under the corrected ingestion, via
     scripts/rebackfill_prices.py.

Report-only by default, matching scripts/backfill_headline_entities.py's
convention -- deleting is a separate, explicit step (--delete, and only
ever applies to #1 above) since removing a `prices` row changes what
check_gaps sees for that symbol and both `prices` and `features` may need a
fresh build_features pass afterward for the affected symbols to fully
clear.

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
# more than everything). The upper bound used to be 5.0 (+500%), which is
# how a real "+252.7% in 20 days" reading (data/ingest/prices.py's
# auto_adjust=False bug -- an unhandled stock split reading as a fake huge
# price jump, see that file's fix for the full explanation) got all the way
# to a client-facing dashboard undetected: it was extreme, but not extreme
# enough to trip a 500% bound. Tightened to 2.0 (+200%) -- still far more
# generous than any genuine S&P 500 constituent move in a 20-day window,
# but tight enough to have caught the actual incident that prompted this.
_MIN_PLAUSIBLE_RETURN = -1.0
_MAX_PLAUSIBLE_RETURN = 2.0

# Same threshold data.validators.checks.check_extreme_single_day_moves uses
# going forward on each new ingest batch -- this queries the *entire*
# stored history instead, to find whatever already made it in before that
# check (or the auto_adjust fix it exists alongside) was live.
_MAX_ABS_SINGLE_DAY_MOVE = 0.60


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


def find_extreme_price_jumps(engine, max_abs_move: float = _MAX_ABS_SINGLE_DAY_MOVE) -> pd.DataFrame:
    """
    Scans the *entire* stored `prices` history (not just one ingest batch,
    unlike data.validators.checks.check_extreme_single_day_moves, which
    only ever sees the rows in front of it) for a symbol whose close moved
    by more than max_abs_move from one stored bar to the next -- the
    signature an unhandled stock split leaves in an unadjusted price series
    (see data/ingest/prices.py's auto_adjust=True / Adjustment.ALL fix).

    Report-only, and deliberately not wired into --delete: the fix for a
    flagged symbol is scripts.rebackfill_prices (a full re-fetch under the
    now-corrected ingestion), not deleting the one jump row -- deleting a
    single row would leave a gap without correcting the stale, unadjusted
    history sitting on either side of it.
    """
    return pd.read_sql(
        text(
            "SELECT symbol, ts, close, prev_close, pct_change FROM ("
            "  SELECT symbol, ts, close, "
            "         LAG(close) OVER (PARTITION BY symbol ORDER BY ts) AS prev_close, "
            "         (close - LAG(close) OVER (PARTITION BY symbol ORDER BY ts)) "
            "           / NULLIF(LAG(close) OVER (PARTITION BY symbol ORDER BY ts), 0) AS pct_change "
            "  FROM prices"
            ") daily_moves "
            "WHERE prev_close IS NOT NULL AND ABS(pct_change) > :max_abs_move "
            "ORDER BY symbol, ts"
        ),
        engine,
        params={"max_abs_move": max_abs_move},
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

    extreme_jumps = find_extreme_price_jumps(engine)
    affected_symbols = sorted(extreme_jumps["symbol"].unique()) if not extreme_jumps.empty else []
    logger.info(
        "%d single-day close move(s) beyond %.0f%% across %d symbol(s) (likely an unhandled split or a "
        "vendor error, not a real move — see scripts/rebackfill_prices.py to fix).",
        len(extreme_jumps), _MAX_ABS_SINGLE_DAY_MOVE * 100, len(affected_symbols),
    )
    for _, row in extreme_jumps.head(20).iterrows():
        logger.info("  %s %s: %+.1f%% (close %.2f, prior %.2f)", row["symbol"], row["ts"], row["pct_change"] * 100, row["close"], row["prev_close"])
    if len(extreme_jumps) > 20:
        logger.info("  ...and %d more.", len(extreme_jumps) - 20)
    if affected_symbols:
        logger.info(
            "Fix with: python -m scripts.rebackfill_prices --symbols %s --backfill-years 5",
            ",".join(affected_symbols),
        )

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
        "extreme_price_jumps": len(extreme_jumps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", action="store_true", help="Delete the bad prices rows found (report-only otherwise).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    audit(delete=args.delete)


if __name__ == "__main__":
    main()
