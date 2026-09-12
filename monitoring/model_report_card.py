"""
Pure computation for the dashboard's "True Report Card" panel -- a
cumulative, real-decisions-only track record of the live system, refreshed
once daily after market close (scripts/refresh_model_report_card.py).

Deliberately separate from monitoring/forecast_accuracy.py (which this
reuses): that module answers "what's the live hit rate right now", called
fresh on every dashboard page load. This module answers "what is the
system's entire track record as of today", computed once a day and stored
(data/schema/018_model_report_card.sql) so the history itself is the point
-- a growing record, not a recomputed-every-refresh number.

Never counts a backtested or simulated number: only real decisions with
mode in ('paper', 'live') -- mode='backfill' (replayed history) is excluded
the same way monitoring/dashboard/server.py's _LIVE_DECISION_MODES already
excludes it from the live-accuracy panel, and walk-forward results
(walk_forward_folds, data/schema/017_walk_forward_folds.sql) never enter
this table at all.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from monitoring.forecast_accuracy import compute_forecast_accuracy

# Real decision modes only -- mode='backfill' (replayed history) is never a
# trade the live system actually took, so it must never count toward a
# "trades taken and capital deployed" report card. Same set server.py's
# live-accuracy/drift panels already use, kept as its own constant here
# rather than imported so this module has no dependency on the dashboard.
LIVE_DECISION_MODES = ("paper", "live")


def real_trades_taken(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    Every real decision that actually sized a trade -- a nonzero
    target_position, from a real (paper/live) mode. Excludes closes
    (target_position == 0.0), holds (target_position IS NULL), and any
    mode='backfill' row, so a count of this frame's rows is exactly
    "trades taken", never inflated by bookkeeping rows or replayed history.

    decisions: columns symbol, ts, mode, target_position (+ whatever else
    the caller wants to carry through, e.g. forecast, executed_position).
    """
    if decisions.empty:
        return decisions
    real = decisions[decisions["mode"].isin(LIVE_DECISION_MODES)]
    return real[real["target_position"].notna() & (real["target_position"] != 0)]


def current_capital_deployed_pct(decisions: pd.DataFrame) -> float:
    """
    What fraction of the book is deployed right now, per the decisions log:
    the LATEST real (paper/live) decision per symbol, summed by absolute
    target_position (a short's capital commitment counts the same as a
    long's). A symbol whose latest decision is a close (target_position ==
    0.0) or a hold (NULL) contributes nothing, exactly as it should --
    that symbol isn't holding capital right now.

    This is a snapshot, not cumulative -- the caller stores one of these per
    day (model_report_card_history.capital_deployed_pct) to build a real
    history of deployment over time out of daily snapshots.
    """
    real = decisions[decisions["mode"].isin(LIVE_DECISION_MODES)]
    if real.empty:
        return 0.0
    latest_per_symbol = real.sort_values("ts").groupby("symbol").tail(1)
    sized = latest_per_symbol["target_position"].dropna()
    return float(sized.abs().sum())


def compute_report_card(
    decisions: pd.DataFrame, prices: pd.DataFrame, horizon_bars: int, as_of_date: dt.date
) -> dict:
    """
    The full cumulative snapshot for one day's row in
    model_report_card_history -- every real trade taken across the
    system's entire history, graded with the exact same
    compute_forecast_accuracy logic the live dashboard already uses.

    decisions: columns symbol, ts, mode, forecast, target_position, from
    the WHOLE decisions table (not pre-filtered by date -- "cumulative"
    means every real decision ever logged, not just today's).
    prices: columns symbol, ts, close, covering every symbol in `decisions`.
    horizon_bars: settings.target_horizon_days -- must match what the live
    forecasts were actually made against (see server.py's own comment on
    this same point).
    """
    trades = real_trades_taken(decisions)
    capital_deployed = current_capital_deployed_pct(decisions)

    if trades.empty:
        return {
            "as_of_date": as_of_date.isoformat(),
            "n_trades_taken": 0,
            "n_matured": 0,
            "n_hits": 0,
            "hit_rate": None,
            "avg_realized_return": None,
            "capital_deployed_pct": capital_deployed,
        }

    scored = compute_forecast_accuracy(
        trades[["symbol", "ts", "forecast"]].dropna(subset=["forecast"]), prices, horizon_bars=horizon_bars
    )
    n_matured = len(scored)
    n_hits = int(scored["hit"].sum()) if n_matured else 0

    return {
        "as_of_date": as_of_date.isoformat(),
        "n_trades_taken": len(trades),
        "n_matured": n_matured,
        "n_hits": n_hits,
        "hit_rate": float(n_hits / n_matured) if n_matured else None,
        "avg_realized_return": float(scored["realized_return"].mean()) if n_matured else None,
        "capital_deployed_pct": capital_deployed,
    }
