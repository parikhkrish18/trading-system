"""
Pure computation for the dashboard's "Model Analysis" / "Model Report Card"
panels -- what the live system's real (paper/live) decisions have actually
scored, bucketed by calendar month, growing forward as more decisions mature.

Replaces the walk-forward-backtest version of these panels (the old
monitoring/dashboard/report_card.py, reading walk_forward_folds -- see
data/schema/017_walk_forward_folds.sql): that measured the model against
2 years of held-out history, computed once by models/train.py. This module
measures nothing but real, live decisions (mode in 'paper'/'live' --
mode='backfill' replays are never counted, same exclusion as everywhere
else this distinction matters), graded after the fact via
monitoring.forecast_accuracy.compute_forecast_accuracy, so the numbers can
only ever describe what the live system actually did -- and there's a new
month to look at every time one closes, rather than a fixed historical set.

models/train.py's walk-forward run still exists and still writes
walk_forward_folds -- that's a separate, valid use (a fast, pre-deployment
sanity check on a candidate model before it goes live) -- this module and
the panels reading it are just no longer how the dashboard reports "how
accurate is the model" to a person.
"""
from __future__ import annotations

import pandas as pd

from monitoring.forecast_accuracy import compute_forecast_accuracy

LIVE_DECISION_MODES = ("paper", "live")


def graded_real_decisions(decisions: pd.DataFrame, prices: pd.DataFrame, horizon_bars: int) -> pd.DataFrame:
    """
    Real (paper/live) decisions that have matured enough to grade -- columns
    symbol, ts, forecast, realized_return, hit. Everything below buckets or
    summarizes this same frame, so there's exactly one place that decides
    what counts as a real, graded decision.
    """
    if decisions.empty:
        return pd.DataFrame(columns=["symbol", "ts", "forecast", "realized_return", "hit"])
    real = decisions[decisions["mode"].isin(LIVE_DECISION_MODES)]
    return compute_forecast_accuracy(real[["symbol", "ts", "forecast"]].dropna(subset=["forecast"]), prices, horizon_bars=horizon_bars)


def monthly_success_buckets(scored: pd.DataFrame) -> pd.DataFrame:
    """
    One row per calendar month that had at least one graded decision, oldest
    first: how many decisions matured that month, what share called
    direction correctly, and the average realized return on them.
    """
    columns = ["month", "n_graded", "success_rate", "avg_realized_return"]
    if scored.empty:
        return pd.DataFrame(columns=columns)

    month = scored["ts"].dt.tz_convert(None).dt.to_period("M").astype(str)
    grouped = (
        scored.assign(month=month)
        .groupby("month")
        .agg(n_graded=("hit", "size"), success_rate=("hit", "mean"), avg_realized_return=("realized_return", "mean"))
        .reset_index()
        .sort_values("month")
    )
    return grouped[columns]


def headline_metrics(scored: pd.DataFrame) -> dict[str, float | int | None]:
    """
    Pooled across every graded decision (not an average of monthly rates,
    which would weight a 2-decision month the same as a 40-decision one) --
    the overall, all-time track record for the stat tiles above the chart.
    """
    if scored.empty:
        return {"n_months": 0, "n_graded": 0, "success_rate": None, "avg_realized_return": None}
    n_months = scored["ts"].dt.tz_convert(None).dt.to_period("M").nunique()
    return {
        "n_months": int(n_months),
        "n_graded": len(scored),
        "success_rate": float(scored["hit"].mean()),
        "avg_realized_return": float(scored["realized_return"].mean()),
    }
