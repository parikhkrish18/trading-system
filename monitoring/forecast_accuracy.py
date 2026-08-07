"""
Pure computation for the dashboard's forecast-accuracy panel, split out of
monitoring/dashboard/app.py so it's unit-testable without a Streamlit/DB context.
"""
from __future__ import annotations

import pandas as pd

# The `mode` written by scripts/backfill_decisions.py. Historical replays are
# scored by this panel exactly like real decisions — they're honest
# out-of-sample predictions — but they were never proposed to anyone at the
# time, so they're kept distinguishable rather than blended in silently.
BACKFILL_MODE = "backfill"

MODE_LABELS = {
    BACKFILL_MODE: "Backfilled (historical replay)",
    "paper": "Paper",
    "live": "Live",
}


def compute_forecast_accuracy(decisions: pd.DataFrame, prices: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """
    For each decision with a non-null forecast, finds the price `horizon_bars`
    trading days after the decision and compares realized return's sign to
    the forecast's sign. Small-data-friendly (loops per symbol, not vectorized
    across the whole table) since dashboard volumes are hundreds of rows, not millions.

    decisions: columns symbol, ts, forecast — plus an optional `mode`, which
    is carried through to the result when present so the caller can break the
    hit rate down by where the decisions came from (see accuracy_by_mode).
    prices: columns symbol, ts, close.
    """
    if decisions.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "forecast", "realized_return", "hit"])

    frames = []
    for symbol, dsub in decisions.groupby("symbol"):
        psub = prices.loc[prices["symbol"] == symbol].sort_values("ts")
        if psub.empty:
            continue
        price_series = psub.set_index("ts")["close"]

        rows = dsub.sort_values("ts").copy()
        price_at_decision = []
        price_future = []
        for t in rows["ts"]:
            price_at_decision.append(price_series.asof(t))
            later = price_series[price_series.index > t]
            price_future.append(later.iloc[horizon_bars - 1] if len(later) >= horizon_bars else None)
        rows["price_at_decision"] = price_at_decision
        rows["price_future"] = price_future
        frames.append(rows)

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "forecast", "realized_return", "hit"])

    result = pd.concat(frames, ignore_index=True).dropna(subset=["price_at_decision", "price_future"])
    result["realized_return"] = result["price_future"] / result["price_at_decision"] - 1
    result["hit"] = (result["forecast"] > 0) == (result["realized_return"] > 0)

    cols = ["symbol", "ts", "forecast", "realized_return", "hit"]
    if "mode" in result.columns:
        cols.append("mode")
    return result[cols]


def accuracy_by_mode(accuracy: pd.DataFrame) -> pd.DataFrame:
    """
    Splits a compute_forecast_accuracy result by `mode`, so a hit rate built
    mostly from backfilled replays is never presented as if it came from live
    trading. Returns columns mode, label, n, hit_rate (most rows first);
    empty if the input has no `mode` column to split on.
    """
    if accuracy.empty or "mode" not in accuracy.columns:
        return pd.DataFrame(columns=["mode", "label", "n", "hit_rate"])

    grouped = (
        accuracy.assign(mode=lambda d: d["mode"].fillna("unknown").astype(str))
        .groupby("mode")["hit"]
        .agg(n="size", hit_rate="mean")
        .reset_index()
    )
    grouped["label"] = grouped["mode"].map(lambda m: MODE_LABELS.get(m, m.title()))
    return grouped.sort_values("n", ascending=False).reset_index(drop=True)[["mode", "label", "n", "hit_rate"]]
