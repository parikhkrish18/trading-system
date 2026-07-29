"""
Pure computation for the dashboard's live forecast-accuracy panel (how
often recent real decisions' predicted direction matched what actually
happened) — kept separate from monitoring/dashboard/server.py so it's
unit-testable without a DB/HTTP context. Complements, not replaces, the
walk-forward backtest metrics in MLflow (server.py's /api/analysis/runs):
this measures live decisions after the fact, walk-forward measures the
model on held-out history before any of them were made.
"""
from __future__ import annotations

import pandas as pd


def compute_forecast_accuracy(decisions: pd.DataFrame, prices: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """
    For each decision with a non-null forecast, finds the price `horizon_bars`
    trading days after the decision and compares realized return's sign to
    the forecast's sign. Small-data-friendly (loops per symbol, not vectorized
    across the whole table) since dashboard volumes are hundreds of rows, not millions.

    decisions: columns symbol, ts, forecast.
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
    return result[["symbol", "ts", "forecast", "realized_return", "hit"]]
