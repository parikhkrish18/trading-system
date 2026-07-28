"""
Combines quant + qualitative + event-risk features into the `features`
table, tagged with a `feature_set_id` so any historical model run can be
reproduced exactly (Phase 2, point 5).

`feature_set_id` should change whenever you change *which* features are
computed or *how* — bump it any time this module's logic changes, don't
reuse an old id with new logic.

Usage:
    python -m features.build_features --symbols SPY,QQQ,TQQQ,SQQQ --feature-set-id v1
"""
from __future__ import annotations

import argparse

import pandas as pd

from data.ingest.db import get_engine, upsert_dataframe
from features.event_risk.calendar_features import days_to_next_macro_event
from features.quant.mean_reversion import bollinger_pct_b, rsi, zscore
from features.quant.momentum import adx, rolling_return
from features.quant.volatility import atr, realized_vol, vol_of_vol

# Macro categories tracked in the macro_calendar table (see
# data/ingest/macro_calendar.py) — one countdown feature per category.
_MACRO_CATEGORIES = ["FOMC", "CPI", "JOBS"]

# Registry of feature functions and how to call them, so the number/shape of
# features is explicit and reviewable in one place rather than scattered.
QUANT_FEATURES = {
    "mom_ret_5d": lambda df: rolling_return(df["close"], 5),
    "mom_ret_20d": lambda df: rolling_return(df["close"], 20),
    "adx_14": lambda df: adx(df["high"], df["low"], df["close"], 14),
    "vol_realized_20d": lambda df: realized_vol(df["close"], 20),
    "vol_atr_14": lambda df: atr(df["high"], df["low"], df["close"], 14),
    "vol_of_vol": lambda df: vol_of_vol(df["close"]),
    "meanrev_zscore_20d": lambda df: zscore(df["close"], 20),
    "meanrev_bollinger_pctb": lambda df: bollinger_pct_b(df["close"]),
    "meanrev_rsi_14": lambda df: rsi(df["close"], 14),
}


def build_quant_features(prices: pd.DataFrame) -> pd.DataFrame:
    """prices: columns [symbol, ts, open, high, low, close, volume], one or more symbols."""
    rows = []
    for symbol, sub in prices.sort_values("ts").groupby("symbol"):
        sub = sub.reset_index(drop=True)
        for name, fn in QUANT_FEATURES.items():
            values = fn(sub)
            out = pd.DataFrame({"symbol": symbol, "ts": sub["ts"], "feature_name": name, "value": values})
            rows.append(out)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["symbol", "ts", "feature_name", "value"]
    )


def build_qualitative_features(prices: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    """
    news: columns [symbol, ts, sentiment] from news_events (unscored rows —
    sentiment IS NULL — carry no signal yet and are dropped). Aggregates
    trailing sentiment into daily features, anchored to each symbol's own
    price dates so this joins cleanly onto build_quant_features' output.
    News ts is naturally point-in-time (when it was published), so no
    look-ahead risk here as long as each date only looks backward.
    """
    if news.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])
    scored = news.dropna(subset=["sentiment"])
    if scored.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    rows: list[tuple] = []
    for symbol, price_dates in prices.groupby("symbol")["ts"]:
        sym_news = scored.loc[scored["symbol"] == symbol].sort_values("ts")
        if sym_news.empty:
            continue
        sentiment_series = sym_news.set_index("ts")["sentiment"]

        for date in price_dates.sort_values():
            trailing_3d = sentiment_series.loc[
                (sentiment_series.index <= date) & (sentiment_series.index > date - pd.Timedelta(days=3))
            ]
            trailing_10d = sentiment_series.loc[
                (sentiment_series.index <= date) & (sentiment_series.index > date - pd.Timedelta(days=10))
            ]
            if trailing_10d.empty:
                continue
            mean_10d = trailing_10d.mean()
            rows.append((symbol, date, "sentiment_mean_10d", mean_10d))
            rows.append((symbol, date, "news_volume_3d", float(len(trailing_3d))))
            if not trailing_3d.empty:
                mean_3d = trailing_3d.mean()
                rows.append((symbol, date, "sentiment_mean_3d", mean_3d))
                rows.append((symbol, date, "sentiment_momentum_3v10", mean_3d - mean_10d))

    return pd.DataFrame(rows, columns=["symbol", "ts", "feature_name", "value"])


def build_event_risk_features(prices: pd.DataFrame, macro_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    macro_calendar: columns [ts, category] from the macro_calendar table.
    Countdown-to-next-event is market-wide, not symbol-specific, but stored
    per symbol/date to match the features table's schema.
    """
    if macro_calendar.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    rows: list[tuple] = []
    for symbol, price_dates in prices.groupby("symbol")["ts"]:
        for date in price_dates.sort_values():
            for category in _MACRO_CATEGORIES:
                days = days_to_next_macro_event(date, macro_calendar, category=category)
                if pd.notna(days):
                    rows.append((symbol, date, f"days_to_next_{category.lower()}", float(days)))

    return pd.DataFrame(rows, columns=["symbol", "ts", "feature_name", "value"])


def build_fundamentals_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    fundamentals: columns [symbol, ts, metric, value] from the fundamentals
    table, where ts is the report's filing_date (see data/ingest/fundamentals.py
    — using end_date instead would leak information: a metric only becomes
    usable on/after the date it was actually filed, not the fiscal period it
    describes). As-of joins the latest known value of each metric onto each
    symbol's price dates.
    """
    if fundamentals.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    rows: list[tuple] = []
    for symbol, price_dates in prices.groupby("symbol")["ts"]:
        sym_fund = fundamentals.loc[fundamentals["symbol"] == symbol]
        if sym_fund.empty:
            continue
        sorted_dates = price_dates.sort_values()
        for metric, msub in sym_fund.groupby("metric"):
            series = msub.sort_values("ts").set_index("ts")["value"]
            for date in sorted_dates:
                latest = series.asof(date)
                if pd.notna(latest):
                    rows.append((symbol, date, f"fund_{metric}_latest", latest))

    return pd.DataFrame(rows, columns=["symbol", "ts", "feature_name", "value"])


def build_and_store(symbols: list[str], feature_set_id: str) -> int:
    engine = get_engine()
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    prices = pd.read_sql(
        f"SELECT symbol, ts, open, high, low, close, volume FROM prices WHERE symbol IN ({symbol_list}) ORDER BY ts",
        engine,
    )
    if prices.empty:
        print("No price data found — run data.ingest.prices first.")
        return 0

    news = pd.read_sql(
        f"SELECT symbol, ts, sentiment FROM news_events WHERE symbol IN ({symbol_list})", engine
    )
    macro_calendar = pd.read_sql("SELECT ts, category FROM macro_calendar", engine)
    fundamentals = pd.read_sql(
        f"SELECT symbol, ts, metric, value FROM fundamentals WHERE symbol IN ({symbol_list})", engine
    )

    feature_frames = [
        frame
        for frame in (
            build_quant_features(prices),
            build_qualitative_features(prices, news),
            build_event_risk_features(prices, macro_calendar),
            build_fundamentals_features(prices, fundamentals),
        )
        if not frame.empty
    ]
    features = pd.concat(
        feature_frames,
        ignore_index=True,
    )
    features["feature_set_id"] = feature_set_id
    features = features.dropna(subset=["value"])

    n = upsert_dataframe(features, table="features", conflict_cols=["symbol", "ts", "feature_set_id", "feature_name"])
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and store versioned feature set.")
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers")
    parser.add_argument("--feature-set-id", required=True, help="e.g. 'v1' — bump when feature logic changes")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    n = build_and_store(symbols, args.feature_set_id)
    print(f"Stored {n} feature rows under feature_set_id={args.feature_set_id!r}.")


if __name__ == "__main__":
    main()
