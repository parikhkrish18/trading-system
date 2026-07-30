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

import numpy as np
import pandas as pd

from data.ingest.db import get_engine, upsert_dataframe
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

    Vectorized via a time-based rolling window rather than a per-date Python
    loop (needed once --universe is scanning ~500 symbols, not 4): per
    symbol, price dates are interleaved into the news timeline as zero-weight
    "anchor" rows, then a single pass of `.rolling("Nd")` computes the
    trailing window ending at every row — anchor rows included — in one
    vectorized call instead of re-filtering the full news series per date.
    """
    if news.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])
    scored = news.dropna(subset=["sentiment"])
    if scored.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    frames: list[pd.DataFrame] = []
    for symbol, price_dates in prices.groupby("symbol")["ts"]:
        sym_news = scored.loc[scored["symbol"] == symbol, ["ts", "sentiment"]]
        if sym_news.empty:
            continue

        anchors = pd.DataFrame(
            {"ts": price_dates.drop_duplicates(), "sentiment": np.nan, "is_anchor": True}
        )
        combined = (
            pd.concat([sym_news.assign(is_anchor=False), anchors], ignore_index=True)
            .sort_values("ts", kind="stable")
            .set_index("ts")
        )

        mean_3d = combined["sentiment"].rolling("3D").mean()
        count_3d = combined["sentiment"].rolling("3D").count()
        mean_10d = combined["sentiment"].rolling("10D").mean()
        count_10d = combined["sentiment"].rolling("10D").count()

        mask = combined["is_anchor"].to_numpy() & (count_10d.to_numpy() > 0)
        if not mask.any():
            continue

        anchor_ts = combined.index[mask]
        out = pd.DataFrame(
            {
                "symbol": symbol,
                "ts": anchor_ts,
                "sentiment_mean_10d": mean_10d.to_numpy()[mask],
                "news_volume_3d": count_3d.to_numpy()[mask],
                "sentiment_mean_3d": mean_3d.to_numpy()[mask],
            }
        )
        out["sentiment_momentum_3v10"] = out["sentiment_mean_3d"] - out["sentiment_mean_10d"]
        # sentiment_mean_3d / sentiment_momentum_3v10 only apply when the 3d
        # window actually had news (matches the original "if trailing_3d" gate).
        out.loc[out["news_volume_3d"] == 0, ["sentiment_mean_3d", "sentiment_momentum_3v10"]] = np.nan
        frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    wide = pd.concat(frames, ignore_index=True)
    long = wide.melt(id_vars=["symbol", "ts"], var_name="feature_name", value_name="value")
    return long.dropna(subset=["value"])


def build_event_risk_features(prices: pd.DataFrame, macro_calendar: pd.DataFrame) -> pd.DataFrame:
    """
    macro_calendar: columns [ts, category] from the macro_calendar table.
    Countdown-to-next-event is market-wide, not symbol-specific — computed
    once per unique trading date (not per symbol) via merge_asof, then
    broadcast onto every symbol that has a price row on that date. This is
    the thing that mattered most to vectorize: a naive per-symbol loop
    recomputes the identical market-wide countdown up to ~500 times.
    """
    if macro_calendar.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    unique_dates = pd.DataFrame({"ts": prices["ts"].drop_duplicates().sort_values()})

    per_date_frames = []
    for category in _MACRO_CATEGORIES:
        event_dates = macro_calendar.loc[macro_calendar["category"] == category, ["ts"]].sort_values("ts")
        if event_dates.empty:
            continue
        joined = pd.merge_asof(
            unique_dates, event_dates.rename(columns={"ts": "event_ts"}), left_on="ts", right_on="event_ts",
            direction="forward",
        )
        joined = joined.dropna(subset=["event_ts"])
        joined["value"] = (joined["event_ts"] - joined["ts"]).dt.total_seconds() / 86400
        joined["feature_name"] = f"days_to_next_{category.lower()}"
        per_date_frames.append(joined[["ts", "feature_name", "value"]])

    if not per_date_frames:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    per_date = pd.concat(per_date_frames, ignore_index=True)
    return prices[["symbol", "ts"]].drop_duplicates().merge(per_date, on="ts", how="inner")


def build_fundamentals_features(prices: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    fundamentals: columns [symbol, ts, metric, value] from the fundamentals
    table, where ts is the report's filing_date (see data/ingest/fundamentals.py
    — using end_date instead would leak information: a metric only becomes
    usable on/after the date it was actually filed, not the fiscal period it
    describes). As-of joins the latest known value of each metric onto each
    symbol's price dates via merge_asof (one call per metric, across all
    symbols at once) instead of a per-symbol Python loop.
    """
    if fundamentals.empty or prices.empty:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])

    price_dates = prices[["symbol", "ts"]].drop_duplicates().sort_values("ts")

    frames = []
    for metric, msub in fundamentals.groupby("metric"):
        metric_sorted = msub[["symbol", "ts", "value"]].sort_values("ts")
        joined = pd.merge_asof(price_dates, metric_sorted, on="ts", by="symbol", direction="backward")
        joined = joined.dropna(subset=["value"])
        if joined.empty:
            continue
        joined["feature_name"] = f"fund_{metric}_latest"
        frames.append(joined[["symbol", "ts", "feature_name", "value"]])

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "feature_name", "value"])
    return pd.concat(frames, ignore_index=True)


# How far back feature computation looks, regardless of how much price
# history the `prices` table actually holds. Hit live: an unbounded pull of
# the full 5-year backfill (503 symbols x ~5yrs) produced a features batch
# large enough to stall a single-transaction upsert for 3+ hours (see
# data/ingest/db.py). 3 years is plenty for the rolling-window quant features
# (max window is 20 days) and the model's own training lookback.
FEATURE_LOOKBACK_YEARS = 3


def build_and_store(symbols: list[str], feature_set_id: str, lookback_years: int = FEATURE_LOOKBACK_YEARS) -> int:
    engine = get_engine()
    symbol_list = ", ".join(f"'{s}'" for s in symbols)
    prices = pd.read_sql(
        f"""SELECT symbol, ts, open, high, low, close, volume FROM prices
            WHERE symbol IN ({symbol_list}) AND ts >= now() - interval '{lookback_years} years'
            ORDER BY ts""",
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
