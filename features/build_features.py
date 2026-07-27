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
from features.quant.mean_reversion import bollinger_pct_b, rsi, zscore
from features.quant.momentum import adx, rolling_return
from features.quant.volatility import atr, realized_vol, vol_of_vol

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

    features = build_quant_features(prices)
    features["feature_set_id"] = feature_set_id
    features = features.dropna(subset=["value"])

    # NOTE: qualitative (sentiment) and event-risk features join in here once
    # features/qualitative/sentiment.py and the macro calendar are populated —
    # left out of the default run until those vendor integrations are wired up,
    # so this doesn't silently produce a feature set missing half its inputs.

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
