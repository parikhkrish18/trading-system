"""
Phase 1 news puller — STUB.

Plug a news/filing vendor in here. The `sentiment` and `surprise` columns
are left NULL at ingest time — they get filled in by
features/qualitative/sentiment.py as a separate pass, so re-scoring with a
better model later doesn't require re-pulling raw news.

Expected output shape:
    symbol | ts | headline | source

Usage (once implemented):
    python -m data.ingest.news --symbols SPY,QQQ --since-hours 24
"""
from __future__ import annotations

import argparse

import pandas as pd

from data.ingest.db import upsert_dataframe


def fetch_news(symbols: list[str], since_hours: int) -> pd.DataFrame:
    """
    Replace with a real vendor call (news API, filings feed, etc).
    Must return columns: symbol, ts (tz-aware), headline, source.
    """
    raise NotImplementedError(
        "Wire up a news/filings vendor here. Expected output shape documented "
        "in this module's docstring. sentiment/surprise are filled in later by "
        "features/qualitative/sentiment.py, not at ingest time."
    )


def ingest_news(symbols: list[str], since_hours: int = 24) -> int:
    df = fetch_news(symbols, since_hours)
    df["sentiment"] = pd.NA
    df["surprise"] = pd.NA
    return upsert_dataframe(df, table="news_events", conflict_cols=["id", "ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest news/filing events into TimescaleDB.")
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers")
    parser.add_argument("--since-hours", type=int, default=24)
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    n = ingest_news(symbols, args.since_hours)
    print(f"Ingested {n} news rows for {symbols}.")


if __name__ == "__main__":
    main()
