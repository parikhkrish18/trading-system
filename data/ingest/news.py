"""
Phase 1 news puller — Polygon.io.

Pulls ticker news via Polygon's news endpoint. The `sentiment` and
`surprise` columns are left NULL at ingest time — they get filled in by
features/qualitative/sentiment.py as a separate pass, so re-scoring with a
better model later doesn't require re-pulling raw news.

Expected output shape:
    symbol | ts | headline | source

Usage:
    python -m data.ingest.news --symbols SPY,QQQ --since-hours 24
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib

import pandas as pd
import requests

from config.settings import settings
from data.ingest.db import upsert_dataframe

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"


def _stable_id(polygon_article_id: str) -> int:
    """
    news_events.id is a BIGSERIAL used as an upsert conflict key. Polygon's
    article id is a string, so hash it into a stable bigint — this makes
    re-pulling the same article idempotent instead of inserting a duplicate
    row with a fresh auto-generated id each time.
    """
    digest = hashlib.sha256(polygon_article_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1  # fits signed bigint


def fetch_news(symbols: list[str], since_hours: int) -> pd.DataFrame:
    """
    Pull recent news per symbol from Polygon.
    Returns columns: id, symbol, ts (tz-aware), headline, source.
    """
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=since_hours)
    rows: list[dict] = []

    for symbol in symbols:
        params = {
            "ticker": symbol,
            "published_utc.gte": since.isoformat(),
            "limit": 1000,
            "apiKey": settings.polygon_api_key,
        }
        resp = requests.get(POLYGON_NEWS_URL, params=params, timeout=30)
        resp.raise_for_status()
        rows.extend(
            {
                "id": _stable_id(article["id"]),
                "symbol": symbol,
                "ts": article["published_utc"],
                "headline": article.get("title", ""),
                "source": "polygon",
            }
            for article in resp.json().get("results", [])
        )

    df = pd.DataFrame(rows, columns=["id", "symbol", "ts", "headline", "source"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def ingest_news(symbols: list[str], since_hours: int = 24) -> int:
    df = fetch_news(symbols, since_hours)
    # float("nan"), not pd.NA: an all-pd.NA column has no numeric dtype, so
    # to_sql would write it as text and the DB (a `double precision` column)
    # would reject the insert with a type mismatch.
    df["sentiment"] = float("nan")
    df["surprise"] = float("nan")
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
