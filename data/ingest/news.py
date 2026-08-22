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
import logging
import time

import pandas as pd
import requests

from config.settings import settings
from data.ingest.db import upsert_dataframe
from data.ingest.http import DEFAULT_SLEEP_SECONDS, polygon_get
from data.ingest.universe import resolve_symbols

logger = logging.getLogger(__name__)

POLYGON_NEWS_URL = "https://api.polygon.io/v2/reference/news"


def _stable_id(polygon_article_id: str, symbol: str) -> int:
    """
    news_events.id is a BIGSERIAL used as an upsert conflict key. Polygon's
    article id is a string, so hash it into a stable bigint — this makes
    re-pulling the same article idempotent instead of inserting a duplicate
    row with a fresh auto-generated id each time.

    Scoped by symbol, not just the article id: a single story often gets
    tagged to multiple tickers (e.g. a "tech stocks rally" piece hitting
    both AAPL and MSFT), and Polygon returns it once per ticker query. Same
    article id + same published_utc across two different symbols would
    otherwise collide on (id, ts) within a single insert batch, which
    Postgres's ON CONFLICT DO UPDATE can't resolve (a single statement can't
    "affect the same row twice"). One row per (article, symbol) is also the
    more useful data model anyway — the shared story should count toward
    each mentioned symbol's sentiment features independently.
    """
    digest = hashlib.sha256(f"{polygon_article_id}:{symbol}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1  # fits signed bigint


def fetch_news(symbols: list[str], since_hours: int, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> pd.DataFrame:
    """
    Pull recent news per symbol from Polygon.
    Returns columns: id, symbol, ts (tz-aware), headline, source.

    `sleep_seconds` paces requests between symbols — matters once --universe
    is scanning hundreds of names against a rate-limited free-tier key; a
    single-digit --symbols list can pass sleep_seconds=0.
    """
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=since_hours)
    rows: list[dict] = []

    for i, symbol in enumerate(symbols):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        params = {
            "ticker": symbol,
            "published_utc.gte": since.isoformat(),
            "limit": 1000,
            "apiKey": settings.polygon_api_key,
        }
        # Hit live twice: a single transient network error (read timeout, DNS
        # blip) on one symbol used to propagate all the way up and discard
        # the whole batch -- losing ~2 hours of already-fetched symbols for
        # the full universe. Skip just the failed symbol instead.
        try:
            resp = polygon_get(POLYGON_NEWS_URL, params)
        except requests.RequestException:
            logger.warning("Failed to fetch news for %s — skipping this symbol.", symbol, exc_info=True)
            continue
        rows.extend(
            {
                "id": _stable_id(article["id"], symbol),
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
        # Belt-and-suspenders against any other source of within-batch (id, ts)
        # collisions — ON CONFLICT DO UPDATE can't handle the same target row
        # twice in one statement, so this must be clean before it reaches SQL.
        df = df.drop_duplicates(subset=["id", "ts"])
    return df


def ingest_news(symbols: list[str], since_hours: int = 24, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> int:
    df = fetch_news(symbols, since_hours, sleep_seconds=sleep_seconds)
    # float("nan"), not pd.NA: an all-pd.NA column has no numeric dtype, so
    # to_sql would write it as text and the DB (a `double precision` column)
    # would reject the insert with a type mismatch.
    df["sentiment"] = float("nan")
    df["surprise"] = float("nan")
    return upsert_dataframe(df, table="news_events", conflict_cols=["id", "ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest news/filing events into TimescaleDB.")
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers")
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument(
        "--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS,
        help="Pause between symbols to stay under Polygon's rate limit (set 0 for a short --symbols list).",
    )
    args = parser.parse_args()
    symbols = resolve_symbols(args.symbols, args.universe)
    n = ingest_news(symbols, args.since_hours, sleep_seconds=args.sleep_seconds)
    print(f"Ingested {n} news rows for {len(symbols)} symbol(s).")


if __name__ == "__main__":
    main()
