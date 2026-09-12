"""
Finnhub news ingestion — company news + SEC filings, replacing Polygon's
REST news puller (data/ingest/news.py, retired) as the polled news source.
data/ingest/news_stream.py's Alpaca/Benzinga websocket keeps running
independently for continuous between-poll coverage; this module is the
polled counterpart, called from the same places ingest_news used to be
(scripts/run_weekly_cycle.py, execution/contradiction_monitor.py).

Two Finnhub endpoints, both mapped into news_events' existing shape so
nothing downstream (features/qualitative/sentiment.py's scoring pass, the
dashboard, the contradiction monitor's news check) needs to know or care
where a row came from:
  - /company-news: ordinary news coverage, source="finnhub".
  - /stock/filings: SEC filings (8-K, 10-K, 10-Q, ...), source="finnhub_filing".
    Represented as a headline ("AAPL filed Form 8-K") plus a summary
    carrying the filing's URL, so it flows through sentiment scoring and
    every existing news view exactly like an ordinary headline — no schema
    change needed (news_events was already documented as "News / filing
    events", see data/schema/001_init.sql).

The sentiment/surprise columns are left NULL at ingest time, same as
data/ingest/news.py did — scored in a separate pass by
features/qualitative/sentiment.py.

Usage:
    python -m data.ingest.finnhub --symbols AAPL,MSFT --since-hours 24
    python -m data.ingest.finnhub --universe
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import logging
import time

import pandas as pd
import requests

from config.settings import settings
from data.ingest.db import upsert_dataframe
from data.ingest.universe import resolve_symbols

logger = logging.getLogger(__name__)

FINNHUB_COMPANY_NEWS_URL = "https://finnhub.io/api/v1/company-news"
FINNHUB_FILINGS_URL = "https://finnhub.io/api/v1/stock/filings"

# Finnhub's free tier is rate-limited around 60 calls/minute -- paced with a
# little margin. Much lighter than Polygon's ~5/min (see data/ingest/http.py),
# so this affords one sleep between every request rather than only between
# symbols.
DEFAULT_SLEEP_SECONDS = 1.1

_NEWS_COLUMNS = ["id", "symbol", "ts", "headline", "summary", "source"]

# Nothing older than this is worth scoring or keeping: this is a swing-
# trading system whose holds typically resolve in days, not weeks (see
# execution/contradiction_monitor.py's docstring), and sentiment_backfill
# scores newest-first anyway, so week-plus-old news would just burn Claude
# calls on stories nothing downstream still cares about. Enforced here, at
# ingestion, regardless of what since_hours a caller passes -- cheaper to
# never write the row than to filter it out later.
_MAX_NEWS_AGE_HOURS = 24 * 7


def finnhub_configured() -> bool:
    """
    Whether a Finnhub key exists at all -- same reasoning as
    data/ingest/http.py::polygon_configured: an unset key otherwise fails
    quietly (every request 401s) rather than loudly, burning the pacing
    sleep on calls that were never going to succeed.
    """
    return bool((settings.finnhub_api_key or "").strip())


def finnhub_get(url: str, params: dict, timeout: int = 30, max_retries: int = 5) -> requests.Response:
    """GET with 429-aware backoff -- same pattern as data/ingest/http.py::polygon_get."""
    for attempt in range(1, max_retries + 1):
        resp = requests.get(url, params={**params, "token": settings.finnhub_api_key}, timeout=timeout)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp
        retry_after = float(resp.headers.get("Retry-After", DEFAULT_SLEEP_SECONDS))
        logger.warning("Finnhub rate limit hit (attempt %s/%s) — sleeping %ss", attempt, max_retries, retry_after)
        time.sleep(retry_after)
    raise RuntimeError(f"Finnhub rate limit exceeded after {max_retries} retries: {url}")


def _stable_id(finnhub_key: str, symbol: str) -> int:
    """
    Same idempotent-id scheme as data/ingest/news.py::_stable_id used:
    news_events.id is a BIGSERIAL upsert conflict key, so re-pulling the
    same article/filing must hash to the same id rather than inserting a
    duplicate row. Scoped by symbol for the same reason the Polygon version
    was -- a story or filing can be pulled once per symbol it's queried
    under.
    """
    digest = hashlib.sha256(f"{finnhub_key}:{symbol}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1  # fits signed bigint


def fetch_company_news(symbols: list[str], since_hours: int, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> pd.DataFrame:
    """
    Pull recent company news per symbol from Finnhub's /company-news.
    Returns columns: id, symbol, ts (tz-aware), headline, summary, source.

    Finnhub's from/to are whole dates, not hours -- since_hours picks a date
    range wide enough to cover the window, and rows older than the real
    since_hours cutoff are filtered out afterward so a --since-hours 1 call
    doesn't return a whole day's backlog.
    """
    if not finnhub_configured():
        logger.warning(
            "FINNHUB_API_KEY is not set — skipping company news for %s symbol(s). "
            "Sentiment features will be absent, which the model tolerates.",
            len(symbols),
        )
        return pd.DataFrame(columns=_NEWS_COLUMNS)

    since_hours = min(since_hours, _MAX_NEWS_AGE_HOURS)
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=since_hours)
    today = dt.datetime.now(tz=dt.UTC).date()
    rows: list[dict] = []

    for i, symbol in enumerate(symbols):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            resp = finnhub_get(
                FINNHUB_COMPANY_NEWS_URL,
                {"symbol": symbol, "from": since.date().isoformat(), "to": today.isoformat()},
            )
        except requests.RequestException:
            logger.warning("Failed to fetch company news for %s — skipping this symbol.", symbol, exc_info=True)
            continue
        for article in resp.json() or []:
            article_id = article.get("id")
            published = article.get("datetime")
            if article_id is None or published is None:
                continue
            rows.append(
                {
                    "id": _stable_id(str(article_id), symbol),
                    "symbol": symbol,
                    "ts": pd.Timestamp(published, unit="s", tz="UTC"),
                    # Seen in practice with Finnhub-aggregated content, same
                    # as Polygon/Benzinga -- decode once here rather than
                    # leaving every downstream reader to notice.
                    "headline": html.unescape(article.get("headline") or ""),
                    "summary": html.unescape(article.get("summary") or ""),
                    "source": "finnhub",
                }
            )

    df = pd.DataFrame(rows, columns=_NEWS_COLUMNS)
    if not df.empty:
        df = df[df["ts"] >= since]
        # Belt-and-suspenders against a within-batch id collision -- same
        # reasoning as data/ingest/news.py's fetch_news.
        df = df.drop_duplicates(subset=["id"], keep="last")
    return df


def fetch_sec_filings(symbols: list[str], since_hours: int, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> pd.DataFrame:
    """
    Pull recent SEC filings per symbol from Finnhub's /stock/filings.
    Represented in the same [id, symbol, ts, headline, summary, source]
    shape as company news -- see this module's docstring for why.
    """
    if not finnhub_configured():
        logger.warning("FINNHUB_API_KEY is not set — skipping SEC filings for %s symbol(s).", len(symbols))
        return pd.DataFrame(columns=_NEWS_COLUMNS)

    since_hours = min(since_hours, _MAX_NEWS_AGE_HOURS)
    since = dt.datetime.now(tz=dt.UTC) - dt.timedelta(hours=since_hours)
    today = dt.datetime.now(tz=dt.UTC).date()
    rows: list[dict] = []

    for i, symbol in enumerate(symbols):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            resp = finnhub_get(
                FINNHUB_FILINGS_URL,
                {"symbol": symbol, "from": since.date().isoformat(), "to": today.isoformat()},
            )
        except requests.RequestException:
            logger.warning("Failed to fetch SEC filings for %s — skipping this symbol.", symbol, exc_info=True)
            continue
        for filing in resp.json() or []:
            access_number = filing.get("accessNumber")
            filed_date = filing.get("filedDate")
            if not access_number or not filed_date:
                continue
            form = filing.get("form")
            url = filing.get("reportUrl") or filing.get("filingUrl") or ""
            rows.append(
                {
                    "id": _stable_id(access_number, symbol),
                    "symbol": symbol,
                    "ts": pd.Timestamp(filed_date, tz="UTC"),
                    "headline": f"{symbol} filed Form {form}" if form else f"{symbol} filed an SEC report",
                    # No separate url column on news_events -- carried in
                    # summary, same slot Finnhub's own article blurb uses
                    # for company news, so it still reaches sentiment
                    # scoring and every existing news view unchanged.
                    "summary": url,
                    "source": "finnhub_filing",
                }
            )

    df = pd.DataFrame(rows, columns=_NEWS_COLUMNS)
    if not df.empty:
        df = df[df["ts"] >= since]
        df = df.drop_duplicates(subset=["id"], keep="last")
    return df


def ingest_finnhub(symbols: list[str], since_hours: int = 24, sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> int:
    """
    Combined news + filings ingest -- same entry-point shape as the
    ingest_news it replaces (data/ingest/news.py, retired), so callers
    (scripts/run_weekly_cycle.py, execution/contradiction_monitor.py) swap
    in with no signature change.
    """
    news_df = fetch_company_news(symbols, since_hours, sleep_seconds=sleep_seconds)
    filings_df = fetch_sec_filings(symbols, since_hours, sleep_seconds=sleep_seconds)
    df = pd.concat([news_df, filings_df], ignore_index=True) if not filings_df.empty else news_df
    if df.empty:
        return 0
    # float("nan"), not pd.NA: an all-pd.NA column has no numeric dtype, so
    # to_sql would write it as text and the DB (a `double precision` column)
    # would reject the insert with a type mismatch.
    df["sentiment"] = float("nan")
    df["surprise"] = float("nan")
    return upsert_dataframe(
        df,
        table="news_events",
        # id alone, not (id, ts) -- see data/ingest/news.py's ingest_news
        # (now retired) and data/schema/014_news_id_unique.sql for why: id
        # is already the stable hash of (article/filing key, symbol), and a
        # vendor redelivering the same item with a corrected timestamp must
        # update the existing row rather than insert a second one.
        conflict_cols=["id"],
        # Never let a re-ingest of an already-scored item wipe out the
        # sentiment/surprise features/qualitative/sentiment.py already
        # computed for it.
        preserve_cols=["sentiment", "surprise"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Finnhub company news + SEC filings into news_events.")
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers")
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--since-hours", type=int, default=24)
    parser.add_argument(
        "--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS,
        help="Pause between requests to stay under Finnhub's rate limit (set 0 for a short --symbols list).",
    )
    args = parser.parse_args()
    symbols = resolve_symbols(args.symbols, args.universe)
    n = ingest_finnhub(symbols, args.since_hours, sleep_seconds=args.sleep_seconds)
    print(f"Ingested {n} news/filing rows for {len(symbols)} symbol(s).")


if __name__ == "__main__":
    main()
