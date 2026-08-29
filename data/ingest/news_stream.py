"""
Continuous, real-time news ingestion via Alpaca's news websocket
(Benzinga-sourced — the same underlying content data/ingest/news.py's
Polygon REST endpoint pulls, pushed instead of polled). Fills the SAME
news_events table, in the same shape (sentiment/surprise left NULL for
features/qualitative/sentiment.py to score in its own pass) — everything
downstream (the weekly screener's sentiment features, the hourly
contradiction monitor's news check) reads this table exactly as before and
needs no changes to know data now arrives continuously instead of on a poll.

Free with any Alpaca account (paper keys are enough) — no separate
market-data subscription needed — and works even when BROKER=ibkr, since
this only reads news, it never places an order.

IMPORTANT — this is a data-freshness change only. It does NOT change how
often the model screens or trades. That stays the weekly cycle
(scripts/run_weekly_cycle.py) plus the existing hourly
execution/contradiction_monitor.py. Deliberately: TARGET_HORIZON_DAYS=20
(config/settings.py) exists so round-trip transaction costs are amortized
over a multi-week move, and hold_rules.py/contradiction_monitor.py exist
specifically to PREVENT reacting to every price or news tick — the
contradiction monitor's own threshold was widened after it "fired on
noise, hourly, against a thesis that needs ~20 trading days to play out."
Continuous news makes that hourly check see fresher sentiment sooner; it
does not make the system trade more often.

Meant to run as one long-lived process (a systemd service or macOS
launchd LaunchAgent — see infra/systemd/news-stream.service and
infra/launchd/com.trading-system.news-stream.plist), not a cron/timer job.

Usage:
    python -m data.ingest.news_stream --universe
    python -m data.ingest.news_stream --symbols AAPL,MSFT,TSLA
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import time
from typing import Any

import pandas as pd

from config.settings import settings
from data.ingest.db import upsert_dataframe
from data.ingest.universe import resolve_symbols

logger = logging.getLogger(__name__)

# Batched rather than one DB write per headline: news arrives in bursts
# around real events (earnings, macro prints) and a write-per-article would
# hammer the DB hardest exactly when it needs to keep up. Mirrors the batch
# write pattern the REST puller already uses, just on a timer instead of a
# single pull-then-write.
DEFAULT_FLUSH_INTERVAL_SECONDS = 15.0
DEFAULT_FLUSH_MAX_BATCH = 200

_NEWS_EVENTS_COLUMNS = ["id", "symbol", "ts", "headline", "source", "sentiment", "surprise"]


def _stream_stable_id(article_id: str, symbol: str) -> int:
    """
    Same idempotent-id scheme as data/ingest/news.py::_stable_id (stable
    hash of article+symbol so a redelivered message upserts instead of
    duplicating) — kept as its own copy rather than importing the other
    module's helper, since that one's docstring and naming are specific to
    Polygon's article id format and this is a different vendor/transport.
    """
    digest = hashlib.sha256(f"{article_id}:{symbol}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) >> 1  # fits signed bigint


def article_to_rows(article: Any) -> list[dict]:
    """
    One Alpaca News article -> one row per symbol it's tagged with (same
    fan-out data/ingest/news.py uses: a story tagged to both AAPL and MSFT
    should count toward each symbol's sentiment features independently).

    Accepts either alpaca.data.models.news.News (attribute access) or a
    plain dict (Alpaca's stream can deliver either depending on
    raw_data=True/False) — checked via getattr with a dict fallback so this
    function needs no import of the alpaca SDK and is testable with plain
    Python objects/dicts.

    Returns [] for a malformed message (missing id/symbols/timestamp)
    rather than raising — one bad message must not take the whole stream
    down; see NewsStreamBuffer.handle_article for where that's logged.
    """

    def _get(name: str):
        if isinstance(article, dict):
            return article.get(name)
        return getattr(article, name, None)

    symbols = _get("symbols") or []
    article_id = _get("id")
    created_at = _get("created_at")
    headline = _get("headline") or ""
    if not symbols or article_id is None or created_at is None:
        return []

    ts = pd.Timestamp(created_at)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    article_id = str(article_id)

    return [
        {
            "id": _stream_stable_id(article_id, symbol),
            "symbol": symbol,
            "ts": ts,
            "headline": headline,
            "source": "alpaca_stream",
            "sentiment": float("nan"),
            "surprise": float("nan"),
        }
        for symbol in symbols
    ]


class NewsStreamBuffer:
    """
    The testable half of the stream: buffers rows from incoming articles and
    decides when to flush, without touching the websocket itself. The
    websocket connection (run_stream, below) is a thin, hard-to-unit-test
    shell around this.
    """

    def __init__(
        self,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
        max_batch: int = DEFAULT_FLUSH_MAX_BATCH,
        writer=upsert_dataframe,
        clock=time.monotonic,
    ):
        self.flush_interval = flush_interval
        self.max_batch = max_batch
        self._writer = writer
        self._clock = clock
        self._buffer: list[dict] = []
        self._last_flush = clock()
        self.total_ingested = 0

    def add_article(self, article: Any) -> int:
        """Buffers one article's rows, flushing first if the batch/interval limit is hit. Returns rows added (0 for a malformed message)."""
        rows = article_to_rows(article)
        if not rows:
            logger.warning("Skipped a malformed news message (missing id/symbols/timestamp).")
            return 0
        self._buffer.extend(rows)
        if len(self._buffer) >= self.max_batch or (self._clock() - self._last_flush) >= self.flush_interval:
            self.flush()
        return len(rows)

    def flush(self) -> int:
        self._last_flush = self._clock()
        if not self._buffer:
            return 0
        df = pd.DataFrame(self._buffer, columns=_NEWS_EVENTS_COLUMNS)
        self._buffer = []
        # Same belt-and-suspenders as data/ingest/news.py's fetch_news: Alpaca
        # redelivers an article (a reconnect, or a corrected/updated version
        # pushed again) often enough in practice that two rows with the same
        # (id, ts) land in one flush window. Postgres's ON CONFLICT DO UPDATE
        # can't touch the same target row twice in a single statement and
        # raises CardinalityViolation -- which was crash-looping this process
        # every time it happened (uncaught inside the flush the websocket
        # handler calls, so run_stream's outer retry saw it as a fatal
        # disconnect and reconnected only to hit the next duplicate). Keep
        # the last delivery of each (id, ts) pair -- it carries any sentiment
        # correction -- and drop the earlier one before it ever reaches SQL.
        df = df.drop_duplicates(subset=["id", "ts"], keep="last")
        n = self._writer(df, table="news_events", conflict_cols=["id", "ts"])
        self.total_ingested += n
        logger.info("Flushed %d row(s) to news_events (%d total this run).", n, self.total_ingested)
        return n

    async def handle_article(self, article: Any) -> None:
        """The coroutine handed to NewsDataStream.subscribe_news."""
        self.add_article(article)


def run_stream(
    symbols: list[str],
    flush_interval: float = DEFAULT_FLUSH_INTERVAL_SECONDS,
    max_reconnect_backoff: float = 120.0,
) -> None:
    """
    Blocks forever (until interrupted). Wraps alpaca-py's own internal
    websocket reconnect logic in one more outer retry loop with exponential
    backoff, so a fatal client-side exception doesn't end the process for
    good — systemd/launchd will also restart it (see the unit files), this
    is belt-and-suspenders for the common transient-disconnect case so a
    flaky connection doesn't also thrash the process supervisor.
    """
    from alpaca.data.live.news import NewsDataStream  # lazy: heavy import, unused by every other ingest script

    api_key = settings.alpaca_paper_api_key or settings.alpaca_live_api_key
    secret_key = settings.alpaca_paper_secret_key or settings.alpaca_live_secret_key
    if not api_key or not secret_key:
        raise RuntimeError(
            "No Alpaca API key configured (ALPACA_PAPER_API_KEY/ALPACA_PAPER_SECRET_KEY, or the "
            "_LIVE_ variants). The news stream is free with any Alpaca account, paper is enough, "
            "even when BROKER=ibkr — it only reads news, it never places an order."
        )

    buffer = NewsStreamBuffer(flush_interval=flush_interval)
    stream = NewsDataStream(api_key, secret_key)
    stream.subscribe_news(buffer.handle_article, *symbols)

    logger.info("Starting continuous news stream for %d symbol(s).", len(symbols))
    backoff = 5.0
    try:
        while True:
            try:
                stream.run()  # blocks; alpaca-py already retries transient drops internally
            except KeyboardInterrupt:
                break
            except Exception:
                logger.exception("News stream disconnected — reconnecting in %.0fs.", backoff)
                buffer.flush()
                time.sleep(backoff)
                backoff = min(backoff * 2, max_reconnect_backoff)
            else:
                break
    finally:
        buffer.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous real-time news ingestion via Alpaca's news websocket.")
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers")
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--flush-interval", type=float, default=DEFAULT_FLUSH_INTERVAL_SECONDS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    symbols = resolve_symbols(args.symbols, args.universe)
    run_stream(symbols, flush_interval=args.flush_interval)


if __name__ == "__main__":
    main()
