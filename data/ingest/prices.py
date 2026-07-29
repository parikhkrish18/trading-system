"""
Phase 1 price puller. Writes daily OHLCV bars into the `prices` hypertable.

Two backends:
  - "alpaca":  uses Alpaca's market data API (needs ALPACA_*_API_KEY set).
  - "yfinance": free fallback, good enough for local dev / initial backfill
                before you've committed to a paid consolidated-tape vendor.

Usage:
    python -m data.ingest.prices --symbols SPY,QQQ,TQQQ,SQQQ --backfill-years 5
    python -m data.ingest.prices --symbols SPY,QQQ --source alpaca
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from config.settings import settings
from data.ingest.db import upsert_dataframe
from data.ingest.universe import resolve_symbols
from data.validators.checks import run_all_validators


def _fetch_yfinance(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf

    # yfinance expects dual-class tickers with a dash (BRK-B), but our
    # canonical symbol everywhere else — Wikipedia's universe scrape,
    # Polygon, Alpaca — uses a dot (BRK.B), matching the SEC's own
    # convention. Translate only for this vendor's API call; store and
    # return the canonical dotted form so downstream code never has to
    # know this quirk exists. Hit live: BRK.B and BF.B both silently
    # failed ("possibly delisted") before this fix.
    yf_symbol = {s: s.replace(".", "-") for s in symbols}

    raw = yf.download(
        list(yf_symbol.values()),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=False,
        group_by="ticker",
        progress=False,
    )

    frames = []
    for sym in symbols:
        try:
            sub = raw[yf_symbol[sym]].copy() if len(symbols) > 1 else raw.copy()
        except KeyError:
            continue
        sub = sub.rename(
            columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
        )
        # how="any", not "all": a row missing even one OHLCV field (e.g. a
        # vendor glitch that returns a null close but valid open/high/low)
        # violates the prices table's NOT NULL constraints and fails the
        # whole batch's single INSERT — better to drop that one row here
        # than poison every symbol's data for the day.
        sub = sub[["open", "high", "low", "close", "volume"]].dropna(how="any")
        sub["symbol"] = sym
        sub["ts"] = pd.to_datetime(sub.index, utc=True)
        sub["source"] = "yfinance"
        frames.append(sub.reset_index(drop=True))

    if not frames:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume", "source"])
    return pd.concat(frames, ignore_index=True)


def _fetch_alpaca(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(settings.alpaca_paper_api_key, settings.alpaca_paper_secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=dt.datetime.combine(start, dt.time.min),
        end=dt.datetime.combine(end, dt.time.min),
    )
    bars = client.get_stock_bars(req).df.reset_index()
    bars = bars.rename(columns={"timestamp": "ts"})
    bars["source"] = "alpaca"
    return bars[["symbol", "ts", "open", "high", "low", "close", "volume", "source"]]


def ingest_prices(symbols: list[str], start: dt.date, end: dt.date, source: str = "yfinance") -> int:
    fetcher = _fetch_alpaca if source == "alpaca" else _fetch_yfinance
    df = fetcher(symbols, start, end)
    if df.empty:
        print("No price data fetched — check symbols/date range/credentials.")
        return 0

    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    n = upsert_dataframe(df, table="prices", conflict_cols=["symbol", "ts"])

    issues = run_all_validators(df, key_cols=["symbol", "ts"], expect_daily=True)
    if issues:
        print(f"[validator] {len(issues)} issue(s) found in this batch — review before downstream use:")
        for issue in issues:
            print(f"  - {issue}")

    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest daily OHLCV bars into TimescaleDB.")
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers, e.g. SPY,QQQ,TQQQ,SQQQ")
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument("--source", default="yfinance", choices=["yfinance", "alpaca"])
    parser.add_argument("--backfill-years", type=int, default=0, help="If set, pulls this many years of history.")
    parser.add_argument("--start", type=str, default=None, help="YYYY-MM-DD, overrides --backfill-years.")
    parser.add_argument("--end", type=str, default=None, help="YYYY-MM-DD, defaults to today.")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, args.universe)
    end = dt.date.fromisoformat(args.end) if args.end else dt.datetime.now(tz=dt.UTC).date()
    if args.start:
        start = dt.date.fromisoformat(args.start)
    elif args.backfill_years:
        start = end.replace(year=end.year - args.backfill_years)
    else:
        start = end - dt.timedelta(days=7)

    n = ingest_prices(symbols, start, end, source=args.source)
    print(f"Ingested/attempted {n} rows for {symbols} from {start} to {end} via {args.source}.")


if __name__ == "__main__":
    main()
