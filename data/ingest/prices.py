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
import tempfile
from pathlib import Path

import pandas as pd

from config.settings import settings
from data.ingest.db import upsert_dataframe
from data.ingest.universe import resolve_symbols
from data.validators.checks import check_nonpositive_prices, run_all_validators
from monitoring.alerts import alert_pipeline_failure


def _fetch_yfinance(symbols: list[str], start: dt.date, end: dt.date) -> pd.DataFrame:
    import yfinance as yf

    # yfinance's default tz cache lives at ~/.cache/py-yfinance and is
    # initialized lazily, per-thread, the first time it's touched — with
    # yf.download()'s internal multi-ticker threading, two threads can race
    # to create that directory at once. yfinance already catches this itself
    # (logs "TzCache will not be used" and carries on, cache just disabled —
    # it never fails the actual price fetch), so this was cosmetic noise,
    # not a real failure. Pointing it at a container-local /tmp path instead
    # of the shared/root cache sidesteps the race entirely rather than just
    # tolerating the warning.
    yf.set_tz_cache_location(str(Path(tempfile.gettempdir()) / "py-yfinance-cache"))

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
            # group_by="ticker" gives (ticker, field) MultiIndex columns even
            # for a single ticker on current yfinance — select the ticker
            # level whenever it exists, not only when len(symbols) > 1.
            # Hit live: every single-symbol ingest (e.g. the SPY regime
            # proxy) crashed on flat-column selection instead.
            sub = raw[yf_symbol[sym]].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
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

    # Checked and dropped *before* the write, not just reported after: a
    # non-positive/missing price is a physical impossibility (see
    # check_nonpositive_prices' docstring for why this matters -- it's what
    # turns into e.g. an "impossible" -118.7% 5-day return once it reaches
    # rolling_return()), so a bad row here must never reach `prices` at all,
    # let alone the features/reasoning built on top of it.
    price_issues = check_nonpositive_prices(df)
    if price_issues:
        price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
        bad_mask = (df[price_cols] <= 0).any(axis=1) | df[price_cols].isna().any(axis=1)
        detail = f"{int(bad_mask.sum())} row(s) with a non-positive or missing price, dropped before write: " + "; ".join(price_issues)
        print(f"[validator] {detail}")
        alert_pipeline_failure("price_ingest", detail)
        df = df.loc[~bad_mask].reset_index(drop=True)
        if df.empty:
            print("Every fetched row failed the price sanity check — nothing written.")
            return 0

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
