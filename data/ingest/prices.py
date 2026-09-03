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
from data.validators.checks import check_extreme_single_day_moves, check_nonpositive_prices, run_all_validators
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

    # auto_adjust=True, not False: split/dividend-adjusted OHLC, not the raw
    # traded price. This is not a style preference -- every rolling-return
    # and volatility feature in this repo (rolling_return(), zscore(), RSI,
    # ATR, realized_vol) is a plain pct_change()/diff() over `close`, with no
    # awareness of corporate actions. Fed a RAW close series, a stock's own
    # ordinary split (forward or reverse) reads as a real, enormous one-day
    # price move: a 4-for-1 forward split shows as a fake ~-75% day, a
    # 1-for-4 reverse split as a fake ~+300% day -- and that one bad day
    # then poisons every rolling window that includes it (a 20-day return
    # reading "+252.7%", a 20-day realized vol reading "144% annualized")
    # for as long as it stays in the window, which is exactly the reasoning
    # a client flagged as obviously wrong. Adjusted OHLC keeps the whole
    # series internally consistent across the corporate action instead, the
    # same total-return basis institutional return/vol calculations use.
    # Hit live 2026-09-03: this was still False, and had been since this
    # file was first written -- every symbol that ever split while in the
    # universe was carrying this exact bug in every rolling feature that
    # touched the split date. See scripts/audit_bad_prices.py for finding
    # and scripts/rebackfill_prices.py for fixing whatever this already
    # wrote to `prices`/`features` before this line was corrected.
    raw = yf.download(
        list(yf_symbol.values()),
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
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
    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(settings.alpaca_paper_api_key, settings.alpaca_paper_secret_key)
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=dt.datetime.combine(start, dt.time.min),
        end=dt.datetime.combine(end, dt.time.min),
        # Same reasoning as _fetch_yfinance's auto_adjust=True just above:
        # Alpaca's default (unset) is "raw", unadjusted for splits or
        # dividends -- Adjustment.ALL matches the total-return-adjusted
        # basis every rolling-return/volatility feature in this repo
        # assumes `close` is already on.
        adjustment=Adjustment.ALL,
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

    # A print() nobody's tailing the logs will ever see isn't a real alert.
    # check_gaps/check_staleness/check_duplicates routinely fire on normal,
    # expected conditions (a holiday, a symbol newly added to the universe)
    # -- alerting on every one of those would be exactly the kind of noise
    # that trains a team to stop reading alerts. An extreme single-day move
    # is different: it's specifically the signature an unhandled corporate
    # action (or a vendor error) leaves behind, and it's what silently
    # produced wrong reasoning on the dashboard before this check existed
    # (see check_extreme_single_day_moves' docstring) -- worth surfacing on
    # its own rather than letting it blend into routine validator chatter.
    extreme_moves = check_extreme_single_day_moves(df)
    if extreme_moves:
        alert_pipeline_failure("price_ingest_extreme_move", "; ".join(extreme_moves))

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
