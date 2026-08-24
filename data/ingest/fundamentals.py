"""
Phase 1 fundamentals puller — Polygon.io.

Pulls quarterly financials via Polygon's vX financials endpoint and reshapes
into the long-format schema downstream code expects:

    symbol | ts | metric | value | source

e.g. one row per (symbol, report_date, "eps_actual"), another for
(symbol, report_date, "revenue_actual"), etc. This keeps the schema stable
regardless of which metrics the vendor exposes.

Usage:
    python -m data.ingest.fundamentals --symbols SPY,QQQ
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import requests

from config.settings import settings
from data.ingest.db import upsert_dataframe
from data.ingest.http import DEFAULT_SLEEP_SECONDS, polygon_configured, polygon_get
from data.ingest.universe import resolve_symbols

logger = logging.getLogger(__name__)

POLYGON_FINANCIALS_URL = "https://api.polygon.io/vX/reference/financials"

# Metric -> path within a Polygon financials result's `financials` block.
_METRIC_PATHS = {
    "eps_actual": ("income_statement", "diluted_earnings_per_share", "value"),
    "revenue_actual": ("income_statement", "revenues", "value"),
    "net_income": ("income_statement", "net_income_loss", "value"),
    "gross_profit": ("income_statement", "gross_profit", "value"),
    "total_assets": ("balance_sheet", "assets", "value"),
    "total_liabilities": ("balance_sheet", "liabilities", "value"),
}


def fetch_fundamentals(symbols: list[str], sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> pd.DataFrame:
    """
    Pull quarterly financials per symbol from Polygon and reshape into
    long format. Returns columns: symbol, ts (tz-aware), metric, value, source.

    `sleep_seconds` paces requests between symbols — matters once --universe
    is scanning hundreds of names against a rate-limited free-tier key; a
    single-digit --symbols list can pass sleep_seconds=0.
    """
    if not polygon_configured():
        # One line instead of 503 slow ones: without a key every request
        # 401s, but the pacing sleep between symbols runs anyway, so a
        # universe pull spends ~109 minutes failing. The model handles
        # missing fundamentals natively (the columns are simply absent),
        # so returning empty is the same outcome, reached immediately.
        logger.warning(
            "POLYGON_API_KEY is not set — skipping fundamentals for %s symbol(s). "
            "Features that depend on fundamentals will be absent, which the model tolerates.",
            len(symbols),
        )
        return pd.DataFrame(columns=["symbol", "ts", "metric", "value", "source"])

    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        params = {
            "ticker": symbol,
            "timeframe": "quarterly",
            "limit": 20,
            "apiKey": settings.polygon_api_key,
        }
        # Same fix as news.py: don't let one symbol's transient network error
        # (read timeout, DNS blip) discard the whole batch of already-fetched
        # symbols — skip just this one and keep going.
        try:
            resp = polygon_get(POLYGON_FINANCIALS_URL, params)
        except requests.RequestException:
            logger.warning("Failed to fetch fundamentals for %s — skipping this symbol.", symbol, exc_info=True)
            continue
        results = resp.json().get("results", [])

        for report in results:
            # Use filing_date (when the report actually became public), not
            # end_date (the fiscal period's end) — using end_date would let a
            # model "see" a quarter's numbers as of the quarter's last day,
            # weeks before they were actually filed. That's look-ahead bias:
            # it would inflate backtested accuracy in a way that can't repeat
            # in live trading, since real-time data has no such head start.
            report_date = report.get("filing_date") or report.get("end_date")
            if not report_date:
                continue
            financials = report.get("financials", {})
            for metric, (statement, field, subfield) in _METRIC_PATHS.items():
                value = financials.get(statement, {}).get(field, {}).get(subfield)
                if value is None:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": report_date,
                        "metric": metric,
                        "value": value,
                        "source": "polygon",
                    }
                )

    df = pd.DataFrame(rows, columns=["symbol", "ts", "metric", "value", "source"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        # Polygon can return more than one report with the same filing_date
        # for the same symbol (e.g. a restated/amended filing) — two rows
        # with an identical (symbol, ts, metric) key in one batch makes
        # upsert_dataframe's ON CONFLICT DO UPDATE fail outright (a single
        # statement can't "affect the same row twice"), the same class of
        # bug fixed for news.py's shared-story case. Keep the last one
        # (Polygon returns revisions after the original in practice).
        df = df.drop_duplicates(subset=["symbol", "ts", "metric"], keep="last")
    return df


def ingest_fundamentals(symbols: list[str], sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> int:
    df = fetch_fundamentals(symbols, sleep_seconds=sleep_seconds)
    return upsert_dataframe(df, table="fundamentals", conflict_cols=["symbol", "ts", "metric"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest fundamentals into TimescaleDB.")
    parser.add_argument("--symbols", default=None, help="Comma-separated tickers")
    parser.add_argument("--universe", action="store_true", help="Use the active S&P 500 universe instead of --symbols.")
    parser.add_argument(
        "--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS,
        help="Pause between symbols to stay under Polygon's rate limit (set 0 for a short --symbols list).",
    )
    args = parser.parse_args()
    symbols = resolve_symbols(args.symbols, args.universe)
    n = ingest_fundamentals(symbols, sleep_seconds=args.sleep_seconds)
    print(f"Ingested {n} fundamentals rows for {len(symbols)} symbol(s).")


if __name__ == "__main__":
    main()
