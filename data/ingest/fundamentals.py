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

import pandas as pd
import requests

from config.settings import settings
from data.ingest.db import upsert_dataframe

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


def fetch_fundamentals(symbols: list[str]) -> pd.DataFrame:
    """
    Pull quarterly financials per symbol from Polygon and reshape into
    long format. Returns columns: symbol, ts (tz-aware), metric, value, source.
    """
    rows: list[dict] = []
    for symbol in symbols:
        params = {
            "ticker": symbol,
            "timeframe": "quarterly",
            "limit": 20,
            "apiKey": settings.polygon_api_key,
        }
        resp = requests.get(POLYGON_FINANCIALS_URL, params=params, timeout=30)
        resp.raise_for_status()
        results = resp.json().get("results", [])

        for report in results:
            report_date = report.get("end_date") or report.get("filing_date")
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
    return df


def ingest_fundamentals(symbols: list[str]) -> int:
    df = fetch_fundamentals(symbols)
    return upsert_dataframe(df, table="fundamentals", conflict_cols=["symbol", "ts", "metric"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest fundamentals into TimescaleDB.")
    parser.add_argument("--symbols", required=True, help="Comma-separated tickers")
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    n = ingest_fundamentals(symbols)
    print(f"Ingested {n} fundamentals rows for {symbols}.")


if __name__ == "__main__":
    main()
