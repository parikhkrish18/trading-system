"""
Phase 1 fundamentals puller — STUB.

The plan calls for a "fundamentals+news bundle" vendor (e.g. a Polygon
add-on, or a dedicated fundamentals API). Plug your vendor's client in here;
the shape downstream code expects is a long-format dataframe:

    symbol | ts | metric | value | source

e.g. one row per (symbol, report_date, "eps_actual"), another for
(symbol, report_date, "revenue_actual"), etc. This keeps the schema stable
regardless of which metrics your vendor exposes.

Usage (once implemented):
    python -m data.ingest.fundamentals --symbols SPY,QQQ
"""
from __future__ import annotations

import argparse

import pandas as pd

from data.ingest.db import upsert_dataframe


def fetch_fundamentals(symbols: list[str]) -> pd.DataFrame:
    """
    Replace this with a real vendor call. Must return columns:
    symbol, ts (tz-aware), metric, value, source.
    """
    raise NotImplementedError(
        "Wire up a fundamentals vendor here (e.g. Polygon fundamentals endpoint, "
        "or your data bundle's equivalent). Expected output shape is documented "
        "in this module's docstring."
    )


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
