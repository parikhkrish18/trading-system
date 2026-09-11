"""
Fundamentals puller — Finnhub's financials-as-reported feed.

Pulls quarterly/annual financial statements via Finnhub's
/stock/financials-reported endpoint and reshapes into the long-format
schema downstream code expects:

    symbol | ts | metric | value | source

e.g. one row per (symbol, filed_date, "eps_actual"), another for
(symbol, filed_date, "revenue_actual"), etc. Same shape this module used
when it pulled from Polygon, so nothing downstream (features/build_features.py,
the dashboard's Ticker Lookup) needs to change.

Replaces Polygon's vX/reference/financials as the fundamentals source:
Polygon's `filing_date` field was missing on most historical reports,
forcing that data to be dropped wholesale (see the look-ahead-bias comment
below for why a missing date can't just fall back to the period-end date).
Finnhub's `filedDate`, coming straight from SEC filing metadata — the same
feed data/ingest/finnhub.py's SEC-filings ingest already relies on — is
reliably present.

Usage:
    python -m data.ingest.fundamentals --symbols SPY,QQQ
"""
from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import requests

from data.ingest.db import upsert_dataframe
from data.ingest.finnhub import DEFAULT_SLEEP_SECONDS, finnhub_configured, finnhub_get
from data.ingest.universe import resolve_symbols

logger = logging.getLogger(__name__)

FINNHUB_FINANCIALS_URL = "https://finnhub.io/api/v1/stock/financials-reported"

# Metric -> XBRL "concept" tags (case-insensitive) that carry it, across the
# handful of us-gaap taxonomy variants companies actually file under (e.g.
# revenue-recognition tags changed industry-wide around ASC 606). Matched
# against every line item Finnhub returns across all statement sections
# present (income statement, balance sheet, cash flow) rather than one fixed
# section, since which section a concept lands in isn't perfectly consistent
# across filers either.
_METRIC_CONCEPTS: dict[str, set[str]] = {
    "eps_actual": {"earningspersharediluted"},
    "revenue_actual": {
        "revenues",
        "revenuefromcontractwithcustomerexcludingassessedtax",
        "revenuefromcontractwithcustomerincludingassessedtax",
        "salesrevenuenet",
        "salesrevenuegoodsnet",
    },
    "net_income": {"netincomeloss", "profitloss"},
    "gross_profit": {"grossprofit"},
    "total_assets": {"assets"},
    "total_liabilities": {"liabilities"},
}


def _extract_metrics(report: dict) -> dict[str, float]:
    """
    Flatten every line item across whichever of report['report']['bs'/'ic'/'cf']
    are present into {metric: value} for the metrics in _METRIC_CONCEPTS,
    keeping the first matching line item per metric.
    """
    found: dict[str, float] = {}
    statements = report.get("report") or {}
    line_items = [item for section in statements.values() if isinstance(section, list) for item in section]
    for metric, concepts in _METRIC_CONCEPTS.items():
        for item in line_items:
            concept = str(item.get("concept") or "").strip().lower()
            if concept in concepts and item.get("value") is not None:
                found[metric] = item["value"]
                break
    return found


def fetch_fundamentals(symbols: list[str], sleep_seconds: float = DEFAULT_SLEEP_SECONDS) -> pd.DataFrame:
    """
    Pull as-reported quarterly/annual financials per symbol from Finnhub and
    reshape into long format. Returns columns: symbol, ts (tz-aware), metric,
    value, source.

    `sleep_seconds` paces requests between symbols, same reasoning as
    data/ingest/finnhub.py's own pacing — matters once --universe is
    scanning hundreds of names; a single-digit --symbols list can pass
    sleep_seconds=0.
    """
    if not finnhub_configured():
        logger.warning(
            "FINNHUB_API_KEY is not set — skipping fundamentals for %s symbol(s). "
            "Features that depend on fundamentals will be absent, which the model tolerates.",
            len(symbols),
        )
        return pd.DataFrame(columns=["symbol", "ts", "metric", "value", "source"])

    rows: list[dict] = []
    for i, symbol in enumerate(symbols):
        if i > 0 and sleep_seconds > 0:
            time.sleep(sleep_seconds)
        try:
            resp = finnhub_get(FINNHUB_FINANCIALS_URL, {"symbol": symbol})
        except requests.RequestException:
            logger.warning("Failed to fetch fundamentals for %s — skipping this symbol.", symbol, exc_info=True)
            continue
        reports = resp.json().get("data", [])

        for report in reports:
            # filedDate (when the filing actually became public), not
            # endDate (the fiscal period's end) -- same look-ahead-bias
            # reasoning this module has always used: using the period end
            # would let a model "see" a quarter's numbers weeks before they
            # were filed, which can't repeat in live trading.
            #
            # No `or report.get("endDate")` fallback: that would leak the
            # exact look-ahead bias this comment warns about, every time
            # filedDate happens to be missing/falsy.
            filed_date = report.get("filedDate")
            if not filed_date:
                logger.warning(
                    "Skipping a %s fundamentals report with no filedDate (would otherwise "
                    "need endDate, which leaks look-ahead bias) — endDate was %r.",
                    symbol, report.get("endDate"),
                )
                continue
            for metric, value in _extract_metrics(report).items():
                rows.append(
                    {"symbol": symbol, "ts": filed_date, "metric": metric, "value": value, "source": "finnhub"}
                )

    df = pd.DataFrame(rows, columns=["symbol", "ts", "metric", "value", "source"])
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        # Finnhub can return more than one report resolving to the same
        # (symbol, filedDate, metric) key (e.g. a restated/amended filing
        # filed the same day as the original) -- two rows with an identical
        # key in one batch makes upsert_dataframe's ON CONFLICT DO UPDATE
        # fail outright. Keep the last one, same convention this module has
        # always used for same-day revisions.
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
        help="Pause between symbols to stay under Finnhub's rate limit (set 0 for a short --symbols list).",
    )
    args = parser.parse_args()
    symbols = resolve_symbols(args.symbols, args.universe)
    n = ingest_fundamentals(symbols, sleep_seconds=args.sleep_seconds)
    print(f"Ingested {n} fundamentals rows for {len(symbols)} symbol(s).")


if __name__ == "__main__":
    main()
