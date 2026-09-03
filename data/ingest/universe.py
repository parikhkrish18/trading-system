"""
Equity universe puller — S&P 500 constituents, scraped from Wikipedia
(en.wikipedia.org/wiki/List_of_S%26P_500_companies). No vendor/API key
needed: it's a public, maintained table and Wikipedia doesn't block
automated requests the way bls.gov does (see data/ingest/macro_calendar.py).

Drives which symbols data.ingest.prices/fundamentals/news and
features.build_features operate on via each script's --universe flag,
instead of typing out symbols by hand.

Usage:
    python -m data.ingest.universe --scrape
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd
import requests
from bs4 import BeautifulSoup
from sqlalchemy import bindparam, text

from data.ingest.db import get_engine, upsert_dataframe, validate_symbols

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_USER_AGENT = "trading-system-universe-ingest/1.0 (personal research bot)"

# The S&P 500 has ~503 tickers (multiple share classes). A scrape returning
# meaningfully fewer means Wikipedia changed its page layout or served an
# error page — NOT that hundreds of companies left the index overnight.
# Refusing matters because refresh_universe deactivates everything absent
# from the scrape: accepting a bad scrape would deactivate most of the
# universe, and the next trading cycle would propose closing every position.
MIN_EXPECTED_CONSTITUENTS = 450


def fetch_sp500_constituents() -> pd.DataFrame:
    """Scrape the current S&P 500 list. Returns columns: symbol, name, gics_sector."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": _USER_AGENT}, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = soup.find("table", id="constituents")
    if table is None:
        return pd.DataFrame(columns=["symbol", "name", "gics_sector"])

    rows: list[dict] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all("td")
        if len(cells) < 3:
            continue  # header row has <th>, not <td>
        symbol = cells[0].get_text(strip=True)
        name = cells[1].get_text(strip=True)
        gics_sector = cells[2].get_text(strip=True)
        if symbol:
            rows.append({"symbol": symbol, "name": name, "gics_sector": gics_sector})

    return pd.DataFrame(rows, columns=["symbol", "name", "gics_sector"])


def refresh_universe() -> int:
    """
    Upserts the current constituent list (is_active=True), then marks any
    previously-tracked symbol NOT in this scrape as is_active=False — so
    ingestion driven by --universe stops chasing names that have been
    removed from the index (acquired, delisted, demoted, etc).

    Fails loudly (ValueError) instead of touching the table when the scrape
    looks wrong: fewer than MIN_EXPECTED_CONSTITUENTS rows, or any scraped
    string that doesn't look like a ticker. Both mean a broken/changed page,
    and deactivating on that basis would tell the next cycle to close every
    position. The weekly cycle's run_job catches the error, alerts, and
    keeps the previous universe.
    """
    constituents = fetch_sp500_constituents()
    if len(constituents) < MIN_EXPECTED_CONSTITUENTS:
        raise ValueError(
            f"S&P 500 scrape returned only {len(constituents)} row(s) "
            f"(expected at least {MIN_EXPECTED_CONSTITUENTS}) — refusing to refresh: "
            "accepting it would deactivate most of the universe."
        )
    # Scraped strings from a public wiki page: nothing that doesn't look
    # like a ticker gets anywhere near the database.
    symbols = validate_symbols(constituents["symbol"])

    now = dt.datetime.now(tz=dt.UTC)
    constituents = constituents.copy()
    constituents["added_at"] = now
    constituents["is_active"] = True

    # Point-in-time membership record: the `universe` table only knows who
    # is in the index *today* (survivorship bias baked in); this snapshot
    # says who was in it on each refresh date, for honest future backtests.
    snapshot = constituents[["symbol", "name", "gics_sector"]].copy()
    snapshot.insert(0, "snapshot_date", now.date())

    deactivate = text("UPDATE universe SET is_active = FALSE WHERE symbol NOT IN :symbols").bindparams(
        bindparam("symbols", expanding=True)
    )

    # All three writes share one transaction: a crash between them used to
    # be able to leave the `universe` table and its `universe_snapshot`
    # point-in-time record inconsistent (e.g. the upsert lands but the
    # process dies before the snapshot or the deactivation runs). Wrapping
    # them in a single engine.begin() means they all commit together or all
    # roll back together.
    engine = get_engine()
    with engine.begin() as conn:
        n = upsert_dataframe(constituents, table="universe", conflict_cols=["symbol"], conn=conn)
        upsert_dataframe(snapshot, table="universe_snapshot", conflict_cols=["snapshot_date", "symbol"], conn=conn)
        conn.execute(deactivate, {"symbols": symbols})
    return n


def load_active_universe() -> list[str]:
    """Symbols currently marked active — what --universe flags resolve to."""
    engine = get_engine()
    df = pd.read_sql("SELECT symbol FROM universe WHERE is_active = TRUE ORDER BY symbol", engine)
    return df["symbol"].tolist()


def resolve_symbols(symbols_arg: str | None, use_universe: bool) -> list[str]:
    """
    Shared CLI helper for data.ingest.prices/fundamentals/news and
    features.build_features: --universe pulls the active S&P 500 list,
    otherwise falls back to a hand-typed --symbols list.
    """
    if use_universe:
        symbols = load_active_universe()
        if not symbols:
            raise SystemExit("universe table is empty — run `python -m data.ingest.universe --scrape` first.")
        return symbols
    if not symbols_arg:
        raise SystemExit("Pass --symbols SYM1,SYM2 or --universe.")
    return [s.strip().upper() for s in symbols_arg.split(",") if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the S&P 500 universe table.")
    parser.add_argument("--scrape", action="store_true", help="Scrape the current S&P 500 constituent list.")
    args = parser.parse_args()
    if args.scrape:
        n = refresh_universe()
        print(f"Upserted {n} universe row(s).")
    else:
        print("Nothing to do — pass --scrape.")


if __name__ == "__main__":
    main()
