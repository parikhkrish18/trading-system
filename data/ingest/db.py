"""
Shared DB plumbing for ingestion scripts. Every puller upserts through
`upsert_dataframe` so the "insert into hypertable, on conflict do nothing/
update" pattern lives in exactly one place — and every symbol-list SQL
filter goes through `symbol_in_clause` so ticker validation does too.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config.settings import settings

_engine: Engine | None = None

# What a ticker is allowed to look like: uppercase letter first, then up to 9
# more uppercase letters/digits/dots/hyphens (BRK.B, BF-B, MMM). Anything
# else — whatever its origin, a bad scrape or a polluted table — has no
# business inside a SQL string.
VALID_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")


def validate_symbols(symbols: Iterable[str]) -> list[str]:
    """
    Return `symbols` as a list, raising ValueError if any entry doesn't look
    like a ticker. Failing loudly is the point: a non-ticker string reaching
    a symbol filter means scraped/external data went somewhere it shouldn't.
    """
    symbols = [str(s) for s in symbols]
    bad = [s for s in symbols if not VALID_SYMBOL_RE.fullmatch(s)]
    if bad:
        preview = ", ".join(repr(s) for s in bad[:5])
        raise ValueError(f"{len(bad)} symbol(s) don't look like tickers: {preview}")
    return symbols


def symbol_in_clause(symbols: Iterable[str]) -> str:
    """
    A quoted SQL IN-list ("'AAPL', 'BRK.B'") built only from validated
    tickers — the one sanctioned way to put a symbol list into a query
    string. The regex admits no quotes or whitespace, so nothing that
    passes validation can escape the literal. An empty input yields "''",
    which matches no real symbol rather than producing invalid `IN ()` SQL.
    """
    validated = validate_symbols(symbols)
    if not validated:
        return "''"
    return ", ".join(f"'{s}'" for s in validated)


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db_url, pool_pre_ping=True)
    return _engine


# Hit live: a single ~9M-row upsert (full 5-year feature rebuild across the
# S&P 500) ran as one INSERT...SELECT...ON CONFLICT inside one transaction
# and stalled for 3+ hours doing random-I/O conflict-checks against a large
# table on default (un-tuned) Postgres settings. Splitting into chunks keeps
# each transaction's working set small enough to stay cache-friendly and
# gives every chunk an independent commit point instead of one all-or-nothing
# multi-hour transaction.
_DEFAULT_CHUNK_ROWS = 200_000


def upsert_dataframe(df: pd.DataFrame, table: str, conflict_cols: list[str], chunk_rows: int = _DEFAULT_CHUNK_ROWS) -> int:
    """
    Upsert a dataframe into `table`, doing nothing on conflict of
    `conflict_cols` (the table's primary key). Returns rows attempted.

    Uses a temp staging table + INSERT ... ON CONFLICT so it works for
    arbitrary column sets without hand-writing SQL per caller. Large
    dataframes are split into `chunk_rows`-sized pieces, each staged/inserted/
    committed independently, rather than one giant single-transaction upsert.
    """
    if df.empty:
        return 0

    engine = get_engine()
    staging_table = f"_staging_{table}"

    cols = list(df.columns)
    col_list = ", ".join(cols)
    conflict_list = ", ".join(conflict_cols)
    update_cols = [c for c in cols if c not in conflict_cols]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols) or "NOTHING"

    # S608: table/column names come from calling code, not user input.
    if update_cols:
        sql = (
            f"INSERT INTO {table} ({col_list}) "  # noqa: S608
            f"SELECT {col_list} FROM {staging_table} "
            f"ON CONFLICT ({conflict_list}) DO UPDATE SET {update_clause}"
        )
    else:
        sql = (
            f"INSERT INTO {table} ({col_list}) "  # noqa: S608
            f"SELECT {col_list} FROM {staging_table} "
            f"ON CONFLICT ({conflict_list}) DO NOTHING"
        )

    for start in range(0, len(df), chunk_rows):
        chunk = df.iloc[start : start + chunk_rows]
        with engine.begin() as conn:
            chunk.to_sql(staging_table, conn, if_exists="replace", index=False, method="multi", chunksize=1000)
            conn.exec_driver_sql(sql)
            conn.exec_driver_sql(f"DROP TABLE {staging_table}")

    return len(df)
