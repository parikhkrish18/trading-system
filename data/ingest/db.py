"""
Shared DB plumbing for ingestion scripts. Every puller upserts through
`upsert_dataframe` so the "insert into hypertable, on conflict do nothing/
update" pattern lives in exactly one place — and every symbol-list SQL
filter goes through `symbol_in_clause` so ticker validation does too.
"""
from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Connection, Engine

from config.settings import settings

logger = logging.getLogger(__name__)

_engine: Engine | None = None

# DB_PASSWORD's literal default (config/settings.py, .env.example) — real on
# purpose, so docker-compose works out of the box for local dev.
_DEFAULT_DB_PASSWORD = "change_me_locally"  # noqa: S105 -- comparison constant, not a real credential

# Same loopback set monitoring/dashboard/server.py's _LOOPBACK_HOSTS uses for
# its request-time bind check. Not imported from there deliberately: this is
# a one-shot settings-level check at engine construction, not a per-request
# one, and duplicating the literal set here keeps it decoupled from however
# that request-time mechanism evolves.
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _warn_if_default_db_password_on_a_non_loopback_deployment() -> None:
    """
    DB_PASSWORD's shipped default is real and known (see above) — harmless
    on loopback, where nothing outside the machine can reach Postgres
    anyway, but a real risk once this deployment is bound beyond loopback
    (DASHBOARD_HOST is the best proxy we have for "is this local dev or a
    real deployment"). A loud warning, not a hard failure: local dev
    genuinely relies on the default working with zero configuration.
    """
    if settings.db_password != _DEFAULT_DB_PASSWORD:
        return
    if settings.dashboard_host in _LOOPBACK_HOSTS:
        return
    logger.warning(
        "DB_PASSWORD is still the default value (%r) and DASHBOARD_HOST=%r "
        "is not loopback — this looks like a non-local deployment running "
        "on the docker-compose default database password. Set a real, "
        "unique DB_PASSWORD before this database is reachable from "
        "anywhere but your own machine.",
        _DEFAULT_DB_PASSWORD,
        settings.dashboard_host,
    )

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
        _warn_if_default_db_password_on_a_non_loopback_deployment()
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


def upsert_dataframe(
    df: pd.DataFrame,
    table: str,
    conflict_cols: list[str],
    chunk_rows: int = _DEFAULT_CHUNK_ROWS,
    preserve_cols: list[str] | None = None,
    conn: Connection | None = None,
) -> int:
    """
    Upsert a dataframe into `table`, doing nothing on conflict of
    `conflict_cols` (the table's primary key). Returns rows attempted.

    Uses a temp staging table + INSERT ... ON CONFLICT so it works for
    arbitrary column sets without hand-writing SQL per caller. Large
    dataframes are split into `chunk_rows`-sized pieces, each staged/inserted/
    committed independently, rather than one giant single-transaction upsert
    -- unless `conn` is given (see below), in which case all chunks share
    that one caller-managed transaction instead.

    `preserve_cols`: columns to leave out of the UPDATE SET clause entirely,
    so an existing row's value for them survives a re-upsert untouched even
    when the incoming dataframe carries a blank/NULL for that column (e.g.
    re-ingesting raw news must never clobber a sentiment score a later pass
    already computed). Still written on first INSERT, same as any other
    column -- only ON CONFLICT UPDATEs skip them.

    `conn`: an existing SQLAlchemy connection/transaction (typically from
    `engine.begin()`) to upsert through instead of opening a new one, so
    this call can participate in a larger caller-managed transaction (e.g.
    several upserts that must all commit or roll back together). When
    omitted (the default), behavior is exactly as before: this function
    opens and commits its own transaction per chunk.
    """
    if df.empty:
        return 0

    preserve_cols = preserve_cols or []
    cols = list(df.columns)
    col_list = ", ".join(cols)
    conflict_list = ", ".join(conflict_cols)
    update_cols = [c for c in cols if c not in conflict_cols and c not in preserve_cols]
    update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols) or "NOTHING"

    def _upsert_chunk(chunk: pd.DataFrame, connection: Connection) -> None:
        # A per-call-unique name, not the fixed f"_staging_{table}" this used
        # to be: two writers upserting into the same table concurrently (e.g.
        # a long-lived streamer flushing every ~15s alongside a cron/poll
        # puller) could otherwise both run CREATE TABLE against the same
        # staging name at the same time and collide.
        staging_table = f"_staging_{table}_{uuid.uuid4().hex}"
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
        chunk.to_sql(staging_table, connection, if_exists="replace", index=False, method="multi", chunksize=1000)
        connection.exec_driver_sql(sql)
        connection.exec_driver_sql(f"DROP TABLE {staging_table}")

    if conn is not None:
        for start in range(0, len(df), chunk_rows):
            _upsert_chunk(df.iloc[start : start + chunk_rows], conn)
    else:
        engine = get_engine()
        for start in range(0, len(df), chunk_rows):
            chunk = df.iloc[start : start + chunk_rows]
            with engine.begin() as new_conn:
                _upsert_chunk(chunk, new_conn)

    return len(df)
