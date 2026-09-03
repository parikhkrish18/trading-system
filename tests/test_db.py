import pandas as pd
import pytest
from sqlalchemy import text

from data.ingest import db

pytestmark = pytest.mark.usefixtures("_widgets_table")


@pytest.fixture
def _widgets_table(monkeypatch):
    """
    Runs against the real local Postgres instance with a throwaway table --
    this code only ever runs against Postgres in production (the staging
    table + ON CONFLICT pattern relies on Postgres-specific syntax that
    SQLite's parser chokes on for the INSERT...SELECT form), so a real
    connection is more faithful than faking one.
    """
    engine = db.get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS widgets"))
        conn.execute(text("CREATE TABLE widgets (symbol TEXT, ts TEXT, value DOUBLE PRECISION, PRIMARY KEY (symbol, ts))"))
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS widgets"))


def test_upsert_dataframe_splits_large_batches_into_chunks(monkeypatch):
    """
    Regression test, hit live: a single ~9M-row upsert ran as one
    INSERT...SELECT...ON CONFLICT inside one transaction and stalled for 3+
    hours on default Postgres settings. Large dataframes must be chunked into
    multiple independent transactions instead of one all-or-nothing insert.
    """
    engine = db.get_engine()
    begin_calls = []
    real_begin = engine.begin

    def counting_begin():
        begin_calls.append(1)
        return real_begin()

    monkeypatch.setattr(engine, "begin", counting_begin)

    df = pd.DataFrame({"symbol": [f"S{i}" for i in range(250)], "ts": ["2026-01-01"] * 250, "value": list(range(250))})

    n = db.upsert_dataframe(df, table="widgets", conflict_cols=["symbol", "ts"], chunk_rows=100)

    assert n == 250
    assert len(begin_calls) == 3  # 100 + 100 + 50 -> 3 separate transactions/commits

    with engine.connect() as conn:
        count = conn.execute(text("SELECT count(*) FROM widgets")).scalar()
    assert count == 250


def test_upsert_dataframe_updates_on_conflict_across_chunks():
    engine = db.get_engine()

    first = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "ts": ["2026-01-01", "2026-01-01"], "value": [1.0, 2.0]})
    db.upsert_dataframe(first, table="widgets", conflict_cols=["symbol", "ts"], chunk_rows=1)

    second = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "ts": ["2026-01-01", "2026-01-01"], "value": [99.0, 98.0]})
    db.upsert_dataframe(second, table="widgets", conflict_cols=["symbol", "ts"], chunk_rows=1)

    with engine.connect() as conn:
        rows = conn.execute(text("SELECT symbol, value FROM widgets ORDER BY symbol")).fetchall()
    assert dict(rows) == {"AAPL": 99.0, "MSFT": 98.0}


def test_upsert_dataframe_empty_df_is_a_noop():
    n = db.upsert_dataframe(pd.DataFrame(), table="widgets", conflict_cols=["symbol", "ts"])
    assert n == 0


def test_upsert_dataframe_uses_a_unique_staging_table_name_per_call(monkeypatch):
    """
    Regression test: a fixed f"_staging_{table}" name meant two writers
    upserting into the same table concurrently (e.g. the news stream's
    ~15s flush overlapping a cron/poll puller) could both run
    CREATE TABLE against the same staging name at the same time and
    collide. Every call must stage under its own unique name.
    """
    staging_names = []
    real_to_sql = pd.DataFrame.to_sql

    def spy_to_sql(self, name, *args, **kwargs):
        staging_names.append(name)
        return real_to_sql(self, name, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_sql", spy_to_sql)

    df1 = pd.DataFrame({"symbol": ["AAPL"], "ts": ["2026-01-01"], "value": [1.0]})
    df2 = pd.DataFrame({"symbol": ["MSFT"], "ts": ["2026-01-01"], "value": [2.0]})

    # Two calls "in a row" without waiting -- simulates the overlap window
    # two concurrent writers would hit, without needing real threads.
    db.upsert_dataframe(df1, table="widgets", conflict_cols=["symbol", "ts"])
    db.upsert_dataframe(df2, table="widgets", conflict_cols=["symbol", "ts"])

    assert len(staging_names) == 2
    assert staging_names[0] != staging_names[1]
    # Neither matches the old fixed, collision-prone name.
    assert "_staging_widgets" not in staging_names


def test_upsert_dataframe_reuses_a_caller_provided_connection():
    """
    `conn=` lets a caller (e.g. universe.py's refresh_universe) fold several
    upserts into one larger transaction it manages itself, instead of each
    upsert opening and committing its own.
    """
    engine = db.get_engine()
    df = pd.DataFrame({"symbol": ["AAPL"], "ts": ["2026-01-01"], "value": [1.0]})

    with engine.begin() as conn:
        n = db.upsert_dataframe(df, table="widgets", conflict_cols=["symbol", "ts"], conn=conn)
        assert n == 1
        # Visible within the same still-open, uncommitted transaction.
        value = conn.execute(text("SELECT value FROM widgets WHERE symbol = 'AAPL'")).scalar()
        assert value == 1.0


@pytest.fixture
def _scored_widgets_table():
    """A throwaway table shaped like news_events' relevant columns, to test
    preserve_cols without depending on the real news_events schema."""
    engine = db.get_engine()
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scored_widgets"))
        conn.execute(
            text("CREATE TABLE scored_widgets (symbol TEXT PRIMARY KEY, headline TEXT, sentiment DOUBLE PRECISION)")
        )
    yield engine
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scored_widgets"))


def test_upsert_dataframe_preserve_cols_survives_a_null_on_re_upsert(_scored_widgets_table):
    """
    Regression test: re-ingesting a row (e.g. news re-pulled from the
    vendor) must not wipe out a value a later pass already computed (e.g.
    a sentiment score), even though the re-ingested row carries NULL/NaN
    for it -- preserve_cols excludes it from the ON CONFLICT UPDATE clause
    entirely, so the existing value survives untouched.
    """
    engine = _scored_widgets_table

    first = pd.DataFrame({"symbol": ["AAPL"], "headline": ["h1"], "sentiment": [0.6]})
    db.upsert_dataframe(first, table="scored_widgets", conflict_cols=["symbol"])

    # Simulates a re-ingest: same row, sentiment blank again (a fresh pull
    # hasn't been scored yet), but preserve_cols says never overwrite it.
    second = pd.DataFrame({"symbol": ["AAPL"], "headline": ["h1 updated"], "sentiment": [float("nan")]})
    db.upsert_dataframe(second, table="scored_widgets", conflict_cols=["symbol"], preserve_cols=["sentiment"])

    with engine.connect() as conn:
        row = conn.execute(text("SELECT headline, sentiment FROM scored_widgets WHERE symbol = 'AAPL'")).fetchone()
    assert row.sentiment == 0.6  # preserved despite the NaN in the re-upsert
    assert row.headline == "h1 updated"  # non-preserved columns still update normally


def test_upsert_dataframe_without_preserve_cols_still_overwrites_everything(_scored_widgets_table):
    """The new parameter must not change behavior for callers who don't pass it."""
    engine = _scored_widgets_table

    first = pd.DataFrame({"symbol": ["AAPL"], "headline": ["h1"], "sentiment": [0.6]})
    db.upsert_dataframe(first, table="scored_widgets", conflict_cols=["symbol"])

    second = pd.DataFrame({"symbol": ["AAPL"], "headline": ["h1 updated"], "sentiment": [float("nan")]})
    db.upsert_dataframe(second, table="scored_widgets", conflict_cols=["symbol"])

    with engine.connect() as conn:
        sentiment = conn.execute(text("SELECT sentiment FROM scored_widgets WHERE symbol = 'AAPL'")).scalar()
    assert sentiment is None  # overwritten to NULL, same as before preserve_cols existed
