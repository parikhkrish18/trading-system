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
