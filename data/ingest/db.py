"""
Shared DB plumbing for ingestion scripts. Every puller upserts through
`upsert_dataframe` so the "insert into hypertable, on conflict do nothing/
update" pattern lives in exactly one place.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config.settings import settings

_engine: Engine | None = None


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(settings.db_url, pool_pre_ping=True)
    return _engine


def upsert_dataframe(df: pd.DataFrame, table: str, conflict_cols: list[str]) -> int:
    """
    Upsert a dataframe into `table`, doing nothing on conflict of
    `conflict_cols` (the table's primary key). Returns rows attempted.

    Uses a temp staging table + INSERT ... ON CONFLICT so it works for
    arbitrary column sets without hand-writing SQL per caller.
    """
    if df.empty:
        return 0

    engine = get_engine()
    staging_table = f"_staging_{table}"

    with engine.begin() as conn:
        df.to_sql(staging_table, conn, if_exists="replace", index=False, method="multi", chunksize=1000)

        cols = list(df.columns)
        col_list = ", ".join(cols)
        conflict_list = ", ".join(conflict_cols)
        update_cols = [c for c in cols if c not in conflict_cols]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols) or "NOTHING"

        if update_cols:
            sql = f"""
                INSERT INTO {table} ({col_list})
                SELECT {col_list} FROM {staging_table}
                ON CONFLICT ({conflict_list}) DO UPDATE SET {update_clause}
            """
        else:
            sql = f"""
                INSERT INTO {table} ({col_list})
                SELECT {col_list} FROM {staging_table}
                ON CONFLICT ({conflict_list}) DO NOTHING
            """
        conn.exec_driver_sql(sql)
        conn.exec_driver_sql(f"DROP TABLE {staging_table}")

    return len(df)
