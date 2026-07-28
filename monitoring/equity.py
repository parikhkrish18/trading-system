"""
Equity curve recording — a snapshot of total portfolio value over time.
Feeds the dashboard's drawdown chart and risk/circuit_breakers.py's
max_drawdown_breaker. Call record_equity_snapshot() from wherever the
execution loop knows current portfolio value (e.g. after each
reconciliation pass in execution/reconciliation.py).
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine, upsert_dataframe


def record_equity_snapshot(equity_value: float, mode: str, ts: dt.datetime | None = None) -> None:
    ts = ts or dt.datetime.now(tz=dt.UTC)
    df = pd.DataFrame([{"ts": ts, "mode": mode, "equity_value": equity_value}])
    upsert_dataframe(df, table="equity_curve", conflict_cols=["ts", "mode"])


def load_equity_curve(mode: str | None = None, limit: int = 1000) -> pd.DataFrame:
    engine = get_engine()
    if mode:
        query = text(
            "SELECT ts, mode, equity_value FROM equity_curve "
            "WHERE mode = :mode ORDER BY ts DESC LIMIT :limit"
        )
        df = pd.read_sql(query, engine, params={"mode": mode, "limit": limit})
    else:
        query = text("SELECT ts, mode, equity_value FROM equity_curve ORDER BY ts DESC LIMIT :limit")
        df = pd.read_sql(query, engine, params={"limit": limit})
    return df.sort_values("ts").reset_index(drop=True)
