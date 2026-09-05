"""
Equity curve recording and live marking.

Persisted snapshots remain the historical source of truth. Readers also get
a best-effort current broker mark appended to the returned curve so the
dashboard and drawdown breaker do not sit on the last weekly/hourly write
while open positions are moving.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine, upsert_dataframe

logger = logging.getLogger(__name__)


def record_equity_snapshot(equity_value: float, mode: str, ts: dt.datetime | None = None) -> None:
    ts = ts or dt.datetime.now(tz=dt.UTC)
    df = pd.DataFrame([{"ts": ts, "mode": mode, "equity_value": equity_value}])
    upsert_dataframe(df, table="equity_curve", conflict_cols=["ts", "mode"])


def _live_equity_mark(mode: str | None) -> dict | None:
    """
    Best-effort current broker mark. Import lazily to avoid making monitoring
    initialization depend on broker construction. A dashboard/history read
    must still work when the broker is temporarily unreachable, so failures
    degrade to persisted snapshots rather than raising.
    """
    if mode not in (None, "paper", "live"):
        return None
    try:
        from execution.broker import get_broker

        broker = get_broker()
        broker_mode = getattr(broker, "mode", None)
        if mode is not None and broker_mode is not None and broker_mode != mode:
            return None
        value = float(broker.get_portfolio_value())
        if value <= 0:
            return None
        return {
            "ts": dt.datetime.now(tz=dt.UTC),
            "mode": broker_mode or mode or "paper",
            "equity_value": value,
        }
    except Exception:
        logger.debug("Could not append a live broker equity mark; returning persisted curve only.", exc_info=True)
        return None


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

    live = _live_equity_mark(mode)
    if live is not None:
        df = pd.concat([df, pd.DataFrame([live])], ignore_index=True)

    if df.empty:
        return df
    return df.sort_values("ts").reset_index(drop=True)
