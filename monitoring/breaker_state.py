"""
Persists circuit-breaker check results (risk/circuit_breakers.py) so the
monitoring dashboard can show current status without recomputing it.
Every check is logged, not just triggers, so the dashboard can show
"last checked at X, all clear" alongside trip history.

Deliberately kept separate from risk/circuit_breakers.py rather than adding
DB writes there — that module is the most conservatively-scoped one in the
repo by design (pure functions, no side effects), so persistence lives here.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
from sqlalchemy import text

from data.ingest.db import get_engine
from risk.circuit_breakers import (
    BreakerResult,
    max_correlated_exposure_breaker,
    max_drawdown_breaker,
    max_single_position_breaker,
)


def check_and_record_breakers(
    equity_curve,
    positions_by_symbol: dict[str, float],
    portfolio_value: float,
    correlation_matrix,
    max_drawdown_pct: float,
    max_single_position_pct: float,
    max_correlated_exposure_pct: float,
) -> list[BreakerResult]:
    """Runs every breaker (tagged by name), logs all results, returns only triggered ones."""
    ts = dt.datetime.now(tz=dt.UTC)
    tagged: list[tuple[str, BreakerResult]] = []

    tagged.append(("max_drawdown", max_drawdown_breaker(equity_curve, max_drawdown_pct)))

    for symbol, value in positions_by_symbol.items():
        tagged.append(
            (f"max_single_position:{symbol}", max_single_position_breaker(value, portfolio_value, max_single_position_pct))
        )

    tagged.extend(
        ("max_correlated_exposure", result)
        for result in max_correlated_exposure_breaker(
            positions_by_symbol, portfolio_value, correlation_matrix, max_correlated_exposure_pct
        )
    )

    _persist(tagged, ts)
    return [r for _, r in tagged if r.triggered]


def _persist(tagged: list[tuple[str, BreakerResult]], ts: dt.datetime) -> None:
    if not tagged:
        return
    rows = [{"ts": ts, "breaker_name": name, "triggered": r.triggered, "reason": r.reason} for name, r in tagged]
    pd.DataFrame(rows).to_sql("circuit_breaker_state", get_engine(), if_exists="append", index=False)


def load_latest_breaker_state(limit: int = 200) -> pd.DataFrame:
    engine = get_engine()
    query = text(
        "SELECT ts, breaker_name, triggered, reason FROM circuit_breaker_state "
        "ORDER BY ts DESC LIMIT :limit"
    )
    return pd.read_sql(query, engine, params={"limit": limit})
