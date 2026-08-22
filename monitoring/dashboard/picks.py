"""
Pure display transforms for the dashboard's "Latest picks" panel, kept apart
from monitoring/dashboard/server.py so the reshaping is testable without an
HTTP or DB context — same split as monitoring/forecast_accuracy.py.

Nothing here touches the DB or the web layer: every function takes a DataFrame
of `decisions` rows (see data/schema/001_init.sql) and returns plain data.
"""
from __future__ import annotations

import pandas as pd

from models.regime.trend_chop_classifier import CHOP, TREND

# Display column names, kept in one place so the frontend's captions can't
# drift from what latest_picks_table() actually emits.
COL_SYMBOL = "Symbol"
COL_DIRECTION = "Direction"
COL_REGIME = "Market regime"
COL_FORECAST = "Predicted move"
COL_SIZE = "Target size"
COL_STATUS = "Placed?"

LONG = "Long"
SHORT = "Short"
FLAT = "Flat"

EXECUTED = "Placed"
NOT_EXECUTED = "Not placed"


def latest_batch(decisions: pd.DataFrame) -> pd.DataFrame:
    """
    The most recent screener run's rows. models.screener.log_candidates stamps
    every candidate in a run with one identical `ts`, so a batch is exactly the
    set of rows sharing the maximum timestamp.
    """
    if decisions.empty or "ts" not in decisions.columns:
        return decisions.iloc[0:0]
    latest_ts = decisions["ts"].max()
    if pd.isna(latest_ts):
        return decisions.iloc[0:0]
    return decisions.loc[decisions["ts"] == latest_ts].copy()


def direction(target_position: float | None) -> str:
    """Long/Short from the sign of the signed target position; Flat for 0 or missing."""
    if target_position is None or pd.isna(target_position) or target_position == 0:
        return FLAT
    return LONG if target_position > 0 else SHORT


def _status(executed_position: float | None) -> str:
    """
    The screener logs picks with executed_position NULL (proposed, not traded);
    the execution loop fills it in. NULL and 0 both mean "no position taken".
    """
    if executed_position is None or pd.isna(executed_position) or executed_position == 0:
        return NOT_EXECUTED
    return EXECUTED


def latest_picks_table(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Reshapes a decisions batch into the display table: one row per pick, sorted
    by strongest predicted move first. `Target size` is unsigned — the sign
    already lives in `Direction`, and a negative percentage reads as a loss to
    anyone who doesn't know it encodes "short".
    """
    if batch.empty:
        return pd.DataFrame(columns=[COL_SYMBOL, COL_DIRECTION, COL_REGIME, COL_FORECAST, COL_SIZE, COL_STATUS])

    out = pd.DataFrame(
        {
            COL_SYMBOL: batch["symbol"].astype(str),
            COL_DIRECTION: batch["target_position"].map(direction),
            COL_REGIME: batch["regime"].fillna("unknown").astype(str),
            COL_FORECAST: pd.to_numeric(batch["forecast"], errors="coerce"),
            COL_SIZE: pd.to_numeric(batch["target_position"], errors="coerce").abs(),
            COL_STATUS: batch["executed_position"].map(_status),
        }
    )
    return out.sort_values(COL_FORECAST, key=lambda s: s.abs(), ascending=False, na_position="last").reset_index(
        drop=True
    )


def batch_summary(batch: pd.DataFrame) -> dict[str, object]:
    """Headline counts for the stat tiles above the picks table."""
    if batch.empty:
        return {"n_picks": 0, "n_long": 0, "n_short": 0, "last_run": None, "mode": None}

    directions = batch["target_position"].map(direction)
    modes = batch["mode"].dropna().unique() if "mode" in batch.columns else []
    return {
        "n_picks": len(batch),
        "n_long": int((directions == LONG).sum()),
        "n_short": int((directions == SHORT).sum()),
        "last_run": batch["ts"].max(),
        "mode": str(modes[0]) if len(modes) else None,
    }


def regime_counts(batch: pd.DataFrame) -> dict[str, int]:
    """How many picks fell in each market regime, for the caption under the table."""
    if batch.empty or "regime" not in batch.columns:
        return {TREND: 0, CHOP: 0}
    counts = batch["regime"].fillna("unknown").astype(str).value_counts()
    return {TREND: int(counts.get(TREND, 0)), CHOP: int(counts.get(CHOP, 0))}
