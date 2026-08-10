"""
The maths behind the dashboard's "What-if thresholds" playground: re-filter and
re-rank the picks the screener already logged, at whatever bar the reader drags
the sliders to.

Read-only in the strongest sense — nothing here retrains, refits or rescores
anything. Every number it uses (`forecast`, `direction_agreement`) is already
sitting in the `decisions` row the screener wrote; the panel only asks "which of
these would still have made the cut". That is what makes it safe to recompute on
every slider tick.

Kept apart from monitoring/dashboard/server.py for the same reason as picks.py:
the filtering rule is the part worth testing, and it shouldn't need an HTTP
server or a database to run.
"""
from __future__ import annotations

import pandas as pd

from monitoring.dashboard.picks import COL_SYMBOL, latest_picks_table

COL_AGREEMENT = "Model agreement"

# The floors the sliders bottom out at. Agreement can't fall below 0.5 by
# construction (it's the share of the ensemble on the majority side, so a dead
# split is the worst case), and a minimum move of 0 asks nothing of the forecast.
AGREEMENT_FLOOR = 0.5
MOVE_FLOOR = 0.0

DEFAULT_MIN_AGREEMENT = 0.8
DEFAULT_MIN_ABS_MOVE = 0.0


def _passes(values: pd.Series, threshold: float, floor: float) -> pd.Series:
    """
    Which rows clear `threshold`, with one rule for missing values: a pick whose
    number was never recorded passes only while the slider sits at its floor —
    i.e. while the slider isn't filtering at all. The moment the reader asks for
    a real bar, a pick that can't be shown to clear it drops out. Silently
    keeping unrecorded picks would overstate the shortlist; silently dropping
    them even at the floor would make an untouched slider look like a filter.
    """
    numeric = pd.to_numeric(values, errors="coerce")
    if threshold <= floor:
        return pd.Series(True, index=numeric.index)
    return numeric >= threshold


def filter_by_thresholds(
    batch: pd.DataFrame,
    min_agreement: float = DEFAULT_MIN_AGREEMENT,
    min_abs_move: float = DEFAULT_MIN_ABS_MOVE,
) -> pd.DataFrame:
    """
    The subset of one screener run's rows that would still be shortlisted under
    a stricter (or looser) pair of thresholds.

    `min_abs_move` is compared against the *size* of the predicted move, not its
    sign — a forecast of -3% is as strong a signal as +3%, it just points the
    other way, and the direction is already carried by the Long/Short column.
    """
    if batch.empty:
        return batch
    keep = _passes(batch["direction_agreement"], min_agreement, AGREEMENT_FLOOR) & _passes(
        pd.to_numeric(batch["forecast"], errors="coerce").abs(), min_abs_move, MOVE_FLOOR
    )
    return batch.loc[keep].copy()


def whatif_table(filtered: pd.DataFrame) -> pd.DataFrame:
    """
    The surviving picks as a display table — the same shape as the Latest picks
    table, plus the agreement figure, since agreement is half of what the reader
    is dragging a slider over and hiding it would make the filter look arbitrary.
    """
    table = latest_picks_table(filtered)
    if table.empty:
        table[COL_AGREEMENT] = pd.Series(dtype=float)
        return table

    # A symbol can in principle appear twice in one batch; take the first, the
    # same way app.py's "Why this pick?" expander does.
    agreement = (
        pd.to_numeric(filtered["direction_agreement"], errors="coerce")
        .groupby(filtered["symbol"].astype(str))
        .first()
    )
    table[COL_AGREEMENT] = table[COL_SYMBOL].map(agreement)
    return table


def _picks(n: int) -> str:
    return "1 pick" if n == 1 else f"{n} picks"


def shortlist_summary(n_before: int, n_after: int) -> str:
    """
    The one-line count under the sliders: "9 picks → 4 picks at these settings".
    Says outright when the sliders aren't doing anything, so an untouched
    playground doesn't read as a filter that happens to agree with the screener.
    """
    if n_before == 0:
        return "No picks in the latest run to filter."
    if n_after == 0:
        return f"{_picks(n_before)} → no picks at these settings"
    if n_after == n_before:
        return f"{_picks(n_before)} → {_picks(n_after)} at these settings — nothing is filtered out"
    return f"{_picks(n_before)} → {_picks(n_after)} at these settings"
