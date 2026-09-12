"""Donchian channel features -- where price sits within its own recent trading range."""
from __future__ import annotations

import numpy as np
import pandas as pd


def donchian_pct(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    Where today's close sits within the trailing `window`-bar high/low range:
    0.0 at the rolling low (support), 1.0 at the rolling high (resistance).
    Includes today's own bar in the range, same as a Donchian channel is
    normally plotted -- this is a continuous "where in the range" read, not
    a breakout event (see donchian_breakout for that). NaN whenever the
    range is flat (upper == lower) rather than a division-by-zero 0 or 1
    that would misread "no range at all" as "sitting at an edge".
    """
    upper = high.rolling(window).max()
    lower = low.rolling(window).min()
    span = upper - lower
    return ((close - lower) / span.replace(0, np.nan)).clip(0.0, 1.0)


def donchian_breakout(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
    """
    +1.0 when today's close clears the prior `window`-bar high (breaking
    resistance), -1.0 when it breaks the prior `window`-bar low (breaking
    support), 0.0 otherwise.

    The prior range is shifted back one bar before today, deliberately: a
    rolling max/min that INCLUDES today's own bar would make "did today
    break out" almost meaningless, since today's own high/low can only ever
    help set that max/min, not clear it. Shifting by one bar compares
    today's close against the range that existed before today could extend
    it -- the standard breakout-system definition (Turtle Trading: buy above
    the highest high of the preceding N days, N excluding today).
    """
    prior_upper = high.shift(1).rolling(window).max()
    prior_lower = low.shift(1).rolling(window).min()
    return pd.Series(
        np.select([close >= prior_upper, close <= prior_lower], [1.0, -1.0], default=0.0),
        index=close.index,
    ).where(prior_upper.notna() & prior_lower.notna())
