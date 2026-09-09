"""Rolling momentum / trend-strength features, computed per symbol."""
from __future__ import annotations

import pandas as pd


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """Simple total return over the trailing `window` bars."""
    return close.pct_change(periods=window)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average Directional Index — a standard trend-strength indicator.
    High ADX = strong trend (either direction); low ADX = choppy/range-bound.
    This is also the base signal the regime classifier stub uses.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
    minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / window, min_periods=window).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / window, min_periods=window).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    # plus_di + minus_di == 0 only when both are exactly zero (both are
    # non-negative sums), i.e. genuinely zero directional movement in
    # either direction over the whole smoothing window (a flat/stale price
    # series) -- the strongest possible non-trending reading, not missing
    # data. Same class of bug as RSI's 0/0 case (mean_reversion.py::rsi):
    # the .replace(0, pd.NA) guard above exists only to avoid a division
    # by zero and ends up turning this into NaN instead of the correct
    # DX=0. Unlike RSI's undefined-direction 50, there's no ambiguity here
    # -- no movement at all means no directional dominance, unambiguously.
    dx = dx.mask((plus_di == 0) & (minus_di == 0), 0.0)
    return dx.ewm(alpha=1 / window, min_periods=window).mean()


def cross_sectional_rank(values_by_symbol: pd.Series) -> pd.Series:
    """
    Rank a single day's cross-section of symbols (e.g. momentum scores) into
    [0, 1]. Call this once per date on a symbol-indexed slice, not on the
    full panel at once.
    """
    return values_by_symbol.rank(pct=True)
