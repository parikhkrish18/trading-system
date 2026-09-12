"""Rolling momentum / trend-strength features, computed per symbol."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_return(close: pd.Series, window: int) -> pd.Series:
    """Simple total return over the trailing `window` bars."""
    return close.pct_change(periods=window)


def trend_pullback_score(close: pd.Series, trend_window: int = 20, pullback_window: int = 5) -> pd.Series:
    """
    Positive when a genuine uptrend (the trend_window return, sized against
    its own volatility) is undergoing a short pullback -- an uptrend-with-
    a-pullback continuation setup, the textbook "buy the dip" read.
    Negative for the mirror case, a downtrend with a bounce. Zero whenever
    the short-term move does NOT oppose the trend -- straight continuation
    without a pullback/bounce is already covered by the plain rolling-
    return features, and isn't the pattern this is meant to flag.

    "Heavy" is read relative to the stock's own day-to-day volatility, not
    a raw return threshold: trend_ret is divided by the trading-day-scale
    move its own daily volatility would predict over trend_window days, so
    a 10% move in a normally-sleepy utility counts as "heavier" than the
    same 10% in a stock that swings that much most weeks. (Day-to-day
    volume isn't folded in here -- no feature in this module reads volume
    at all yet, and volatility already captures the day-to-day-moves half
    of what "heavy" means; a volume-confirmation term would be a separate,
    larger addition.)
    """
    trend_ret = close.pct_change(periods=trend_window)
    pullback_ret = close.pct_change(periods=pullback_window)
    daily_vol = np.log(close / close.shift(1)).rolling(trend_window).std()
    expected_move = daily_vol * np.sqrt(trend_window)
    risk_adj_trend = trend_ret / expected_move.replace(0, np.nan)

    opposes = np.sign(pullback_ret) == -np.sign(risk_adj_trend)
    return risk_adj_trend.where(opposes, 0.0)


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
