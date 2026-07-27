"""Realized volatility features — also feeds risk/sizing.py directly."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def realized_vol(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling realized volatility from close-to-close log returns."""
    log_ret = np.log(close / close.shift(1))
    vol = log_ret.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(TRADING_DAYS_PER_YEAR)
    return vol


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range — non-directional volatility, useful for sizing."""
    tr = pd.concat(
        [
            (high - low),
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def vol_of_vol(close: pd.Series, short_window: int = 10, long_window: int = 60) -> pd.Series:
    """Ratio of short-term to long-term realized vol — flags regime shifts in vol itself."""
    short_vol = realized_vol(close, short_window, annualize=False)
    long_vol = realized_vol(close, long_window, annualize=False)
    return short_vol / long_vol.replace(0, np.nan)
