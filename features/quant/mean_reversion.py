"""Mean-reversion / z-score features."""
from __future__ import annotations

import pandas as pd


def zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Standard rolling z-score of price vs. its own trailing mean."""
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std.replace(0, pd.NA)


def bollinger_pct_b(close: pd.Series, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """
    %b: where price sits within its Bollinger Band, in [0, 1] under normal
    conditions (can exceed the range during a breakout — that's meaningful,
    not a bug).
    """
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    return (close - lower) / (upper - lower).replace(0, pd.NA)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Standard RSI, in [0, 100]."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))
