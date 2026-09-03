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
    result = 100 - (100 / (1 + rs))
    # avg_loss == 0 isn't "no data" (that's the window guard above via
    # min_periods) -- inside a filled window it means zero down-moves,
    # i.e. the strongest possible bullish reading. The .replace(0, pd.NA)
    # guard above exists only to avoid a division by zero and ends up
    # producing NaN here instead of 100, silently dropping RSI's single
    # most informative case (a strict uptrend) into missing data -- which
    # is exactly the shape of bug models/screener.py and risk/sizing.py
    # both guard against elsewhere (a NaN quietly bypassing a downstream
    # filter rather than being a real, usable value). Fix it at the
    # source: 100 when there were gains and zero losses, 50 (undefined
    # direction) only when both are zero.
    result = result.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    result = result.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return result
