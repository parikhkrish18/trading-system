"""
Leveraged-ETF daily-reset decay simulator.

Per the plan (Phase 4, point 2): "since these products reset leverage daily,
your backtest must simulate the actual daily compounding, not just multiply
the underlying's return by 3. This is the single most common modeling
mistake with these instruments."

The mechanism: a 3x fund targets 3x the underlying's return EACH DAY, then
resets. Compounding a fixed daily multiple is NOT the same as the multiple
of the compounded underlying return over any period longer than one day —
the difference (volatility decay / beta-slippage) grows with both realized
volatility and holding period, and is worst in choppy, mean-reverting
markets (exactly the environment the regime classifier exists to detect).

    WRONG:  lev_cum_return_over_N_days = 3 * underlying_cum_return_over_N_days
    RIGHT:  lev_cum_return_over_N_days = product_over_days(1 + 3*r_t) - 1

This module also nets out the fund's expense ratio and an approximate
financing cost for the leveraged notional (funds achieve leverage largely
through swaps/futures, which carry an implicit financing cost roughly
tracking a short-term rate) — both are real, continuous drags independent
of volatility decay, and backtests that ignore them will look too good.
"""
from __future__ import annotations

import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def underlying_daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def daily_cost_drag(
    leverage: float,
    expense_ratio_annual: float = 0.0095,
    financing_rate_annual: float = 0.0,
) -> float:
    """
    Approximate daily cost drag from (a) the fund's stated expense ratio and
    (b) financing cost on the leveraged notional beyond 1x exposure. Both are
    real 3x-ETF prospectus / total-cost-of-ownership considerations, kept
    here as explicit, overridable parameters rather than hardcoded constants.
    """
    financing_drag_annual = max(leverage - 1.0, 0.0) * financing_rate_annual
    return (expense_ratio_annual + financing_drag_annual) / TRADING_DAYS_PER_YEAR


def simulate_leveraged_daily_returns(
    underlying_returns: pd.Series,
    leverage: float = 3.0,
    expense_ratio_annual: float = 0.0095,
    financing_rate_annual: float = 0.0,
) -> pd.Series:
    """
    The correct daily return series for a daily-reset leveraged ETF, given
    the underlying's daily returns. This is what should feed the backtest's
    bar-by-bar equity update — never the naive "multiply the total return"
    shortcut.
    """
    drag = daily_cost_drag(leverage, expense_ratio_annual, financing_rate_annual)
    return leverage * underlying_returns - drag


def compound(daily_returns: pd.Series, start_value: float = 1.0) -> pd.Series:
    """Cumulative growth-of-$1 (or growth-of-start_value) series from daily returns."""
    return start_value * (1.0 + daily_returns.fillna(0)).cumprod()


def naive_leveraged_cumulative_return(underlying_returns: pd.Series, leverage: float = 3.0) -> pd.Series:
    """
    THE WRONG WAY — included only so you can quantify how wrong it is, and
    so a regression test can assert the real simulator diverges from it.
    Multiplies the *cumulative* underlying return by the leverage factor,
    ignoring daily-reset compounding entirely.
    """
    underlying_cum_return = compound(underlying_returns) - 1.0
    return leverage * underlying_cum_return


def decay_report(
    close: pd.Series,
    leverage: float = 3.0,
    expense_ratio_annual: float = 0.0095,
    financing_rate_annual: float = 0.0,
) -> pd.DataFrame:
    """
    Given an underlying's close price series, returns a comparison table:
    correctly-compounded leveraged path vs. the naive (wrong) shortcut, so
    the size of the modeling error is visible and reviewable, not implicit.
    """
    u_ret = underlying_daily_returns(close)
    lev_ret = simulate_leveraged_daily_returns(u_ret, leverage, expense_ratio_annual, financing_rate_annual)

    correct_growth = compound(lev_ret)
    naive_cum_return = naive_leveraged_cumulative_return(u_ret, leverage)

    return pd.DataFrame(
        {
            "close": close,
            "underlying_daily_return": u_ret,
            "leveraged_daily_return": lev_ret,
            "correct_leveraged_growth": correct_growth,
            "correct_leveraged_cum_return": correct_growth - 1.0,
            "naive_leveraged_cum_return": naive_cum_return,
            "decay_gap": naive_cum_return - (correct_growth - 1.0),
        }
    )
