import numpy as np
import pandas as pd
import pytest

from backtest.decay_sim import (
    compound,
    daily_cost_drag,
    decay_report,
    naive_leveraged_cumulative_return,
    simulate_leveraged_daily_returns,
    underlying_daily_returns,
)


def _choppy_price_series(n=252, seed=0):
    """Mean-reverting (choppy) synthetic price path — where decay bites hardest."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, n)
    # Alternate sign every ~5 days to force chop rather than trend.
    chop = np.array([0.01 if (i // 5) % 2 == 0 else -0.01 for i in range(n)])
    returns = noise + chop
    price = 100 * np.cumprod(1 + returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(price, index=idx)


def _trending_price_series(n=252, seed=0):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0008, 0.01, n)  # steady upward drift
    price = 100 * np.cumprod(1 + returns)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(price, index=idx)


def test_naive_and_correct_are_identical_for_a_single_day():
    """Over exactly one day, daily-reset compounding and the naive shortcut must agree."""
    close = pd.Series([100.0, 103.0])
    u_ret = underlying_daily_returns(close)
    correct = compound(simulate_leveraged_daily_returns(u_ret, leverage=3.0, expense_ratio_annual=0.0))
    naive = 1 + naive_leveraged_cumulative_return(u_ret, leverage=3.0)
    # Only the second element is a real one-day return (first is NaN from pct_change).
    assert correct.iloc[1] == pytest.approx(naive.iloc[1], rel=1e-9)


def test_decay_diverges_in_choppy_market():
    """
    Core assertion: in a choppy/mean-reverting market, the correctly
    compounded leveraged path must underperform the naive (wrong) shortcut.
    This is the exact failure mode the plan warns about.
    """
    close = _choppy_price_series()
    report = decay_report(close, leverage=3.0, expense_ratio_annual=0.0, financing_rate_annual=0.0)
    final_correct = report["correct_leveraged_cum_return"].iloc[-1]
    final_naive = report["naive_leveraged_cum_return"].iloc[-1]

    assert final_correct < final_naive, (
        "Expected daily-reset compounding to underperform the naive 3x-multiply "
        "shortcut in a choppy market — decay should make the correct path worse."
    )
    # And the gap should be economically meaningful, not a rounding artifact.
    assert (final_naive - final_correct) > 0.02


def test_naive_shortcut_overstates_return_or_matches_in_trend():
    """
    In a strongly trending market, the naive shortcut should still be >= the
    correctly compounded path (decay drag never *helps* the naive number
    look worse than reality) — the sign of the gap shouldn't flip.
    """
    close = _trending_price_series()
    report = decay_report(close, leverage=3.0, expense_ratio_annual=0.0, financing_rate_annual=0.0)
    assert report["decay_gap"].iloc[-1] >= -1e-9


def test_daily_cost_drag_scales_with_leverage_and_rates():
    drag_1x = daily_cost_drag(leverage=1.0, expense_ratio_annual=0.01, financing_rate_annual=0.05)
    drag_3x = daily_cost_drag(leverage=3.0, expense_ratio_annual=0.01, financing_rate_annual=0.05)
    assert drag_3x > drag_1x, "Financing drag on the extra 2x notional should increase total daily cost."


def test_expense_ratio_reduces_growth_even_with_zero_underlying_return():
    flat_close = pd.Series([100.0] * 30)
    u_ret = underlying_daily_returns(flat_close)
    lev_ret = simulate_leveraged_daily_returns(u_ret, leverage=3.0, expense_ratio_annual=0.0095)
    growth = compound(lev_ret)
    assert growth.iloc[-1] < 1.0, "A flat underlying should still show a fund value decline from expense drag."
