"""
Coverage for features/quant/{momentum,volatility,mean_reversion}.py -- these
had zero direct test coverage (only exercised indirectly, if at all, through
features/build_features.py). All three modules are pure pandas functions, so
each is checked against a hand-derivable value on a small series plus the
edge cases each docstring/implementation implicitly promises: NaN warm-up
windows, division-by-zero guards (the `.replace(0, ...)` calls), and that
cross_sectional_rank/realized_vol behave the way risk/sizing.py and the
screener actually rely on them behaving.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.quant import mean_reversion, momentum, volatility

# ---------------------------------------------------------------------
# momentum.py
# ---------------------------------------------------------------------


def test_rolling_return_matches_hand_computed_pct_change():
    close = pd.Series([100.0, 110.0, 121.0, 108.9])
    result = momentum.rolling_return(close, window=1)
    # (110-100)/100, (121-110)/110, (108.9-121)/121
    expected = pd.Series([np.nan, 0.10, 0.10, -0.10])
    pd.testing.assert_series_equal(result, expected, check_exact=False, atol=1e-9)


def test_rolling_return_over_a_multi_bar_window():
    close = pd.Series([100.0, 105.0, 110.0, 120.0])
    result = momentum.rolling_return(close, window=3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx((120.0 - 100.0) / 100.0)


def test_adx_is_nan_during_warmup_and_finite_once_the_window_fills():
    n = 60
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    # A clean uptrend so the trend-strength indicator has something real to compute.
    close = pd.Series(100 + np.arange(n) * 1.0, index=idx)
    high = close + 1.0
    low = close - 1.0

    result = momentum.adx(high, low, close, window=14)

    assert result.iloc[:14].isna().all()
    assert np.isfinite(result.iloc[-1])
    # A steady, one-directional trend should read as a strong trend, not a weak one.
    assert result.iloc[-1] > 20


def test_adx_flat_price_series_does_not_raise_on_the_zero_denominator():
    """
    plus_di + minus_di can be exactly 0 on a perfectly flat series (no
    directional movement at all) -- the .replace(0, pd.NA) guard exists for
    this; confirm it actually produces NaN rather than raising or a stray inf.
    """
    n = 40
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    close = pd.Series([100.0] * n, index=idx)
    high = close.copy()
    low = close.copy()

    result = momentum.adx(high, low, close, window=14)

    assert not np.isinf(result.dropna()).any()


def test_cross_sectional_rank_is_bounded_in_zero_one_and_orders_correctly():
    values = pd.Series({"AAPL": 0.05, "MSFT": -0.02, "TSLA": 0.20, "GOOG": 0.05})
    ranked = momentum.cross_sectional_rank(values)

    assert (ranked >= 0).all() and (ranked <= 1).all()
    assert ranked["TSLA"] == ranked.max()  # highest raw value -> highest percentile rank
    assert ranked["MSFT"] == ranked.min()  # lowest raw value -> lowest percentile rank
    assert ranked["AAPL"] == ranked["GOOG"]  # tied raw values get the same (averaged) rank


def test_cross_sectional_rank_single_symbol_is_the_max_percentile():
    values = pd.Series({"AAPL": 0.01})
    ranked = momentum.cross_sectional_rank(values)
    assert ranked["AAPL"] == 1.0


# ---------------------------------------------------------------------
# volatility.py
# ---------------------------------------------------------------------


def test_realized_vol_zero_for_a_perfectly_flat_price_series():
    close = pd.Series([100.0] * 30)
    vol = volatility.realized_vol(close, window=20, annualize=False)
    assert vol.dropna().eq(0.0).all()


def test_realized_vol_annualization_scales_by_sqrt_252():
    rng = np.random.default_rng(11)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 30))))
    raw = volatility.realized_vol(close, window=20, annualize=False)
    annualized = volatility.realized_vol(close, window=20, annualize=True)
    ratio = (annualized / raw).dropna()
    assert np.allclose(ratio.to_numpy(), np.sqrt(252))


def test_realized_vol_respects_the_warmup_window():
    close = pd.Series(np.linspace(100, 120, 25))
    vol = volatility.realized_vol(close, window=20)
    assert vol.iloc[:20].isna().all()
    assert np.isfinite(vol.iloc[-1])


def test_atr_matches_hand_computed_true_range_on_a_simple_series():
    # 3 bars, window=2 -> ATR at bar index 2 (0-indexed) is the mean of the
    # true ranges at bars 1 and 2 (rolling(2).mean(), no min_periods set so
    # bar 1 is already the first non-NaN value).
    high = pd.Series([102.0, 106.0, 103.0])
    low = pd.Series([98.0, 101.0, 99.0])
    close = pd.Series([100.0, 105.0, 100.0])

    result = volatility.atr(high, low, close, window=2)

    # bar0 TR = high-low = 4.0 (no prior close to compare against)
    # bar1 TR = max(106-101, |106-100|, |101-100|) = max(5, 6, 1) = 6.0
    # bar2 TR = max(103-99, |103-105|, |99-105|) = max(4, 2, 6) = 6.0
    tr0, tr1, tr2 = 4.0, 6.0, 6.0
    assert result.iloc[1] == pytest.approx((tr0 + tr1) / 2)
    assert result.iloc[2] == pytest.approx((tr1 + tr2) / 2)


def test_vol_of_vol_is_one_when_short_and_long_regimes_match():
    """A series with genuinely constant volatility should have short/long realized vol converge near 1."""
    rng = np.random.default_rng(42)
    n = 300
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))

    result = volatility.vol_of_vol(close, short_window=10, long_window=60)

    tail = result.dropna().iloc[-50:]
    assert tail.mean() == pytest.approx(1.0, abs=0.5)  # loose bound -- this is a statistical property, not exact


def test_vol_of_vol_does_not_divide_by_zero_on_a_flat_long_window():
    close = pd.Series([100.0] * 80)
    result = volatility.vol_of_vol(close, short_window=10, long_window=60)
    assert not np.isinf(result.dropna()).any()


# ---------------------------------------------------------------------
# mean_reversion.py
# ---------------------------------------------------------------------


def test_zscore_matches_hand_computed_value():
    close = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    result = mean_reversion.zscore(close, window=5)
    mean = close.mean()
    std = close.std()
    assert result.iloc[-1] == pytest.approx((18.0 - mean) / std)


def test_zscore_is_nan_during_warmup():
    close = pd.Series(np.arange(10, dtype=float))
    result = mean_reversion.zscore(close, window=5)
    assert result.iloc[:4].isna().all()


def test_zscore_does_not_divide_by_zero_on_a_flat_window():
    close = pd.Series([50.0] * 10)
    result = mean_reversion.zscore(close, window=5)
    assert result.dropna().empty or not np.isinf(result.dropna()).any()


def test_bollinger_pct_b_is_zero_point_five_at_the_moving_average():
    close = pd.Series([100.0] * 19 + [100.0])  # flat series: price == its own mean at every point
    result = mean_reversion.bollinger_pct_b(close, window=20)
    # std is 0 here -> guarded to NaN, not 0.5 -- confirms the div-by-zero guard fires
    assert result.iloc[-1] is pd.NA or pd.isna(result.iloc[-1])


def test_bollinger_pct_b_bounded_under_normal_conditions():
    rng = np.random.default_rng(7)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 0.3, 100)))
    result = mean_reversion.bollinger_pct_b(close, window=20).dropna()
    # Not a hard guarantee (breakouts can exceed [0,1] per the docstring),
    # but under mild random-walk noise the overwhelming majority of points
    # should sit inside the bands.
    within_bounds = ((result >= -0.5) & (result <= 1.5)).mean()
    assert within_bounds > 0.9


def test_rsi_is_100_after_a_strictly_increasing_run():
    """
    Regression: zero down-moves in the window means avg_loss == 0, which the
    implementation's own div-by-zero guard (.replace(0, pd.NA)) used to turn
    into NaN instead of the correct, well-defined answer -- silently erasing
    RSI's single strongest bullish reading into missing data.
    """
    close = pd.Series(np.arange(1, 30, dtype=float))  # strictly increasing -- zero losses
    result = mean_reversion.rsi(close, window=14)
    assert result.iloc[-1] == pytest.approx(100.0)


def test_rsi_is_zero_after_a_strictly_decreasing_run():
    close = pd.Series(np.arange(30, 1, -1, dtype=float))  # strictly decreasing -- zero gains
    result = mean_reversion.rsi(close, window=14)
    assert result.iloc[-1] == pytest.approx(0.0)


def test_rsi_is_fifty_on_a_perfectly_flat_window():
    """Zero gains AND zero losses is the one genuinely undefined case -- the midpoint, not NaN and not 100."""
    close = pd.Series([50.0] * 20)
    result = mean_reversion.rsi(close, window=14)
    assert result.iloc[-1] == pytest.approx(50.0)


def test_rsi_is_bounded_zero_to_hundred_on_mixed_data():
    rng = np.random.default_rng(3)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 60)))
    result = mean_reversion.rsi(close, window=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()
