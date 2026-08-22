"""
The honest-evaluation pieces of models/train.py: the purge gap that stops
training labels from overlapping the test window, the per-fold spread
report, and the transaction-cost hurdle shared with the screener.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.cost_model import round_trip_cost_fraction
from models.screener import DEFAULT_MIN_ABS_RETURN
from models.train import headline_verdict, make_expanding_folds, purged_train_cutoff, spread_summary


def _dates(n: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.bdate_range("2025-01-01", periods=n))


# --------------------------------------------------------------------------
# purged_train_cutoff
# --------------------------------------------------------------------------


def test_purge_moves_the_cutoff_back_by_the_horizon():
    dates = _dates(100)
    train_end = dates[60]
    cutoff = purged_train_cutoff(dates, train_end, horizon_days=5)
    assert cutoff == dates[55]


def test_purge_of_zero_disables_nothing_is_cut():
    dates = _dates(100)
    assert purged_train_cutoff(dates, dates[60], horizon_days=0) == dates[60]


def test_purge_never_walks_off_the_start_of_history():
    dates = _dates(10)
    cutoff = purged_train_cutoff(dates, dates[3], horizon_days=8)
    assert cutoff == dates[0]


def test_purged_training_window_has_no_label_overlap_with_test():
    """
    The actual property that matters: with a 5-day horizon, every surviving
    training row's label (built from the price 5 trading days later) must be
    fully realized before the test window starts.
    """
    dates = _dates(100)
    horizon = 5
    df = pd.DataFrame({"ts": dates, "close": np.linspace(100, 120, len(dates))})
    df["fwd_return"] = df["close"].shift(-horizon) / df["close"] - 1

    folds = make_expanding_folds(dates, n_folds=4)
    for fold in folds:
        cutoff = purged_train_cutoff(dates, fold.train_end, horizon)
        train = df[(df["ts"] >= fold.train_start) & (df["ts"] < cutoff)]
        # The label of the last surviving training row matures at
        # ts + horizon trading days; that maturity date must precede
        # test_start (== train_end).
        last_ts = train["ts"].max()
        maturity_idx = dates.get_loc(last_ts) + horizon
        assert dates[maturity_idx] <= fold.train_end, (
            f"fold {fold.fold_id}: training label matures at {dates[maturity_idx]}, "
            f"inside the test window starting {fold.test_start}"
        )


def test_unpurged_training_window_does_overlap_proving_the_bug_existed():
    dates = _dates(100)
    horizon = 5
    folds = make_expanding_folds(dates, n_folds=4)
    fold = folds[0]
    # Without the purge, the last training row sits one day before train_end
    # and its label needs prices horizon days into the test window.
    last_unpurged = dates[dates.get_loc(fold.train_end) - 1]
    maturity = dates[dates.get_loc(last_unpurged) + horizon]
    assert maturity > fold.test_start


# --------------------------------------------------------------------------
# costs
# --------------------------------------------------------------------------


def test_round_trip_cost_is_twice_the_per_side_spread_floor():
    # No ADV data -> participation 0 -> per-side cost is base_spread_bps.
    assert round_trip_cost_fraction(base_spread_bps=1.0) == pytest.approx(2 / 10_000)
    assert round_trip_cost_fraction(base_spread_bps=5.0) == pytest.approx(10 / 10_000)


def test_round_trip_cost_grows_with_participation():
    small = round_trip_cost_fraction(trade_shares=1_000, avg_daily_volume=10_000_000)
    big = round_trip_cost_fraction(trade_shares=1_000_000, avg_daily_volume=10_000_000)
    assert big > small


def test_commission_is_included_per_share():
    free = round_trip_cost_fraction(price=100.0, commission_per_share=0.0)
    paid = round_trip_cost_fraction(price=100.0, commission_per_share=0.01)
    assert paid == pytest.approx(free + 2 * 0.01 / 100.0)


def test_screener_min_abs_return_defaults_to_the_cost_hurdle():
    """The screener and the eval harness must agree on what a trade costs."""
    assert DEFAULT_MIN_ABS_RETURN == pytest.approx(round_trip_cost_fraction())
    assert DEFAULT_MIN_ABS_RETURN > 0.0


# --------------------------------------------------------------------------
# spread reporting
# --------------------------------------------------------------------------


def test_spread_summary_reports_variability_not_just_the_mean():
    results = pd.DataFrame(
        {
            "directional_accuracy": [0.60, 0.40, 0.55],
            "directional_accuracy_when_confident": [0.7, float("nan"), 0.5],
            "model_return_net": [0.001, -0.002, 0.0005],
            "benchmark_return": [0.002, 0.001, 0.003],
            "excess_return": [-0.001, -0.003, -0.0025],
        }
    )
    text = spread_summary(results)
    assert "std" in text and "min" in text and "max" in text
    assert "directional_accuracy" in text
    assert "model_return_net" in text
    # NaN folds are dropped, not averaged in as zeros.
    assert "n_folds 2" in text


def test_spread_summary_never_shows_a_return_without_its_benchmark():
    """
    The whole point of the benchmark work: a reader must not be able to see
    what the model returned without seeing what doing nothing returned.
    """
    results = pd.DataFrame(
        {
            "directional_accuracy": [0.55, 0.52],
            "directional_accuracy_when_confident": [0.55, 0.52],
            "benchmark_return": [0.0032, 0.0028],
            "model_return_net": [0.0012, 0.0009],
            "excess_return": [-0.0020, -0.0019],
        }
    )
    text = spread_summary(results)
    assert "benchmark_return" in text
    assert "excess_return" in text
    assert text.index("benchmark_return") < text.index("model_return_net")


def test_spread_summary_handles_all_nan_columns():
    results = pd.DataFrame(
        {
            "directional_accuracy": [0.5],
            "directional_accuracy_when_confident": [float("nan")],
            "model_return_net": [float("nan")],
            "excess_return": [float("nan")],
        }
    )
    assert "no folds produced a value" in spread_summary(results)


def test_spread_summary_skips_columns_an_older_results_frame_lacks():
    results = pd.DataFrame({"directional_accuracy": [0.55, 0.51]})
    text = spread_summary(results)
    assert "directional_accuracy" in text
    assert "excess_return" not in text


# --------------------------------------------------------------------------
# headline verdict
# --------------------------------------------------------------------------


def test_headline_verdict_calls_a_profitable_but_lagging_model_a_loss():
    """
    The exact failure this project shipped for months: every fold made money
    and every fold trailed buy-and-hold. The verdict must say LOSES TO.
    """
    results = pd.DataFrame(
        {
            "model_return_net": [0.0012, 0.0009, 0.0015],
            "benchmark_return": [0.0032, 0.0028, 0.0035],
            "excess_return": [-0.0020, -0.0019, -0.0020],
            "pct_long": [0.95, 0.94, 0.96],
        }
    )
    text = headline_verdict(results)
    assert "LOSES TO" in text
    assert "folds with positive excess: 0/3" in text


def test_headline_verdict_credits_a_model_that_actually_beats_the_baseline():
    results = pd.DataFrame(
        {
            "model_return_net": [0.005, 0.004, 0.006],
            "benchmark_return": [0.002, 0.002, 0.002],
            "excess_return": [0.003, 0.002, 0.004],
            "pct_long": [0.5, 0.5, 0.5],
        }
    )
    text = headline_verdict(results)
    assert "BEATS" in text
    assert "folds with positive excess: 3/3" in text
