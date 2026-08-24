"""
The do-nothing baseline (models/evaluation.py).

Every one of these tests exists because the harness previously reported a
return with nothing to compare it to, and a +0.16%/trade result that was
actually losing to a +0.32% market read as a success for months.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.evaluation import (
    benchmark_return,
    cross_sectional_excess,
    cross_sectional_zscore,
    describe_book,
    production_book_mask,
    trade_metrics,
)

# --------------------------------------------------------------------------
# benchmark_return
# --------------------------------------------------------------------------


def test_benchmark_is_the_equal_weight_mean_of_every_candidate_row():
    assert benchmark_return([0.01, 0.02, 0.03]) == pytest.approx(0.02)


def test_benchmark_ignores_missing_rows_rather_than_treating_them_as_zero():
    assert benchmark_return([0.02, float("nan"), 0.04]) == pytest.approx(0.03)


def test_benchmark_of_nothing_is_nan_not_zero():
    """Zero would silently read as 'the market went nowhere', which is a claim."""
    assert np.isnan(benchmark_return([]))


# --------------------------------------------------------------------------
# trade_metrics — the headline
# --------------------------------------------------------------------------


def test_excess_return_is_model_net_minus_benchmark():
    preds = np.array([1.0, 1.0, 1.0, 1.0])
    realized = np.array([0.01, 0.02, 0.03, 0.04])
    universe = np.array([0.01, 0.02, 0.03, 0.04, 0.10, 0.10])

    m = trade_metrics(preds, realized, universe, cost=0.0002)

    assert m["benchmark_return"] == pytest.approx(universe.mean())
    assert m["model_return_gross"] == pytest.approx(0.025)
    assert m["model_return_net"] == pytest.approx(0.025 - 0.0002)
    assert m["excess_return"] == pytest.approx(m["model_return_net"] - m["benchmark_return"])


def test_a_profitable_model_that_trails_the_market_reports_negative_excess():
    """
    This project's actual result, in miniature: the model made money and
    still lost to buying everything.
    """
    m = trade_metrics(
        predictions=np.ones(3),
        realized_returns=np.array([0.001, 0.002, 0.001]),
        universe_returns=np.array([0.001, 0.002, 0.001, 0.02, 0.02, 0.02]),
        cost=0.0002,
    )
    assert m["model_return_net"] > 0
    assert m["excess_return"] < 0


def test_shorts_earn_the_negative_of_the_realized_return():
    m = trade_metrics(
        predictions=np.array([-1.0, -1.0]),
        realized_returns=np.array([-0.05, 0.01]),
        universe_returns=np.array([-0.05, 0.01]),
        cost=0.0,
    )
    # Short a stock that fell 5% -> +5%; short one that rose 1% -> -1%.
    assert m["model_return_gross"] == pytest.approx(0.02)
    assert m["n_short"] == 2 and m["n_long"] == 0
    assert m["pct_long"] == pytest.approx(0.0)


def test_long_short_split_separates_two_very_different_populations():
    """
    Aggregate +0.1% hid longs at +0.225% and shorts at -1.069%. The split is
    the whole reason shorts were caught at all.
    """
    preds = np.array([1.0, 1.0, 1.0, -1.0])
    realized = np.array([0.02, 0.03, 0.01, 0.04])  # the short is against a riser
    m = trade_metrics(preds, realized, realized, cost=0.0)

    assert m["n_long"] == 3 and m["n_short"] == 1
    assert m["pct_long"] == pytest.approx(0.75)
    assert m["long_return_net"] == pytest.approx(0.02)
    assert m["long_win_rate"] == pytest.approx(1.0)
    assert m["short_return_net"] == pytest.approx(-0.04)
    assert m["short_win_rate"] == pytest.approx(0.0)


def test_win_rate_is_measured_after_costs():
    """A trade that gained less than the round trip cost did not win."""
    m = trade_metrics(
        predictions=np.array([1.0, 1.0]),
        realized_returns=np.array([0.0001, 0.05]),  # first gain < 2bp cost
        universe_returns=np.array([0.0001, 0.05]),
        cost=0.0002,
    )
    assert m["long_win_rate"] == pytest.approx(0.5)


def test_a_side_with_no_trades_reports_nan_not_a_fake_zero():
    m = trade_metrics(np.ones(3), np.full(3, 0.01), np.full(3, 0.01), cost=0.0)
    assert m["n_short"] == 0
    assert np.isnan(m["short_return_net"])
    assert np.isnan(m["short_win_rate"])


def test_a_zero_prediction_is_treated_as_long_matching_the_screener():
    """models/screener.py: side = "long" if forecast >= 0 else "short"."""
    m = trade_metrics(np.array([0.0]), np.array([0.01]), np.array([0.01]), cost=0.0)
    assert m["n_long"] == 1 and m["n_short"] == 0


def test_no_trades_still_reports_the_benchmark():
    """A fold where the filter kept nothing still tells you what the market did."""
    m = trade_metrics(np.array([]), np.array([]), np.array([0.01, 0.03]), cost=0.0002)
    assert m["n_trades"] == 0
    assert m["benchmark_return"] == pytest.approx(0.02)
    assert np.isnan(m["excess_return"])


def test_benchmark_is_gross_while_the_model_pays_costs():
    """
    The baseline is charged nothing on purpose — buy-and-hold pays the spread
    once, a trader pays it every round trip. Charging the benchmark nothing
    is the harder test to pass.
    """
    same = np.array([0.01, 0.01])
    m = trade_metrics(np.ones(2), same, same, cost=0.0002)
    assert m["benchmark_return"] == pytest.approx(0.01)
    assert m["excess_return"] == pytest.approx(-0.0002)


# --------------------------------------------------------------------------
# cross-sectional transforms (TARGET_MODE=relative)
# --------------------------------------------------------------------------


def test_excess_label_removes_the_market_move_on_each_date():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-03"] * 3),
            "fwd_return": [0.01, 0.02, 0.03, 0.11, 0.12, 0.13],
        }
    )
    excess = cross_sectional_excess(df)
    # Day two is a +10% market shift of day one; the labels are identical.
    assert list(excess[:3]) == pytest.approx(list(excess[3:]))
    assert excess.groupby(df["ts"]).mean().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_excess_label_is_zero_when_every_stock_moves_together():
    """A pure market move contains no stock-picking information at all."""
    df = pd.DataFrame(
        {"ts": pd.to_datetime(["2025-01-02"] * 4), "fwd_return": [0.05] * 4}
    )
    assert cross_sectional_excess(df).abs().max() == pytest.approx(0.0)


def test_zscore_ranks_within_a_date_not_across_dates():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-02"] * 3 + ["2025-01-03"] * 3),
            "rsi": [10.0, 20.0, 30.0, 110.0, 120.0, 130.0],
        }
    )
    out = cross_sectional_zscore(df, ["rsi"])
    # The second date is uniformly 100 higher; relative position is identical.
    assert list(out["rsi"][:3]) == pytest.approx(list(out["rsi"][3:]))
    assert out.groupby("ts")["rsi"].mean().abs().max() == pytest.approx(0.0, abs=1e-12)


def test_zscore_of_a_constant_feature_is_zero_not_infinite():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-02"] * 3), "flat": [7.0, 7.0, 7.0]})
    out = cross_sectional_zscore(df, ["flat"])
    assert list(out["flat"]) == [0.0, 0.0, 0.0]


def test_zscore_of_a_single_symbol_on_a_date_is_zero_not_nan():
    df = pd.DataFrame({"ts": pd.to_datetime(["2025-01-02"]), "x": [3.0]})
    assert list(cross_sectional_zscore(df, ["x"])["x"]) == [0.0]


def test_zscore_leaves_non_feature_columns_untouched():
    df = pd.DataFrame(
        {
            "ts": pd.to_datetime(["2025-01-02"] * 2),
            "symbol": ["AAPL", "MSFT"],
            "close": [100.0, 200.0],
            "rsi": [30.0, 70.0],
        }
    )
    out = cross_sectional_zscore(df, ["rsi"])
    assert list(out["close"]) == [100.0, 200.0]
    assert list(out["symbol"]) == ["AAPL", "MSFT"]


# --------------------------------------------------------------------------
# The harness must measure the book that actually trades
# --------------------------------------------------------------------------


def _mask(preds, dates=None, cost=0.0, allow_shorts=True, top_k=100):
    dates = dates if dates is not None else ["2026-01-01"] * len(preds)
    return production_book_mask(
        np.array(preds, dtype=float), np.array(dates), cost=cost, allow_shorts=allow_shorts, top_k=top_k
    )


def test_shorts_are_dropped_when_production_forbids_them():
    """
    The reason this exists. Production runs ALLOW_SHORTS=false, but the
    harness scored both sides, so ~65% of its picks were trades that could
    never have been placed and the headline described a book nobody runs.
    """
    kept = _mask([0.05, -0.05, 0.03, -0.03], allow_shorts=False)

    assert list(kept) == [True, False, True, False]


def test_shorts_are_kept_when_production_allows_them():
    kept = _mask([0.05, -0.05], allow_shorts=True)

    assert list(kept) == [True, True]


def test_a_zero_prediction_counts_as_long_not_as_a_dropped_short():
    """
    sign(0) is treated as long everywhere else. If it were dropped here and
    traded there, the measurement would drift from the system again.
    """
    assert list(_mask([0.0], allow_shorts=False)) == [True]


def test_moves_smaller_than_the_round_trip_cost_are_not_tradeable():
    """A predicted move that doesn't cover the cost is a guaranteed loser."""
    kept = _mask([0.05, 0.001], cost=0.01)

    assert list(kept) == [True, False]


def test_only_the_top_k_are_traded_on_any_given_date():
    kept = _mask([0.09, 0.07, 0.05, 0.03], top_k=2)

    assert list(kept) == [True, True, False, False]


def test_top_k_is_applied_per_date_not_across_the_whole_window():
    """
    The screener picks a fresh book each run. Ranking the whole test window
    at once would let a strong March pick crowd out every April pick, which
    is not a book anyone could have held.
    """
    preds = [0.09, 0.08, 0.02, 0.01]
    dates = ["2026-01-01", "2026-01-01", "2026-02-01", "2026-02-01"]

    kept = _mask(preds, dates=dates, top_k=1)

    assert list(kept) == [True, False, True, False]


def test_ranking_uses_the_size_of_the_move_regardless_of_direction():
    """A -6% forecast is a stronger call than a +2% one."""
    kept = _mask([0.02, -0.06], top_k=1, allow_shorts=True)

    assert list(kept) == [False, True]


def test_the_cost_floor_applies_before_top_k_not_after():
    """
    Otherwise a thin day fills the book with picks that cannot pay for
    themselves, purely because nothing better was available.
    """
    kept = _mask([0.05, 0.001, 0.0005], cost=0.01, top_k=3)

    assert list(kept) == [True, False, False]


def test_an_empty_window_produces_an_empty_mask():
    assert len(_mask([])) == 0


def test_describe_book_states_which_system_was_measured():
    """
    Two runs of this harness can disagree purely by measuring different
    systems, and nothing in the numbers themselves would say which.
    """
    long_only = describe_book(allow_shorts=False, strategy_mode="diversified", top_k=10, cost=0.004)
    both = describe_book(allow_shorts=True, strategy_mode="concentrated", top_k=2, cost=0.004)

    assert "long only" in long_only
    assert "top 10" in long_only
    assert "long and short" in both
    assert "concentrated" in both
