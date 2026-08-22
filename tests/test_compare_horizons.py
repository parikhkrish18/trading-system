"""
The horizon comparison harness (scripts/compare_horizons.py) plus the
TARGET_HORIZON_DAYS config wiring: the horizon must be a setting the whole
pipeline actually reads, not a hardcoded 5 scattered around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.compare_horizons import (
    comparison_table,
    excess_vs_zero,
    paired_test,
    summarize_horizon,
)


def _fold_frame(excess_returns, accuracies=None, pct_confident=0.3, benchmark=0.003):
    """
    A horizon's per-fold results. `excess_returns` is the headline metric
    (model net minus benchmark); model_return_net is derived from it so the
    three columns stay arithmetically consistent, the way a real fold's do.
    """
    n = len(excess_returns)
    accuracies = accuracies if accuracies is not None else [0.52] * n
    excess = pd.Series(excess_returns, dtype=float)
    return pd.DataFrame(
        {
            "fold_id": range(n),
            "directional_accuracy": accuracies,
            "directional_accuracy_when_confident": accuracies,
            "pct_rows_confident": [pct_confident] * n,
            "benchmark_return": [benchmark] * n,
            "model_return_net": excess + benchmark,
            "excess_return": excess,
            "pct_long": [0.95] * n,
            "long_return_net": [0.002] * n,
            "long_win_rate": [0.52] * n,
            "short_return_net": [-0.01] * n,
            "short_win_rate": [0.42] * n,
        }
    )


def test_summarize_horizon_reports_means_spread_and_folds_with_positive_excess():
    frame = _fold_frame([0.01, -0.01, 0.02, 0.03])

    summary = summarize_horizon(frame)

    assert summary["n_folds"] == 4
    assert summary["folds_positive_excess"] == 3
    assert summary["excess_return"] == pytest.approx(0.0125)
    assert summary["excess_return_std"] > 0


def test_summarize_horizon_always_carries_the_benchmark_beside_the_return():
    summary = summarize_horizon(_fold_frame([0.01, 0.02]))
    assert "benchmark_return" in summary
    assert "model_return_net" in summary
    assert "excess_return" in summary


def test_a_horizon_that_makes_money_but_trails_the_market_counts_zero_good_folds():
    """
    Every fold returns +0.1% against a +0.3% benchmark: profitable, and a
    total failure. folds_positive_excess must be 0.
    """
    frame = _fold_frame([-0.002] * 10, benchmark=0.003)

    summary = summarize_horizon(frame)

    assert summary["model_return_net"] > 0
    assert summary["folds_positive_excess"] == 0
    assert summary["excess_return"] < 0


def test_summarize_horizon_ignores_folds_with_no_confident_rows():
    frame = _fold_frame([0.01, float("nan"), 0.02])

    summary = summarize_horizon(frame)

    assert summary["excess_return"] == pytest.approx(0.015)
    assert summary["folds_positive_excess"] == 2


# --- excess vs zero ---------------------------------------------------------


def test_excess_vs_zero_detects_a_consistent_edge():
    result = excess_vs_zero(_fold_frame([0.01, 0.011, 0.009, 0.0105, 0.0095, 0.010]))
    assert result["mean"] > 0
    assert result["t_pvalue"] < 0.001


def test_excess_vs_zero_finds_nothing_in_noise():
    rng = np.random.default_rng(7)
    result = excess_vs_zero(_fold_frame(rng.normal(0.0, 0.01, size=10)))
    assert result["t_pvalue"] > 0.05


def test_excess_vs_zero_is_paired_by_fold_not_by_row():
    """
    Ten folds means ten observations — never the tens of thousands of
    overlapping rows behind them. Guards against someone 'improving' the
    test by feeding it row-level data.
    """
    result = excess_vs_zero(_fold_frame([0.001] * 10))
    assert result["n_folds"] == 10


def test_excess_vs_zero_needs_at_least_two_folds():
    result = excess_vs_zero(_fold_frame([0.01]))
    assert np.isnan(result["t_pvalue"])


def test_paired_test_pairs_by_fold_and_detects_a_consistent_edge():
    rng = np.random.default_rng(0)
    base_vals = rng.normal(0.0, 0.001, size=10)
    baseline = _fold_frame(base_vals)
    candidate = _fold_frame(base_vals + 0.01)  # every fold better by exactly 1%

    result = paired_test(baseline, candidate, "excess_return")

    assert result["n_pairs"] == 10
    assert result["mean_diff"] == pytest.approx(0.01)
    assert result["t_pvalue"] < 0.001
    assert result["wilcoxon_pvalue"] < 0.01


def test_paired_test_finds_no_signal_in_noise():
    rng = np.random.default_rng(1)
    baseline = _fold_frame(rng.normal(0.0, 0.01, size=10))
    candidate = _fold_frame(rng.normal(0.0, 0.01, size=10))

    result = paired_test(baseline, candidate, "excess_return")

    assert result["t_pvalue"] > 0.05


def test_paired_test_drops_folds_missing_on_either_side():
    baseline = _fold_frame([0.01, 0.02, float("nan"), 0.01])
    candidate = _fold_frame([0.02, float("nan"), 0.03, 0.02])

    result = paired_test(baseline, candidate, "excess_return")

    assert result["n_pairs"] == 2  # folds 0 and 3 only


def test_comparison_table_has_one_row_per_horizon_with_pvalues_vs_baseline():
    results = {
        5: _fold_frame([0.001] * 10),
        20: _fold_frame([0.004] * 10),
    }

    table = comparison_table(results, baseline_horizon=5)

    assert list(table["horizon_days"]) == [5, 20]
    row20 = table.loc[table["horizon_days"] == 20].iloc[0]
    assert row20["excess_diff_vs_5d"] == pytest.approx(0.003)
    assert "excess_pvalue" in table.columns
    baseline_row = table.loc[table["horizon_days"] == 5].iloc[0]
    assert pd.isna(baseline_row.get("excess_diff_vs_5d"))


def test_comparison_table_shows_every_horizon_against_doing_nothing():
    """
    The bug this change fixes: the old table let 40d look ten times better
    than 5d because both raw returns grew with the horizon. Every row must
    now carry its own benchmark and its own test against zero excess.
    """
    results = {
        5: _fold_frame([-0.002] * 10, benchmark=0.003),
        40: _fold_frame([-0.003] * 10, benchmark=0.023),
    }

    table = comparison_table(results, baseline_horizon=5)

    for col in ("benchmark_return", "model_return_net", "excess_return", "excess_p_vs_zero"):
        assert col in table.columns
    # Both horizons are profitable in raw terms and both lose to the market.
    assert (table["model_return_net"] > 0).all()
    assert (table["excess_return"] < 0).all()
    assert (table["folds_positive_excess"] == 0).all()


def test_comparison_table_reports_the_long_short_split():
    table = comparison_table({5: _fold_frame([0.001] * 10)}, baseline_horizon=5)
    for col in ("pct_long", "long_return_net", "long_win_rate", "short_return_net", "short_win_rate"):
        assert col in table.columns


# --- TARGET_HORIZON_DAYS wiring ---------------------------------------------


def test_run_walk_forward_defaults_to_the_configured_horizon(monkeypatch):
    from models import train

    monkeypatch.setattr(train.settings, "target_horizon_days", 7)
    captured = {}

    def _capture(feature_set_id, symbols, horizon):
        captured["horizon"] = horizon
        raise RuntimeError("stop here — only the horizon resolution is under test")

    monkeypatch.setattr(train, "load_training_frame", _capture)

    with pytest.raises(RuntimeError):
        train.run_walk_forward("v4", ["AAPL"])

    assert captured["horizon"] == 7


def test_run_screen_defaults_to_the_configured_horizon(monkeypatch):
    from models import screener

    monkeypatch.setattr(screener.settings, "target_horizon_days", 13)
    captured = {}

    def _capture(feature_set_id, symbols, horizon):
        captured["horizon"] = horizon
        raise RuntimeError("stop here — only the horizon resolution is under test")

    monkeypatch.setattr(screener, "load_training_frame", _capture)

    with pytest.raises(RuntimeError):
        screener.run_screen("v4", ["AAPL"])

    assert captured["horizon"] == 13


def test_explicit_horizon_still_wins_over_the_config(monkeypatch):
    from models import train

    monkeypatch.setattr(train.settings, "target_horizon_days", 20)
    captured = {}

    def _capture(feature_set_id, symbols, horizon):
        captured["horizon"] = horizon
        raise RuntimeError("stop")

    monkeypatch.setattr(train, "load_training_frame", _capture)

    with pytest.raises(RuntimeError):
        train.run_walk_forward("v4", ["AAPL"], target_horizon_days=5)

    assert captured["horizon"] == 5
