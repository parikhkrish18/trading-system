"""
The horizon comparison harness (scripts/compare_horizons.py) plus the
TARGET_HORIZON_DAYS config wiring: the horizon must be a setting the whole
pipeline actually reads, not a hardcoded 5 scattered around.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.compare_horizons import comparison_table, paired_test, summarize_horizon


def _fold_frame(net_returns, accuracies=None, pct_confident=0.3):
    n = len(net_returns)
    accuracies = accuracies if accuracies is not None else [0.52] * n
    return pd.DataFrame(
        {
            "fold_id": range(n),
            "directional_accuracy": accuracies,
            "directional_accuracy_when_confident": accuracies,
            "pct_rows_confident": [pct_confident] * n,
            "mean_return_confident_net": net_returns,
        }
    )


def test_summarize_horizon_reports_means_spread_and_profitable_folds():
    frame = _fold_frame([0.01, -0.01, 0.02, 0.03])

    summary = summarize_horizon(frame)

    assert summary["n_folds"] == 4
    assert summary["folds_profitable"] == 3
    assert summary["mean_return_confident_net"] == pytest.approx(0.0125)
    assert summary["mean_return_confident_net_std"] > 0


def test_summarize_horizon_ignores_folds_with_no_confident_rows():
    frame = _fold_frame([0.01, float("nan"), 0.02])

    summary = summarize_horizon(frame)

    assert summary["mean_return_confident_net"] == pytest.approx(0.015)
    assert summary["folds_profitable"] == 2


def test_paired_test_pairs_by_fold_and_detects_a_consistent_edge():
    rng = np.random.default_rng(0)
    base_vals = rng.normal(0.0, 0.001, size=10)
    baseline = _fold_frame(base_vals)
    candidate = _fold_frame(base_vals + 0.01)  # every fold better by exactly 1%

    result = paired_test(baseline, candidate, "mean_return_confident_net")

    assert result["n_pairs"] == 10
    assert result["mean_diff"] == pytest.approx(0.01)
    assert result["t_pvalue"] < 0.001
    assert result["wilcoxon_pvalue"] < 0.01


def test_paired_test_finds_no_signal_in_noise():
    rng = np.random.default_rng(1)
    baseline = _fold_frame(rng.normal(0.0, 0.01, size=10))
    candidate = _fold_frame(rng.normal(0.0, 0.01, size=10))

    result = paired_test(baseline, candidate, "mean_return_confident_net")

    assert result["t_pvalue"] > 0.05


def test_paired_test_drops_folds_missing_on_either_side():
    baseline = _fold_frame([0.01, 0.02, float("nan"), 0.01])
    candidate = _fold_frame([0.02, float("nan"), 0.03, 0.02])

    result = paired_test(baseline, candidate, "mean_return_confident_net")

    assert result["n_pairs"] == 2  # folds 0 and 3 only


def test_comparison_table_has_one_row_per_horizon_with_pvalues_vs_baseline():
    results = {
        5: _fold_frame([0.001] * 10),
        20: _fold_frame([0.004] * 10),
    }

    table = comparison_table(results, baseline_horizon=5)

    assert list(table["horizon_days"]) == [5, 20]
    row20 = table.loc[table["horizon_days"] == 20].iloc[0]
    assert row20["net_diff_vs_5d"] == pytest.approx(0.003)
    assert "net_pvalue" in table.columns
    baseline_row = table.loc[table["horizon_days"] == 5].iloc[0]
    assert pd.isna(baseline_row.get("net_diff_vs_5d"))


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
