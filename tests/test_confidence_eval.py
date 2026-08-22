"""
The confidence-filter evaluation harness has to be trustworthy before its
verdict on any filter is: these tests pin the mask definitions, the
per-fold summaries, and the paired-test plumbing on small synthetic frames
where the right answer is checkable by hand.
"""
import numpy as np
import pandas as pd
import pytest

from models.confidence_eval import (
    _per_fold_top_quantile,
    add_summary_columns,
    build_variants,
    collect_fold_predictions,
    evaluate,
    paired_comparison,
    summarize_variant,
)


def _prediction_frame():
    """
    Two folds x 10 rows. Members mostly agree on sign; magnitudes and
    spreads are chosen so each variant keeps a knowable subset.
    """
    rng = np.random.default_rng(7)
    n = 20
    base = rng.normal(scale=0.01, size=n)
    frame = pd.DataFrame(
        {
            "fold_id": [0] * 10 + [1] * 10,
            "symbol": [f"S{i}" for i in range(n)],
            "ts": pd.date_range("2026-01-01", periods=n),
            "fwd_return": rng.normal(scale=0.02, size=n),
            "member_0": base,
            "member_1": base + rng.normal(scale=0.001, size=n),
            "member_2": base + rng.normal(scale=0.001, size=n),
        }
    )
    return add_summary_columns(frame)


def test_add_summary_columns_matches_members():
    frame = _prediction_frame()
    members = frame[["member_0", "member_1", "member_2"]].to_numpy()
    np.testing.assert_allclose(frame["mean_prediction"], members.mean(axis=1))
    np.testing.assert_allclose(frame["std_prediction"], members.std(axis=1))
    assert (frame["direction_agreement"] >= 0.5).all()


def test_signal_to_noise_is_inf_when_members_identical():
    frame = pd.DataFrame(
        {
            "fold_id": [0],
            "fwd_return": [0.01],
            "member_0": [0.005],
            "member_1": [0.005],
        }
    )
    out = add_summary_columns(frame)
    assert np.isinf(out.loc[0, "signal_to_noise"])


def test_per_fold_top_quantile_selects_within_each_fold():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    fold_id = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])
    mask = _per_fold_top_quantile(values, fold_id, q=0.75)
    # Top 25% of each fold — the largest value per fold, not the largest
    # globally (which would put all four winners in fold 1).
    assert mask.tolist() == [False, False, False, True, False, False, False, True]


def test_per_fold_top_quantile_handles_inf():
    values = pd.Series([np.inf, 1.0, 2.0, 3.0])
    fold_id = pd.Series([0, 0, 0, 0])
    mask = _per_fold_top_quantile(values, fold_id, q=0.75)
    assert mask.tolist() == [True, False, False, False]


def test_build_variants_baseline_is_permissive_and_snr_is_selective():
    frame = _prediction_frame()
    variants = build_variants(cost=0.0002)
    baseline_pass = variants["agree80"](frame).mean()
    snr10_pass = variants["snr_top10"](frame).mean()
    assert baseline_pass > 0.5  # near-identical members almost always agree
    assert snr10_pass == pytest.approx(0.1, abs=0.06)  # top decile per fold


def test_summarize_variant_reports_per_fold_and_nan_on_empty_folds():
    frame = _prediction_frame()
    mask = pd.Series(False, index=frame.index)
    mask.iloc[:3] = True  # only fold 0 keeps rows
    table = summarize_variant(frame, mask, cost=0.0002)

    assert set(table["fold_id"]) == {0, 1}
    fold0 = table.set_index("fold_id").loc[0]
    fold1 = table.set_index("fold_id").loc[1]
    assert fold0["pct_pass"] == pytest.approx(0.3)
    assert np.isnan(fold1["acc_conf"]) and np.isnan(fold1["net_conf"])
    # Net is gross minus the round-trip cost, exactly.
    assert fold0["net_conf"] == pytest.approx(fold0["gross_conf"] - 0.0002)
    # A fold that kept no rows still reports what the market did, so an
    # empty variant can't read as "no information about this window".
    assert not np.isnan(fold1["benchmark"])


def test_summarize_variant_reports_the_benchmark_and_excess_for_every_fold():
    """No return column without the do-nothing baseline beside it."""
    frame = _prediction_frame()
    table = summarize_variant(frame, pd.Series(True, index=frame.index), cost=0.0002)

    for col in ("benchmark", "net_conf", "excess_conf", "pct_long", "long_net", "short_net"):
        assert col in table.columns
    np.testing.assert_allclose(
        table["excess_conf"].to_numpy(),
        (table["net_conf"] - table["benchmark"]).to_numpy(),
        atol=1e-12,
    )


def test_a_variant_that_only_keeps_longs_in_a_rally_shows_no_excess():
    """
    The trap the benchmark exists to catch: keeping the long half of a
    universe that is uniformly rising lifts net_conf while adding nothing.
    Every stock returns +2%, so beating the market is impossible and
    excess_conf must be negative by exactly the transaction cost.
    """
    frame = pd.DataFrame(
        {
            "fold_id": [0] * 4,
            "fwd_return": [0.02] * 4,
            "member_0": [0.01, 0.01, -0.01, -0.01],
            "member_1": [0.01, 0.01, -0.01, -0.01],
        }
    )
    frame = add_summary_columns(frame)
    longs_only = frame["mean_prediction"] > 0

    table = summarize_variant(frame, longs_only, cost=0.0002)

    assert table.loc[0, "net_conf"] == pytest.approx(0.02 - 0.0002)  # looks great
    assert table.loc[0, "benchmark"] == pytest.approx(0.02)
    assert table.loc[0, "excess_conf"] == pytest.approx(-0.0002)  # adds nothing
    assert table.loc[0, "pct_long"] == pytest.approx(1.0)


def test_summarize_variant_accuracy_is_sign_match():
    frame = pd.DataFrame(
        {
            "fold_id": [0, 0],
            "fwd_return": [0.01, -0.01],
            "member_0": [0.02, 0.02],
            "member_1": [0.02, 0.02],
        }
    )
    frame = add_summary_columns(frame)
    table = summarize_variant(frame, pd.Series([True, True]), cost=0.0)
    assert table.loc[0, "acc_all"] == pytest.approx(0.5)
    # Long both: +1% and -1% cancel out.
    assert table.loc[0, "gross_conf"] == pytest.approx(0.0)


def test_paired_comparison_detects_identical_and_shifted():
    folds = pd.DataFrame({"fold_id": range(6), "excess_conf": [0.01, 0.02, 0.0, -0.01, 0.03, 0.01]})
    same = paired_comparison(folds, folds)
    assert same["mean_diff"] == pytest.approx(0.0)
    assert same["wilcoxon_pvalue"] == pytest.approx(1.0)

    shifted = folds.copy()
    shifted["excess_conf"] += 0.005
    better = paired_comparison(shifted, folds)
    assert better["mean_diff"] == pytest.approx(0.005)
    assert better["t_pvalue"] < 0.01  # constant shift → tiny paired p


def test_paired_comparison_compares_excess_not_raw_return_by_default():
    """
    Two filters can differ in net_conf purely because one carries more
    market exposure. Excess is what isolates filter quality, so it is what
    the default comparison must use.
    """
    folds = pd.DataFrame(
        {
            "fold_id": range(4),
            "net_conf": [0.01, 0.01, 0.01, 0.01],
            "excess_conf": [-0.002, -0.002, -0.002, -0.002],
        }
    )
    better_net_worse_excess = pd.DataFrame(
        {
            "fold_id": range(4),
            "net_conf": [0.02, 0.02, 0.02, 0.02],  # twice the raw return...
            "excess_conf": [-0.004, -0.004, -0.004, -0.004],  # ...and further behind
        }
    )
    result = paired_comparison(better_net_worse_excess, folds)
    assert result["mean_diff"] == pytest.approx(-0.002)


def test_paired_comparison_drops_nan_folds_pairwise():
    a = pd.DataFrame({"fold_id": [0, 1, 2], "excess_conf": [0.01, np.nan, 0.03]})
    b = pd.DataFrame({"fold_id": [0, 1, 2], "excess_conf": [0.0, 0.0, 0.0]})
    result = paired_comparison(a, b)
    assert result["n_folds"] == 2


def test_evaluate_includes_all_diversity_variant_combos():
    frame = _prediction_frame()
    aggregate, fold_tables = evaluate({"seed": frame, "structural": frame}, cost=0.0002)
    n_variants = len(build_variants(0.0002))
    assert len(aggregate) == 2 * n_variants
    assert len(fold_tables) == 2 * n_variants
    baseline_row = aggregate[(aggregate["diversity"] == "seed") & (aggregate["variant"] == "agree80")]
    # The baseline compared to itself must show a zero difference.
    assert baseline_row["vs_baseline_mean_diff"].iloc[0] == pytest.approx(0.0)


def _training_frame(n_days=60, n_symbols=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=n_days)
    rows = []
    for s in range(n_symbols):
        f1 = rng.normal(size=n_days)
        rows.append(
            pd.DataFrame(
                {
                    "symbol": f"S{s}",
                    "ts": dates,
                    "f1": f1,
                    "f2": rng.normal(size=n_days),
                    "fwd_return": 0.01 * f1 + rng.normal(scale=0.005, size=n_days),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_collect_fold_predictions_walks_folds_without_leakage():
    df = _training_frame()
    preds = collect_fold_predictions(
        df, ["f1", "f2"], n_folds=3, purge_days=2, n_ensemble_models=2, diversity="seed"
    )
    assert {"fold_id", "symbol", "ts", "fwd_return", "member_0", "member_1"} <= set(preds.columns)
    assert preds["fold_id"].nunique() >= 2
    # Every test row lies strictly after the first 40% training reserve.
    dates = pd.DatetimeIndex(sorted(df["ts"].unique()))
    assert preds["ts"].min() >= dates[int(len(dates) * 0.4)]


def test_collect_fold_predictions_structural_runs():
    df = _training_frame()
    preds = collect_fold_predictions(
        df, ["f1", "f2"], n_folds=2, purge_days=2, n_ensemble_models=5, diversity="structural"
    )
    assert "member_4" in preds.columns
    assert len(preds) > 0
