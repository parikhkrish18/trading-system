import numpy as np
import pandas as pd
import pytest

from models.forecast.ensemble import EnsembleForecastModel, recent_window_mask


def _synthetic_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "f3": rng.normal(size=n),
        }
    )
    y = 0.5 * X["f1"] - 0.3 * X["f2"] + rng.normal(scale=0.1, size=n)
    return X, y


def test_ensemble_trains_n_models():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=4, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)
    assert len(ensemble.models) == 4


def test_ensemble_predict_returns_expected_columns_and_index():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=3, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    result = ensemble.predict(X)

    assert list(result.columns) == ["mean_prediction", "std_prediction", "direction_agreement"]
    assert list(result.index) == list(X.index)
    assert len(result) == len(X)


def test_ensemble_direction_agreement_is_bounded():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=5, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    result = ensemble.predict(X)

    assert (result["direction_agreement"] >= 0.5).all()
    assert (result["direction_agreement"] <= 1.0).all()
    assert (result["std_prediction"] >= 0).all()


def test_ensemble_members_actually_differ_across_seeds():
    """
    If different seeds didn't change anything, this would just be an
    expensive way to train the same model N times — the whole point is
    getting genuine disagreement to measure.
    """
    X, y = _synthetic_data(n=300)
    ensemble = EnsembleForecastModel(n_models=5, num_boost_round=50, base_seed=1)
    ensemble.fit(X, y)

    result = ensemble.predict(X)
    assert result["std_prediction"].mean() > 0


def test_ensemble_predict_members_matches_predict_mean():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=3, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    members = ensemble.predict_members(X)
    summary = ensemble.predict(X)

    assert list(members.columns) == ["member_0", "member_1", "member_2"]
    np.testing.assert_allclose(members.mean(axis=1), summary["mean_prediction"])


def test_structural_diversity_varies_member_params():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=5, num_boost_round=20, base_seed=1, diversity="structural")
    ensemble.fit(X, y)

    num_leaves = {m.params["num_leaves"] for m in ensemble.models}
    feature_fractions = {m.params["feature_fraction"] for m in ensemble.models}
    assert len(num_leaves) > 1
    assert len(feature_fractions) > 1


def test_structural_diversity_disagrees_more_than_seed_only():
    """
    The entire reason "structural" exists: seed-only members are near
    clones, so their spread understates real model uncertainty.
    """
    X, y = _synthetic_data(n=400)
    ts = pd.Series(pd.date_range("2025-01-01", periods=len(X)))

    seed_only = EnsembleForecastModel(n_models=5, num_boost_round=50, base_seed=1)
    seed_only.fit(X, y)
    structural = EnsembleForecastModel(n_models=5, num_boost_round=50, base_seed=1, diversity="structural")
    structural.fit(X, y, ts=ts)

    assert structural.predict(X)["std_prediction"].mean() > seed_only.predict(X)["std_prediction"].mean()


def test_recent_window_mask_keeps_recent_dates_across_symbols():
    # Two symbols interleaved: date-based masking must keep the recent 40%
    # of *dates* for both symbols, not the tail 40% of rows (which would be
    # entirely the second symbol).
    dates = pd.date_range("2025-01-01", periods=10)
    ts = pd.Series(list(dates) + list(dates))  # symbol A rows then symbol B rows
    mask = recent_window_mask(ts, window_frac=0.4)

    assert mask.sum() == 8  # 4 recent dates x 2 symbols
    kept = ts[mask]
    assert kept.min() == dates[6]
    assert set(kept) == set(dates[6:])


def test_invalid_diversity_raises():
    with pytest.raises(ValueError, match="diversity"):
        EnsembleForecastModel(diversity="chaos")


def test_ensemble_predict_before_fit_raises():
    ensemble = EnsembleForecastModel()
    with pytest.raises(RuntimeError, match="not trained"):
        ensemble.predict(pd.DataFrame({"f1": [1.0]}))


def test_ensemble_feature_importance_averages_across_members():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=3, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    importances = ensemble.feature_importance()

    assert set(importances.index) == {"f1", "f2", "f3"}
    assert importances.iloc[0] >= importances.iloc[-1]  # sorted descending


def test_ensemble_feature_importance_before_fit_raises():
    ensemble = EnsembleForecastModel()
    with pytest.raises(RuntimeError, match="not trained"):
        ensemble.feature_importance()


def test_ensemble_predict_contributions_shape_and_columns():
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=3, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    contrib = ensemble.predict_contributions(X)

    assert list(contrib.columns) == ["f1", "f2", "f3", "base_value"]
    assert list(contrib.index) == list(X.index)


def test_ensemble_predict_contributions_sum_to_mean_prediction():
    """
    The core correctness property of Tree SHAP: contributions + base_value
    reconstruct the raw prediction exactly (up to floating point) — this is
    what makes "reasoning" honest rather than decorative.
    """
    X, y = _synthetic_data()
    ensemble = EnsembleForecastModel(n_models=3, num_boost_round=20, base_seed=1)
    ensemble.fit(X, y)

    contrib = ensemble.predict_contributions(X)
    preds = ensemble.predict(X)["mean_prediction"]

    reconstructed = contrib.sum(axis=1)
    assert (reconstructed - preds).abs().max() < 1e-6


def test_ensemble_predict_contributions_before_fit_raises():
    ensemble = EnsembleForecastModel()
    with pytest.raises(RuntimeError, match="not trained"):
        ensemble.predict_contributions(pd.DataFrame({"f1": [1.0]}))
