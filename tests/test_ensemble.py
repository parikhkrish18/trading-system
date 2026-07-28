import numpy as np
import pandas as pd
import pytest

from models.forecast.ensemble import EnsembleForecastModel


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
