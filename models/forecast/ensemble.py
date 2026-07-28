"""
Bagged ensemble around ForecastModel — a way to get a genuine confidence
signal out of an otherwise point-estimate regressor. Trains K models with
different bagging/feature-sampling seeds; a symbol only counts as
"confident" (see models/screener.py) if the members agree on direction and
clear a magnitude threshold, not just because one point prediction happens
to be large. A single LightGBM model can't tell you "how sure" it is —
disagreement across an ensemble is the honest proxy for that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.forecast.lgbm_forecast import ForecastModel


class EnsembleForecastModel:
    def __init__(
        self,
        n_models: int = 5,
        params: dict | None = None,
        num_boost_round: int = 200,
        base_seed: int = 42,
    ):
        self.n_models = n_models
        self.base_seed = base_seed
        self._params = params or {}
        self._num_boost_round = num_boost_round
        self.models: list[ForecastModel] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.models = []
        for i in range(self.n_models):
            seed = self.base_seed + i
            params = {
                **self._params,
                "bagging_seed": seed,
                "feature_fraction_seed": seed,
                "data_random_seed": seed,
            }
            model = ForecastModel(params=params, num_boost_round=self._num_boost_round)
            model.fit(X, y)
            self.models.append(model)

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame indexed like X with columns:
          mean_prediction    — average predicted forward return across members
          std_prediction     — spread across members (higher = less agreement)
          direction_agreement — fraction of members agreeing with the
                                 majority sign, in [0.5, 1.0]. 1.0 means every
                                 member predicts the same direction.
        """
        if not self.models:
            raise RuntimeError("Ensemble not trained yet — call fit() first.")

        preds = np.column_stack([m.predict(X) for m in self.models])  # (n_rows, n_models)
        signs = np.sign(preds)
        positive_frac = (signs > 0).mean(axis=1)
        agreement = np.maximum(positive_frac, 1 - positive_frac)

        return pd.DataFrame(
            {
                "mean_prediction": preds.mean(axis=1),
                "std_prediction": preds.std(axis=1),
                "direction_agreement": agreement,
            },
            index=X.index,
        )

    def feature_importance(self) -> pd.Series:
        """Average gain-based feature importance across ensemble members."""
        if not self.models:
            raise RuntimeError("Ensemble not trained yet.")
        importances = pd.concat([m.feature_importance() for m in self.models], axis=1)
        return importances.mean(axis=1).sort_values(ascending=False)
